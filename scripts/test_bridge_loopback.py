#!/usr/bin/env python3
"""test_bridge_loopback.py - regression test for scripts/webgpu_bridge.py.

Starts the bridge on loopback (port fallback engaged automatically), drives
the full phone-attn contract through it with a Python fake tab:

    HELLO -> HELLO_OK (version 3) | CONFIG -> OK | APPEND -> OK + n_now
    ATTN (1 head, 64 keys) -> ATTN_OK + O f16 + lse (sizes + zero-row lse)
    ATTN_BIG (2 groups) -> ATTN_OK (grouped sizes) | STATS -> OK text
    PING -> OK bytes | TRUNCATE -> OK | wrong-token WS -> 403

The fake tab answers ATTN with zero O / -inf lse (contract shapes only;
math correctness is covered by pa-tool on hardware + the JS codec tests).
Exit 0 on pass, 1 with FAIL lines otherwise. Stdlib only.
"""
from __future__ import annotations

import base64
import json
import math
import re
import socket
import struct
import subprocess
import sys
import threading
import time

PATN = 0x4E544150
HELLO, HELLO_OK, CONFIG, APPEND, TRUNCATE, ATTN = 1, 2, 3, 4, 5, 6
ATTN_OK, STATS, OK, ERR, BYE, PING, ATTN_BIG = 7, 8, 9, 10, 11, 12, 13
NR, HD = 48, 256

fails = []


def ok(cond, name):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        fails.append(name)


def recv_exact(s, n):
    data = b""
    while len(data) < n:
        chunk = s.recv(n - len(data))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return data


def send_msg(s, typ, payload=b""):
    s.sendall(struct.pack("<IIQ", PATN, typ, len(payload)) + payload)


def recv_msg(s):
    h = recv_exact(s, 16)
    magic, typ, ln = struct.unpack("<IIQ", h)
    assert magic == PATN, f"bad magic {magic:#x}"
    return typ, recv_exact(s, ln) if ln else b""


# ---------------- fake tab (mirrors web/webgpu-phone/app.js handling) ----------------

def fake_tab(port, token, ready):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    key = base64.b64encode(b"0" * 16).decode()
    s.sendall(f"GET /?token={token} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
              f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
              "Sec-WebSocket-Version: 13\r\n\r\n".encode())
    if b"101" not in s.recv(1024):
        raise RuntimeError("tab handshake failed")
    ready.append(True)
    held = {}
    cfg = {}

    def ws_send(payload):
        m = b"\xaa\xbb\xcc\xdd"
        n = len(payload)
        h = bytes([0x82]) + (bytes([0x80 | n]) if n < 126
                             else bytes([0x80 | 126]) + struct.pack(">H", n))
        s.sendall(h + m + bytes(p ^ m[i % 4] for i, p in enumerate(payload)))

    def ws_recv():
        h = recv_exact(s, 2)
        ln = h[1] & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", recv_exact(s, 2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", recv_exact(s, 8))[0]
        data = recv_exact(s, ln)
        return h[0] & 0x0F, data

    def handle(typ, pay):
        if typ == HELLO:
            return HELLO_OK, struct.pack("<II", 3, 0) + b"Z" * 64
        if typ == CONFIG:
            n_layer, nkv, rs = struct.unpack("<III", pay[:12])
            hb = struct.unpack("<I", pay[12:16])[0]
            cfg.update(n_layer=n_layer, nkv=nkv, rs=rs, hb=hb,
                       is_q8=struct.unpack("<I", pay[16:20])[0])
            held.clear()
            for l in range(n_layer):
                held[l] = 0
            return OK, b""
        if typ == APPEND:
            layer, pos0, n = struct.unpack("<III", pay[:12])
            assert pos0 == held[layer], "pos guard"
            held[layer] += n
            return OK, struct.pack("<I", held[layer])
        if typ == TRUNCATE:
            (n,) = struct.unpack("<I", pay[:4])
            for l in held:
                held[l] = min(held[l], n)
            return OK, b""
        if typ in (ATTN, ATTN_BIG):
            layer, n_tok, nk = struct.unpack("<III", pay[:12])
            scale = struct.unpack("<f", pay[12:16])[0]
            assert scale > 0
            nkv = cfg["nkv"]
            qn = nkv * NR * HD
            ng = (n_tok + 7) // 8 if typ == ATTN_BIG else 1
            assert len(pay) == 16 + ng * qn * 2, (len(pay), ng, qn)
            eff = nk or held[layer]
            rep = struct.pack("<IfffII", eff, 1.0, 1.0, 0.0, 0, 1)
            o = b"\x00" * (ng * qn * 2)
            lse = struct.pack("<%df" % (ng * nkv * NR), *([-float("inf")] * ng * nkv * NR))
            return ATTN_OK, rep + o + lse
        if typ == STATS:
            return OK, b"state=idle attn_calls=1"
        if typ == PING:
            (n,) = struct.unpack("<I", pay[:4])
            return OK, b"\x5a" * n
        return ERR, b"unknown"

    while True:
        op, frame = ws_recv()
        if op != 0x2:
            continue
        magic, typ, ln = struct.unpack("<IIQ", frame[:16])
        assert magic == PATN
        assert len(frame) == 16 + ln
        rtyp, rpay = handle(typ, frame[16:])
        ws_send(struct.pack("<IIQ", PATN, rtyp, len(rpay)) + rpay)


def main():
    token = "loopback-test-token"
    br = subprocess.Popen(
        [sys.executable, "scripts/webgpu_bridge.py", "--listen", "127.0.0.1",
         "--tcp-port", "50062", "--ws-port", "50063", "--cmd-port", "50061",
         "--token", token, "--avail-mb", "2000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tcp_p = ws_p = cmd_p = None
    try:
        # read the machine-readable ports line (fallback-proof)
        start = time.time()
        while time.time() - start < 30 and (tcp_p is None or ws_p is None or cmd_p is None):
            line = br.stdout.readline()
            m = re.search(r"ports tcp=(\d+) ws=(\d+) cmd=(\d+)", line)
            if m:
                tcp_p, ws_p, cmd_p = int(m[1]), int(m[2]), int(m[3])
        ok(tcp_p and ws_p and cmd_p and len({tcp_p, ws_p, cmd_p}) == 3,
           f"distinct ports tcp={tcp_p} ws={ws_p} cmd={cmd_p}")

        # wrong token rejected
        b = socket.create_connection(("127.0.0.1", ws_p), timeout=5)
        b.sendall(b"GET /?token=nope HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
                  b"Connection: Upgrade\r\nSec-WebSocket-Key: eA==\r\n"
                  b"Sec-WebSocket-Version: 13\r\n\r\n")
        ok(b"403" in b.recv(256), "wrong-token 403")
        b.close()

        # mem sizing
        c = socket.create_connection(("127.0.0.1", cmd_p), timeout=5)
        c.sendall(b"mem\n")
        m = json.loads(c.makefile().readline())
        c.close()
        ok(m["avail_mb"] == 2000 and m["pa_ane"] is False, "mem avail/pa_ane")

        ready = []
        threading.Thread(target=fake_tab, args=(ws_p, token, ready), daemon=True).start()
        start = time.time()
        while not ready and time.time() - start < 15:
            time.sleep(0.1)
        ok(bool(ready), "tab connected")

        t = socket.create_connection(("127.0.0.1", tcp_p), timeout=20)
        send_msg(t, HELLO)
        typ, pay = recv_msg(t)
        v, _sme2 = struct.unpack("<II", pay[:8])
        ok(typ == HELLO_OK and v == 3, "HELLO version 3")

        nkv, nk = 1, 64
        rs = hb = 272  # q8_0 rows, 1 head
        send_msg(t, CONFIG, struct.pack("<11I", 1, nkv, rs, hb, 1, 0, 0, 0, 1024, 0, 0))
        ok(recv_msg(t)[0] == OK, "CONFIG ok")

        kv = b"\x07" * (2 * nk * rs)
        send_msg(t, APPEND, struct.pack("<III", 0, 0, nk) + kv)
        typ, pay = recv_msg(t)
        ok(typ == OK and struct.unpack("<I", pay)[0] == nk, "APPEND n_now")

        qn = nkv * NR * HD
        send_msg(t, ATTN, struct.pack("<IIIf", 0, 8, 0, 0.0625) + b"\x00" * (qn * 2))
        typ, pay = recv_msg(t)
        eff_nk, phone_ms = struct.unpack("<If", pay[:8])
        o_len = qn * 2
        lse_len = (qn // HD) * 4
        ok(typ == ATTN_OK and len(pay) == 24 + o_len + lse_len, "ATTN sizes")
        lse = struct.unpack("<%df" % (qn // HD), pay[24 + o_len:])
        ok(all(v == -float("inf") for v in lse), "ATTN zero-Q lse -inf")

        send_msg(t, ATTN_BIG, struct.pack("<IIIf", 0, 16, 0, 0.0625) + b"\x02" * (2 * qn * 2))
        typ, pay = recv_msg(t)
        ok(typ == ATTN_OK and len(pay) == 24 + 2 * o_len + 2 * lse_len, "ATTN_BIG sizes")

        send_msg(t, STATS)
        ok(recv_msg(t)[0] == OK, "STATS ok")
        send_msg(t, PING, struct.pack("<I", 7) + b"\x00" * 7)
        typ, pay = recv_msg(t)
        ok(typ == OK and pay == b"\x5a" * 7, "PING echo")
        send_msg(t, TRUNCATE, struct.pack("<I", 32))
        ok(recv_msg(t)[0] == OK, "TRUNCATE ok")
        send_msg(t, BYE)
        t.close()
    finally:
        br.terminate()
    print("RESULT:", "ALL LOOPBACK TESTS PASS" if not fails else fails, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
