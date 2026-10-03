#!/usr/bin/env python3
"""test_webgpu_live.py - live accuracy + timing test against the real tab.

Drives the full phone-attn contract through the bridge to a connected
browser tab (WebGPU kernels on real phone hardware):

    HELLO -> CONFIG -> APPEND -> ATTN -> ATTN_BIG -> STATS

then checks the returned O/lse against a numpy reference computed here.
Same accuracy gate as pa-tool.cpp: max|O-ref|/max|ref| < 5e-3.

Usage:
    python3 scripts/test_webgpu_live.py --tcp-port 50361 --cmd-port 50362
    python3 scripts/test_webgpu_live.py --is-q8 1 --nk 1024 --seed 7

Needs a tab connected to the bridge (it answers HELLO with version 3 and a
'webgpu-phone' device string; the script refuses fakes).
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys

import numpy as np

PATN = 0x4E544150
HELLO, CONFIG, APPEND, TRUNCATE, ATTN = 1, 3, 4, 5, 6
ATTN_OK, STATS, OK, PING, ATTN_BIG = 7, 8, 9, 12, 13
NR, HD = 48, 256


def recv_exact(s, n):
    data = b""
    while len(data) < n:
        chunk = s.recv(n - len(data))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return data


def call(s, typ, payload=b""):
    s.sendall(struct.pack("<IIQ", PATN, typ, len(payload)) + payload)
    h = recv_exact(s, 16)
    magic, rtyp, ln = struct.unpack("<IIQ", h)
    assert magic == PATN, f"bad magic {magic:#x}"
    return rtyp, recv_exact(s, ln) if ln else b""


def f16(x):
    return np.asarray(x, dtype=np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tcp-port", type=int, default=50062)
    ap.add_argument("--nkv", type=int, default=1)
    ap.add_argument("--nk", type=int, default=1024)
    ap.add_argument("--is-q8", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    nkv, nk = args.nkv, args.nk
    assert nk % 64 == 0, "server contract: nk multiple of 64"

    s = socket.create_connection((args.host, args.tcp_port), timeout=120)
    fails = []

    def ok(cond, name):
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        if not cond:
            fails.append(name)

    # HELLO: must be the real tab, not a fake
    typ, pay = call(s, HELLO)
    ver, sme2 = struct.unpack("<II", pay[:8])
    dev = pay[8:72].split(b"\x00")[0].decode("latin1", "replace")
    ok(typ == 2 and ver == 3, f"HELLO v3 (got {ver})")
    ok(dev.startswith("webgpu-phone"), f"real tab ({dev[:60]})")
    ok(sme2 == 0, "sme2=0 honest (browser)")

    # CONFIG: 2 layers, test layer 1 (non-zero index exercises indexing)
    rs = {0: nkv * HD * 2, 1: nkv * 272, 2: nkv * 144}[args.is_q8]
    hb = {0: HD * 2, 1: 272, 2: 144}[args.is_q8]
    typ, _ = call(s, CONFIG, struct.pack("<11I", 2, nkv, rs, hb, args.is_q8, 0, 0, 0, 1024, 0, 0))
    ok(typ == OK, "CONFIG ok")

    # keys: fp32 master, wire bytes per is_q8
    Kf = rng.normal(0, 0.5, (nk, nkv, HD)).astype(np.float32)
    Vf = rng.normal(0, 0.5, (nk, nkv, HD)).astype(np.float32)

    def enc_row(mat):
        if args.is_q8 == 0:
            return f16(mat).tobytes()
        out = bytearray()
        flat = mat.reshape(nkv, -1)
        if args.is_q8 == 1:
            for h in range(nkv):
                for b in range(8):
                    grp = flat[h, b * 32:(b + 1) * 32]
                    amax = float(np.max(np.abs(grp))) or 1e-9
                    d = amax / 127.0
                    out += struct.pack("<e", np.float16(d))
                    out += np.clip(np.round(grp / d), -127, 127).astype(np.int8).tobytes()
        else:
            for h in range(nkv):
                for b in range(8):
                    grp = flat[h, b * 32:(b + 1) * 32]
                    amax = float(np.max(np.abs(grp))) or 1e-9
                    d = amax / 7.0
                    out += struct.pack("<e", np.float16(d))
                    qs = np.clip(np.round(grp / d), -8, 7).astype(np.int8) + 8
                    lo = qs[0:32:2] & 15
                    hi = qs[1:32:2] & 15
                    out += ((np.asarray(lo) | (np.asarray(hi) << 4)) & 255).astype(np.uint8).tobytes()
        return bytes(out)

    kpart = b"".join(enc_row(Kf[j]) for j in range(nk))
    vpart = b"".join(enc_row(Vf[j]) for j in range(nk))
    typ, pay = call(s, APPEND, struct.pack("<III", 1, 0, nk) + kpart + vpart)
    ok(typ == OK and struct.unpack("<I", pay)[0] == nk, f"APPEND {nk} keys")

    # reference uses the DEQUANTIZED values (what the tab actually stored)
    def dec_row(data, off):
        mat = np.zeros((nkv, HD), np.float32)
        if args.is_q8 == 0:
            mat[:] = np.frombuffer(data[off:off + nkv * HD * 2], dtype=np.float16).reshape(nkv, HD)
            return mat, off + nkv * HD * 2
        for h in range(nkv):
            if args.is_q8 == 1:
                for b in range(8):
                    (d,) = struct.unpack("<e", data[off:off + 2])
                    qs = np.frombuffer(data[off + 2:off + 34], dtype=np.int8).astype(np.float32)
                    mat[h, b * 32:(b + 1) * 32] = float(d) * qs
                    off += 34
            else:
                for b in range(8):
                    (d,) = struct.unpack("<e", data[off:off + 2])
                    qb = np.frombuffer(data[off + 2:off + 18], dtype=np.uint8)
                    qs = np.empty(32, np.float32)
                    qs[0::2] = (qb & 15).astype(np.float32) - 8
                    qs[1::2] = ((qb >> 4) & 15).astype(np.float32) - 8
                    mat[h, b * 32:(b + 1) * 32] = float(d) * qs
                    off += 18
        return mat, off

    Kd = np.zeros_like(Kf)
    off = 0
    for j in range(nk):
        Kd[j], off = dec_row(kpart, off)
    Vd = np.zeros_like(Vf)
    off = 0
    for j in range(nk):
        Vd[j], off = dec_row(vpart, off)

    # ATTN: n_tok=8 -> all 48 rows live; scale like a real head
    scale = 1.0 / 16.0
    Qf = rng.normal(0, 0.5, (nkv, NR, HD)).astype(np.float32)
    qn = nkv * NR * HD
    typ, pay = call(s, ATTN, struct.pack("<IIIf", 1, 8, 0, scale) + f16(Qf).tobytes())
    ok(typ == ATTN_OK, "ATTN ok")
    rnk, phone_ms, gpu_ms, sme_ms, gpu_pages, pages = struct.unpack("<IfffII", pay[:24])
    O = np.frombuffer(pay[24:24 + qn * 2], dtype=np.float16).astype(np.float32).reshape(nkv, NR, HD)
    lse = np.frombuffer(pay[24 + qn * 2:], dtype=np.float32).reshape(nkv, NR)
    print(f"  phone {phone_ms:.1f} ms, gpu {gpu_ms:.1f} ms, nk={rnk}", flush=True)

    worst_o, worst_l = 0.0, 0.0
    for h in range(nkv):
        scores = (Qf[h] * scale) @ Kd[:, h, :].T  # [48, nk]
        m = scores.max(axis=1, keepdims=True)
        l = np.exp(scores - m).sum(axis=1, keepdims=True)
        P = np.exp(scores - m) / l
        Oref = P @ Vd[:, h, :]
        lseref = (m[:, 0] + np.log(l[:, 0]))
        denom = max(float(np.max(np.abs(Oref))), 1e-9)
        worst_o = max(worst_o, float(np.max(np.abs(O[h] - Oref))) / denom)
        worst_l = max(worst_l, float(np.max(np.abs(lse[h] - lseref))))
    print(f"  max|O-ref|/max|ref| = {worst_o:.2e}, max|lse-ref| = {worst_l:.2e}", flush=True)
    ok(worst_o < 5e-3, "O accuracy gate 5e-3")
    ok(worst_l < 5e-2, "lse accuracy gate 5e-2")

    # ATTN_BIG: n_tok=16 -> 2 groups; zero-pad group rows past n_tok? here all live
    Qb = rng.normal(0, 0.5, (2, nkv, NR, HD)).astype(np.float32)
    typ, pay = call(s, ATTN_BIG, struct.pack("<IIIf", 1, 16, 0, scale) + f16(Qb).tobytes())
    ok(typ == ATTN_OK and len(pay) == 24 + 2 * (qn * 2 + (qn // HD) * 4), "ATTN_BIG sizes")
    rnk2, bms = struct.unpack("<If", pay[:8])
    print(f"  big phone {bms:.1f} ms", flush=True)

    typ, pay = call(s, STATS)
    ok(typ == OK, f"STATS ({pay.decode('latin1', 'replace')[:80]})")
    typ, pay = call(s, TRUNCATE, struct.pack("<I", 0))
    ok(typ == OK, "TRUNCATE ok")
    s.close()
    print("RESULT:", "ALL LIVE TESTS PASS" if not fails else fails, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
