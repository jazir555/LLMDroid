# Wireless WebGPU node (no cable)

Runs the phone-held attention job in a browser tab over Wi-Fi/LAN instead of a
USB cable, using the real [`@huggingface/kernels`](https://huggingface.co/blog/webgpu-kernels)
WebGPU kernels (MatMul + Softmax) for the math. Status: **v1 — phone-attn only**.

## How it fits

```
tab (WebGPU: MatMul, Softmax) <--ws 50063--> webgpu_bridge.py <--tcp 50062--> host
         ^-- :50061 mem/mac served by the bridge locally --^
```

- The tab speaks the exact `phone-attn/phone-attn.h` framing (`PATN` magic,
  HELLO/CONFIG/APPEND/TRUNCATE/ATTN/ATTN_BIG/STATS/PING/BYE) as WebSocket
  binary frames. The bridge (`scripts/webgpu_bridge.py`, stdlib only) is a dumb
  authenticated pipe, so the host treats the tab like any other phone-attn
  server: `PHONE_KV=<bridge>:50062`, no serve changes.
- Per KV head, per 8-token group: `S = (Q*scale) @ K^T` (MatMul `[48,256] x
  [256,ck]`), `P = softmax(S)` (Softmax), `O = P @ V` (MatMul) — the blog's
  Demo 01 transformer-attention recipe. `lse` comes from the CPU copy of `S`
  (the kernel returns probs only); key chunks merge with the same log-sum-exp
  math as `pa::merge_partial`, so chunking changes speed, not answers.
- K/V dequant (q8_0/q4_0/f16 → f16 storage) mirrors `pa-tool.cpp`; zero-padded
  Q rows (`t >= n_tok`) are skipped with `O=0, lse=-inf`, matching the server.
- No WebGPU → exact CPU fallback (slow, correct). No SME2 in a browser (HELLO
  reports `sme2=0`, honestly).

## Run it

```bash
# 0. host bridge (token printed if not given)
python3 scripts/webgpu_bridge.py --listen 0.0.0.0 --token SECRET
#    TCP :50062 (host) / WS :50063 (tab) / CMD :50061 (mem+mac notes)

# 1. serve the tab (secure context required for WebGPU: localhost is fine)
python3 -m http.server 8000 --directory web/webgpu-phone
#    open http://<host-lan>:8000 from the wireless machine, enter
#    ws://<host-lan>:50063 + token, Connect

# 2. probe (same line contract as phone-up.sh)
python3 scripts/phone_up_webgpu.py --host 127.0.0.1
#    127.0.0.1 0 <ver> <avail> <wired> webgpu-wireless

# 3. serve (no split tail wirelessly: TAIL 0, attention only)
PHONE_KV=127.0.0.1:50062 scripts/serve.ps1        # Windows
PHONE_KV=127.0.0.1:50062 scripts/serve.sh        # Mac
#    or auto-detect the tab on Windows: scripts\serve.ps1 -Webgpu
#    (WEBGPU_HOST/TCP_PORT/CMD_PORT env match a floated bridge; see ports)
```

## Ports (Windows)

Hyper-V reserves ranges like 50060–50159 (`netsh int ipv4 show
excludedportrange protocol=tcp`), so the classic phone ports may refuse to
bind. The bridge tries `--tcp-port`/`--ws-port`/`--cmd-port` upward until each
binds and prints a machine-readable line:

```
webgpu-bridge: ports tcp=50361 ws=50360 cmd=50362
```

Point the tab, the probe, and serve at the actual ports:
`ws://host:50360`, `phone_up_webgpu.py --tcp-port 50361 --cmd-port 50362`,
`$env:WEBGPU_TCP_PORT='50361'; $env:WEBGPU_CMD_PORT='50362'; scripts\serve.ps1 -Webgpu`.
Regression: `python3 scripts/test_bridge_loopback.py` (starts its own bridge,
drives HELLO→ATTN_BIG through a fake tab, checks the 403/mem/sizes).

## Tab options (URL params)

- `?cap=N` — keys/layer KV cap, 1024–65536 (default 8192). f16 store ≈
  `nkv·512 B`/key/layer; size the tab before serve sizes `CTX_TOTAL`.
- `?rev=pin` — freeze both kernels to the `KERNELS.md` revisions instead of the
  moving `version: 1` (or `?matmul_rev=<40-char>` / `?softmax_rev=` singly).
- The tab takes a screen Wake Lock while connected and reconnects immediately
  when returning to visible (background throttling is what stalls ATTN and
  forces the host's 60 s Mac-only fallback).

`--avail-mb` (default 2048) is what the bridge reports for `mem`, so serve
sizes `CTX_TOTAL` from the tab's budget the same way it sizes from a phone's
(`docs/ANDROID.md`). Tab KV cap is 8192 keys/layer (f16 store, ~0.5 GB at
`nkv=4`); beyond that APPEND answers `ERR out of memory` and the host sizes
down, like the iOS jetsam guard.

## Security

LAN-only by default (`--listen 127.0.0.1`); `--listen 0.0.0.0` binds Wi-Fi, so
always set `--token` (a random one prints when omitted), keep the token off
screenshots, and prefer `wss://` via a reverse proxy for anything beyond a
trusted LAN. One tab and one host client at a time; extras are rejected.

## Multiple tabs (same device)

Opening the page twice on one machine used to wedge the bridge: both tabs
auto-connected, the first held the slot, and the second hammered it with
rejected retries. Mitigations, both live:

- **Page (same device):** tabs elect one leader over `BroadcastChannel` + a
  `localStorage` heartbeat lock. Only the leader opens the socket; the rest
  show `standby` and take over — with a host key re-send — if the leader goes
  stale (>6 s) or announces release on unload. No more duplicate connections
  from one device.
- **Bridge (across devices / ghosts):** a newcomer with a valid token triggers
  a 5 s ping-challenge of the holder. A live tab pongs and keeps its slot
  (newcomer gets 1013); a network-switched ghost stays silent and is evicted.
  Same-IP re-challenges are rate-limited to one per 30 s.

## Limits (v1, honest)

- **Attention only.** Split prefill (`:50060` tail, 24 layers) is not served
  wirelessly — that needs full-layer execution, not attention primitives.
  `fetch` on `:50061` answers an error saying so.
- **No ANE/page engine** (`pa_ane:false`); every key runs the
  MatMul→Softmax→MatMul path (or CPU).
- **Throughput is link-bound.** USB does ~GB/s with sub-ms gaps; Wi-Fi adds
  milliseconds per layer per token. Expect the 140k-class decode help to shrink
  and prefill `ATTN_BIG` calls (MBs per layer per ubatch) to be the bottleneck.
  Best wireless use: mid-context sessions where the tab adds memory (more total
  context), not speed. Measure with `bench/long-bench.py` before trusting it.
- **Tab must stay alive.** Backgrounded/throttled tabs stall ATTN; the host's
  60 s phone-failure path then drops to Mac-only, same as a locked phone.
- Kernel pins: `web/webgpu-phone/KERNELS.md` (MatMul/Softmax/Add v1 revisions,
  vendored manifests + Add `test.json`). The tab loads `version: 1` at runtime
  via `getKernel`; pass a 40-char revision in `attention.js` to freeze bytes.
