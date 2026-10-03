/* app.js - wireless WebGPU node runtime.
 *
 * Connects to the host bridge over WebSocket (binary frames carry the exact
 * PATN bytes from phone-attn.h, so the bridge is a dumb pipe). Strict
 * request -> reply, one at a time; messages may arrive coalesced or split,
 * hence the reassembly buffer. Query: ?token= must match the bridge token.
 */

import {
  MAGIC, VERSION, NR, HD, T, encodeMsg, decodeHdr, errMsg, okMsg,
  rowBytes, KVStore,
} from './phone-attn.js';
import { ensureKernels, kernelStatus, computeAttn } from './attention.js';

const $ = (id) => document.getElementById(id);
const logEl = () => $('log');
export function log(s) {
  const el = logEl();
  const line = `${new Date().toLocaleTimeString()} ${s}`;
  el.textContent = (line + '\n' + el.textContent).slice(0, 8000);
}

// ?cap=N overrides the keys/layer cap (default 8192, clamp 1024..65536).
// f16 store ~= nkv*512B/key/layer, so size the tab before serve sizes CTX_TOTAL.
function kvCap() {
  const raw = parseInt(new URLSearchParams(location.search).get('cap') || '', 10);
  if (Number.isFinite(raw)) return Math.min(65536, Math.max(1024, raw));
  return 8192;
}

const store = new KVStore(kvCap());
const stats = { calls: 0, lastMs: 0, gpuMs: 0, startMs: 0 };
let ws = null;
let rxBuf = new Uint8Array(0);
let macPhase = '';
let wakeLock = null;

async function holdWakeLock() {
  // Screen Wake Lock keeps the tab's GPU timers alive while held keys exist;
  // without it a backgrounded tab stalls ATTN and the host drops to Mac-only.
  if (!('wakeLock' in navigator)) return;
  try {
    wakeLock = await navigator.wakeLock.request('screen');
    wakeLock.addEventListener('release', () => { wakeLock = null; });
  } catch (_) {
    wakeLock = null; // denied: host-side 60 s fallback still covers us
  }
}

function heldKeys() {
  return store.layers.length ? store.layers[0].n : 0;
}

function refresh() {
  $('held').textContent = heldKeys().toLocaleString();
  $('calls').textContent = String(stats.calls);
  $('lastms').textContent = stats.lastMs ? stats.lastMs.toFixed(1) + ' ms' : '-';
  $('kernels').textContent = kernelStatus();
  $('phase').textContent = macPhase || '-';
}

function send(bytes) {
  ws.send(bytes);
}

async function handleMessage(type, payload) {
  const dv = new DataView(payload.buffer, payload.byteOffset);
  switch (type) {
    case T.HELLO: {
      const rep = new Uint8Array(4 + 4 + 64);
      const rdv = new DataView(rep.buffer);
      rdv.setUint32(0, VERSION, true);
      rdv.setUint32(4, 0, true); // sme2: browser has none; honest
      const dev = `webgpu-phone kernels=matmul,softmax ${navigator.userAgent.slice(0, 40)}`;
      new TextEncoder().encodeInto(dev, new Uint8Array(rep.buffer, 8, 64));
      send(encodeMsg(T.HELLO_OK, rep));
      break;
    }
    case T.CONFIG: {
      if (payload.length < 44) { send(errMsg('short CONFIG')); break; }
      const cfg = {
        n_layer: dv.getUint32(0, true), n_head_kv: dv.getUint32(4, true),
        rs: dv.getUint32(8, true), hb: dv.getUint32(12, true),
        is_q8: dv.getUint32(16, true), sme_workers: dv.getUint32(20, true),
        sme_helpers: dv.getUint32(24, true), gpu_permille: dv.getUint32(28, true),
        gpu_chunk: dv.getUint32(32, true), store_f16: dv.getUint32(36, true),
        gpu_variant: dv.getUint32(40, true),
      };
      if (cfg.rs < cfg.n_head_kv * cfg.hb) { send(errMsg('bad CONFIG')); break; }
      const bad = store.configure(cfg);
      if (bad) { send(errMsg(bad)); break; }
      stats.calls = 0;
      log(`CONFIG: ${cfg.n_layer} layers x ${cfg.n_head_kv} heads, ${['f16', 'q8_0', 'q4_0'][cfg.is_q8] || '?'}`);
      send(okMsg());
      break;
    }
    case T.APPEND: {
      if (payload.length < 12) { send(errMsg('short APPEND')); break; }
      const layer = dv.getUint32(0, true), pos0 = dv.getUint32(4, true), n = dv.getUint32(8, true);
      const rs = store.cfg ? store.cfg.rs : 0;
      if (!store.cfg || payload.length !== 12 + 2 * n * rs) { send(errMsg('bad APPEND')); break; }
      const r = store.append(layer, pos0, n, payload.subarray(12));
      if (r.err) send(errMsg(r.err));
      else send(okMsg(r.ok));
      break;
    }
    case T.TRUNCATE: {
      const n = payload.length >= 4 ? dv.getUint32(0, true) : 0;
      store.truncate(n);
      send(okMsg());
      break;
    }
    case T.ATTN:
    case T.ATTN_BIG: {
      const big = type === T.ATTN_BIG;
      if (payload.length < 12 || !store.cfg) { send(errMsg('bad ATTN')); break; }
      const layer = dv.getUint32(0, true), nTok = dv.getUint32(4, true);
      const nk = dv.getUint32(8, true), scale = dv.getFloat32(12, true);
      const nkv = store.cfg.n_head_kv;
      const qn = nkv * NR * HD;
      const ng = big ? Math.ceil(nTok / 8) : 1;
      if (ng < 1 || ng > 64 || payload.length !== 16 + ng * qn * 2) { send(errMsg('bad ATTN')); break; }
      if (nk > store.layers[layer].n || nk % 1 !== 0) { send(errMsg(`ATTN nk ${nk}`)); break; }
      const Qw = new Uint16Array(payload.buffer, payload.byteOffset + 16, ng * qn);
      const t0 = performance.now();
      const { Owords, lse, gpu } = await computeAttn(store, layer, nTok, nk, scale, Qw, ng);
      const ms = performance.now() - t0;
      stats.calls++; stats.lastMs = ms; stats.gpuMs = gpu ? ms : 0;
      const effNk = nk || store.layers[layer].n;
      const pages = Math.ceil(effNk / 4096);
      // attn_rep: u32 nk; float phone_ms, gpu_ms, sme_ms; u32 gpu_pages, pages
      const rep = new Uint8Array(24);
      const rdv = new DataView(rep.buffer);
      rdv.setUint32(0, effNk, true);
      rdv.setFloat32(4, ms, true);
      rdv.setFloat32(8, gpu ? ms : 0, true);
      rdv.setFloat32(12, 0, true);
      rdv.setUint32(16, 0, true); // gpu_pages: keys run on the WebGPU path
      rdv.setUint32(20, pages, true);
      const head = encodeMsg(T.ATTN_OK, rep);
      const out = new Uint8Array(head.length + Owords.byteLength + lse.byteLength);
      out.set(head, 0);
      out.set(new Uint8Array(Owords.buffer, Owords.byteOffset, Owords.byteLength), head.length);
      out.set(new Uint8Array(lse.buffer, lse.byteOffset, lse.byteLength), head.length + Owords.byteLength);
      send(out);
      break;
    }
    case T.STATS: {
      const s = `state=idle attn_calls=${stats.calls} last_ms=${stats.lastMs.toFixed(3)} held=${heldKeys()} appended=${store.appended} ${kernelStatus()}`;
      send(okMsg(new TextEncoder().encode(s)));
      break;
    }
    case T.PING: {
      let n = 0;
      if (payload.length >= 4) n = dv.getUint32(0, true);
      send(okMsg(new Uint8Array(n).fill(0x5a)));
      break;
    }
    case T.BYE:
      ws.close();
      break;
    default:
      send(errMsg(`unknown message ${type}`));
  }
  refresh();
}

function onData(chunk) {
  const merged = new Uint8Array(rxBuf.length + chunk.length);
  merged.set(rxBuf, 0); merged.set(chunk, rxBuf.length);
  rxBuf = merged;
  // process complete messages in order (strict request -> reply preserved)
  const pump = async () => {
    while (rxBuf.length >= 16) {
      const h = decodeHdr(rxBuf, 0);
      if (h.magic !== MAGIC) { log('bad magic, closing'); ws.close(); return; }
      if (h.len > (1 << 31)) { log('oversize message, closing'); ws.close(); return; }
      if (rxBuf.length < 16 + h.len) return; // wait for more
      const type = h.type;
      const payload = rxBuf.slice(16, 16 + h.len);
      rxBuf = rxBuf.slice(16 + h.len);
      try {
        await handleMessage(type, payload);
      } catch (e) {
        try { send(errMsg(String((e && e.message) || e))); } catch (_) {}
      }
    }
  };
  pump();
}

/* ---- single-flight per device: only the leader tab holds the bridge slot.
 * Tabs coordinate over BroadcastChannel + a localStorage heartbeat lock, so
 * opening the page twice on one machine no longer wedges the bridge with a
 * 1013 retry storm. Standbys show their state and take over (with a host
 * key re-send) if the leader goes stale. Cross-device contention (phone +
 * PC) is still first-wins at the bridge — this only covers tabs on one device. */
const TAB_ID = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random())).slice(0, 8);
const LEADER_KEY = 'webgpu-phone-leader';
const LEADER_TTL_MS = 6000;
const leaderBus = ('BroadcastChannel' in window) ? new BroadcastChannel('webgpu-phone') : null;
let isLeader = false;

function readLock() {
  try {
    const raw = localStorage.getItem(LEADER_KEY);
    if (!raw) return null;
    const o = JSON.parse(raw);
    if (!o || typeof o.ts !== 'number') return null;
    return o;
  } catch (_) {
    return null;
  }
}

function claimLeadership() {
  const lock = { id: TAB_ID, ts: Date.now() };
  try {
    localStorage.setItem(LEADER_KEY, JSON.stringify(lock));
  } catch (_) {
    return false;
  }
  // Re-read to lose races between tabs that loaded simultaneously: last
  // writer wins, everyone else stands by.
  const back = readLock();
  if (!back || back.id !== TAB_ID) return false;
  if (!isLeader) {
    isLeader = true;
    log('leader on this device — connecting');
    leaderBus && leaderBus.postMessage({ type: 'leader', id: TAB_ID });
  }
  return true;
}

function heartbeatLeadership() {
  if (!isLeader) return;
  try {
    localStorage.setItem(LEADER_KEY, JSON.stringify({ id: TAB_ID, ts: Date.now() }));
  } catch (_) {}
}

function resignLeadership() {
  if (!isLeader) return;
  isLeader = false;
  try {
    const cur = readLock();
    if (cur && cur.id === TAB_ID) localStorage.removeItem(LEADER_KEY);
  } catch (_) {}
  leaderBus && leaderBus.postMessage({ type: 'release', id: TAB_ID });
  try { ws && ws.close(1000, 'resign'); } catch (_) {}
}

function standbyCheck() {
  if (isLeader) { heartbeatLeadership(); return; }
  const lock = readLock();
  const stale = !lock || (Date.now() - lock.ts > LEADER_TTL_MS);
  if (!stale) return;
  // Leader gone (closed tab, crashed renderer): take over. The bridge
  // challenge evicts its ghost socket if one lingers; the host re-sends
  // CONFIG + keys to the fresh store, so answers stay exact.
  if (claimLeadership()) {
    log('takeover: previous leader stale, host will re-send keys');
    connect();
  }
}

async function connect() {
  if (!isLeader) return; // standbys never open the socket
  if (ws && ws.readyState !== WebSocket.CLOSED) return; // OPEN/CONNECTING/CLOSING: busy
  const url = $('url').value.trim();
  const token = $('token').value.trim();
  if (!url) { log('enter the bridge ws:// URL first'); return; }
  const full = url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token);
  $('state').textContent = 'connecting…';
  await ensureKernels(log);
  ws = new WebSocket(full);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    $('state').textContent = 'connected'; rxBuf = new Uint8Array(0);
    log(`connected to ${url} (cap ${store.cap} keys/layer)`);
    holdWakeLock();
    refresh();
  };
  // Binary frames are PATN messages; text frames are host notes ("mac PHASE...").
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') { macNote(ev.data); return; }
    onData(new Uint8Array(ev.data));
  };
  ws.onclose = (ev) => {
    const why = ev.code === 1013 ? ' (bridge held elsewhere — this tab stays standby)' : '';
    $('state').textContent = isLeader ? `closed (${ev.code})` : 'standby';
    log(`closed (${ev.code})${why}; reconnect in 3 s`);
    setTimeout(() => { if ($('auto').checked) connect(); }, 3000);
  };
  ws.onerror = () => { $('state').textContent = 'error'; };
  refresh();
}

function macNote(line) {
  // :50061-style mac PHASE reports arrive tunneled as TEXT frames "mac ...".
  const m = /^mac (\S+)(.*)$/.exec(line);
  if (m) {
    macPhase = m[1];
    log('mac: ' + line);
    refresh();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  $('connect').addEventListener('click', () => { if (claimLeadership()) connect(); });
  setInterval(refresh, 1000);
  setInterval(standbyCheck, 2000);
  if (leaderBus) {
    // A released slot is an instant takeover opportunity, no 6 s stale wait.
    leaderBus.onmessage = (ev) => {
      if (ev.data && ev.data.type === 'release') standbyCheck();
    };
  }
  window.addEventListener('beforeunload', resignLeadership);
  // A tab returning to visible gets its timers back: reconnect now instead of
  // waiting out the 3 s backoff, and re-take the wake lock (it releases on hide).
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    holdWakeLock();
    if (isLeader && $('auto').checked && (!ws || ws.readyState === WebSocket.CLOSED)) connect();
  });
  // Zero-config entry: visit ?ws=ws://host:port&token=SECRET&connect=1 and the
  // leader tab fills the fields and connects by itself — nothing to type on
  // the phone. Non-leaders stay standby even with connect=1.
  const q = new URLSearchParams(location.search);
  if (q.get('ws')) $('url').value = q.get('ws');
  if (q.get('token')) $('token').value = q.get('token');
  if (claimLeadership()) {
    $('state').textContent = 'leader';
    if (q.get('connect') === '1' && $('url').value) {
      log('auto-connecting…');
      connect();
    } else {
      log('enter ws://host:50063 ?token=… and Connect. Needs WebGPU (Chrome/Edge 113+, Safari 26+).');
    }
  } else {
    $('state').textContent = 'standby (leader active on this device)';
    log('standby: another tab on this device holds the bridge. Close it to take over, or just wait — this tab steps in if it goes stale.');
  }
  log(`KV cap ${store.cap} keys/layer (?cap=N to change); kernel pins ?rev=pin.`);
});
