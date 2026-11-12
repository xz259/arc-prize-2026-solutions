"""stage2_v1 'dflashtaps' part (2026-10-10, DFlash plan Phase 1; design dflash_20261006/EXPORTER_DESIGN_2026-10-10.md).
Target taps for the DFlash drafter: the attention sublayer's `block_input` (the single-stream mix each layer consumes,
[T, 2560]) of the layers in ARC3_S2_TAPS (library ids: id i = the input of layer i = the state after layer i - 1),
captured into a preallocated buffer inside the compiled forward (as the MTP's _mtp_hidden_buffer), plus the final state
(the model's return value: the final mixer's output for every row, which lm_head consumes with no norm).

Export mode (ARC3_S2_TAPS_DIR set): after every executed step the runner copies the real rows (taps, final state, the
1-D input_batch.positions the DFlash speculator consumes, input ids) to pinned host buffers; a writer thread appends
them per engine request in host memory and, when the request finishes, writes one file per request:
<dir>/<request id>.safetensors with hidden_states [N, n_taps + 1, 2560] (bf16; slot order = ARC3_S2_TAPS then final),
token_ids [N], position_ids [N] (int64: the row's arrival index among all rows the request forwarded, which in the
forced replay is the stream index, the training position) and served_position_ids [N] (the engine's 1-D positions),
rows in arrival order, never deduplicated (2026-10-10, job 399d: the served 1-D positions are compacted after an
eviction, t - removed, so they repeat within a session and cannot key the rows; every token is forwarded exactly once in
stream order in the forced replay). The file is already a Speculators sample file, so the packer hard-links a window
that is exactly one segment instead of copying it (it checks ids and the expected served positions first).
The device copies are enqueued on the forward's stream right after execute_model, so the next step's forward (same
stream) cannot overwrite the tap buffer before they complete; the pinned ring applies back-pressure (STATS stalls).
Receipt: STATS."""
import os
import queue
import re
import threading
import time
from pathlib import Path

PINS = {'vllm.models.qwen3_8_flash_next.nvidia.model': 'd900cd6fcacba18f460e00b3f018fbf36fbe6ecc310692b6adb1451f7f53cc17',
        'vllm.v1.worker.gpu.model_runner': 'f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8'}
TAPS = tuple(int(x) for x in os.environ.get('ARC3_S2_TAPS', '2,4,12,13,24,35,36,44,46').split(',') if x.strip())
OUT_DIR = os.environ.get('ARC3_S2_TAPS_DIR', '')
RING = int(os.environ.get('ARC3_S2_TAPS_RING', '8'))
# ARC3_S2_TAPS_KEEP=<json {session id: [[start, end), ...]}>: rows whose arrival index (the stream index in the forced
# replay) is outside a request's session ranges are dropped at capture (windowed caches: replay runs up to a window's
# end, only the window is stored); sessions not listed keep all
KEEP_FILE = os.environ.get('ARC3_S2_TAPS_KEEP', '')
# 2026-10-10 (Phase 6 streaming, PHASE6_STREAMING_DESIGN_2026-10-10.md): ARC3_S2_TAPS_STORE = slot indices to store (0..n_taps-1
# the taps, n_taps the final state; default all, in that order); the capture and the compiled forward keep every tap, so
# the compile cache is unaffected. ARC3_S2_TAPS_SEGMENT = rows per segment file (0: one file per request at finish, as
# before): a request's rows are flushed as soon as a segment fills, consecutive segments overlapping by ARC3_S2_TAPS_HEAD
# rows (the anchor-free head, default 8,192), so whole sessions never sit in host memory; files <name>_s<NNNN>.safetensors.
STORE = tuple(int(x) for x in os.environ.get('ARC3_S2_TAPS_STORE', '').split(',') if x.strip())
SEGMENT = int(os.environ.get('ARC3_S2_TAPS_SEGMENT', '0') or 0)
HEAD = int(os.environ.get('ARC3_S2_TAPS_HEAD', '8192') or 0)
# 2026-10-10 07:10 (Phase 6): ARC3_S2_TAPS_PREFIX = only requests whose id starts with it are stored (Phase 6:
# 'arc3-persistent-', so the server warm-up and the NLL panel leave no files in the RAM directory); default all.
PREFIX = os.environ.get('ARC3_S2_TAPS_PREFIX', '')
# 2026-10-11 (MTP fine-tune backup plan, mtp_ft_20261011/README.md): ARC3_S2_TAPS_MTP=1 also exports the target's
# pre-final-mixer multi-stream state (the MTP drafter's input, `_mtp_hidden_buffer` [rows, hc*H]) as hc slots after the
# final state; needs the MTP drafter configured (the buffer exists only then)
MTP = os.environ.get('ARC3_S2_TAPS_MTP', '') == '1'
assert not (SEGMENT and KEEP_FILE), 'ARC3_S2_TAPS_SEGMENT and ARC3_S2_TAPS_KEEP are exclusive'
assert not SEGMENT or 0 <= HEAD < SEGMENT, 'ARC3_S2_TAPS_HEAD must be below ARC3_S2_TAPS_SEGMENT'
STATS = dict(taps=list(TAPS), out_dir=OUT_DIR, buffer_rows=0, steps=0, rows=0, files=0, file_rows=0,
             overwritten_rows=0, writer_queue_max=0, stalls=0, stall_seconds=0.0, writer_errors=0, last_error=None,
             open_requests=0, kept_rows=0, dropped_rows=0, store=list(STORE), segment=SEGMENT, head=HEAD, segments=0,
             prefix=PREFIX, mtp=MTP, mtp_streams=0)


def patch_model(mod):
    Model, Layer = mod.Qwen3_8FlashNextModel, mod.Qwen3_8FlashNextDecoderLayer
    orig_init = Model.__init__

    def __init__(self, *, vllm_config, prefix=''):
        import torch
        orig_init(self, vllm_config=vllm_config, prefix=prefix)
        rows = vllm_config.scheduler_config.max_num_batched_tokens
        self._dflash_tap_buffer = torch.empty(len(TAPS), rows, self.config.hidden_size,
                                              dtype=vllm_config.model_config.dtype)
        STATS['buffer_rows'] = int(rows)
        for slot, layer_id in enumerate(TAPS):
            layer = self.layers[layer_id]
            layer._dflash_tap_slot = slot
            layer.__dict__['_dflash_tap_buffer'] = self._dflash_tap_buffer   # plain attribute, not a submodule/buffer

    Model.__init__ = __init__

    def forward(self, hidden_states, prev_block_output, prev_injection, positions, *, input_ids, query_start_loc,
                ngram_context):
        # Qwen3_8FlashNextDecoderLayer.forward (pinned source) with one added copy of the attention block_input
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = attn_hc.combine(hidden_states, prev_block_output, prev_injection)
                prev_block_output = prev_injection = None
            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = hidden_states + self.ple(hidden_states, input_ids, query_start_loc, ngram_context)
        if prev_block_output is not None and prev_injection is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(hidden_states, prev_block_output,
                                                                            prev_injection)
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)
        slot = getattr(self, '_dflash_tap_slot', None)
        if slot is not None:                                          # dflashtaps: the drafter's target feature
            self._dflash_tap_buffer[slot, :block_input.shape[0]].copy_(block_input)
        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        elif self.layer_type == "full_attention":
            attn_out = self.self_attn(hidden_states=block_input, positions=positions)
        else:
            raise ValueError("Invalid layer_type")
        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(hidden_states, attn_out, injection)
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection

    Layer.forward = forward


class _Writer:
    """Host side of export mode: pinned ring, a writer thread, per-request accumulation, one file per finished request."""

    def __init__(self, n_slots, hidden, rows, device):
        import torch
        self.torch = torch
        self.dir = Path(OUT_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ring = [dict(h=torch.empty(n_slots, rows, hidden, dtype=torch.bfloat16, pin_memory=True),
                          pos=torch.empty(rows, dtype=torch.int64, pin_memory=True),
                          ids=torch.empty(rows, dtype=torch.int64, pin_memory=True),
                          free=threading.Event()) for _ in range(RING)]
        for r in self.ring:
            r['free'].set()
        self.next = 0
        self.q = queue.Queue()
        self.acc = {}
        self.keep = {}
        if KEEP_FILE:
            import json
            self.keep = {k: [(int(a), int(b)) for a, b in v] for k, v in json.load(open(KEEP_FILE)).items()}
        self.rid_ranges = {}
        self.arrived = {}                                                 # request -> rows forwarded so far
        self.seg_idx = {}                                                 # request -> next segment number
        self.thread = threading.Thread(target=self._run, name='dflashtaps-writer', daemon=True)
        self.thread.start()

    def capture(self, model, state):
        torch = self.torch
        ib = state.input_batch
        n = int(ib.num_tokens)
        if n <= 0 or state.hidden_states is None:
            return
        slot = self.ring[self.next]
        self.next = (self.next + 1) % len(self.ring)
        if not slot['free'].is_set():                                 # back-pressure: the writer has not drained it
            t0 = time.perf_counter()
            slot['free'].wait()
            STATS['stalls'] += 1
            STATS['stall_seconds'] += time.perf_counter() - t0
        slot['free'].clear()
        k = len(TAPS)
        slot['h'][:k, :n].copy_(model._dflash_tap_buffer[:, :n], non_blocking=True)
        slot['h'][k, :n].copy_(state.hidden_states[:n], non_blocking=True)
        if MTP:                                                       # hc streams of the MTP input, after the final state
            mb = model._mtp_hidden_buffer
            hc = mb.shape[-1] // slot['h'].shape[-1]
            slot['h'][k + 1:k + 1 + hc, :n].copy_(mb[:n].view(n, hc, -1).transpose(0, 1), non_blocking=True)
        slot['pos'][:n].copy_(ib.positions[:n], non_blocking=True)
        slot['ids'][:n].copy_(ib.input_ids[:n], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        qsl = ib.query_start_loc_np[:ib.num_reqs + 1].copy()
        self.q.put(('rows', slot, ev, n, list(ib.req_ids[:ib.num_reqs]), qsl))
        STATS['steps'] += 1
        STATS['rows'] += n
        STATS['writer_queue_max'] = max(STATS['writer_queue_max'], self.q.qsize())

    def finish(self, req_ids):
        for r in req_ids:
            self.q.put(('finish', r))

    def _run(self):
        import numpy as np
        torch = self.torch
        while True:
            item = self.q.get()
            try:
                if item[0] == 'rows':
                    _, slot, ev, n, req_ids, qsl = item
                    ev.synchronize()
                    src = slot['h'][list(STORE), :n] if STORE else slot['h'][:, :n]
                    h = src.transpose(0, 1).contiguous()                  # [n, stored slots, hidden]
                    pos = slot['pos'][:n].numpy().copy()
                    ids = slot['ids'][:n].numpy().copy()
                    slot['free'].set()
                    for i, rid in enumerate(req_ids):
                        a, b = int(qsl[i]), int(qsl[i + 1])
                        if b <= a or (PREFIX and not rid.startswith(PREFIX)):
                            continue
                        ar0 = self.arrived.get(rid, 0)
                        self.arrived[rid] = ar0 + (b - a)
                        ar = np.arange(ar0, ar0 + (b - a), dtype=np.int64)
                        ranges = self._ranges(rid)
                        if ranges is None:
                            self.acc.setdefault(rid, []).append((ar, pos[a:b], ids[a:b], h[a:b]))
                            STATS['kept_rows'] += b - a
                            if SEGMENT:
                                self._flush_segments(rid, np)
                            continue
                        sel = np.zeros(b - a, dtype=bool)
                        for lo, hi in ranges:
                            sel |= (ar >= lo) & (ar < hi)
                        k = int(sel.sum())
                        STATS['kept_rows'] += k
                        STATS['dropped_rows'] += b - a - k
                        if k:
                            idx = np.nonzero(sel)[0]
                            self.acc.setdefault(rid, []).append((ar[idx], pos[a:b][idx], ids[a:b][idx],
                                                                 h[a:b][torch.from_numpy(idx)]))
                    STATS['open_requests'] = len(self.acc)
                elif item[0] == 'finish':
                    self._write(item[1], np)
            except Exception as exc:                                      # noqa: BLE001
                STATS['writer_errors'] += 1
                STATS['last_error'] = repr(exc)[:300]

    def _ranges(self, rid):
        if not self.keep:
            return None
        if rid not in self.rid_ranges:
            self.rid_ranges[rid] = next((v for k, v in self.keep.items() if k in rid), None)
        return self.rid_ranges[rid]

    def _save(self, name, ar, pos, ids, h, np):
        from safetensors.torch import save_file
        torch = self.torch
        tmp = self.dir / (name + '.safetensors.part')
        save_file(dict(hidden_states=h.contiguous(), token_ids=torch.from_numpy(ids.astype(np.int64)),
                       position_ids=torch.from_numpy(ar.astype(np.int64)),
                       served_position_ids=torch.from_numpy(pos.astype(np.int64))), str(tmp))
        tmp.rename(self.dir / (name + '.safetensors'))
        STATS['files'] += 1
        STATS['file_rows'] += int(len(ar))

    def _flush_segments(self, rid, np, final=False):
        """Write every full segment buffered for rid (final: also the remainder); keep the HEAD-row overlap."""
        torch = self.torch
        base = re.sub(r'[^A-Za-z0-9._-]', '_', rid)
        while True:
            chunks = self.acc.get(rid)
            if not chunks:
                return
            total = sum(len(c[0]) for c in chunks)
            k = self.seg_idx.get(rid, 0)
            if not final and total < SEGMENT:
                return
            if final and k > 0 and total <= HEAD:                     # only the overlap is left: already written
                self.acc.pop(rid, None)
                return
            take = total if final else SEGMENT
            ar = np.concatenate([c[0] for c in chunks])
            pos = np.concatenate([c[1] for c in chunks])
            ids = np.concatenate([c[2] for c in chunks])
            h = torch.cat([c[3] for c in chunks])
            self._save('%s_s%04d' % (base, k), ar[:take], pos[:take], ids[:take], h[:take], np)
            self.seg_idx[rid] = k + 1
            STATS['segments'] += 1
            if final:
                self.acc.pop(rid, None)
                return
            cut = take - HEAD                                         # the next segment starts HEAD rows back
            self.acc[rid] = [(ar[cut:].copy(), pos[cut:].copy(), ids[cut:].copy(), h[cut:].clone())]

    def _write(self, rid, np):
        torch = self.torch
        if SEGMENT:
            self._flush_segments(rid, np, final=True)
            self.arrived.pop(rid, None)
            self.seg_idx.pop(rid, None)
            STATS['open_requests'] = len(self.acc)
            return
        chunks = self.acc.pop(rid, None)
        self.arrived.pop(rid, None)
        STATS['open_requests'] = len(self.acc)
        if not chunks:
            return
        ar = np.concatenate([c[0] for c in chunks])
        pos = np.concatenate([c[1] for c in chunks])
        ids = np.concatenate([c[2] for c in chunks])
        h = torch.cat([c[3] for c in chunks])
        self._save(re.sub(r'[^A-Za-z0-9._-]', '_', rid), ar, pos, ids, h, np)


_W = []


def _tap_model(runner):
    """The Qwen3_8FlashNextModel holding the tap buffer (runner.model is the multimodal wrapper: its language_model.model);
    found once by type name and cached on the runner (the decoder layers also carry the buffer, so match the type)."""
    m = runner.__dict__.get('_dflash_tap_model')
    if m is None:
        m = next((x for x in runner.model.modules() if type(x).__name__ == 'Qwen3_8FlashNextModel'
                  and hasattr(x, '_dflash_tap_buffer')), False)
        runner.__dict__['_dflash_tap_model'] = m
        STATS['tap_model_found'] = bool(m)
    return m or None


def patch_model_runner(mod):
    cls = mod.GPUModelRunner
    orig = cls.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        out = orig(self, scheduler_output, *args, **kwargs)
        if not OUT_DIR:
            return out
        try:
            state = getattr(self, 'execute_model_state', None)
            model = _tap_model(self)
            buf = getattr(model, '_dflash_tap_buffer', None)
            if state is not None and buf is not None and state is not getattr(self, '_dflash_last_state', None):
                self._dflash_last_state = state
                if not _W:
                    extra = 0
                    if MTP:
                        mb = getattr(model, '_mtp_hidden_buffer', None)
                        if mb is None:
                            raise RuntimeError('ARC3_S2_TAPS_MTP=1 needs the MTP drafter (no _mtp_hidden_buffer)')
                        extra = int(mb.shape[-1]) // int(buf.shape[-1])
                        STATS['mtp_streams'] = extra
                    _W.append(_Writer(len(TAPS) + 1 + extra, int(buf.shape[-1]), int(buf.shape[1]), buf.device))
                _W[0].capture(model, state)
            if _W and getattr(scheduler_output, 'finished_req_ids', None):
                _W[0].finish(sorted(scheduler_output.finished_req_ids))
        except Exception as exc:                                          # noqa: BLE001
            STATS['writer_errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
            raise
        return out

    cls.execute_model = execute_model


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.model': patch_model,
            'vllm.v1.worker.gpu.model_runner': patch_model_runner}
