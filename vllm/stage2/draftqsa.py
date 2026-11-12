"""stage2_v1 'draftqsa' part: the MTP drafter under the eviction runner's QSA and position mapping (M2).

1. Draft decode steps (draft positions >= 1) build their attention metadata with the speculator's own builders
   (DraftModelSpeculator._build_draft_attn_metadata), outside the model state's prepare_attn, so the eviction runner's
   QSA shift (_ACTIVE_QSA) is absent: after an eviction the build-time side-cache positions (ring and compressed slot
   mappings, selection query positions) miss the lane's deleted / padding shift. For batches with an evicted lane the
   build gets the shift in the non-graph form (shifted copies of the draft seq lens). Batches without one stay stock.
   - With graph mapping (production) the replayed decode graphs read the stable raw/virtual seq-len buffers staged by
     the target step (the verify-window end, shift included): a draft query past it loses its own tail tokens. A first
     version restaged them with the draft lens; an on/off A/B on one server (6 pairs, 25 lanes, 2026-09-30) showed no
     gain (tau 2.75 vs 2.81, not significant), so the stock stable-buffer semantics are kept.
2. The drafter's RoPE positions are the 1-D resident index (stock). After an eviction its kept keys carry their
   original index, so new queries would be shifted by the deleted count. The drafter forward adds the lane's deleted
   count (the raw index, i.e. the original stream index: the stock value without eviction) per token, from a
   persistent buffer filled before each drafter forward (step 0: the target's token layout; decode steps: one row
   per request). The add runs in the speculator's eager _run_model (outside torch.compile; inside the captured region
   of full CUDA graphs) into a persistent positions buffer that the model call reads instead of input_buffers
   .positions. The slot mappings and sampling keep the resident positions.
Draft numerics only: the target verifies every draft.

A/B switch (measurement only): with ARC3_S2_DRAFTQSA_SWITCH=<file>, a file containing 'off' restores the stock draft
decode metadata (item 1) on the running server; read at most once a second."""
import os
import time
from types import SimpleNamespace

PINS = {
    'vllm.v1.worker.gpu.spec_decode.speculator': '218e9a73eaa090947ab748fa4923821399b1f1086442a4a12986f425bb3ae8d9',
    'vllm.v1.worker.gpu.spec_decode.autoregressive.speculator':
        '575f39930f7b3a89402c385885d598416137b72e51fea83f2320a3212b5b99e1',
}
STATS = dict(buffers=0, proposes=0, shifted_proposes=0, decode_builds=0, decode_staged=0, decode_shifted=0,
             decode_switched_off=0)
_SWITCH = dict(path=os.environ.get('ARC3_S2_DRAFTQSA_SWITCH'), t=0.0, off=False)


def _switched_off():
    if not _SWITCH['path']:
        return False
    now = time.monotonic()
    if now - _SWITCH['t'] > 1.0:
        _SWITCH['t'] = now
        try:
            with open(_SWITCH['path']) as f:
                _SWITCH['off'] = f.read().strip() == 'off'
        except OSError:
            _SWITCH['off'] = False
    return _SWITCH['off']
_BUF = {}          # 'shift', 'pos': int64 [max_num_tokens] (one drafter per worker process); 'dirty': shift non-zero


def _eviction(spec):
    state = getattr(spec, 'model_state', None)          # absent in the profile run (before set_attn)
    return getattr(state, '_arc3_eviction_runner', None), state


def _evicted(runner, req_ids):
    states = runner._arc3_eviction_states
    return any((states.get(r) or {}).get('deleted_tokens', 0) for r in req_ids)


def patch_ar_speculator(mod):
    cls = mod.AutoRegressiveSpeculator
    orig_load, orig_propose, orig_run = cls.load_model, cls.propose, cls._run_model

    def load_model(self, target_model):
        import torch
        orig_load(self, target_model)
        _BUF['shift'] = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
        _BUF['pos'] = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
        _BUF['dirty'] = False
        STATS['buffers'] += 1

    def propose(self, input_batch, *args, **kwargs):
        STATS['proposes'] += 1
        runner, state = _eviction(self)
        self._arc3_batch = input_batch
        self._arc3_evicted = bool(runner is not None and _evicted(runner, input_batch.req_ids))
        shift = _BUF.get('shift')
        if shift is not None:
            if self._arc3_evicted:                      # step 0: per token of the target layout
                import torch
                n, r = input_batch.num_tokens, input_batch.num_reqs
                ends = input_batch.query_start_loc[1:r + 1].long()
                rows = torch.searchsorted(ends, torch.arange(n, device=shift.device), right=True)
                slots = input_batch.idx_mapping[:r].long().index_select(0, rows)
                shift[:n].copy_(state._arc3_eviction_buffers.deleted.index_select(0, slots))
                shift[n:input_batch.num_tokens_after_padding].zero_()
                _BUF['dirty'] = True
                STATS['shifted_proposes'] += 1
            elif _BUF['dirty']:
                shift.zero_()
                _BUF['dirty'] = False
        try:
            return orig_propose(self, input_batch, *args, **kwargs)
        finally:
            self._arc3_batch = None

    def _run_model(self, num_tokens, *args, **kwargs):
        import torch
        buffers = self.input_buffers
        stock = buffers.positions
        pos = _BUF.get('pos')
        if pos is None or stock.ndim != 1:
            return orig_run(self, num_tokens, *args, **kwargs)
        torch.add(stock[:num_tokens], _BUF['shift'][:num_tokens], out=pos[:num_tokens])
        buffers.positions = pos                         # the model call reads positions[:num_tokens] from here
        try:
            return orig_run(self, num_tokens, *args, **kwargs)
        finally:
            buffers.positions = stock

    cls.load_model, cls.propose, cls._run_model = load_model, propose, _run_model


def patch_speculator(mod):
    cls = mod.DraftModelSpeculator
    orig = cls._build_draft_attn_metadata

    def _build_draft_attn_metadata(self, num_reqs, num_reqs_padded, num_tokens_padded, seq_lens_cpu_upper_bound,
                                   step, *args, **kwargs):
        runner, state = _eviction(self)
        batch = getattr(self, '_arc3_batch', None)
        evicted = getattr(self, '_arc3_evicted', False)
        if runner is None or batch is None or not evicted or _switched_off():   # stock: no evicted lane (or A/B off)
            if batch is not None and runner is not None and evicted:
                STATS['decode_switched_off'] += 1
            return orig(self, num_reqs, num_reqs_padded, num_tokens_padded, seq_lens_cpu_upper_bound, step,
                        *args, **kwargs)
        import torch
        import eviction_runner as er
        STATS['decode_builds'] += 1
        buffers = state._arc3_eviction_buffers
        n, p = num_reqs, num_reqs_padded
        slots = self.idx_mapping[:n].long()
        shift = _BUF.get('shift')
        if shift is not None:                            # decode steps: one RoPE shift row per request
            shift[:n].copy_(buffers.deleted.index_select(0, slots))
            shift[n:num_tokens_padded].zero_()
            _BUF['dirty'] = True
            STATS['decode_shifted'] += 1
        # Build-time QSA positions (ring and compressed slots, selection positions) take each lane's shift, as the
        # target's prepare_attn gives them (non-graph form: shifted copies, padded rows unshifted). The replayed graphs
        # keep reading the stable seq-len buffers the target step staged, which already carry the shift.
        deleted = torch.zeros(p, dtype=torch.int64, device=slots.device)
        padding = torch.zeros(p, dtype=torch.int64, device=slots.device)
        pslots = torch.full((p,), -1, dtype=torch.int64, device=slots.device)
        deleted[:n] = buffers.deleted.index_select(0, slots)
        padding[:n] = buffers.padding.index_select(0, slots)
        pslots[:n] = slots
        active = SimpleNamespace(deleted=deleted, padding=padding, lookup=buffers.lookup, cutoffs=buffers.cutoffs,
                                 state_padding=buffers.padding, slots=pslots, mapping=None)
        STATS['decode_staged'] += 1
        token = er._ACTIVE_QSA.set(active)
        try:
            return orig(self, num_reqs, num_reqs_padded, num_tokens_padded, seq_lens_cpu_upper_bound, step,
                        *args, **kwargs)
        finally:
            er._ACTIVE_QSA.reset(token)

    cls._build_draft_attn_metadata = _build_draft_attn_metadata


PATCHERS = {'vllm.v1.worker.gpu.spec_decode.speculator': patch_speculator,
            'vllm.v1.worker.gpu.spec_decode.autoregressive.speculator': patch_ar_speculator}
