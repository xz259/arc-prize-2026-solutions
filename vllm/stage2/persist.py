"""stage2_v1 'persist' part: MTP k in the persistent streaming-session engine (M2, option C: no one-step lookahead).

Design: stage2/M2_DESIGN.md §2, §8. Needs 'rollback' (one GDN slot per lane, pending window committed by the next
step) and persistent requests without one-step lookahead (the 'guards' endpoint patch drops the client's per-turn
lookahead request when this part is on; the server keeps the production canary configuration), so a persistent lane
never has a step in flight when its stop is processed.

Engine core (source transforms of engine_patch and persistent_batch_policy; the files are unchanged):
  - validate_config and the batch policy admit MTP k (only with this part and 'rollback');
  - a persistent stop drops the drafts scheduled for a next step, releases the output placeholders of the outputs the
    stop dropped, and sets num_computed_tokens to the kept stream minus one: the MTP0 boundary (exactly one accepted,
    uncomputed final token), so append, empty continuation, history eviction and their audits are production's. vLLM
    subtracts only rejected drafts: a stop at accepted draft d_j (or a length stop inside the window) leaves L+n+1 >
    L+j; that d_j was an input of the step and is computed again by the resumed chunk (one token, idempotent KV).
Worker (vllm.v1.worker.gpu.model_runner, MRv2):
  - The kept count m of the stopping window (1..k+1) is computed by the scheduler (window start noted when the step is
    scheduled; m = frontier - start, the window inputs before the uncomputed final token) and travels with the resumed
    chunk in sampling_params.extra_args['arc3_s2_kept']: a count, so history eviction between the pause and the resume
    (which shifts resident positions) does not change it.
  - add_requests: a streaming resume (the request exists, frontier > 0) carrying m commits m pending tokens into every
    GDN layer's state with the exact recurrence, moves the GDN and PLE conv states from the speculative rolling layout
    (offset m - 1) to the prefill layout (column 0), and clears the GDN flags. Eager, before the forward, no host
    synchronisation. A rebase restart (frontier 0) recomputes everything and commits nothing. The rollback part's
    counter of prefill rows that resume with a pending window must stay 0."""
import os
import time

PINS = {
    'engine_patch': '23f5a4d83de7254c7df9076aa162f25a88082801c8508417ae1fc6d7ae45c310',
    'persistent_batch_policy': '2ff199cc697f1c4b3bbf43777ac23a00596872127f8472137937ba9d2fd5b281',
    'vllm.v1.worker.gpu.model_runner': 'f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8',
}
STATS = dict(stops=0, stop_clamped=0, stop_last_requeued=0, stop_drafts_cleared=0, stop_placeholders_released=0,
             kept_passed=0, resumes=0, resume_commits=0, resume_tokens=0, resume_skipped_restart=0, resume_ms=0.0,
             warmup_pending_resumes=None)


def _enabled():
    parts = set(os.environ.get('ARC3_STAGE2', '0').split(','))
    return {'persist', 'rollback'} <= parts


_FLUSHER = []


def _flush_receipts():
    """Engine core (no CUDA context): write the receipt within ~5 s of a change (daemon thread, off the step path)."""
    if _FLUSHER:
        return
    import threading

    def run():
        from .install import save
        last = None
        while True:
            time.sleep(5)
            now = repr(sorted(STATS.items()))
            if now != last:
                save()
                last = now

    _FLUSHER.append(threading.Thread(target=run, name='arc3-s2-persist-receipts', daemon=True))
    _FLUSHER[0].start()


# ---------------------------------------------------------------- engine core helpers (injected into engine_patch)
def arc3_s2_persist_mtp(config):
    return _enabled() and bool(config.num_speculative_tokens)


def arc3_s2_spec(scheduler):
    return bool(getattr(scheduler.vllm_config, 'num_speculative_tokens', 0))


def arc3_s2_stop_fixup(request, scheduler):
    """Persistent stop under MTP: drop the drafts scheduled for a next step and leave the MTP0 boundary (the final
    kept token uncomputed; the resumed chunk computes it again if it was a verify input).

    Async scheduling holds 1 + k output placeholders per verify step and releases the rejected drafts and the tokens
    returned; a stop inside the window drops n + 1 - j outputs that keep their placeholders. With no other step in
    flight (option C) that count is exactly (frontier excess) + 1, where the excess is num_computed_tokens (vLLM's
    L + n + 1) minus the kept stream (L + 1 + j); anything else means work in flight and is refused. The kept count
    m = (L + j) - L = j window inputs are committed at resume."""
    STATS['stops'] += 1
    _flush_receipts()
    if request.spec_token_ids:
        request.spec_token_ids = []
        STATS['stop_drafts_cleared'] += 1
    n = len(request._all_token_ids)
    held, excess = request.num_output_placeholders, request.num_computed_tokens - n
    if request.num_in_flight_tokens or held != max(excess + 1, 0):
        raise RuntimeError(f'stage2 persist: stop with {held} output placeholders, {request.num_in_flight_tokens} '
                           f'tokens in flight, frontier excess {excess}')
    if held:
        request.num_output_placeholders = 0
        STATS['stop_placeholders_released'] += held
    if request.num_computed_tokens > n - 1:           # the MTP0 boundary: the final kept token is uncomputed
        STATS['stop_clamped'] += 1
        STATS['stop_last_requeued'] += request.num_computed_tokens >= n
        request.num_computed_tokens = n - 1
    win = getattr(request, '_arc3_s2_win', None)
    request._arc3_s2_kept = None
    if win is not None:
        kept = request.num_computed_tokens - win
        if not 1 <= kept <= 1 + int(scheduler.vllm_config.num_speculative_tokens):
            raise RuntimeError(f'stage2 persist: stop keeps {kept} tokens of its window (start {win})')
        request._arc3_s2_kept = kept


def arc3_s2_note_step(request, spec_ids):
    """Scheduler, before the step's num_computed_tokens update: remember the verify window start of a speculative step
    (None for a prefill or non-speculative step)."""
    request._arc3_s2_win = request.num_computed_tokens if request.request_id in spec_ids else None


def arc3_s2_pass_kept(session, update):
    """append_session: hand the kept count of the paused window to the worker with the resumed chunk."""
    kept = getattr(session, '_arc3_s2_kept', None)
    session._arc3_s2_kept = None
    if kept:
        params = update.sampling_params
        if params.extra_args is None:
            params.extra_args = {}
        params.extra_args['arc3_s2_kept'] = int(kept)
        STATS['kept_passed'] += 1


def transform_engine_patch(src):
    reps = [
        ('from __future__ import annotations\n',
         'from __future__ import annotations\n'
         'from stage2_v1.persist import (arc3_s2_persist_mtp, arc3_s2_spec, arc3_s2_stop_fixup, arc3_s2_note_step,\n'
         '                               arc3_s2_pass_kept)\n'),
        ('    if config.speculative_config is not None or config.num_speculative_tokens != 0:\n'
         '        raise PersistentStateError("Persistent phase 1 requires MTP0 without a draft model")\n',
         '    if ((config.speculative_config is not None or config.num_speculative_tokens != 0)\n'
         '            and not arc3_s2_persist_mtp(config)):\n'
         '        raise PersistentStateError("Persistent phase 1 requires MTP0 without a draft model")\n'),
        ('        "mtp": 0,\n', '        "mtp": int(config.num_speculative_tokens or 0),\n'),
        ('        if is_persistent(request):\n'
         '            assert_settled(request)\n',
         '        if is_persistent(request):\n'
         '            if arc3_s2_spec(self):\n'
         '                arc3_s2_stop_fixup(request, self)\n'
         '            assert_settled(request)\n'),
        ('    validate_sampling(update.sampling_params)\n    assert_settled(session)\n',
         '    validate_sampling(update.sampling_params)\n    assert_settled(session)\n'
         '    if arc3_s2_spec(scheduler):\n        arc3_s2_pass_kept(session, update)\n'),
        ('            import one_lookahead\n            if one_lookahead.marked(request):\n'
         '                one_lookahead.after_schedule_check(request, count)\n',
         '            import one_lookahead\n            if arc3_s2_spec(self):\n'
         '                arc3_s2_note_step(request, scheduler_output.scheduled_spec_decode_tokens or {})\n'
         '            if one_lookahead.marked(request):\n                one_lookahead.after_schedule_check(request, count)\n'),
    ]
    for a, b in reps:
        assert src.count(a) == 1, ('stage2 persist: engine_patch anchor', a[:60])
        src = src.replace(a, b)
    return src


def transform_batch_policy(src):
    a = ('                or pc.data_parallel_size != 1 or config.num_speculative_tokens != 0):\n')
    b = ('                or pc.data_parallel_size != 1\n'
         '                or (config.num_speculative_tokens != 0 and not __import__("stage2_v1.persist").persist'
         '.arc3_s2_persist_mtp(config))):\n')
    assert src.count(a) == 1, 'stage2 persist: batch policy anchor'
    return src.replace(a, b)


# ---------------------------------------------------------------- worker (MRv2 model runner)
def _layers(runner):
    """GDN layers (with rollback flags) and the PLE layer, each with its KV cache group index."""
    cached = getattr(runner, '_s2_persist_layers', None)
    if cached is not None:
        return cached
    ctx = runner.vllm_config.compilation_config.static_forward_context
    group_of = {}
    for g, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            group_of[name] = g
    gdn, ple = [], []
    for name, layer in ctx.items():
        kind = type(layer).__name__
        if kind == 'QwenGatedDeltaNetAttention' and getattr(layer, '_s2_flags', None) is not None:
            gdn.append((name, layer, group_of[name]))
        elif kind == 'Qwen3_8FlashNextPLELayer':
            ple.append((name, layer, group_of[name]))
    runner._s2_persist_layers = (gdn, ple)
    return gdn, ple


def _gdn_conv_view(layer):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    conv = layer.kv_cache[0]
    return conv if is_conv_state_dim_first() else conv.transpose(-1, -2)          # [slots, dim, L]


def _ple_conv_view(layer):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    conv = layer.kv_cache[0]
    conv = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
    cap = layer.conv_state_len + layer.num_spec_tokens
    return conv[..., -cap:] if cap > 0 else conv                                    # the view the layer reads


def commit_resumed(runner, resumed):
    """resumed: list of (block_ids per group, m). Commit the pending GDN windows and realign the conv states."""
    import torch
    from .persist_kernels import gdn_resume_commit, realign_conv
    from . import rollback
    t0 = time.perf_counter()
    gdn, ple = _layers(runner)
    dev = runner.device
    ms = torch.tensor([m for _, m in resumed], dtype=torch.int64)
    ms_dev = ms.to(dev, non_blocking=True)
    offs_dev = (ms - 1).to(dev, non_blocking=True)
    for name, layer, g in gdn:
        slots = torch.tensor([ids[g][0] for ids, _ in resumed], dtype=torch.int64).to(dev, non_blocking=True)
        ssm = layer.kv_cache[1]
        pend, flags, W, T1 = rollback._REG[ssm.data_ptr()]
        H = layer.num_k_heads // layer.tp_size
        gdn_resume_commit(layer.A_log, layer.dt_bias, ssm, pend, flags, slots, ms_dev, H, layer.head_k_dim, W, T1,
                          scale_state=rollback._SCALE.get(ssm.data_ptr()))        # scaled16 row scales (None when off)
        realign_conv(_gdn_conv_view(layer), slots, offs_dev, layer.conv_kernel_size - 1)
        flags.index_fill_(0, slots, 0)
    for name, layer, g in ple:
        if layer.conv_state_len > 0:
            slots = torch.tensor([ids[g][0] for ids, _ in resumed], dtype=torch.int64).to(dev, non_blocking=True)
            realign_conv(_ple_conv_view(layer), slots, offs_dev, layer.conv_state_len)
    STATS['resume_commits'] += len(resumed)
    STATS['resume_tokens'] += int(ms.sum())
    STATS['resume_ms'] += (time.perf_counter() - t0) * 1e3


def _group_ids(nd):
    """Per-group block ids of a resumed request. 2026-10-15 (4b): with the 'draftring' part the PLE state sits in a worker-only
    group the scheduler sends no ids for; draftring appends the request's static rows."""
    from . import install
    if 'draftring' not in install.PARTS:
        return nd.block_ids
    from . import draftring
    return draftring.group_block_ids(nd.req_id, nd.block_ids)


def patch_model_runner(mod):
    cls = mod.GPUModelRunner
    orig_add = cls.add_requests

    def add_requests(self, scheduler_output):
        if STATS['warmup_pending_resumes'] is None and scheduler_output.scheduled_new_reqs:
            from . import rollback            # first served request: the counter covers served work only
            STATS['warmup_pending_resumes'] = rollback.reset_bad_resumes()
        resumed = []
        k1 = self.vllm_config.num_speculative_tokens + 1
        for nd in scheduler_output.scheduled_new_reqs:
            params = nd.sampling_params
            kept = (getattr(params, 'extra_args', None) or {}).get('arc3_s2_kept') if params is not None else None
            if not kept or nd.req_id not in self.req_states.req_id_to_index:
                continue
            STATS['resumes'] += 1
            if int(nd.num_computed_tokens) == 0:          # rebase restart: recomputed from position 0
                STATS['resume_skipped_restart'] += 1
                continue
            if not 1 <= int(kept) <= k1:
                raise RuntimeError(f'stage2 persist: resume of {nd.req_id} keeps {kept} tokens (k+1 = {k1})')
            resumed.append((_group_ids(nd), int(kept)))
        if resumed:
            commit_resumed(self, resumed)
        return orig_add(self, scheduler_output)

    cls.add_requests = add_requests


PATCHERS = {'engine_patch': ('source', transform_engine_patch),
            'persistent_batch_policy': ('source', transform_batch_policy),
            'vllm.v1.worker.gpu.model_runner': patch_model_runner}
