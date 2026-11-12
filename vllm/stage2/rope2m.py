"""stage2_v1 'rope2m' part (2026-10-03; user decision 03:41 "2M with one rebase is the way to go", corrected 04:30: adoption
gated by a split-shift NLL test; built 15:40): logical positions up to ARC3_S2_ROPE_ROWS (default 2,097,152) without a
rebase, instead of rebasing near the trained 262,144.

Table (every process that builds a rotary table; the worker in practice):
  RotaryEmbeddingBase._compute_cos_sin_cache is called unchanged (today's table: 4 x 262,144 rows for M-RoPE, FP32),
  then the table is extended to max(rows, ARC3_S2_ROPE_ROWS). Rows below ARC3_S2_ROPE_TRAINED (262,144) are kept
  bit-identical; rows from there on are recomputed with the same float32 inverse frequencies converted exactly to
  float64, the angle t * inv_freq and its cos / sin in float64, then cast to the table's float32 (and to BF16 at first
  use, as today). Frequencies are not diluted (no YaRN, interpolation or NTK). The attention, the QSA indexer and the
  drafter index the same tables (the indexer reuses the attention's rotary module; get_rope caches instances), so no
  kernel changes. Tables smaller than the trained length (vision rotary embeddings) are left alone.
Limits (the three readers of hf_text_config.max_position_embeddings that bound logical positions):
  session_service.SessionManager.__init__ (API server: max_logical_tokens; the rebase threshold follows),
  engine_eviction.validate_turn_capacity (engine core: the per-turn logical limit),
  eviction_runner._limits (worker: max_logical_tokens), each raised to ARC3_S2_ROPE_ROWS.
Split-shift test hook (test servers only; ARC3_S2_ROPE_SHIFT_SWITCH=<file> holding "<shift> <split token id>"):
  RopeState.init_prefill_positions shifts the positions of every prompt token from the first QSA compression-group
  boundary (4 tokens) after the first <split token id> (the end of the system turn) by <shift>, a multiple of 4, and the
  decode delta with them, so no compressed-key group mixes shifted and unshifted tokens (Fable's review, 17:40);
  "<shift> -1" shifts every token (uniform shift). The file is re-read on each new request, so one server tests several shifts. Absent or "0", or a prompt
  without the split token: no change.
Cost (GPU): the M-RoPE table grows from 1,048,576 to 2,097,152 rows (+128 MiB in BF16 at run time; up to +512 MiB of
FP32 transient while the model is built, before the KV cache is allocated), and the worker's QSA virtual lookup
(eviction_runner._allocate_buffers: max_num_reqs x max_logical_tokens int32) would grow by 7 MiB per lane (+175 MiB at 25
lanes, +350 MiB at 50); since 2026-10-04 it is allocated at 4 x max_resident_tokens instead (its true bound: retained
groups x 4 slots, cut at resident + padding): 382,896 entries (1.5 MiB per lane) at 95,724, 146,992 (0.6 MiB) at 36,748. Host: per-resume validation and lookup builds and the session's logical token lists scale with
the logical length, which is no longer bounded by the rebase at 237,568."""
import os

ROWS = int(os.environ.get('ARC3_S2_ROPE_ROWS', '2097152'))
# 2026-10-04 (Fable's review: force one rebase with the full part set): test only. A positive value replaces the API
# server's session limit (the rebase threshold follows: limit - 24,576), so sessions rebase early; the engine and worker
# limits stay at ROWS (more permissive), so no turn fails before the rebase. Unset in every production run.
TEST_SESSION_LIMIT = int(os.environ.get('ARC3_S2_ROPE_TEST_SESSION_LIMIT', '0') or 0)
# test servers only (2026-10-06, deadlinefix's capacity abort): the n-th turn-capacity check fails once with the QSA
# page-table message, as at 381's engine death (0: off); ARC3_S2_TEST_TURN_TRIP_MSG another EngineEvictionError message
TEST_TURN_TRIP = int(os.environ.get('ARC3_S2_TEST_TURN_TRIP', '0') or 0)
TEST_TURN_TRIP_MSG = os.environ.get('ARC3_S2_TEST_TURN_TRIP_MSG', '') or 'Turn exceeds the QSA descriptor page-table capacity'
TRAINED = int(os.environ.get('ARC3_S2_ROPE_TRAINED', '262144'))
CHUNK = 131072
GROUP = 4                                   # indexer_compress_ratio: QSA compressed-key groups of 4 tokens
SHIFT_SWITCH = os.environ.get('ARC3_S2_ROPE_SHIFT_SWITCH', '')
# rft78 (2026-10-21, the FP64 table test): rows from FP64_FROM on are built from FP64 angles (default TRAINED = production's table: FP32-angle
# rows below 262,144 kept bit-identical; 0 = the whole table from FP64 angles, then the table's float32 and BF16 at first use as before).
FP64_FROM = min(int(os.environ.get('ARC3_S2_ROPE_FP64_FROM', str(TRAINED)) or TRAINED), TRAINED)
STATS = dict(tables=0, rows=0, fp64_rows=0, session_limit=None, turn_limit_calls=0, runner_limit=None,
             shifted_requests=0, unsplit_requests=0, shift=None)

PINS = {'vllm.model_executor.layers.rotary_embedding.base':
        'c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5',
        'session_service': '7ce0eaef072e45354cfae8476719313a401387ea379b3fe9c3ca9c1ab40193e8',
        'engine_eviction': '1c49513acc9543b6f2392cdcf1f8c6eb172f84f343d853ff2464314f558ae2b5',
        'eviction_runner': '03ecae34a987b0b3203df42cd609f41c051226371c517ae85d69591851253ed8',
        'vllm.v1.worker.gpu.mm.rope': '8d1de6397e64e657a70a6ed4fc9c0fb64e7ca71c470d559bc7b7647281581d94'}


def _save():
    try:
        from . import install
        install.save()
    except Exception:                               # receipts are diagnostics; never fail the model build for them
        pass


def extend_table(cache, inv_freq32, rows=None, trained=None):
    """The extended table: cache[:trained] kept, rows trained..rows-1 from float64 angles (cos | sin)."""
    import torch
    rows = ROWS if rows is None else rows
    trained = TRAINED if trained is None else trained
    n = max(int(cache.shape[0]), rows)
    if cache.is_meta or inv_freq32.is_meta:          # modules built on the meta device (the PLE offload worker): shape only
        return torch.empty((n, cache.shape[1]), dtype=cache.dtype, device='meta')
    out = torch.empty((n, cache.shape[1]), dtype=cache.dtype, device=cache.device)
    out[:trained] = cache[:trained]
    inv = inv_freq32.detach().to('cpu', torch.float64)
    for a in range(trained, n, CHUNK):
        b = min(n, a + CHUNK)
        f = torch.outer(torch.arange(a, b, dtype=torch.float64, device='cpu'), inv)   # CPU even under a CUDA default device
        out[a:b] = torch.cat((f.cos(), f.sin()), dim=-1).to(cache.dtype).to(out.device)
    return out


def patch_base(mod):
    cls = mod.RotaryEmbeddingBase
    original = cls._compute_cos_sin_cache

    def _compute_cos_sin_cache(self):
        cache = original(self)
        if cache.shape[0] < TRAINED or cache.shape[1] != self.rotary_dim:
            return cache                                        # vision tables and unusual layouts: unchanged
        out = extend_table(cache, self._compute_inv_freq(self.base), trained=FP64_FROM)
        self.max_position_embeddings = int(out.shape[0])
        STATS['tables'] += 1
        STATS['rows'] = int(out.shape[0])
        STATS['fp64_rows'] = int(out.shape[0]) - FP64_FROM
        STATS['fp64_from'] = FP64_FROM
        _save()
        return out

    cls._compute_cos_sin_cache = _compute_cos_sin_cache


def patch_session_service(mod):
    original = mod.SessionManager.__init__

    def __init__(self, *args, max_logical_tokens=262144, **kwargs):
        limit = TEST_SESSION_LIMIT if TEST_SESSION_LIMIT > 0 else max(int(max_logical_tokens), ROWS)
        STATS['session_limit'] = limit
        original(self, *args, max_logical_tokens=limit, **kwargs)

    mod.SessionManager.__init__ = __init__


def patch_engine_eviction(mod):
    def validate_turn_capacity(scheduler, *, logical_tokens, resident_tokens, max_tokens, state=None):
        """engine_eviction.validate_turn_capacity with the logical limit raised to ARC3_S2_ROPE_ROWS (otherwise as
        production)."""
        STATS['turn_limit_calls'] += 1
        if TEST_TURN_TRIP and STATS['turn_limit_calls'] == TEST_TURN_TRIP:
            raise mod.EngineEvictionError(TEST_TURN_TRIP_MSG)
        logical_limit = max(int(scheduler.vllm_config.model_config.hf_text_config.max_position_embeddings), ROWS)
        if (type(max_tokens) is not int or max_tokens <= 0 or logical_tokens + max_tokens > logical_limit):
            raise mod.EngineEvictionError("Turn exceeds the original model position limit")
        padding = state.group_padding if state is not None else 0
        for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
            if getattr(manager, "_arc3_has_compressed", False):
                if resident_tokens + max_tokens + padding > manager._arc3_table_capacity:
                    raise mod.EngineEvictionError("Turn exceeds the QSA descriptor page-table capacity")

    mod.validate_turn_capacity = validate_turn_capacity


def patch_eviction_runner(mod):
    original = mod._limits

    def _limits(runner):
        out = original(runner)
        out = dict(out, max_logical_tokens=max(int(out['max_logical_tokens']), ROWS))
        STATS['runner_limit'] = out['max_logical_tokens']
        return out

    mod._limits = _limits
    original_alloc = mod._allocate_buffers

    def _allocate_buffers(runner):
        # 2026-10-04 (Fable's review): the QSA virtual lookup holds the retained groups in row order (4 slots each, -1 for
        # removed members) cut at resident + padding (historical_virtual_lookup), and every retained group has a resident
        # member, so its width never exceeds 4 x max_resident_tokens. Allocate that instead of the 2M logical limit
        # (8 MiB per lane); staging's own width check (cutoff <= lookup.shape[1]) still guards it.
        full = mod._limits

        def capped(r):
            out = full(r)
            width = min(int(out['max_logical_tokens']), 4 * int(out['max_resident_tokens']))
            STATS['lookup_width'] = width
            return dict(out, max_logical_tokens=width)
        mod._limits = capped
        try:
            return original_alloc(runner)
        finally:
            mod._limits = full

    mod._allocate_buffers = _allocate_buffers


def _read_shift():
    """(shift, split token id, prefix shift, from row) from the switch file
    '<shift> [<split id> [<prefix shift> [<from row>]]]'; (0, None, 0, 0) when absent. The prefix shift (2026-10-03 22:52,
    the rebase-design panel: the system turn moved to an effective distance D = shift - prefix shift from the rest)
    applies to the tokens before the split; 0 keeps them in place. From row (2026-10-04, the promptmove gate panel): the
    prefix shift applies only from that row on; the rows before it keep their positions (promptmove's test hook moves
    them, after they are computed, by the same amount)."""
    try:
        with open(SHIFT_SWITCH) as f:
            parts = f.read().split()
    except OSError:
        return 0, None, 0, 0
    if not parts:
        return 0, None, 0, 0
    shift = int(parts[0])
    split = int(parts[1]) if len(parts) > 1 else -1
    prefix_shift = int(parts[2]) if len(parts) > 2 else 0
    from_row = int(parts[3]) if len(parts) > 3 else 0
    return shift, split, prefix_shift, from_row


def patch_rope_state(mod):
    cls = mod.RopeState
    original = cls.init_prefill_positions

    def init_prefill_positions(self, req_idx, model, prefill_token_ids, mm_features):
        shift, split_id, prefix_shift, from_row = _read_shift()
        if not shift or not self.has_delta:
            return original(self, req_idx, model, prefill_token_ids, mm_features)
        if shift % GROUP or prefix_shift % GROUP or from_row % GROUP:
            raise ValueError('rope2m shifts %d / %d are not multiples of the QSA compression group (%d)'
                             % (shift, prefix_shift, GROUP))
        if prefix_shift and (split_id is None or split_id < 0):
            raise ValueError('rope2m prefix shift needs a split token')
        if split_id is None or split_id < 0:
            start = 0
        else:
            try:
                start = list(prefill_token_ids).index(split_id) + 1
                start = -(-start // GROUP) * GROUP                 # whole compression groups: none straddles the split
            except ValueError:                                  # no split token: production's positions, unshifted
                STATS['unsplit_requests'] += 1
                _save()
                return original(self, req_idx, model, prefill_token_ids, mm_features)
        positions, delta = model.get_mrope_input_positions(prefill_token_ids, mm_features)
        positions = positions.clone()
        positions[:, start:] += shift
        if prefix_shift:
            positions[:, from_row:start] += prefix_shift
        self.prefill_delta.np[req_idx] = delta + shift
        for i in range(self.num_dims):
            self.prefill_positions.stage_write(self.num_dims * req_idx + i, 0, positions[i].tolist())
        STATS['shifted_requests'] += 1
        STATS['shift'] = [shift, split_id, start, prefix_shift, from_row]
        _save()                                                 # test servers only: one receipt write per request

    cls.init_prefill_positions = init_prefill_positions


PATCHERS = {'vllm.model_executor.layers.rotary_embedding.base': patch_base,
            'session_service': patch_session_service,
            'engine_eviction': patch_engine_eviction,
            'eviction_runner': patch_eviction_runner}
if SHIFT_SWITCH:
    PATCHERS['vllm.v1.worker.gpu.mm.rope'] = patch_rope_state
