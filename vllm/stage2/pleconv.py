"""stage2_v1 pleconv: memory-bounded PLE short-conv prefill (stage-2 memory optimisation, 2026-09-29).

Qwen3_8FlashNextPLELayer._short_conv_dilated_prefill_batched packs every prefill request of a step into a zero-padded
[num_prefills, max_len, 10,240] BF16 tensor and then holds about five tensors of that size (pack, transposed copy,
history, conv output, SiLU and its transposed copy). Two ~8K prefills in one step cost ~1.9 GiB (memdiag: the 26-lane
burst peak), and the padded size is unbounded: one long turn plus several short ones pads every short one to the long
length.

This part keeps the same operations on each group of consecutive prefills whose padded size fits a token budget
(BUDGET, default max_num_batched_tokens), and releases the pack and the history as soon as they are consumed; SiLU runs
in place and the output rows are gathered from the untransposed conv output. Per request the arithmetic is unchanged
(same depthwise conv1d kernel, same state rows), which test_pleconv.py checks bitwise against the original.

The group layout needs the prefill lengths on the host: the metadata builder already reads query_start_loc_cpu for
max_prefill_query_len; its wrapper attaches the prefill lengths (the last num_prefills requests of the reordered
batch). Without them, or on any mismatch, the original function runs (counted).

Sync-free speculative build (2026-09-30, GAP_PLAN item 3): the builder's spec path runs with non-blocking index copies and
index_fill_ scalar puts (identical tensors, no stream synchronisation); switch and sampled verification below."""
import inspect
import os
import textwrap
import time

import torch
import torch.nn.functional as F

PINS = {
    'vllm.models.qwen3_8_flash_next.nvidia.ple_layer': '160d553ec81bae0990934cfe7c14f98d2d65b6ee3d5c5a85ff8421b7ddb17952',
    'vllm.v1.attention.backends.short_conv_attn': '985fd9f320eff615f2bc4b4d5576cd7f301a751380709979a251f92101bda14d',
}
BUDGET = int(os.environ.get('ARC3_S2_PLECONV_BUDGET', '0') or 0)     # tokens per padded group; 0 = max_num_batched_tokens
STATS = dict(calls=0, grouped_calls=0, groups=0, fallback_no_lens=0, fallback_mismatch=0, max_padded_tokens=0,
             lens_attached=0, lens_mismatch=0)
_NULL = [None]
_ORIGINAL = [None]


def _budget(self, num_prefill_tokens):
    if BUDGET > 0:
        return BUDGET
    return getattr(self, '_arc3_pleconv_budget', None) or max(8192, num_prefill_tokens)


def _groups(lens, budget):
    out, a, n = [], 0, len(lens)
    while a < n:
        b, m = a + 1, lens[a]
        while b < n and (b + 1 - a) * max(m, lens[b]) <= budget:
            m = max(m, lens[b]); b += 1
        out.append((a, b, m)); a = b
    return out


def _one_group(self, x_g, q_starts, max_len, conv_state, conv_weights, state_indices_g, has_initial_g, output_g):
    """The original per-step computation restricted to one group; q_starts[0] == 0 (int64, on device)."""
    num_prefills = state_indices_g.numel()
    num_tokens = x_g.shape[0]
    lengths = q_starts[1:] - q_starts[:-1]
    hidden_size = x_g.shape[1]
    positions = torch.arange(num_tokens, device=x_g.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    packed_tokens = x_g.new_zeros((num_prefills, max_len, hidden_size))
    packed_tokens[req_indices, col_indices] = x_g
    packed_tokens = packed_tokens.transpose(1, 2).contiguous()

    state_indices = state_indices_g.to(device=conv_state.device, dtype=torch.int64)
    valid_state = state_indices != _NULL[0]
    state_indices = torch.where(valid_state, state_indices, torch.zeros_like(state_indices))
    has_initial = has_initial_g.to(device=conv_state.device, dtype=torch.bool)
    if self.conv_state_len > 0:
        if conv_state.shape[0] == 0:
            state = conv_state.new_zeros((num_prefills, hidden_size, self.conv_state_len), dtype=x_g.dtype)
        else:
            state = conv_state.index_select(0, state_indices)[..., : self.conv_state_len].to(x_g.dtype)
        use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
        initial_state = torch.where(use_initial_mask, state, torch.zeros_like(state))
        history = torch.cat((initial_state, packed_tokens), dim=-1)
        del packed_tokens, state, initial_state
    else:
        history = packed_tokens
        del packed_tokens

    next_state = None
    if self.conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(num_prefills, 1, 1)
        state_offsets = torch.arange(self.conv_state_len, device=history.device,
                                     dtype=torch.int64).view(1, 1, self.conv_state_len)
        next_state = history.gather(dim=2, index=(state_starts + state_offsets).expand(-1, history.size(1), -1))

    conv_output = F.conv1d(history, conv_weights.unsqueeze(1).contiguous(), groups=history.size(1),
                           dilation=self.short_conv_dilation)
    del history
    F.silu(conv_output, inplace=True)

    token_positions = torch.arange(max_len, device=x_g.device, dtype=torch.int64)
    valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
    valid_output_mask = valid_tokens & valid_state.to(device=x_g.device).view(num_prefills, 1)
    conv_output.masked_fill_(~valid_output_mask.unsqueeze(1), 0)
    output_g.copy_(conv_output[req_indices, :, col_indices])
    del conv_output

    if next_state is not None:
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., : self.conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(update_mask.view(num_prefills, 1, 1), next_state.to(conv_state.dtype),
                                      existing_base_state)
        existing_state[..., : self.conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)


def prefill_bounded(self, x_p, metadata, conv_state, conv_weights, state_indices_tensor_p, num_prefills,
                    num_decode_tokens, num_prefill_tokens):
    STATS['calls'] += 1
    original = _ORIGINAL[0]
    lens = getattr(metadata, 'arc3_prefill_lens', None)
    if lens is None:
        STATS['fallback_no_lens'] += 1
        return original(self, x_p, metadata, conv_state, conv_weights, state_indices_tensor_p, num_prefills,
                        num_decode_tokens, num_prefill_tokens)
    if (num_prefills == 0 or x_p.numel() == 0 or len(lens) != num_prefills or sum(lens) != num_prefill_tokens
            or max(lens) != metadata.max_prefill_query_len or metadata.non_spec_query_start_loc is None
            or metadata.has_initial_states_p is None or state_indices_tensor_p.numel() < num_prefills):
        STATS['fallback_mismatch'] += 1
        return original(self, x_p, metadata, conv_state, conv_weights, state_indices_tensor_p, num_prefills,
                        num_decode_tokens, num_prefill_tokens)
    STATS['grouped_calls'] += 1
    groups = _groups(lens, _budget(self, num_prefill_tokens))
    STATS['groups'] += len(groups)
    STATS['max_padded_tokens'] = max(STATS['max_padded_tokens'], max((b - a) * m for a, b, m in groups))
    q_starts_all = (metadata.non_spec_query_start_loc[-num_prefills - 1:] - num_decode_tokens).to(torch.int64)
    has_initial_all = metadata.has_initial_states_p[:num_prefills]
    output = torch.empty_like(x_p)
    offs = [0]
    for n in lens:
        offs.append(offs[-1] + n)
    for a, b, m in groups:
        t0, t1 = offs[a], offs[b]
        _one_group(self, x_p[t0:t1], q_starts_all[a:b + 1] - q_starts_all[a], m, conv_state, conv_weights,
                   state_indices_tensor_p[a:b], has_initial_all[a:b], output[t0:t1])
    try:
        from .install import save
        save(force=False)
    except Exception:
        pass
    return output


def patch_ple_layer(mod):
    cls = mod.Qwen3_8FlashNextPLELayer
    _ORIGINAL[0] = cls._short_conv_dilated_prefill_batched
    _NULL[0] = mod.NULL_BLOCK_ID
    cls._short_conv_dilated_prefill_batched = prefill_bounded
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        from vllm.config import get_current_vllm_config
        self._arc3_pleconv_budget = int(get_current_vllm_config().scheduler_config.max_num_batched_tokens)
        STATS['budget'] = self._arc3_pleconv_budget

    cls.__init__ = __init__


# Sync-free speculative build (GAP_PLAN item 3, 2026-09-30). The fork's spec path copies CPU index tensors to the GPU
# with blocking .to(device) (memcpy_and_sync: a stream synchronisation each) and puts Python scalars through advanced
# indexing (a pageable value copy and a synchronisation each): 3 synchronisations in a decode step's build, 6 in a
# mixed step's. In the pipelined engine the worker then waits inside the builder for the previous step's GPU work
# (41.8 of 43.3 ms per execute_model in a decode wave). The same copies with non_blocking=True (pageable host data is
# staged at call time; no stream drain) and index_fill_ for the scalar puts give identical tensors.
SYNCFREE = [
    ('spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)',
     'spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device, non_blocking=True)', 1),
    ('non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)',
     'non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device, non_blocking=True)', 1),
    ('req_group[spec_req_idx] = 0', 'req_group.index_fill_(0, spec_req_idx, 0)', 1),
    ('req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1',
     'req_group.index_fill_(0, decode_req_idx_cpu.to(query_start_loc.device, non_blocking=True), 1)', 1),
    ('spec_req_idx_cpu.to(num_accepted_tokens.device)',
     'spec_req_idx_cpu.to(num_accepted_tokens.device, non_blocking=True)', 1),
    ('non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)',
     'non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device, non_blocking=True)', 1),
]
# Default since 2026-09-30 21:10 (ARC3_S2_PLECONV_SYNCFREE=0 restores the fork's build). Exact, 3-6 syncs fewer per step.
# A between-run comparison showed nothing (g4d against g4sf); the in-run A/B (g4f, GAP_PLAN item 4b) measured plain steps
# 0.45 ms shorter at p50 and +0.62% +/- 0.65 tok/s: the synced build held the worker until the drafter's GPU tail ended.
# The index copies' sources are fresh pageable tensors (nonzero / cat), so non_blocking H2D copies are staged safely.
# A/B switch (measurement only): ARC3_S2_PLECONV_SWITCH=<file>, a file containing 'synced' restores the fork's build;
# read at most once a second. ARC3_S2_PLECONV_VERIFY=N: one build in N also runs the fork's build on the same inputs and
# compares every tensor field of the result (a difference raises).
SYNCFREE_ON = os.environ.get('ARC3_S2_PLECONV_SYNCFREE', '1') != '0'
_SWITCH = dict(path=os.environ.get('ARC3_S2_PLECONV_SWITCH'), t=0.0, synced=False)
VERIFY = int(os.environ.get('ARC3_S2_PLECONV_VERIFY', '0') or 0)
STATS.update(syncfree_builds=0, synced_builds=0, verified=0)


def _synced():
    if not _SWITCH['path']:
        return False
    now = time.monotonic()
    if now - _SWITCH['t'] > 1.0:
        _SWITCH['t'] = now
        try:
            with open(_SWITCH['path']) as f:
                _SWITCH['synced'] = f.read().strip() == 'synced'
        except OSError:
            _SWITCH['synced'] = False
    return _SWITCH['synced']


def _syncfree_build(mod, original):
    src = textwrap.dedent(inspect.getsource(original))
    assert 'super()' not in src, 'pleconv: the fork build uses super(); cannot recompile outside the class'
    for old, new, n in SYNCFREE:
        assert src.count(old) == n, ('pleconv sync-free anchor', old, src.count(old))
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, '<stage2 pleconv sync-free PleShortConvAttentionMetadataBuilder.build>', 'exec'), mod.__dict__, ns)
    return ns['build']


def _same(a, b):
    for k, va in vars(a).items():
        vb = getattr(b, k, None)
        if isinstance(va, torch.Tensor):
            if not (isinstance(vb, torch.Tensor) and va.shape == vb.shape and torch.equal(va, vb)):
                return k
        elif isinstance(va, (int, float, bool, str)) or va is None:
            if va != vb:
                return k
    return None


def patch_short_conv_attn(mod):
    cls = mod.PleShortConvAttentionMetadataBuilder
    original = cls.build
    fast = _syncfree_build(mod, original) if SYNCFREE_ON else None
    tick = [0]

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        if fast is None or _synced():
            STATS['synced_builds'] += 1
            md = original(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
        else:
            STATS['syncfree_builds'] += 1
            md = fast(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
            tick[0] += 1
            if VERIFY and tick[0] % VERIFY == 0:
                ref = original(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
                bad = _same(md, ref) or _same(ref, md)
                if bad:
                    raise RuntimeError(f'pleconv sync-free build differs from the fork build in {bad}')
                STATS['verified'] += 1
        n_p = getattr(md, 'num_prefills', 0) or 0
        if n_p > 0:
            n = int(common_attn_metadata.num_reqs)
            qsl = common_attn_metadata.query_start_loc_cpu
            lens = torch.diff(qsl[n - n_p: n + 1]).tolist()
            if max(lens) == md.max_prefill_query_len:
                md.arc3_prefill_lens = lens
                STATS['lens_attached'] += 1
            else:
                STATS['lens_mismatch'] += 1
        return md

    cls.build = build


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.ple_layer': patch_ple_layer,
            'vllm.v1.attention.backends.short_conv_attn': patch_short_conv_attn}
