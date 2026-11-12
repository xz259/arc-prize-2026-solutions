"""stage2_v1 'rollback' part: GDN one-slot rollback by lazy replay on the production (fold B) layout.

- MambaBase.get_kv_cache_spec: num_speculative_blocks = 0 for every Mamba-type layer (GDN and PLE), so each request
  holds one state page per group. The GDN and PLE conv states keep their k extra columns and the stock
  num_accepted shift (both read column 0 only).
- Qwen3_8FlashNextForCausalLM.get_mamba_specs_from_config: the GDN spec's page grows by the pending area (two banks
  of (k + 1) tokens x W BF16 values), so vLLM's page alignment picks an attention block size whose page holds it.
- MambaBase.bind_kv_cache (GDN layers): after the stock unpack, a BF16 view of the page bytes that follow the conv
  and recurrent states is the layer's pending area; int32 flags per state slot live outside the pool.
- GDNAttentionMetadataBuilder.build: spec_state_indices is broadcast to k + 1 columns (the GDN block table has one);
  the conv update takes its width as the maximum query length.
- qwen_gdn_linear_attn: speculative fused_sigmoid_gating_delta_rule_update calls go to gdn_fused_db_step (exact,
  stock tiling) or, with the 'wy' part, gdn_wy_db_step; _forward_core clears the flags of prefill rows (M1: fresh
  requests, whose slots are reused; index_fill_, no host synchronisation) and counts prefill rows that resume with a pending window (M2 must commit
  those); the fused CUDA MTP kernel is refused.
- MambaHybridModelState.prepare_attn (2026-09-30, M2): a decode row scheduled with 0 drafts is a non-spec row (MRv2
  marks spec rows by drafts > 0), which every GDN path updates with the stock recurrence from the committed state,
  and whose conv updates read column 0. The scheduler gives a decoding lane a single token when the step's token
  budget is nearly spent (mixed prefill steps). After a verify step that lane's state lags by its pending window and
  its conv states are in the rolling layout, so before the forward (eager, outside CUDA graphs) such rows commit the
  pending window with the lane's last accepted count and realign the GDN and PLE conv states (persist_kernels; the
  kernels act only on slots with a pending window). Counted in STATS / the transition counter.
- scaled16 (2026-10-15, Claude fork 5d20b23b; ARC3_S2_GDN_STATE=scaled16, needs ARC3_S2_SSM_DTYPE=float16 and the 'wy' part; off by
  default): the GDN page grows by an FP32 block-scale area [HV, V / 32] after the pending area (768 bytes, inside the existing page
  padding; s16.py: S = s * M per value head and 32-row block). Every
  state site passes it: WY verify / commit (scale_state), the transition commit, persist's resume commit, 0-draft decode rows (routed
  from the stock update to s16.gdn_s16_decode) and the prefill read / store (the layer's chunk op wrapped by s16.wrap_chunk_op).
- gdn32 (2026-10-22, Claude session 099768dc; ARC3_S2_SSM_DTYPE=float32 with the 'wy' part, scaled16 off): an FP32 recurrent state.
  WY verify / commit take wy_kernels' F32 path; the resume / transition commit (persist_kernels, FP32 recurrence) and the stock
  prefill and 0-draft decode kernels already carry the state's dtype. The page doubles, so vLLM doubles the attention block.
"""
import dataclasses
import os
import time
from math import prod

import torch

from . import s16

WY = False
S16 = s16.ENABLED
_SCALE = {}        # ssm state data_ptr -> scaled16 block-scale view [blocks, HV, V / 32] FP32 (scaled16 only)
STATS = dict(registered=0, spec_calls=0, prefill_flag_clears=0, gdn_page_bytes=None, pending_bytes=None,
             block_table_broadcasts=0, transition_launches=0, transition_rows=0)
_REG = {}          # ssm state data_ptr -> (pend view, flags, W, max_tokens)
_BAD = []          # device counters: prefill rows with initial state and a pending window (one per layer)
_TRANSITION = []   # device counter: 0-draft decode rows whose lane had a pending window (committed before the forward)
# A/B switch (measurement only, 2026-09-30): with ARC3_S2_ROLLBACK_SWITCH=<file>, a file containing 'synced' restores the
# old prefill flag clear flags[idx] = 0, whose Python scalar value index_put_ copies from pageable host memory with a
# stream synchronisation in every GDN layer (36 per prefill step); read at most once a second.
_SWITCH = dict(path=os.environ.get('ARC3_S2_ROLLBACK_SWITCH'), t=0.0, synced=False)


def _synced_clear():
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


def pending_width(H, HV, K, V):
    return H * K + HV * V + 2 * HV


def pending_bytes_from_config(vllm_config):
    k = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
    if not k:
        return 0
    hf = vllm_config.model_config.hf_text_config
    tp = vllm_config.parallel_config.tensor_parallel_size
    W = pending_width(hf.linear_num_key_heads // tp, hf.linear_num_value_heads // tp, hf.linear_key_head_dim,
                      hf.linear_value_head_dim)
    extra = s16.scale_bytes(hf.linear_num_value_heads // tp, hf.linear_value_head_dim) if S16 else 0
    return 2 * (k + 1) * W * 2 + extra


def install_abstract(module):
    base = module.MambaBase
    orig_spec = base.get_kv_cache_spec
    orig_bind = base.bind_kv_cache

    def get_kv_cache_spec(self, vllm_config):
        spec = orig_spec(self, vllm_config)
        if spec is not None and getattr(spec, 'num_speculative_blocks', 0):
            spec = dataclasses.replace(spec, num_speculative_blocks=0)
        return spec

    def bind_kv_cache(self, kv_cache):
        orig_bind(self, kv_cache)
        if type(self).__name__ != 'QwenGatedDeltaNetAttention' or getattr(self, 'num_spec', 0) <= 0:
            return
        pages = kv_cache.squeeze(dim=(1, 2))
        off = sum(prod(s) * torch.empty((), dtype=d).element_size()
                  for s, d in zip(self.get_state_shape(), self.get_state_dtype()))
        H, HV = self.num_k_heads // self.tp_size, self.num_v_heads // self.tp_size
        W = pending_width(H, HV, self.head_k_dim, self.head_v_dim)
        T1 = self.num_spec + 1
        nbytes = 2 * T1 * W * 2
        assert pages.shape[1] >= off + nbytes, ('stage2 rollback: page has no room for the pending area',
                                                pages.shape[1], off, nbytes)
        assert off % 2 == 0 and pages.shape[1] % 2 == 0
        pend = pages[:, off:off + nbytes].view(torch.bfloat16)
        ssm = self.kv_cache[1]
        assert ssm.dtype in (torch.bfloat16, torch.float16, torch.float32), (
            'stage2 rollback needs BF16, FP16 or FP32 recurrent states', ssm.dtype)
        if ssm.dtype == torch.float32:                  # gdn32: the WY F32 path only (the fused_db kernels are untested on FP32)
            assert WY and not S16, ('gdn32: an FP32 state needs the wy part and no scaled16', WY, S16)
            STATS['gdn32'] = True
        if S16:
            assert WY, 'scaled16 needs the wy part (the fused_db kernels have no scaled path)'
            assert ssm.dtype == torch.float16, ('scaled16 needs ARC3_S2_SSM_DTYPE=float16', ssm.dtype)
            sb = s16.scale_bytes(HV, self.head_v_dim)
            o2 = off + nbytes
            assert pages.shape[1] >= o2 + sb and o2 % 4 == 0 and pages.shape[1] % 4 == 0, (
                'scaled16: page has no aligned room for the row scales', pages.shape[1], o2, sb)
            sc = pages[:, o2:o2 + sb].view(torch.float32).view(pages.shape[0], HV, self.head_v_dim // s16.BLK)
            if ssm.data_ptr() in _SCALE:
                assert _SCALE[ssm.data_ptr()].data_ptr() == sc.data_ptr()
            _SCALE[ssm.data_ptr()] = sc
            self._s16_scale = sc
            from vllm.forward_context import get_forward_context
            s16.wrap_chunk_op(self, sc, get_forward_context)
            STATS['s16'] = s16.STATS
        # Fold B (and the stock CSA layout) put the i-th layer of every GDN group in main tensor i: those layers have
        # identical state views (same address, shape and strides) and differ only in their block ids. They share one
        # pending view and one flag tensor, indexed by block id; the kernel (looked up by the state address) and the
        # prefill clearing (self._s2_flags) must see the same flags.
        key = ssm.data_ptr()
        if key in _REG:
            pend0, flags, W0, T10 = _REG[key]
            assert (W0, T10) == (W, T1) and pend0.data_ptr() == pend.data_ptr() and pend0.shape == pend.shape
            STATS['shared_state_tensors'] = STATS.get('shared_state_tensors', 0) + 1
        else:
            flags = torch.zeros(pages.shape[0], dtype=torch.int32, device=pages.device)
            _REG[key] = (pend, flags, W, T1)
        bad = torch.zeros(1, dtype=torch.int64, device=pages.device)
        self._s2_flags, self._s2_bad = flags, bad
        _BAD.append(bad)
        STATS['registered'] += 1
        STATS['gdn_page_bytes'] = int(pages.shape[1]); STATS['pending_bytes'] = int(nbytes)

    base.get_kv_cache_spec = get_kv_cache_spec
    base.bind_kv_cache = bind_kv_cache


def install_model(module):
    cls = module.Qwen3_8FlashNextForCausalLM
    orig = cls.get_mamba_specs_from_config.__func__

    def get_mamba_specs_from_config(klass, vllm_config):
        specs = orig(klass, vllm_config)
        extra = pending_bytes_from_config(vllm_config)
        if not extra:
            return specs
        gdn = specs[0]
        return (dataclasses.replace(gdn, page_size_padded=gdn.page_size_bytes + extra), *specs[1:])

    cls.get_mamba_specs_from_config = classmethod(get_mamba_specs_from_config)


def install_gdn_attn(module):
    cls = module.GDNAttentionMetadataBuilder
    orig_build = cls.build

    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
              num_decode_draft_tokens_cpu=None, fast_build=False):
        meta = orig_build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=num_accepted_tokens,
                          num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu, fast_build=fast_build)
        s = meta.spec_state_indices_tensor
        if s is not None and s.dim() == 2 and s.shape[1] != self.num_spec + 1:
            meta.spec_state_indices_tensor = s[:, :1].expand(-1, self.num_spec + 1).contiguous()
            STATS['block_table_broadcasts'] += 1
        return meta

    cls.build = build


def install_gdn(module):
    from .rb_kernels import gdn_fused_db_step
    from .wy_kernels import gdn_wy_db_step
    cls = module.QwenGatedDeltaNetAttention
    orig_core = cls._forward_core
    stock_update = module.fused_sigmoid_gating_delta_rule_update
    get_forward_context = module.get_forward_context

    def _forward_core(self, *args, **kwargs):
        assert not getattr(self, 'enable_fused_gdn_decode', False), 'stage2 rollback needs VLLM_GDN_DECODE_KERNEL=triton'
        flags = getattr(self, '_s2_flags', None)
        if flags is not None:
            md = get_forward_context().attn_metadata
            md = md.get(self.prefix) if isinstance(md, dict) else None
            if md is not None and md.num_prefills > 0 and md.prefill_state_indices is not None:
                idx = md.prefill_state_indices.long()
                if md.prefill_has_initial_state is not None:
                    resumed = md.prefill_has_initial_state & (flags[idx] > 0)
                    self._s2_bad += resumed.sum()
                if _synced_clear():
                    flags[idx] = 0
                else:
                    flags.index_fill_(0, idx, 0)        # the scalar is a kernel argument: no host copy, no sync
                STATS['prefill_flag_clears'] += 1
                from .install import save
                save(force=False)                   # eager prefill steps only; throttled to one write a minute
        return orig_core(self, *args, **kwargs)

    def update(*args, **kwargs):
        num_accepted = kwargs.get('num_accepted_tokens')
        idx = kwargs.get('ssm_state_indices')
        state = kwargs.get('initial_state')
        if num_accepted is None or idx is None or idx.ndim != 2 or state is None:
            if S16 and state is not None and state.data_ptr() in _SCALE:      # 0-draft decode rows on a scaled16 state
                assert not args, 'scaled16: the GDN update is called with keyword arguments'
                return s16.gdn_s16_decode(scale_state=_SCALE[state.data_ptr()], **kwargs)
            return stock_update(*args, **kwargs)
        assert state.data_ptr() in _REG, 'stage2 rollback: speculative call without a registered pending area'
        pend, flags, W, T1 = _REG[state.data_ptr()]
        if S16:
            o = gdn_wy_db_step(kwargs['A_log'], kwargs['a'], kwargs['b'], kwargs['dt_bias'], kwargs['q'], kwargs['k'],
                               kwargs['v'], state, pend, flags, kwargs['cu_seqlens'], idx, num_accepted, W, T1, pcol=0,
                               scale=kwargs.get('scale'), scale_state=_SCALE[state.data_ptr()])
            STATS['spec_calls'] += 1
            return o.unsqueeze(0), state
        step = gdn_wy_db_step if WY else gdn_fused_db_step
        o = step(kwargs['A_log'], kwargs['a'], kwargs['b'], kwargs['dt_bias'], kwargs['q'], kwargs['k'], kwargs['v'],
                 state, pend, flags, kwargs['cu_seqlens'], idx, num_accepted, W, T1, pcol=0, scale=kwargs.get('scale'))
        STATS['spec_calls'] += 1
        return o.unsqueeze(0), state

    cls._forward_core = _forward_core
    cls._can_use_fused_gdn_mtp_decode = lambda self, attn_metadata: False
    module.fused_sigmoid_gating_delta_rule_update = update


def _state_layers(state, kv_cache_config):
    """GDN layers with rollback flags and the PLE layer, each with its KV cache group index (cached on the state)."""
    cached = getattr(state, '_s2_rb_layers', None)
    if cached is not None:
        return cached
    ctx = state.vllm_config.compilation_config.static_forward_context
    group_of = {name: g for g, group in enumerate(kv_cache_config.kv_cache_groups) for name in group.layer_names}
    gdn = [(n, l, group_of[n]) for n, l in ctx.items()
           if type(l).__name__ == 'QwenGatedDeltaNetAttention' and getattr(l, '_s2_flags', None) is not None]
    ple = [(n, l, group_of[n]) for n, l in ctx.items() if type(l).__name__ == 'Qwen3_8FlashNextPLELayer']
    state._s2_rb_layers = (gdn, ple)
    return state._s2_rb_layers


def commit_transition(state, input_batch, block_tables, kv_cache_config, rows_np):
    """0-draft decode rows: commit a pending verify window (m = the lane's last accepted count) and realign the GDN and
    PLE conv states from the rolling layout (offset m - 1), only where the lane has a pending window."""
    import torch
    from .persist_kernels import gdn_resume_commit, realign_conv
    from .persist import _gdn_conv_view, _ple_conv_view
    gdn, ple = _state_layers(state, kv_cache_config)
    if not gdn:
        return
    dev = state.num_accepted_tokens_gpu.device
    rows = torch.from_numpy(rows_np.astype('int64')).to(dev, non_blocking=True)
    m = state.num_accepted_tokens_gpu.index_select(0, input_batch.idx_mapping.index_select(0, rows).long()).long()
    _, layer0, g0 = gdn[0]
    flags0 = _REG[layer0.kv_cache[1].data_ptr()][1]
    pending = flags0.index_select(0, block_tables[g0].index_select(0, rows)[:, 0].long()) > 0
    offs = torch.where(pending, m - 1, torch.zeros_like(m))
    if not _TRANSITION:
        _TRANSITION.append(torch.zeros(1, dtype=torch.int64, device=dev))
    _TRANSITION[0] += pending.sum()
    for _, layer, g in gdn:
        slots = block_tables[g].index_select(0, rows)[:, 0].long()
        ssm = layer.kv_cache[1]
        pend, flags, W, T1 = _REG[ssm.data_ptr()]
        gdn_resume_commit(layer.A_log, layer.dt_bias, ssm, pend, flags, slots, m, layer.num_k_heads // layer.tp_size,
                          layer.head_k_dim, W, T1, scale_state=_SCALE.get(ssm.data_ptr()))
        realign_conv(_gdn_conv_view(layer), slots, offs, layer.conv_kernel_size - 1)
        flags.index_fill_(0, slots, 0)
    for _, layer, g in ple:
        if layer.conv_state_len > 0:
            realign_conv(_ple_conv_view(layer), block_tables[g].index_select(0, rows)[:, 0].long(), offs,
                         layer.conv_state_len)
    STATS['transition_launches'] += 1
    STATS['transition_rows'] += int(len(rows_np))


def install_mamba_state(module):
    """last_spec (CPU, per request state index): the lane's latest scheduled step was a verify step, i.e. its GDN slot
    holds a pending window when the next step runs (stream order); prefill rows, re-added requests and committed
    transitions reset it. Only 0-draft decode rows of such lanes launch the commit."""
    cls = module.MambaHybridModelState
    orig, orig_add = cls.prepare_attn, cls.add_request

    def _last_spec(self):
        v = getattr(self, '_s2_last_spec', None)
        if v is None:
            import numpy as np
            v = self._s2_last_spec = np.zeros(self.max_num_reqs, dtype=bool)
        return v

    def add_request(self, req_index, new_req_data):
        _last_spec(self)[req_index] = False
        return orig_add(self, req_index, new_req_data)

    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                     for_capture=False):
        if not for_capture and _REG and self.vllm_config.num_speculative_tokens:
            import numpy as np
            n = input_batch.num_reqs
            idx = np.asarray(input_batch.idx_mapping_np[:n])
            sched = np.asarray(input_batch.num_scheduled_tokens[:n])
            drafts = input_batch.num_draft_tokens_per_req
            drafts = np.zeros(n, dtype=np.int64) if drafts is None else np.asarray(drafts[:n])
            spec = (drafts > 0) & (sched == drafts + 1)
            last = _last_spec(self)
            rows = np.nonzero((sched == 1) & (drafts == 0) & last[idx]
                              & ~np.asarray(input_batch.is_prefilling_np[:n], dtype=bool))[0]
            last[idx] = spec
            if len(rows):
                commit_transition(self, input_batch, block_tables, kv_cache_config, rows)
        return orig(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                    for_capture=for_capture)

    cls.prepare_attn, cls.add_request = prepare_attn, add_request


def transition_pending_rows():
    return int(_TRANSITION[0].item()) if _TRANSITION else 0


def bad_resumes():
    return int(sum(int(b.item()) for b in _BAD)) if _BAD else 0


def reset_bad_resumes():
    """Zero the counters and return their total (the persist part calls this before the first served request: the
    startup dummy runs leave flagged slots that dummy prefill rows then count)."""
    total = bad_resumes()
    for b in _BAD:
        b.zero_()
    return total


PATCHERS = {
    'vllm.model_executor.layers.mamba.abstract': install_abstract,
    'vllm.models.qwen3_8_flash_next.nvidia.model': install_model,
    'vllm.v1.attention.backends.gdn_attn': install_gdn_attn,
    'vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn': install_gdn,
    'vllm.v1.worker.gpu.model_states.mamba_hybrid': install_mamba_state,
}
PINS = {
    'vllm.model_executor.layers.mamba.abstract': 'f0c1093395d3be1e7d26cbf20ffcfe877d5048212332c487d879c11bcdad5617',
    'vllm.models.qwen3_8_flash_next.nvidia.model': 'd900cd6fcacba18f460e00b3f018fbf36fbe6ecc310692b6adb1451f7f53cc17',
    'vllm.v1.attention.backends.gdn_attn': 'fda86b96ab5daaf50bd02d022518779c220401dbedc7b28cf478f4c48e72d3d3',
    'vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn': '81b4dcd0952492375c93bffc2cdf45f10b45ab5e117f2e1d949a147d144e64f0',
    'vllm.v1.worker.gpu.model_states.mamba_hybrid': '2e65f1da0a440f11a784bd36afe9206a7a2f674d522c047b838ad604e4c6acc0',
}
