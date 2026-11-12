"""stage2_v1 'vsampler' part: MRv2 RejectionSampler._verify through the candidate verify sampler (kernel work block 2,
item 5; cand_verify.py = sampling/cand_sampler.py) for eligible batches, else the stock path.

Eligible: probabilistic drafts (draft logits present), standard rejection, no logprobs, every row with temperature > 0
(2026-10-04; was = 1 until cand_verify stopped re-dividing the processed logits by T) and 1 <= top_k <= KMAX - 1 (the
production gameplay sampling). The value_bf16 shortcut (raw logits) covers any T > 0 since 2026-10-16 (the candidate kernel
divides by T as the stock _temperature_kernel; ARC3_S2_VSAMPLER_VALUE_BF16_T=0 restores T = 1 only). Penalties, logit bias and min_p still go through the
stock apply_sampling_params (skip_top_k_top_p=True); the candidate kernels then apply the reference top-k / top-p
rule and run the stock acceptance test and residual Gumbel-max with the stock noise, so the output distribution is
the stock one (token-identical offline on reference-masked logits; ties beyond KMAX are flagged and counted).

2026-09-30: both this sampler and the stock fallback key the resample (correction / bonus) Gumbel noise by pos + 2^30.
Stock keys it by pos, the same key the drafter used to sample the draft, so a correction reused the noise that
selected the rejected draft and the output distribution was not the target's (cand_verify.py docstring).

2026-09-30, backport of upstream vLLM #58784 (fedbc3b): draft slots that were never proposed are verified as -1. The
scheduler pads a 1-token waiting request (our continuation resume, num_computed_tokens = len - 1) to 1 + k rows while
other lanes decode, and MRv2 fills the k draft inputs from req_states.draft_tokens, which add_request zeroes. Verified as
a draft, that token 0 made the first token after the resume a draw from max(p - q, 0) with the request slot's stale
draft-logits row as q (on rejection), or token 0 itself (on acceptance, p(0) > 0). Slots with local_pos > 0 in a step
that starts inside the prefill (pos - local_pos < prefill_len) are placeholders; the rejection kernels (stock and
candidate) stop at a -1 draft and resample that row from the target. apply_sampling_params still receives the input
tokens (row 0, the only row sampled from such a step, never depends on later draft slots), so processed logits are
unchanged.

2026-10-03 (Fable's review): eligible batches whose rows need no logits processing besides top-k / top-p skip
apply_sampling_params's [rows, V] FP32 copy and pass the raw logits with value_bf16=True (bitwise-identical candidate
keys; see _plain_rows). ARC3_S2_VSAMPLER_VALUE_BF16=0 restores the copy.

2026-10-04 (overnight item b): ARC3_S2_BLOCKVERIFY=1 runs block verification (cand_verify._block_verify_kernel; lossless,
expected accepted length >= token verification's) on eligible batches; ARC3_S2_BLOCKVERIFY_SWITCH=<file> selects it per
ab_switch.py window ('off' = token verification, anything else = block; read at most once a second). The stock fallback
keeps token verification (both are exact).

2026-10-08 (user: "If it is something that should not appear in decoding, let's just block it"): ARC3_S2_BAN_TOKENS=<ids>
(comma list; lc_run3.sh sets 248044 = <|endoftext|> of the Qwen3.8 tokenizer) gives those ids -inf in the target's logits at
the top of _verify, before both the candidate and the stock path, so they are never accepted as drafts, drawn as a
correction or drawn as the bonus token: the output is exact for the masked target. Three gameplay turns of run 396 ended in
<|endoftext|> instead of <|im_end|> (the harness then restarts the level). vLLM refuses logit_bias under speculative
decoding, hence the mask here. Prompt logprobs (the NLL panel) are computed elsewhere and are unchanged."""
import os
import time

import numpy as np
import torch

BAN = tuple(int(t) for t in os.environ.get('ARC3_S2_BAN_TOKENS', '').split(',') if t.strip())
STATS = dict(calls=0, fallbacks={}, workspace_rows=0, value_bf16_calls=0, block_verify_calls=0, block_verify_switches=0,
             banned_tokens=list(BAN), ban_calls=0)
_STATE = dict(ws=None, overflow=None, placeholders=None)
VALUE_BF16 = os.environ.get('ARC3_S2_VSAMPLER_VALUE_BF16', '1') != '0'
VALUE_BF16_T = os.environ.get('ARC3_S2_VSAMPLER_VALUE_BF16_T', '1') != '0'    # 2026-10-16: the fast path at T != 1 too
BLOCK_VERIFY = os.environ.get('ARC3_S2_BLOCKVERIFY', '0') == '1'
_BV = dict(path=os.environ.get('ARC3_S2_BLOCKVERIFY_SWITCH'), t=0.0, on=BLOCK_VERIFY)


def _block_verify_on():
    """Block or token verification for this call: the env default, or the switch file's level (at most one read a
    second; a missing file keeps the default)."""
    s = _BV
    if s['path'] is None:
        return s['on']
    now = time.monotonic()
    if now - s['t'] >= 1.0:
        s['t'] = now
        try:
            with open(s['path']) as f:
                on = f.read().strip() != 'off'
        except OSError:
            on = BLOCK_VERIFY
        if on != s['on']:
            STATS['block_verify_switches'] += 1
        s['on'] = on
    return s['on']


def _sync_drafthead():
    """drafthead's in-run truncation switch (a no-op unless ARC3_S2_DRAFTTRUNC_SWITCH is set)."""
    try:
        from .drafthead import sync_trunc_switch
    except ImportError:
        return
    sync_trunc_switch()


def _plain_rows(sampler, idx_np):
    """2026-10-03 (Fable's review): True when apply_sampling_params(skip_top_k_top_p=True) would only copy the logits
    to FP32 and apply the temperature for every row: no logit bias, penalties, bad words, thinking budget or min_p. Its
    needs_logits_processing flag is set by top-k / top-p (and T != 1) alone, which the candidate kernels apply. The raw
    logits then go to the candidate kernels with value_bf16=True: same keys bit for bit (the BF16 top-k threshold converts
    exactly; at T != 1 the kernel divides as the stock _temperature_kernel, since 2026-10-16), no [rows, V] FP32 copy. The processed logits returned by _verify are only read
    for logprobs, which make a batch ineligible."""
    tb = sampler.thinking_budget_state
    return not (np.any(sampler.logit_bias_state.use_logit_bias[idx_np])
                or np.any(sampler.penalties_state.use_penalty[idx_np])
                or np.any(sampler.bad_words_state.num_bad_words.np[idx_np] > 0)
                or (tb.enabled and np.any(tb.use_thinking_budget[idx_np]))
                or np.any(sampler.sampling_states.min_p.np[idx_np] != 0.0))


def ban_logits(logits):
    """The banned token ids (ARC3_S2_BAN_TOKENS) set to -inf in place in every row of the target's logits."""
    if not BAN:
        return logits
    idx = _STATE.get('ban_idx')
    if idx is None or idx.device != logits.device:
        idx = _STATE['ban_idx'] = torch.tensor(BAN, dtype=torch.long, device=logits.device)
    logits.index_fill_(1, idx, float('-inf'))
    STATS['ban_calls'] += 1
    return logits


def _fallback(reason):
    STATS['fallbacks'][reason] = STATS['fallbacks'].get(reason, 0) + 1


MIX_BINS = ('0', '1', '2', '3', '4-7', '8+')


def _mix_account(cu_num_logits, num_sampled, k):
    """Acceptance diagnostics (overnight step 4): drafts verified in this step were proposed at the end of the previous
    step; bin them by that step's distance from the last mixed step (a step where some request has fewer than 1 + k
    logit rows: a prefill row). GPU counters only, no host synchronisation."""
    m = _STATE.get('mix')
    if m is None:
        dev = cu_num_logits.device
        m = _STATE['mix'] = dict(since=torch.full((1,), 99, dtype=torch.int64, device=dev),
                                 acc=torch.zeros(len(MIX_BINS), dtype=torch.int64, device=dev),
                                 dr=torch.zeros(len(MIX_BINS), dtype=torch.int64, device=dev))
    n = (cu_num_logits[1:] - cu_num_logits[:-1]).long()
    drafts = (n - 1).clamp(min=0)
    acc = torch.minimum((num_sampled.long() - 1).clamp(min=0), drafts)
    s = m['since']
    b = torch.where(s >= 8, torch.full_like(s, 5), torch.where(s >= 4, torch.full_like(s, 4), s))
    m['acc'].index_add_(0, b, acc.sum().view(1))
    m['dr'].index_add_(0, b, drafts.sum().view(1))
    m['since'].copy_(torch.where((n < k + 1).any(), torch.zeros_like(s), s + 1))


def mix_report():
    m = _STATE.get('mix')
    if m is None:
        return None
    acc, dr = m['acc'].tolist(), m['dr'].tolist()
    return {b: dict(drafts=d, accept=round(a / d, 4) if d else None) for b, a, d in zip(MIX_BINS, acc, dr)}


def mask_unproposed(draft_sampled, pos, expanded_idx_mapping, expanded_local_pos, prefill_len):
    """Draft tokens as the rejection kernels must see them: -1 in the draft slots of a step that starts inside the
    prefill (upstream gather_draft_sampled). Counts the masked slots on the GPU (no host sync)."""
    placeholder = (expanded_local_pos > 0) & (pos - expanded_local_pos < prefill_len[expanded_idx_mapping])
    if _STATE['placeholders'] is None:
        _STATE['placeholders'] = torch.zeros(1, dtype=torch.int64, device=pos.device)
    _STATE['placeholders'] += placeholder.sum()
    return draft_sampled.masked_fill(placeholder, -1)


def install(module):
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
    from .cand_verify import KMAX, Workspace, cand_rejection_sample
    cls = module.RejectionSampler

    def _verify(self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                expanded_idx_mapping, expanded_local_pos):
        st = self.sampler.sampling_states
        _sync_drafthead()
        if BAN:
            ban_logits(logits)                      # 10-08: banned ids (ARC3_S2_BAN_TOKENS) never sampled
        verify_drafts = mask_unproposed(draft_sampled, pos, expanded_idx_mapping, expanded_local_pos,
                                        self.sampler.req_states.prefill_len.gpu)
        if draft_logits is None:
            _fallback('no_draft_logits')
        elif self.use_block_verification or self.synthetic_conditional_rates is not None:
            _fallback('rejection_method')
        elif st.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS:
            _fallback('logprobs')
        elif not bool(np.all(st.temperature.np[idx_mapping_np] > 0.0)):      # 10-04: any T > 0 (cand_verify fix)
            _fallback('temperature')
        elif not bool(np.all((st.top_k.np[idx_mapping_np] >= 1) & (st.top_k.np[idx_mapping_np] <= KMAX - 1))):
            _fallback('top_k')
        else:
            t_one = bool(np.all(st.temperature.np[idx_mapping_np] == 1.0))
            if VALUE_BF16 and (t_one or VALUE_BF16_T) and _plain_rows(self.sampler, idx_mapping_np):
                processed, value_bf16 = logits, True    # the stock pass would only copy to FP32 (and divide by T)
                STATS['value_bf16_calls'] += 1
                if not t_one:
                    STATS['value_bf16_scaled_calls'] = STATS.get('value_bf16_scaled_calls', 0) + 1
            else:
                processed, value_bf16 = self.sampler.apply_sampling_params(
                    logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos, draft_sampled, expanded_local_pos,
                    skip_top_k_top_p=True), False
            top_k, top_p = st.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
            if top_p is None:
                top_p = torch.ones(processed.shape[0], dtype=torch.float32, device=processed.device)
            rows = processed.shape[0]
            if _STATE['ws'] is None or STATS['workspace_rows'] < rows:
                n = max(rows, 512)
                _STATE['ws'] = Workspace(n, processed.shape[1], processed.device)
                _STATE['overflow'] = torch.zeros(1, dtype=torch.int64, device=processed.device)
                STATS['workspace_rows'] = n
            bv = _block_verify_on()
            STATS['block_verify_calls'] += int(bv)
            sampled, num_sampled, flags = cand_rejection_sample(
                processed, draft_logits, verify_drafts, cu_num_logits, pos, idx_mapping, expanded_idx_mapping,
                expanded_local_pos, st.temperature.gpu, st.seeds.gpu, top_k, top_p, self.num_speculative_steps,
                _STATE['ws'], value_bf16=value_bf16, block_verify=bv)
            _STATE['overflow'] += flags.sum()
            STATS['calls'] += 1
            _mix_account(cu_num_logits, num_sampled, self.num_speculative_steps)
            from .install import save
            save(force=False)                       # throttled to one write a minute
            return processed, sampled, num_sampled
        # The pinned stock _verify, except that the rejection kernels receive verify_drafts.
        processed = self.sampler.apply_sampling_params(logits, expanded_idx_mapping, idx_mapping, idx_mapping_np,
                                                       pos, draft_sampled, expanded_local_pos)
        sampled, num_sampled = module.rejection_sample(
            processed, draft_logits, verify_drafts, cu_num_logits, pos, idx_mapping, expanded_idx_mapping,
            expanded_local_pos, st.temperature.gpu, st.seeds.gpu, self.num_speculative_steps,
            self.synthetic_conditional_rates, use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification)
        _mix_account(cu_num_logits, num_sampled, self.num_speculative_steps)
        from .install import save
        save(force=False)
        return processed, sampled, num_sampled

    cls._verify = _verify


def overflow_rows():
    return int(_STATE['overflow'].item()) if _STATE['overflow'] is not None else 0


def placeholder_slots():
    return int(_STATE['placeholders'].item()) if _STATE['placeholders'] is not None else 0


def transform_rejection_utils(src):
    """Stock fallback path (logprobs, other temperatures, ...): the resample kernel's Gumbel noise is keyed by
    pos + 2^30 instead of pos, independent of the drafter's (seed, pos) draft noise (see cand_verify.py)."""
    a = ('        seed,\n        pos,\n        cumulative_log_p,\n        vocab_size,\n'
         '        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,\n')
    b = ('        seed,\n        pos + (1 << 30),   # stage 2: independent of the draft noise\n        cumulative_log_p,\n'
         '        vocab_size,\n        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,\n')
    assert src.count(a) == 1, 'stage2 vsampler: resample launch anchor'
    return src.replace(a, b)


PATCHERS = {'vllm.v1.worker.gpu.spec_decode.rejection_sampler': install,
            'vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils': ('source', transform_rejection_utils)}
PINS = {'vllm.v1.worker.gpu.spec_decode.rejection_sampler':
        '308d4171f892f27231655289ceb03cd4566533c7d26b8f98bd62ef0a34f986f0',
        'vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils':
        '659e82c2ce1249a6e614f7adf62e3824329a7b79e7a4caa55cfadf254068a98a'}
