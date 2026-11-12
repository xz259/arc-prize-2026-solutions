"""stage2_v1 'topkfix' part: vLLM's top-k / top-p masking through an exact implementation, so the stock Triton kernel
with the heavy-tie mis-truncation (M2_DESIGN §13) is unreachable from the stack (GAP_PLAN, 2026-09-30).

vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p sends GPU batches of 8 or more rows to apply_top_k_top_p_triton
(the "Qrita" pivot kernel). diag_tie_rows.py measured it against an exact float64 reference: 5-6 wrong rows per
16,384 in heavy-tie BF16 rows, keeping too few tokens (11 where 19 were needed) or more than k (21 and 25 with k = 20).
Its duplicate handling rebuilds the tied logit through a log/exp round trip and matches it with a 1e-9 absolute
tolerance; no upstream fix exists. In the MTP3 candidate the verify sampler (vsampler, exact tie rule) serves every
gameplay row and the drafter samples with temperature only, so only vsampler's stock fallback (logprobs requests,
other sampling parameters) still reached the kernel.

Replacement (same signature and in-place contract, no host synchronisation): the rule of vsampler and of the exact
reference: order by logit descending, token id ascending (a stable sort of the negated logits); keep exactly the first
k; then keep the minimal prefix whose float64 probability mass, renormalised over the kept top-k, reaches p (a position
stays while the mass before it is below p); the first token always stays. Installed at the defining module, before
any importer binds the name (MRv2's sampler and sampling states, the v1 rejection sampler bind it by name at import).
Production serves apply_top_k_top_p_triton from arc3_sampler_topk_guard (ARC3_SKIP_REDUNDANT_TOPK=1); both kernels
become unreachable.

Rows are masked ROWS at a time: the sort, the float64 softmax and cumsum take about 45 bytes per logit, so a full
verify batch at once (120 rows x 248,320) would need about 1.3 GB transient against 643 MiB minimum free (g4v); 8 rows
take about 90 MB. Rows are independent, so the result does not depend on ROWS."""
import torch

PINS = {'vllm.v1.sample.ops.topk_topp_sampler': 'a660990ec2e84cd8e0e3ff47146dadd2757399228de09a7c2b9959a26d77d882'}
STATS = dict(calls=0, rows=0)
ROWS = 8


def exact_top_k_top_p(logits, k, p):
    if k is None and p is None:
        return logits
    STATS['calls'] += 1
    STATS['rows'] += int(logits.shape[0])
    for s in range(0, logits.shape[0], ROWS):
        _mask_rows(logits[s:s + ROWS], None if k is None else k[s:s + ROWS], None if p is None else p[s:s + ROWS])
    return logits


def _mask_rows(logits, k, p):
    """In place on a row slice of the caller's logits."""
    V = logits.shape[-1]
    neg, order = torch.sort(-logits, dim=-1, stable=True)       # logit descending, id ascending among ties
    ranks = torch.arange(V, device=logits.device)
    keep = torch.ones(logits.shape, dtype=torch.bool, device=logits.device)
    if k is not None:
        keep &= ranks[None, :] < k.to(torch.long)[:, None]
    if p is not None:
        masked = (-neg).double().masked_fill(~keep, float('-inf'))
        probs = torch.softmax(masked, dim=-1)
        before = torch.cumsum(probs, dim=-1) - probs
        p64 = p.to(torch.float64)[:, None]
        keep &= (before < p64) | (p64 >= 1.0)                     # p = 1: no truncation (rows mixed with top-p rows)
        keep[:, 0] = True
    unsorted = torch.empty_like(keep)
    unsorted.scatter_(1, order, keep)
    logits.masked_fill_(~unsorted, float('-inf'))


def patch_topk_topp_sampler(mod):
    mod._s2_stock_apply_top_k_top_p = mod.apply_top_k_top_p
    mod.apply_top_k_top_p = exact_top_k_top_p


PATCHERS = {'vllm.v1.sample.ops.topk_topp_sampler': patch_topk_topp_sampler}
