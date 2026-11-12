"""Candidate-based verify sampler (kernel work block 2, item 5): the stock MRv2 verify path (FP32 copy of the target
logits, Triton top-k/top-p masking over the vocabulary, then rejection sampling and resampling over the vocabulary)
reduced to one pass over the target and draft logits plus work on at most KMAX candidates per row.

Scope: probabilistic drafts (draft logits present), standard rejection, temperature > 0, top_k <= KMAX - 1, no
penalties / logit bias / bad words / min_p / logprobs (the production sampling parameters). Semantics follow the stock
kernels: target p = softmax of the top-k (threshold: every logit >= the k-th largest) then top-p (ascending
cumulative mass <= 1 - p removed, the largest kept) logits; draft q = softmax(draft_logits / T) over the vocabulary;
accept draft x iff log p(x) > log u + log q(x) with u = rand(seed, pos); resample the first rejected position from
max(p - q, 0) (or p for the bonus / a -1 placeholder) by Gumbel-max with noise rand(randint(seed, pos + 2^30), token
id). The noise for a token depends only on (seed, position, token id), so evaluating it on the candidates reproduces
the (patched) stock argmax; only summation orders (log-sum-exp) and tie orders differ.

2026-09-30: the resample noise is keyed by pos + 2^30. Stock MRv2 keys it by pos, and the drafter samples each draft
with Gumbel noise keyed by the same (seed, pos) ("positions + 1 ... to match the Gumbel noise used for draft and
target sampling"), so a correction draw reused the noise that selected the rejected draft: conditioned on the draft
having won, it is not a sample of max(p - q, 0), and the output is not p (T=1 PIT test on gameplay contexts: KS 0.0094
vs 0.0056 critical; simulation: 2-3% total variation per rejecting position). Acceptance keeps rand(seed, pos), which
is independent of the draft noise. vsampler.py applies the same offset to the stock resample kernel (fallback path).

2026-09-30 (overnight step 2c): ties follow the stock Triton top-k / top-p kernel (vllm/v1/sample/ops/topk_topp_triton.py,
which production MTP0 and the stock verify path use for batches of 8+ rows). Candidates are ordered by (logit
descending, token id ascending); top-k keeps exactly the first k; top-p keeps the minimal prefix of that order reaching
p. Previously top-k kept every token tied with the k-th value and the key order put higher ids first, so with BF16 ties
(ubiquitous: logit steps of 0.0625-0.125) the nucleus could differ from production's. Sort keys carry the inverted
token id (INV - id) so that a larger key means a larger logit, then a smaller id; the candidate set can no longer
overflow (flag always 0).

2026-10-03 (Fable's review): the block kernel's width and warps are configurable (block_config); vsampler passes the
raw logits with value_bf16=True when no row needs processing besides top-k / top-p (no FP32 copy; bitwise-identical
keys at temperature 1).

2026-10-04 (temperature, user ~11:32: 64K runs at T = 0.6 / 0.7): the target logits reach these kernels as vsampler's
`processed` = apply_sampling_params(skip_top_k_top_p=True), which already divides by the temperature (stock MRv2 order:
temperature, min_p, then top-k / top-p), or raw with value_bf16 at T = 1; the block kernels no longer divide them by T
again (they did, harmless at T = 1, a T^2 scaling otherwise; vsampler had gated T != 1 to the stock path). Draft logits
are stored pre-temperature and are still divided by T here, as the stock kernels do.

2026-10-16 (user: build the temperature-scaled fast path before the T 0.6 safety replica): value_bf16 at any T > 0. The block
kernel loads the raw BF16 logits, finds the block's KC-th largest and compacts the elements at or above it on the raw values
(x / T is strictly monotone on distinct BF16 values, so the same elements as on the scaled ones), then divides the kept keys'
values by T with the stock _temperature_kernel's FP32 division (skipped at T = 1, as there): the same keys, bit for bit, as
the FP32 path (apply_sampling_params' copy + temperature), without the [rows, V] FP32 copy.

2026-10-04 (overnight item b, user: "adopt if there is a gain without any hazard"): block verification (Sun et al. 2024,
arXiv 2403.10444; vLLM's stock USE_BLOCK_VERIFICATION path) in candidate space, _block_verify_kernel. With P_i the
running joint ratio min(P_{i-1} p_i(x_i) / q_i(x_i), 1), the block length is the last i with u_i <= h_i, where
h_i = m_i / (m_i + 1 - P_i), m_i = sum_x max(P_i p_{i+1}(x) - q_{i+1}(x), 0) (h = P_i at the last draft or before a -1
placeholder); the next token is drawn from max(P_tau p(x) - q(x), 0) (P_0 = 1), or from p after a full block. The target
p has support on the kept candidates, so m_i and the residual are exact sums over candidates. Same randomness as the
token kernel: u_i = rand(seed, pos), resample noise keyed by pos + 2^30. Lossless (exact float64 enumeration: the output
extended by target sampling equals the target joint to 3e-16 on 200 random cases); expected accepted length >= token
verification's.
"""
import torch
import triton
import triton.language as tl

from vllm.triton_utils import tldevice
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32

RESAMPLE_NOISE_OFFSET = tl.constexpr(1 << 30)
INV = tl.constexpr(2147483647)          # sort keys hold INV - token id: ties rank lower ids first

KMAX = 32
BLOCK = 4096


@triton.jit
def _sortable(v):
    b = v.to(tl.int32, bitcast=True)
    return b ^ ((b >> 31) & 0x7FFFFFFF)


@triton.jit
def _unsortable(k):
    b = k ^ ((k >> 31) & 0x7FFFFFFF)
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _block_topk_kernel(logits_ptr, logits_stride, draft_ptr, draft_s0, draft_s1, exp_idx_ptr, exp_pos_ptr, temp_ptr,
                       ckey_ptr, dmax_ptr, dsum_ptr, V, NB, K_SPEC, BLOCK: tl.constexpr, KC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < V
    req = tl.load(exp_idx_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=m, other=float('-inf')).to(tl.float32)
    # (2026-10-04) the target logits arrive temperature-scaled (apply_sampling_params) or raw at T = 1: no division here
    key = (_sortable(x).to(tl.int64) << 32) | (INV - offs).to(tl.int64)
    top = tl.topk(key, KC)
    tl.store(ckey_ptr + (row * NB + blk) * KC + tl.arange(0, KC), top)
    step = tl.load(exp_pos_ptr + row)
    if step < K_SPEC:
        d = tl.load(draft_ptr + req * draft_s0 + step * draft_s1 + offs, mask=m, other=float('-inf')).to(tl.float32)
        d = d / temp
        dm = tl.max(d, axis=0)
        # a block whose draft logits are all -inf (a vocabulary-subset draft head) contributes 0, not NaN (as the stock
        # rejection_sampler_utils._compute_max_and_sumexp; 2026-10-02, drafthead v1's tau 1.000)
        ds = tl.where(dm > float('-inf'), tl.sum(tl.exp(d - dm), axis=0), 0.0)
        tl.store(dmax_ptr + row * NB + blk, dm)
        tl.store(dsum_ptr + row * NB + blk, ds)


@triton.jit
def _block_topk_v2_kernel(logits_ptr, logits_stride, draft_ptr, draft_s0, draft_s1, exp_idx_ptr, exp_pos_ptr,
                          temp_ptr, ckey_ptr, dmax_ptr, dsum_ptr, V, NB, K_SPEC, BLOCK: tl.constexpr, KC: tl.constexpr,
                          VALUE_BF16: tl.constexpr):
    """Same output as _block_topk_kernel: the block's KC largest (value, index) keys. The KC-th largest value comes
    from a top-k over the values (the raw BF16 ones with VALUE_BF16), then every element at or above it is compacted by
    prefix sums (strictly larger elements first, then ties at the threshold by index; dropped ties can only matter when
    the row's KMAX-th largest reaches its k-th, which the row select flags). VALUE_BF16 at T != 1 (2026-10-16): the
    selection on the raw values, the stored values divided by T as the stock _temperature_kernel does."""
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < V
    req = tl.load(exp_idx_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    raw = tl.load(logits_ptr + row * logits_stride + offs, mask=m, other=float('-inf'))
    if VALUE_BF16:
        thr = tl.min(tl.topk(raw, KC), axis=0).to(tl.float32)
        xr = raw.to(tl.float32)
        # every element above the threshold (fewer than KC), then ties at it in index order up to KC: compared on the raw
        # values, which select the same elements as the scaled ones (x / T is strictly monotone on distinct BF16 values)
        gt = xr > thr
        eq = xr == thr
        if temp != 1.0:
            # 2026-10-16: the stock _temperature_kernel's division. MUST stay the same Triton fp32 `/` as
            # vllm/v1/worker/gpu/sample/gumbel.py _temperature_kernel (div.full.f32; IE): a torch or div_rn division on either
            # side changes bits (<= 2 ulp) and breaks test_cand_temp2's T5 (keep T5 in the permanent test set)
            x = xr / temp
        else:
            x = xr
    else:
        x = raw.to(tl.float32)                  # temperature already applied by apply_sampling_params (10-04)
        thr = tl.min(tl.topk(x, KC), axis=0)
        # every element above the threshold (fewer than KC), then ties at it in index order up to KC
        gt = x > thr
        eq = x == thr
    n_gt = tl.sum(gt.to(tl.int32), axis=0)
    rank = tl.where(gt, tl.cumsum(gt.to(tl.int32), axis=0) - 1, n_gt + tl.cumsum(eq.to(tl.int32), axis=0) - 1)
    key = (_sortable(x).to(tl.int64) << 32) | (INV - offs).to(tl.int64)
    base = ckey_ptr + (row * NB + blk) * KC
    tl.store(base + rank, key, mask=(gt | eq) & (rank < KC))
    step = tl.load(exp_pos_ptr + row)
    if step < K_SPEC:
        d = tl.load(draft_ptr + req * draft_s0 + step * draft_s1 + offs, mask=m, other=float('-inf')).to(tl.float32)
        d = d / temp
        dm = tl.max(d, axis=0)
        # a block whose draft logits are all -inf (a vocabulary-subset draft head) contributes 0, not NaN (as the stock
        # rejection_sampler_utils._compute_max_and_sumexp; 2026-10-02, drafthead v1's tau 1.000)
        ds = tl.where(dm > float('-inf'), tl.sum(tl.exp(d - dm), axis=0), 0.0)
        tl.store(dmax_ptr + row * NB + blk, dm)
        tl.store(dsum_ptr + row * NB + blk, ds)


@triton.jit
def _row_select_kernel(ckey_ptr, dmax_ptr, dsum_ptr, draft_ptr, draft_s0, draft_s1, exp_idx_ptr, exp_pos_ptr,
                       temp_ptr, topk_ptr, topp_ptr, cid_ptr, clog_ptr, cdraft_ptr, lse_t_ptr, lse_d_ptr, flag_ptr,
                       NB, K_SPEC, NBK: tl.constexpr, KC: tl.constexpr, KMAX: tl.constexpr, NBP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, NBK)
    keys = tl.load(ckey_ptr + row * NB * KC + o, mask=o < NB * KC, other=-9223372036854775807 - 1)
    top = tl.sort(tl.topk(keys, KMAX), descending=True)          # largest first
    vals = _unsortable((top >> 32).to(tl.int32))
    ids = (INV - (top & 0xFFFFFFFF)).to(tl.int32)
    j = tl.arange(0, KMAX)
    k = tl.load(topk_ptr + row)
    p = tl.load(topp_ptr + row)
    keep = (j < k) & (vals > float('-inf'))                       # top-k: exactly the first k (stock Triton ties)
    overflow = j < 0                                              # cannot overflow any more (flag 0)
    overflow = tl.max(overflow.to(tl.int32), axis=0)
    vmax = tl.max(vals, axis=0)
    e = tl.where(keep, tl.exp(vals - vmax), 0.0)
    total = tl.sum(e, axis=0)
    # stock top-p: ascending order, remove while the cumulative mass (this and all smaller) <= 1 - p; keep the largest
    suffix = total - tl.cumsum(e, axis=0) + e
    keep = keep & ((suffix / total > 1.0 - p) | (j == 0))
    logit = tl.where(keep, vals, float('-inf'))
    ek = tl.where(keep, tl.exp(vals - vmax), 0.0)
    lse_t = vmax + tl.log(tl.sum(ek, axis=0))
    tl.store(cid_ptr + row * KMAX + j, tl.where(keep, ids, -1))
    tl.store(clog_ptr + row * KMAX + j, logit)
    tl.store(lse_t_ptr + row, lse_t)
    tl.store(flag_ptr + row, overflow.to(tl.int32))
    step = tl.load(exp_pos_ptr + row)
    if step < K_SPEC:
        req = tl.load(exp_idx_ptr + row).to(tl.int64)
        temp = tl.load(temp_ptr + req).to(tl.float32)
        b = tl.arange(0, NBP)
        dm = tl.load(dmax_ptr + row * NB + b, mask=b < NB, other=float('-inf'))
        ds = tl.load(dsum_ptr + row * NB + b, mask=b < NB, other=0.0)
        gm = tl.max(dm, axis=0)
        lse_d = gm + tl.log(tl.sum(ds * tl.exp(dm - gm), axis=0))
        tl.store(lse_d_ptr + row, lse_d)
        dq = tl.load(draft_ptr + req * draft_s0 + step * draft_s1 + tl.maximum(ids, 0)).to(tl.float32) / temp
        tl.store(cdraft_ptr + row * KMAX + j, dq)


@triton.jit
def _verify_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, draft_sampled_ptr, draft_ptr, draft_s0, draft_s1,
                   cid_ptr, clog_ptr, cdraft_ptr, lse_t_ptr, lse_d_ptr, cu_ptr, idx_mapping_ptr, temp_ptr, seed_ptr,
                   pos_ptr, KMAX: tl.constexpr):
    req_idx = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start = tl.load(cu_ptr + req_idx).to(tl.int64)
    end = tl.load(cu_ptr + req_idx + 1)
    n_draft = end - start - 1
    seed = tl.load(seed_ptr + req)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    j = tl.arange(0, KMAX)
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(n_draft):
        li = start + i
        x = tl.load(draft_sampled_ptr + li + 1).to(tl.int64)
        valid = x >= 0
        x = tl.maximum(0, x)
        verifying &= valid
        if verifying:
            pos = tl.load(pos_ptr + li)
            u = tl_rand32(seed, pos, includes_zero=False)
            cid = tl.load(cid_ptr + li * KMAX + j)
            clog = tl.load(clog_ptr + li * KMAX + j)
            tl_x = tl.max(tl.where(cid == x, clog, float('-inf')), axis=0)
            target_logprob = tl_x - tl.load(lse_t_ptr + li)
            dl_x = tl.load(draft_ptr + req * draft_s0 + i * draft_s1 + x).to(tl.float32) / temp
            draft_logprob = dl_x - tl.load(lse_d_ptr + li)
            accepted = target_logprob > tl.log(u) + draft_logprob
            verifying = accepted
            accepted_length += accepted
            tl.store(sampled_ptr + req_idx * sampled_stride + i, x)
    # resample the first rejected position, or the bonus position
    r = start + accepted_length
    is_bonus = accepted_length == n_draft
    xr = tl.load(draft_sampled_ptr + r + 1, mask=not is_bonus, other=0)
    cid = tl.load(cid_ptr + r * KMAX + j)
    clog = tl.load(clog_ptr + r * KMAX + j)
    if is_bonus or xr < 0:
        residual = clog
    else:
        tlp = clog - tl.load(lse_t_ptr + r)
        dlp = tl.load(cdraft_ptr + r * KMAX + j) - tl.load(lse_d_ptr + r)
        ratio = tl.exp(dlp - tlp)
        residual = tl.where(ratio < 1.0, tlp + tldevice.log1p(-ratio), float('-inf'))
    pos_r = tl.load(pos_ptr + r)
    gseed = tl.randint(seed, pos_r + RESAMPLE_NOISE_OFFSET)       # independent of the draft's (seed, pos) noise
    ug = tl_rand32(gseed, tl.maximum(cid, 0), includes_zero=False)
    noise = -tl.log(-tldevice.log1p(-ug))
    score = tl.where(cid >= 0, residual + noise, float('-inf'))
    best = tl.argmax(score, axis=0)
    token = tl.sum(tl.where(j == best, cid, 0), axis=0).to(tl.int64)
    tl.store(sampled_ptr + req_idx * sampled_stride + accepted_length, token)
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))



@triton.jit
def _block_verify_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, draft_sampled_ptr, draft_ptr, draft_s0, draft_s1,
                         cid_ptr, clog_ptr, cdraft_ptr, lse_t_ptr, lse_d_ptr, cu_ptr, idx_mapping_ptr, temp_ptr, seed_ptr,
                         pos_ptr, KMAX: tl.constexpr):
    """Block verification over the same candidate rows as _verify_kernel (see the module docstring)."""
    req_idx = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start = tl.load(cu_ptr + req_idx).to(tl.int64)
    end = tl.load(cu_ptr + req_idx + 1)
    n_draft = end - start - 1
    seed = tl.load(seed_ptr + req)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    j = tl.arange(0, KMAX)
    accepted_length = tl.zeros((), tl.int64)
    log_p = tl.zeros((), tl.float32)              # log P_i, the running joint ratio
    log_p_tau = tl.zeros((), tl.float32)          # log P at the block's end (0 = P_0 for an empty block)
    verifying = True
    for i in range(n_draft):
        li = start + i
        x = tl.load(draft_sampled_ptr + li + 1).to(tl.int64)
        valid = x >= 0
        x = tl.maximum(0, x)
        verifying &= valid
        if verifying:
            pos = tl.load(pos_ptr + li)
            u = tl_rand32(seed, pos, includes_zero=False)
            cid = tl.load(cid_ptr + li * KMAX + j)
            clog = tl.load(clog_ptr + li * KMAX + j)
            tl_x = tl.max(tl.where(cid == x, clog, float('-inf')), axis=0)
            target_logprob = tl_x - tl.load(lse_t_ptr + li)
            dl_x = tl.load(draft_ptr + req * draft_s0 + i * draft_s1 + x).to(tl.float32) / temp
            draft_logprob = dl_x - tl.load(lse_d_ptr + li)
            log_p = tl.minimum(log_p + (target_logprob - draft_logprob), 0.0)
            prefix = tl.exp(log_p)
            nxt = tl.load(draft_sampled_ptr + li + 2, mask=i < n_draft - 1, other=-1)
            if nxt >= 0:
                # residual mass one position ahead: sum_x max(P_i p_{i+1}(x) - q_{i+1}(x), 0) over row li + 1's candidates
                cid1 = tl.load(cid_ptr + (li + 1) * KMAX + j)
                tlp1 = tl.load(clog_ptr + (li + 1) * KMAX + j) - tl.load(lse_t_ptr + li + 1)
                dlp1 = tl.load(cdraft_ptr + (li + 1) * KMAX + j) - tl.load(lse_d_ptr + li + 1)
                mass = tl.sum(tl.where(cid1 >= 0, tl.maximum(prefix * tl.exp(tlp1) - tl.exp(dlp1), 0.0), 0.0), axis=0)
                denom = mass + 1.0 - prefix
                h = tl.where(denom > 0.0, mass / denom, 1.0)
            else:
                h = prefix
            take = u <= h
            accepted_length = tl.where(take, i + 1, accepted_length)
            log_p_tau = tl.where(take, log_p, log_p_tau)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, x)
    # the next token: from max(P_tau p - q, 0) at the block's end, or from p after a full block / before a placeholder
    r = start + accepted_length
    is_bonus = accepted_length == n_draft
    xr = tl.load(draft_sampled_ptr + r + 1, mask=not is_bonus, other=0)
    cid = tl.load(cid_ptr + r * KMAX + j)
    clog = tl.load(clog_ptr + r * KMAX + j)
    if is_bonus or xr < 0:
        residual = clog
    else:
        tlp = clog - tl.load(lse_t_ptr + r) + log_p_tau
        dlp = tl.load(cdraft_ptr + r * KMAX + j) - tl.load(lse_d_ptr + r)
        ratio = tl.exp(dlp - tlp)
        residual = tl.where(ratio < 1.0, tlp + tldevice.log1p(-ratio), float('-inf'))
    pos_r = tl.load(pos_ptr + r)
    gseed = tl.randint(seed, pos_r + RESAMPLE_NOISE_OFFSET)       # independent of the draft's (seed, pos) noise
    ug = tl_rand32(gseed, tl.maximum(cid, 0), includes_zero=False)
    noise = -tl.log(-tldevice.log1p(-ug))
    score = tl.where(cid >= 0, residual + noise, float('-inf'))
    best = tl.argmax(score, axis=0)
    token = tl.sum(tl.where(j == best, cid, 0), axis=0).to(tl.int64)
    tl.store(sampled_ptr + req_idx * sampled_stride + accepted_length, token)
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))

def block_config(block=None, warps=None):
    """Vocabulary block width and warps of _block_topk_v2 (2026-10-03, Fable's review): ARC3_S2_CANDVERIFY_BLOCK
    (default 2048, the deployed value) and ARC3_S2_CANDVERIFY_WARPS (default 4 up to 2048, 8 up to 8192, else 16).
    The candidate set is the row's KMAX largest (value, id) keys for any block width (a row's top-KMAX key lies in its
    block's top-KMAX), so cid / clog / lse_t do not change; the draft log-sum-exp (per-block sums, then their merge)
    changes only in summation order (rounding-level)."""
    import os
    block = int(block or os.environ.get('ARC3_S2_CANDVERIFY_BLOCK', '2048'))
    warps = int(warps or os.environ.get('ARC3_S2_CANDVERIFY_WARPS', '0') or 0) or (
        4 if block <= 2048 else 8 if block <= 8192 else 16)
    return block, warps


class Workspace:
    def __init__(self, max_rows, vocab, device, block=None, warps=None):
        block, self.warps = block_config(block, warps)
        self.block = block
        self.nb = triton.cdiv(vocab, block)
        self.nbk = triton.next_power_of_2(self.nb * KMAX)
        self.ckey = torch.empty(max_rows * self.nb * KMAX, dtype=torch.int64, device=device)
        self.dmax = torch.empty(max_rows * self.nb, dtype=torch.float32, device=device)
        self.dsum = torch.empty(max_rows * self.nb, dtype=torch.float32, device=device)
        self.cid = torch.empty(max_rows, KMAX, dtype=torch.int32, device=device)
        self.clog = torch.empty(max_rows, KMAX, dtype=torch.float32, device=device)
        self.cdraft = torch.empty(max_rows, KMAX, dtype=torch.float32, device=device)
        self.lse_t = torch.empty(max_rows, dtype=torch.float32, device=device)
        self.lse_d = torch.empty(max_rows, dtype=torch.float32, device=device)
        self.flag = torch.empty(max_rows, dtype=torch.int32, device=device)


def cand_rejection_sample(logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping, expanded_idx_mapping,
                          expanded_local_pos, temperature, seed, top_k, top_p, num_speculative_steps, ws,
                          value_bf16=False, block_verify=False):
    """Same inputs as the stock verify path (raw target logits instead of processed ones, plus per-row top_k/top_p);
    returns (sampled [num_reqs, K + 1] int64, num_sampled [num_reqs] int32, overflow flags [rows]). value_bf16: raw BF16
    logits that need no processing besides temperature / top-k / top-p (the block threshold is found on the raw values and
    the kernel applies each row's temperature as the stock kernel does; 2026-10-16: any T > 0). block_verify: block
    verification (_block_verify_kernel) instead of token-by-token rejection."""
    R, V = logits.shape
    num_reqs = cu_num_logits.numel() - 1
    nb = ws.nb
    _block_topk_v2_kernel[(R, nb)](logits, logits.stride(0), draft_logits, draft_logits.stride(0),
                                   draft_logits.stride(1), expanded_idx_mapping, expanded_local_pos, temperature,
                                   ws.ckey, ws.dmax, ws.dsum, V, nb, num_speculative_steps, BLOCK=ws.block, KC=KMAX,
                                   VALUE_BF16=bool(value_bf16), num_warps=ws.warps)
    _row_select_kernel[(R,)](ws.ckey, ws.dmax, ws.dsum, draft_logits, draft_logits.stride(0), draft_logits.stride(1),
                             expanded_idx_mapping, expanded_local_pos, temperature, top_k, top_p, ws.cid, ws.clog,
                             ws.cdraft, ws.lse_t, ws.lse_d, ws.flag, nb, num_speculative_steps, NBK=ws.nbk, KC=KMAX,
                             KMAX=KMAX, NBP=triton.next_power_of_2(nb), num_warps=4)
    sampled = draft_sampled.new_empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    kernel = _block_verify_kernel if block_verify else _verify_kernel
    kernel[(num_reqs,)](sampled, sampled.stride(0), num_sampled, draft_sampled, draft_logits,
                        draft_logits.stride(0), draft_logits.stride(1), ws.cid, ws.clog, ws.cdraft, ws.lse_t,
                        ws.lse_d, cu_num_logits, idx_mapping, temperature, seed, pos, KMAX=KMAX, num_warps=1)
    return sampled, num_sampled, ws.flag[:R]
