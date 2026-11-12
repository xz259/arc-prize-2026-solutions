"""Opt-in serving diagnostics; defaults preserve the stock selector/finalizer.

Exact QSA selection is the retained PyTorch correctness reference. The opt-in
radix implementation follows the same visibility, tie and output-order contract.
Mode changes are allowed only through the idle, eager-only worker RPC.
"""
import os

QSA_MODE = os.environ.get('ARC3_QSA_MODE', 'native')
MOE_FUSED_FINALIZE = os.environ.get('ARC3_MOE_FINALIZE', 'fused') == 'fused'
if QSA_MODE not in {'native', 'exact', 'radix'}:
    raise ValueError('Unknown QSA mode')
if os.environ.get('ARC3_MOE_FINALIZE', 'fused') not in {'fused', 'unfused'}:
    raise ValueError('Unknown MoE finalizer mode')
TOPK_THREADS = int(os.environ.get('ARC3_TOPK_THREADS', '256'))
TOPK_CACHE_ROW = os.environ.get('ARC3_TOPK_CACHE_ROW', '1') == '1'
PREFILL_PROFILE = os.environ.get('ARC3_QSA_PREFILL_PROFILE', 'stock')
PREFILL_MIN_ROWS = 1024
if PREFILL_PROFILE not in {'stock', 'tile32'}:
    raise ValueError('Unsupported BF16 sparse-prefill profile')
PREFILL_COUNTERS = {'stock':0, 'tile32':0}
if TOPK_THREADS not in {128, 256, 512, 1024}:
    raise ValueError('Unsupported radix selector thread count')
COUNTERS = {'qsa_native': 0, 'qsa_exact': 0, 'qsa_radix': 0, 'moe_fused': 0, 'moe_unfused': 0}
CAPTURE_PENDING = False
CAPTURE = None


def exact_topk(logits, lengths, output, k, columns):
    import torch
    output.fill_(-1)
    width = min(k, columns)
    if not width:
        return
    indices = torch.arange(columns, device=logits.device)
    # Bound extra sort workspace; the original QSA logit workspace already
    # exists. No data-dependent host synchronization is needed for selection.
    for start in range(0, logits.shape[0], 128):
        end = min(start + 128, logits.shape[0])
        valid = indices[None, :] < lengths[start:end, None]
        scores = logits[start:end, :columns].masked_fill(~valid, -float('inf'))
        selected = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :width]
        selected = selected.masked_fill(selected >= lengths[start:end, None], columns)
        selected = selected.sort(dim=-1).values
        selected = selected.masked_fill(selected == columns, -1)
        output[start:end, :width].copy_(selected)


def select_topk(native_op, logits, lengths, output, workspace, k, columns):
    global CAPTURE_PENDING, CAPTURE
    COUNTERS['qsa_' + QSA_MODE] += 1
    if QSA_MODE == 'native':
        native_op(logits, lengths, output, workspace, k, columns)
    elif QSA_MODE == 'radix':
        from arc3_topk_radix import radix_topk
        radix_topk(logits, lengths, output, k, columns, TOPK_THREADS, TOPK_CACHE_ROW)
    else:
        exact_topk(logits, lengths, output, k, columns)
    if CAPTURE_PENDING and logits.shape[0] > 32:
        # Diagnostic only: retain a bounded real prefill slice, with its original
        # row stride. CPU transfer intentionally synchronizes this one request.
        CAPTURE = dict(logits=logits[-32:, :columns].detach().cpu(),
                       lengths=lengths[-32:].detach().cpu(),
                       output=output[-32:].detach().cpu(), k=k, columns=columns,
                       original_rows=logits.shape[0], stride=logits.stride(0))
        CAPTURE_PENDING = False


def fused_finalize():
    COUNTERS['moe_fused' if MOE_FUSED_FINALIZE else 'moe_unfused'] += 1
    return MOE_FUSED_FINALIZE


def sparse_prefill_profile(rows, block_n, splits, warps):
    """Only large BF16 QSA calls; captured decode profiles remain unchanged."""
    if rows < PREFILL_MIN_ROWS:
        return block_n, splits, warps
    PREFILL_COUNTERS[PREFILL_PROFILE] += 1
    return (32, 1, 4) if PREFILL_PROFILE == 'tile32' else (block_n, splits, warps)


def status():
    return dict(qsa=QSA_MODE, moe='fused' if MOE_FUSED_FINALIZE else 'unfused',
                counters=dict(COUNTERS), captured=CAPTURE is not None,
                radix_threads=TOPK_THREADS, radix_cache_row=TOPK_CACHE_ROW,
                prefill_profile=PREFILL_PROFILE,prefill_counters=dict(PREFILL_COUNTERS))


def check_captured_topk():
    """Compare native output and selected values on identical captured inputs."""
    import torch
    if CAPTURE is None:
        return {'error': 'No QSA input has been captured'}
    c = CAPTURE
    data = c['logits'].to('cuda')
    logits = torch.empty_strided(data.shape, (c['stride'], 1), device='cuda', dtype=data.dtype)
    logits.copy_(data)
    lengths = c['lengths'].to('cuda')
    output = torch.empty((len(lengths), c['k']), device='cuda', dtype=torch.int32)
    reference = torch.empty_like(output)
    workspace = torch.empty(1024 * 1024, device='cuda', dtype=torch.uint8)
    exact_topk(logits, lengths, reference, c['k'], c['columns'])
    expected = reference.cpu()
    calls = []
    previous = None
    for _ in range(6):
        torch.ops._C.persistent_topk(logits, lengths, output, workspace, c['k'], c['columns'])
        actual = output.cpu()
        invalid = ((actual < -1) | ((actual >= c['lengths'][:, None]) & (actual >= 0)))
        def sorted_values(ids):
            safe = ids.clamp(min=0, max=c['columns']-1).long()
            values = c['logits'].gather(1, safe).masked_fill(ids < 0, -float('inf'))
            return values.sort(dim=-1).values
        calls.append(dict(identical_to_previous=None if previous is None else torch.equal(previous, actual),
                          selected_set_matches_reference=torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values),
                          selected_values_match_reference=torch.equal(sorted_values(actual), sorted_values(expected)),
                          invalid_indices=int(invalid.sum())))
        previous = actual
    return dict(rows=len(lengths), original_rows=c['original_rows'], columns=c['columns'],
                k=c['k'], min_visible=int(c['lengths'].min()), max_visible=int(c['lengths'].max()), calls=calls)
