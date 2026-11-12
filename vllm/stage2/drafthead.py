"""stage2_v1 'drafthead' part (milestone 2 A3, 2026-10-01): the MTP drafter scores only a fixed vocabulary subset S.

Draft logits outside S are -inf, so the probabilistic draft distribution q is renormalised over S. vsampler verifies
each draft against the same cached draft logits, so the output distribution stays exactly the target's (rejection
sampling with a matching q); only acceptance can change (a target token outside S can never be drafted).
S = ARC3_S2_DRAFTHEAD_VOCAB (JSON with half-open 'token_ranges', lordhansolo's draft_vocab_32k.json format; same
tokenizer hash as ours; it covers 99.82% of our run16 tokens, PORTING_POINTERS A3).

The subset head is production's FP8 lm_head restricted to the rows of S: mtp0x fp8w per-channel E4M3 rows, run through
the registered fp8w op with the full head's (n, k) config rows, so every subset logit is computed exactly as the full
head computes that column (checked on the first call, recorded as first_call_max_abs_diff). The drafter's
compute_logits runs eagerly (outside its CUDA graphs): ARC3_S2_DRAFTHEAD_SWITCH=<file> containing 'full' uses the stock
full head at run time (in-run A/B); no file or anything else = subset.

v2 (2026-10-02 19:55, B7): (1) v1's tau 1.000 came from vsampler (cand_verify) lacking the all -inf block guard of the
stock rejection sampler; fixed there (10685d16). (2) The drafter's compute_logits runs inside its CUDA graphs (W5), so
the runtime switch only reaches eager steps: leave ARC3_S2_DRAFTHEAD_SWITCH unset. (3) The drafter shares the target's
lm_head; with the BF16 head (no fp8w 'lm_head' registered) the subset is the BF16 weight's rows S ([|S|, 2560] copy)
through F.linear (mode 'bf16'); with production's FP8 head, v1's fp8w rows (mode 'fp8w', per-column exact). A BF16
subset GEMM may round a column differently from the full GEMM; the output distribution stays exact (the cached q is the
distribution the drafts were sampled from); first_call_max_abs_diff records it.

v3 (2026-10-03, Fable's review): optional draft truncation, ARC3_S2_DRAFTTRUNC='k,p' (see _trunc_config); default off.
v4 (2026-10-20, item 20 / IE-D3): ARC3_S2_DRAFTTRUNC_T = the temperature of the top-p prefix (default 1: as v3); at the serving T the
draft nucleus is the target's construction (T -> top-k -> top-p)."""
import json
import os
import time

PINS = {'vllm.models.qwen3_8_flash_next.nvidia.mtp': 'd4bbe292ead768e9e3b302a03fd07ecf979f5ea7d79f3a48598d50f15ffc9fd7'}
STATS = dict(calls=0, subset_calls=0, full_calls=0, vocab=0, subset=0, first_call_max_abs_diff=None, out_dtype=None, mode=None,
             trunc=None)


def _trunc_config():
    """ARC3_S2_DRAFTTRUNC='k,p' (2026-10-03, Fable's review; PLAN A4's draft truncation): the drafter's subset logits
    keep only their own top-k, then the smallest prefix of those (largest first) whose softmax mass reaches p; the rest
    are -inf. The drafts are sampled from, and verified against, the same cached logits, so the output distribution is
    still exactly the target's for any k and p; only acceptance changes (draft mass outside the target's top-k / top-p
    nucleus is moved onto likelier tokens). Unset or '0' = off."""
    v = os.environ.get('ARC3_S2_DRAFTTRUNC', '0').strip()
    if v in ('', '0'):
        return None
    k, p = v.split(',')
    k, p = int(k), float(p)
    assert 1 <= k and 0.0 < p <= 1.0, v
    return k, p


TRUNC = _trunc_config()
STATS['trunc'] = list(TRUNC) if TRUNC else None
TRUNC_T = float(os.environ.get('ARC3_S2_DRAFTTRUNC_T', '1') or 1)   # v4 (item 20): the prefix's temperature; 1 = v3
assert TRUNC_T > 0.0, TRUNC_T
STATS['trunc_t'] = TRUNC_T
STATS['trunc_switches'] = 0
# ARC3_S2_DRAFTTRUNC_SWITCH=<file> (in-run A/B): the captured draft step computes both logits and selects by a device
# flag; vsampler's _verify (eager, every step) calls sync_trunc_switch, which reads the file at most once a second
# ('off' = untruncated, anything else = truncated) and writes the flag. Unset = truncation without the selection.
_TS = dict(path=os.environ.get('ARC3_S2_DRAFTTRUNC_SWITCH'), t=0.0, on=True, flag=None)


def _trunc_flag(device):
    import torch
    if _TS['flag'] is None:
        _TS['flag'] = torch.ones((), dtype=torch.bool, device=device)
    return _TS['flag']


def sync_trunc_switch():
    s = _TS
    if s['path'] is None or s['flag'] is None:
        return
    now = time.monotonic()
    if now - s['t'] < 1.0:
        return
    s['t'] = now
    try:
        with open(s['path']) as f:
            on = f.read().strip() != 'off'
    except OSError:
        on = True
    if on != s['on']:
        s['on'] = on
        s['flag'].fill_(on)
        STATS['trunc_switches'] += 1


def truncate(y, k, p):
    """y [M, n] subset logits -> same shape and dtype, -inf outside each row's top-k / top-p prefix (CUDA-graph safe:
    static shapes, no host synchronisation)."""
    import torch
    v, i = torch.topk(y.float(), min(k, y.shape[1]), dim=1)          # largest first
    if p < 1.0:
        q = torch.softmax(v if TRUNC_T == 1.0 else v / TRUNC_T, dim=1)     # v4 (item 20): the prefix at TRUNC_T
        v = torch.where(torch.cumsum(q, dim=1) - q < p, v, float('-inf'))    # mass before this token < p: keep
    return torch.full_like(y, float('-inf')).scatter_(1, i, v.to(y.dtype))
_SWITCH = dict(path=os.environ.get('ARC3_S2_DRAFTHEAD_SWITCH'), t=0.0, full=False)
STATE = dict(ready=False, idx=None, prefix='arc3_draft_head_subset', vocab=None, dtype=None, mode=None, weight=None)


def _full():
    s = _SWITCH
    if s['path'] is None:
        return False
    now = time.monotonic()
    if now - s['t'] > 1.0:
        s['t'] = now
        try:
            with open(s['path']) as f:
                s['full'] = f.read().strip() == 'full'
        except OSError:
            s['full'] = False
    return s['full']


def subset_ids(path):
    d = json.load(open(path))
    ids = sorted({i for a, b in d['token_ranges'] for i in range(int(a), int(b))} | set(d.get('extra_ids', [])))
    assert ids and ids[0] >= 0, path
    return ids


def _fp8w_head():
    try:
        import mtp0x_v1.install as mx                     # production's fp8w state (FP8 lm_head rows, config table)
    except ImportError:
        return None, None
    return mx, mx.FP8W.get('layers', {}).get('lm_head')


def _prepare(model):
    import torch
    ids = subset_ids(os.environ['ARC3_S2_DRAFTHEAD_VOCAB'])
    n = len(ids)
    mx, st = _fp8w_head()
    if st is not None:                                    # FP8 head (production's fp8w rows): v1's path
        V, K = st['n'], st['k']
        assert ids[-1] < V, (ids[-1], V)
        idx = torch.tensor(ids, dtype=torch.long, device=st['q'].device)
        mx.FP8W['table'][(n, K)] = mx.FP8W['table'][(V, K)]    # the full head's configs: identical per-column arithmetic
        mx.FP8W['layers'][STATE['prefix']] = dict(q=st['q'].index_select(0, idx).contiguous(),
                                                  scale=st['scale'].index_select(0, idx).contiguous(), n=n, k=K)
        STATE.update(ready=True, idx=idx, vocab=V, mode='fp8w')
    else:                                                 # BF16 head: the shared lm_head weight's rows S
        w = model.lm_head.weight
        V = model.logits_processor.vocab_size
        assert ids[-1] < V <= w.shape[0], (ids[-1], V, tuple(w.shape))
        idx = torch.tensor(ids, dtype=torch.long, device=w.device)
        STATE.update(ready=True, idx=idx, vocab=V, mode='bf16', weight=w.index_select(0, idx).contiguous())
    STATS.update(vocab=V, subset=n, mode=STATE['mode'])


def patch_mtp(mod):
    cls = mod.Qwen3_8FlashNextMTP
    original = cls.compute_logits

    def compute_logits(self, hidden_states, spec_step_idx=0):
        import torch
        STATS['calls'] += 1
        if _full():
            STATS['full_calls'] += 1
            return original(self, hidden_states, spec_step_idx)
        if not STATE['ready']:
            _prepare(self)                                  # first call: eager (it synchronises), before any capture
            if TRUNC is not None and _TS['path'] is not None:
                _trunc_flag(hidden_states.device)           # allocated here, never inside a captured graph
        if STATE['mode'] == 'fp8w':
            y = torch.ops.arc3_mtp0x.fp8w_linear(hidden_states, STATE['prefix'])
        else:
            y = torch.nn.functional.linear(hidden_states, STATE['weight'])
        if STATE['dtype'] is None:                          # first call: dtype and per-column identity with the full head
            ref = original(self, hidden_states, spec_step_idx)
            STATE['dtype'] = ref.dtype
            STATS['out_dtype'] = str(ref.dtype)
            STATS['first_call_max_abs_diff'] = float((ref.index_select(1, STATE['idx']).float() - y.float()).abs().max())
            from . import install
            install.save()
        if TRUNC is not None:
            yt = truncate(y, *TRUNC)
            y = torch.where(_trunc_flag(y.device), yt, y) if _TS['path'] is not None else yt
        out = torch.full((hidden_states.shape[0], STATE['vocab']), float('-inf'), dtype=STATE['dtype'],
                         device=hidden_states.device)
        out.index_copy_(1, STATE['idx'], y.to(STATE['dtype']))
        STATS['subset_calls'] += 1
        return out

    compute_logits.__wrapped__ = original
    cls.compute_logits = compute_logits


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.mtp': patch_mtp}
