"""stage2_v1 'promptmove' part (2026-10-04 03:30; rebase design option 3, user ~00:09 "we do option 3", ~02:45 "Let's build
Option 3 before the runs"). The system prompt's stored keys follow the context instead of the context being rebased: at
every worker staging of a session (its re-add after a history eviction), when the oldest resident row after the leading
block has moved far from it, the block's stored keys are rotated forward so the block sits just behind that row.

Block. The leading contiguous run of resident rows [0, R) at the first move: the system prompt (protected, never
evicted), plus any first-interaction rows still adjacent to it (the age cap normally removes the first interaction whole
first, so R is the prompt). R0 is fixed per request from then on; later moves move the block's rows that are still
resident (original positions < R0, a prefix of the rows) by the same amount, so the block stays internally intact.
Rows after the block are never touched.

Target. With t the original M-RoPE position of the first resident row after the block (staged rows), the block takes
positions [t - R0, t): delta = t - R0. A move happens when delta - current >= ARC3_S2_PMOVE_STEP (default 1,024) and
delta >= ARC3_S2_PMOVE_MIN_GAP (default 4,096). The block's own positions are 0..R0-1 (text; checked).

What moves (every QSA owner: the 12 target layers and the MTP drafter's layer):
  main KV, K side only (148 of 296 bytes per head row): the original record (captured once per request, before its first
     move, from the request's own rows, kept in pinned host memory: ~23 MB per session for a 5K prompt) is decoded
     (codebook levels x E4M3 block scale x head scale), turned back by the normalized Hadamard, the layer's K mean added,
     the 64 rotary dims (NeoX pairs i, i+32) rotated by delta x inv_freq (float64 angles from the model's float32 inverse
     frequencies; for text tokens all three M-RoPE axes share the position, so one angle per pair), then written by the
     production codebook store (kvlut_kernels.store: centering, Hadamard, block-MSE scales) through a scratch page;
     the K bytes are copied into the request's rows. Every move starts from the original: always two roundings.
  QSA compressed keys (BF16, one row per 4-token group; RoPE'd at the group's first position): rotated by delta from
     the captured originals, for the retained groups whose first position is < R0 (ranks 0..m-1).
  V, the raw-key ring and the recorded positions are unchanged (V carries no position; the ring only holds the open
     group at the frontier; the block's compressed groups are complete, so nothing recomputes them).
Requires ARC3_PREFIX_SHARE=0 (a shared first page would move every lane's prompt), codebook (lut*) K formats on every
owner, and rope2m for positions past 262,144. A rebase restart is a new request: fresh state, positions from 0.
ARC3_S2_PMOVE_VERIFY=1: every move re-reads what it wrote; every capture compares its K rows with the first capture's
(the same system prompt in every lane). Lanes compute the prompt in different batch shapes, so the stored bytes differ
(gameplay 10-04: byte agreement 0.43-0.46; 1.0 only at a fixed shape, as in the gate panel) and the decoded (centered)
values differ by 0.04 (layer 3) to 0.28 (layer 47), growing with depth (job 310a); rows of another context give ~1.4.
A capture whose decoded error against the first exceeds ARC3_S2_PMOVE_CAPTURE_MAX_ERR (default 0.6) on any owner counts
as a capture mismatch and a verify failure.
Errors are counted; a failed move leaves the rows unmodified (all writes happen after every computation succeeded).
Test hook (gate panel; ARC3_S2_PMOVE_TEST=<file> holding '<delta> <split id> <rows>'): a request admitted without an
eviction context whose first group boundary after the split token is rows has its rows [0, rows) captured and moved by
delta once rows of it are computed (before the next step's forward), once per request; other requests are counted in
test_skipped. Used with rope2m's split shift (the rest far away) and a prefill chunk limit of rows, so the first chunk
is exactly the system turn."""
import os
import time

import numpy as np

PINS = {'eviction_runner': '03ecae34a987b0b3203df42cd609f41c051226371c517ae85d69591851253ed8'}
MIN_GAP = int(os.environ.get('ARC3_S2_PMOVE_MIN_GAP', '4096'))
STEP = int(os.environ.get('ARC3_S2_PMOVE_STEP', '1024'))
PREFIX_MAX = int(os.environ.get('ARC3_S2_PMOVE_PREFIX_MAX', '16384'))
VERIFY = os.environ.get('ARC3_S2_PMOVE_VERIFY', '0') == '1'
CAPTURE_MAX_ERR = float(os.environ.get('ARC3_S2_PMOVE_CAPTURE_MAX_ERR', '0.6'))
TEST_SWITCH = os.environ.get('ARC3_S2_PMOVE_TEST', '')
STATS = dict(stagings=0, moves=0, captures=0, rows_moved=0, groups_moved=0, move_ms=0.0, capture_ms=0.0,
             skipped_gap=0, skipped_step=0, skipped_layout=0, skipped_block=0, freed=0, errors=0, last_error=None,
             last=None, verify_moves=0, verify_failures=0, capture_mismatches=0, capture_agreement=None, test_moves=0, test_skipped=0, test_aligned=0, owners=None)
_CTX = {}


def _err(exc):
    STATS['errors'] += 1
    STATS['last_error'] = repr(exc)[:400]


def _drafter_out_of_pool(name):
    """True for the MTP drafter's owner when the 'draftring' part keeps its KV in a private ring (pool block ids do not address it)."""
    from . import install
    if 'draftring' not in install.PARTS:
        return False
    from .kvrot import _DRAFT_NAME
    return bool(_DRAFT_NAME.search(name))


def _tensor(x):
    import torch
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
    return x if torch.is_tensor(x) and x.numel() > 0 else None


def _context(runner):
    """Owners (main and compressed caches, groups, formats, means) and the float32 inverse frequencies, once per runner."""
    ctx = _CTX.get(id(runner))
    if ctx is not None:
        return ctx
    import torch
    if os.environ.get('ARC3_PREFIX_SHARE', '1') != '0':
        raise RuntimeError('promptmove: needs ARC3_PREFIX_SHARE=0 (a shared prefix page would move every lane)')
    sfc = runner.compilation_config.static_forward_context
    gidx = {}
    for i, g in enumerate(runner.kv_cache_config.kv_cache_groups):
        for n in g.layer_names:
            gidx[n] = i
    owners = []
    for name, layer in sfc.items():
        if not name.endswith('.attn'):
            continue
        if _drafter_out_of_pool(name):                   # 2026-10-15 (4b, IE review): the drafter's KV is draftring's ring
            STATS['skipped_drafter'] = name
            continue
        cache = _tensor(getattr(layer, 'kv_cache', None))
        if cache is None or cache.dtype != torch.uint8 or cache.ndim != 4 or cache.shape[3] != 296:
            continue
        cname = name[:-len('.attn')] + '.indexer.compressed_key_cache'
        ccache = _tensor(getattr(sfc.get(cname), 'kv_cache', None)) if cname in sfc else None
        if ccache is not None and (ccache.dtype != torch.bfloat16 or ccache.ndim != 4 or ccache.shape[1] * 4 != cache.shape[2]):
            raise RuntimeError('promptmove: unexpected compressed key cache for %s: %s' % (name, tuple(ccache.shape)))
        owners.append(dict(name=name, layer=layer, cache=cache, g=gidx[name], page=int(cache.shape[2]), heads=int(cache.shape[1]),
                           cname=cname, ccache=ccache, cg=gidx.get(cname), crows=int(ccache.shape[1]) if ccache is not None else 0))
    if not owners:
        raise RuntimeError('promptmove: no QSA owners found')
    rot = None
    for m in runner.model.modules():
        if getattr(m, 'rotary_dim', None) == 64 and getattr(m, 'head_size', None) == 256 and hasattr(m, 'base'):
            rot = m
            break
    if rot is None or not getattr(rot, 'is_neox_style', True):
        raise RuntimeError('promptmove: attention rotary embedding (rotary_dim 64, NeoX) not found')
    inv = (1.0 / (float(rot.base) ** (torch.arange(0, 64, 2, dtype=torch.float) / 64))).to(torch.float64)
    from . import kvlut_kernels
    lut = torch.tensor([-v for v in reversed(kvlut_kernels.LUTP)] + list(kvlut_kernels.LUTP), dtype=torch.float32)
    dev = owners[0]['cache'].device
    ctx = _CTX[id(runner)] = dict(owners=owners, inv=inv, lut=lut.to(dev), dev=dev, scratch=None, ref=None)
    STATS['owners'] = len(owners)
    STATS['owners_without_compressed'] = [o['name'] for o in owners if o['ccache'] is None]
    return ctx


def _role(owner):
    r = getattr(owner['layer'], '_arc3_kvrot_role', None)
    if r is None or not str(r[0]).startswith('lut'):
        raise RuntimeError('promptmove: %s is not in a codebook K format (role %r)' % (owner['name'], None if r is None else r[0]))
    return r


def _main_index(block_ids, rows, page, dev):
    import torch
    bids = np.asarray(block_ids, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    return (torch.from_numpy(bids[rows // page]).to(dev), torch.from_numpy(rows % page).to(dev))


def _decode(ctx, rec):
    """Codebook K records [n, H, 148] uint8 -> the stored (centered, Hadamard-domain) values [n, H, 256] FP32."""
    import torch
    n, h = rec.shape[0], rec.shape[1]
    codes = rec[..., :128].long()
    lv = torch.stack((ctx['lut'][codes & 15], ctx['lut'][codes >> 4]), dim=-1).reshape(n, h, 16, 16)
    bs = rec[..., 128:144].contiguous().view(torch.float8_e4m3fn).float()
    hs = rec[..., 144:148].contiguous().view(torch.float32)
    return (lv * (bs * hs)[..., None]).reshape(n, h, 256)


def _angles(ctx, delta):
    import torch
    a = ctx['inv'] * float(delta)
    return a.cos().float().to(ctx['dev']), a.sin().float().to(ctx['dev'])


def _rope_(x, cs):
    """Rotate the 64 rotary dims (NeoX pairs i, i+32) of x [..., D] FP32 in place by the angles cs = (cos, sin)."""
    cos, sin = cs
    a = x[..., :32].clone()
    b = x[..., 32:64].clone()
    x[..., :32] = a * cos - b * sin
    x[..., 32:64] = a * sin + b * cos
    return x


def _encode(ctx, owner, kfull, role):
    """Production codebook store of BF16 K [n, H, 256] through a scratch page; returns the K bytes [n, H, 148]."""
    import torch
    from . import kvlut_kernels
    n, page, heads = kfull.shape[0], owner['page'], owner['heads']
    nb = (n + page - 1) // page
    s = ctx['scratch']
    if s is None or s.shape[0] < nb or s.shape[1] != heads or s.shape[2] != page:
        s = ctx['scratch'] = torch.empty((nb, heads, page, 296), dtype=torch.uint8, device=ctx['dev'])
    slots = torch.arange(n, dtype=torch.int64, device=ctx['dev'])
    mu = role[2]
    mk, mv = (mu[0], mu[1]) if mu is not None else (None, None)
    kvlut_kernels.store(kfull, kfull, s, slots, mk, mv, unbias=str(role[0]).endswith('u'))
    return s[slots // page, :, slots % page, :148]


def _capture(ctx, block_ids, R):
    """Originals of rows [0, R) and groups [0, ceil(R / 4)) of every owner, in pinned host memory."""
    import torch
    t0 = time.perf_counter()
    out = {}
    G = (R + 3) // 4
    for o in ctx['owners']:
        blk, off = _main_index(block_ids[o['g']], np.arange(R), o['page'], ctx['dev'])
        k = o['cache'][blk, :, off, :148].to('cpu').pin_memory()
        c = None
        if o['ccache'] is not None:
            cblk, coff = _main_index(block_ids[o['cg']], np.arange(G), o['crows'], ctx['dev'])
            c = o['ccache'][cblk, coff, 0, :].to('cpu').pin_memory()
        out[o['name']] = (k, c)
    STATS['captures'] += 1
    STATS['capture_ms'] += 1000 * (time.perf_counter() - t0)
    if VERIFY:
        # 10-04 04:45 / 05:00 (gameplay smokes: byte agreement 0.456 / 0.443 against 1.0 in the one-request-at-a-time gate
        # panel): also the decoded values (codebook bytes flip on small numeric differences; rows of another context give
        # ~1.4), by row range and by owner, with the block's first page and R, per capture.
        ref = ctx['ref']
        g0 = ctx['owners'][0]['g']
        diag = dict(R=int(R), page0=int(block_ids[g0][0]))
        STATS.setdefault('capture_diag', [])
        if ref is None:
            ctx['ref'] = {n: v[0][:3104].clone() for n, v in out.items()}
        else:
            bins = ((0, 16), (16, 256), (256, 1024), (1024, 3104))
            bs, bt, bn, bd = [0] * 4, [0] * 4, [0.0] * 4, [0.0] * 4
            own = {}
            for n, (k, _) in out.items():
                m = min(len(k), len(ref[n]))
                eq = k[:m] == ref[n][:m]
                a, b = _decode(ctx, k[:m].to(ctx['dev'])), _decode(ctx, ref[n][:m].to(ctx['dev']))
                d2, r2 = ((a - b) ** 2).sum(dim=(1, 2)), (b ** 2).sum(dim=(1, 2))
                own[n.replace('model.', '').replace('.attn', '')] = round(float(d2.sum() / r2.sum().clamp_min(1e-30)) ** 0.5, 4)
                for j, (lo, hi) in enumerate(bins):
                    if lo < m:
                        h = min(hi, m)
                        bs[j] += int(eq[lo:h].sum())
                        bt[j] += eq[lo:h].numel()
                        bn[j] += float(d2[lo:h].sum())
                        bd[j] += float(r2[lo:h].sum())
            agree, err = sum(bs) / max(1, sum(bt)), (sum(bn) / max(sum(bd), 1e-30)) ** 0.5
            STATS['capture_agreement'] = round(agree, 4)
            STATS.setdefault('capture_agreements', []).append(round(agree, 3))
            STATS.setdefault('capture_rel_err', []).append(round(err, 4))
            STATS.setdefault('capture_rel_err_head16', []).append(round((bn[0] / max(bd[0], 1e-30)) ** 0.5, 4))
            diag.update(agree=[round(bs[j] / max(1, bt[j]), 3) for j in range(4)],
                        err=[round((bn[j] / max(bd[j], 1e-30)) ** 0.5, 4) for j in range(4)], owners=own)
            if own and max(own.values()) > CAPTURE_MAX_ERR:          # wrong rows, not batch-shape numerics
                STATS['capture_mismatches'] += 1
                STATS['verify_failures'] += 1
                diag['mismatch'] = True
        STATS['capture_diag'].append(diag)
    return out


def _move(ctx, block_ids, orig, rows, src_pos, ranks, src_gid, delta):
    """Rotate the originals src_pos (rows) and src_gid (compressed ranks) by delta; all computation before any write."""
    import torch
    cs = _angles(ctx, delta)
    writes = []
    for o in ctx['owners']:
        role = _role(o)
        k0, c0 = orig[o['name']]
        rec = k0[torch.from_numpy(np.asarray(src_pos, dtype=np.int64))].to(ctx['dev'], non_blocking=True)
        from .kvrot import rotate_rows
        x = rotate_rows(_decode(ctx, rec))
        mu = role[2]
        if mu is not None:
            x = x + mu[0][None]
        kb = _encode(ctx, o, _rope_(x, cs).to(torch.bfloat16), role)
        blk, off = _main_index(block_ids[o['g']], rows, o['page'], ctx['dev'])
        c = cblk = coff = None
        if c0 is not None:
            c = _rope_(c0[torch.from_numpy(np.asarray(src_gid, dtype=np.int64))].to(ctx['dev'], non_blocking=True).float(), cs)
            c = c.to(torch.bfloat16)
            cblk, coff = _main_index(block_ids[o['cg']], ranks, o['crows'], ctx['dev'])
        writes.append((o, blk, off, kb.clone(), cblk, coff, c))
    for o, blk, off, kb, cblk, coff, c in writes:
        o['cache'][blk, :, off, :148] = kb
        if c is not None:
            o['ccache'][cblk, coff, 0, :] = c
    if VERIFY:
        STATS['verify_moves'] += 1
        for o, blk, off, kb, cblk, coff, c in writes:
            if not torch.equal(o['cache'][blk, :, off, :148], kb) or (c is not None and not torch.equal(o['ccache'][cblk, coff, 0, :], c)):
                STATS['verify_failures'] += 1


def maybe_move(runner, req_id, block_ids, rows):
    """The gameplay path: after the request's staging (its context and M-RoPE rows are current)."""
    state = runner._arc3_eviction_states.get(req_id)
    if state is None:
        return
    store = runner.__dict__.setdefault('_arc3_pmove', {})
    for rid in [r for r in store if r not in runner._arc3_eviction_states]:
        del store[rid]
        STATS['freed'] += 1
    pos = np.asarray(state['resident_positions'], dtype=np.int64)
    t = rows[0]
    N = len(pos)
    if N == 0:
        return
    pm = store.get(req_id)
    if pm is None:
        lead = pos == np.arange(N)
        R = int(N if lead.all() else np.argmin(lead))
        if R == N:
            return                                                  # nothing evicted after the block yet
        if R < 64 or R > PREFIX_MAX:
            STATS['skipped_block'] += 1
            return
        if list(np.asarray(t[:R], dtype=np.int64)) != list(range(R)):
            STATS['skipped_layout'] += 1
            return
        target = int(t[R]) - R
        if target < MIN_GAP:
            STATS['skipped_gap'] += 1
            return
        orig = _capture(_context(runner), block_ids, R)
        pm = store[req_id] = dict(R0=R, delta=0, orig=orig)
        k = R
    else:
        R = pm['R0']
        k = int(np.searchsorted(pos, R))
        if k >= N:
            return
        target = int(t[k]) - R
        if target - pm['delta'] < STEP:
            STATS['skipped_step'] += 1
            return
    ctx = _context(runner)
    t0 = time.perf_counter()
    gids = np.unique(pos[:k] // 4)
    _move(ctx, block_ids, pm['orig'], np.arange(k), pos[:k], np.arange(len(gids)), gids, target)
    STATS['move_ms'] += 1000 * (time.perf_counter() - t0)
    STATS['moves'] += 1
    STATS['rows_moved'] += k
    STATS['groups_moved'] += len(gids)
    STATS['last'] = dict(req=req_id[-24:], R0=R, rows=k, groups=len(gids), delta=target, previous=pm['delta'], next_position=int(t[k]))
    pm['delta'] = target
    _save()


def _save():
    try:
        from . import install
        install.REPORT['promptmove'] = STATS
        install.save(force=os.environ.get('ARC3_S2_PMOVE_SAVE_FORCE', '1') == '1')   # v7.5 C2: 0 = install's 60-s throttle
    except Exception:                                   # noqa: BLE001  receipts are diagnostics
        pass


def patch_eviction_runner(mod):
    original = mod._stage_request

    def _stage_request(runner, req_index, new_req_data, context):
        rows, delta = original(runner, req_index, new_req_data, context)
        STATS['stagings'] += 1
        try:
            maybe_move(runner, new_req_data.req_id, new_req_data.block_ids, rows)
        except Exception as exc:                        # noqa: BLE001  the turn proceeds with the rows as they were
            _err(exc)
            _save()
        if STATS['stagings'] % 200 == 1:
            _save()
        return rows, delta

    mod._stage_request = _stage_request

    if TEST_SWITCH:
        install = mod.install_runner_hooks

        def install_runner_hooks():
            out = install()
            _install_test_hook()
            return out

        mod.install_runner_hooks = install_runner_hooks


def _read_test():
    try:
        with open(TEST_SWITCH) as f:
            parts = f.read().split()
        return (int(parts[0]), int(parts[1]), int(parts[2])) if len(parts) >= 3 else None
    except (OSError, ValueError):
        return None


def _install_test_hook():
    """Gate-panel hook: capture block ids at admission (no eviction context), move once rows are computed."""
    from vllm.models.qwen3_8_flash_next.nvidia.model_state import Qwen3_8FlashNextModelState as MS
    add, prep = MS.add_request, MS.prepare_inputs

    def add_request(self, req_index, new_req_data):
        out = add(self, req_index, new_req_data)
        runner = getattr(self, '_arc3_eviction_runner', None)
        extra = getattr(new_req_data.sampling_params, 'extra_args', None) or {}
        spec = _read_test()
        if runner is not None and spec is not None and 'arc3_eviction_context' not in extra and spec[0] != 0:
            ids = list(new_req_data.prompt_token_ids or [])
            rows = -(-(ids.index(spec[1]) + 1) // 4) * 4 if spec[1] in ids else None
            if rows != spec[2]:
                STATS['test_skipped'] += 1
            else:
                runner.__dict__.setdefault('_arc3_pmove_test', {})[new_req_data.req_id] = dict(
                    delta=spec[0], rows=rows, block_ids=[list(b) for b in new_req_data.block_ids], done=False)
        return out

    def prepare_inputs(self, input_batch, req_states):
        runner = getattr(self, '_arc3_eviction_runner', None)
        tests = getattr(runner, '_arc3_pmove_test', None) if runner is not None else None
        if tests:
            for rid in list(tests):
                if rid not in req_states.req_id_to_index:
                    del tests[rid]
                    continue
                tst = tests[rid]
                idx = req_states.req_id_to_index[rid]
                computed = int(req_states.num_computed_tokens_np[idx])
                if not tst['done'] and computed >= tst['rows']:
                    tst['done'] = True
                    STATS['test_aligned'] += int(computed == tst['rows'])   # the first chunk ended exactly at the block
                    try:
                        ctx = _context(runner)
                        n = tst['rows']
                        orig = _capture(ctx, tst['block_ids'], n)
                        _move(ctx, tst['block_ids'], orig, np.arange(n), np.arange(n), np.arange((n + 3) // 4),
                              np.arange((n + 3) // 4), tst['delta'])
                        STATS['test_moves'] += 1
                        STATS['last'] = dict(req=rid[-24:], test=True, rows=n, delta=tst['delta'], computed=computed)
                    except Exception as exc:            # noqa: BLE001
                        _err(exc)
                    _save()
        return prep(self, input_batch, req_states)

    MS.add_request = add_request
    MS.prepare_inputs = prepare_inputs


PATCHERS = {'eviction_runner': patch_eviction_runner}
