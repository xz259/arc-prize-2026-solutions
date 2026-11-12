"""apply_dflashtaps_mtp.py (2026-10-11; MTP fine-tune backup plan; NOT applied while Phase 6 uses the replay): writes a
patched copy of stage2_v1/dflashtaps.py (pinned source md5 31a39ca9) with the option ARC3_S2_TAPS_MTP=1: the target's
pre-final-mixer multi-stream state (`_mtp_hidden_buffer` [rows, hc*H], what the MTP drafter consumes) is exported as hc
extra slots after the final state, so a file's hidden_states is [N, n_taps + 1 + hc, H] (slot order: ARC3_S2_TAPS, the
final state, then the hc streams, HC outer / HS inner as in the buffer). The per-request accumulation, segments and files
are unchanged (they carry whatever slots the tensor has; ARC3_S2_TAPS_STORE selects slots as before, e.g. the final
state + the hc streams). With the option unset the code path is the original's. Requires a server launched with the MTP
drafter (the production K = 3 launch), which allocates the buffer.
usage: python apply_dflashtaps_mtp.py SRC_dflashtaps.py OUT_dflashtaps.py   (then deploy OUT with a backup of SRC)"""
import hashlib
import sys

src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
assert hashlib.md5(t.encode()).hexdigest().startswith('31a39ca9'), 'unexpected dflashtaps.py version'


def rep(old, new):
    global t
    assert t.count(old) == 1, old[:80]
    t = t.replace(old, new)


rep("PREFIX = os.environ.get('ARC3_S2_TAPS_PREFIX', '')\n",
    "PREFIX = os.environ.get('ARC3_S2_TAPS_PREFIX', '')\n"
    "# 2026-10-11 (MTP fine-tune backup plan, mtp_ft_20261011/README.md): ARC3_S2_TAPS_MTP=1 also exports the target's\n"
    "# pre-final-mixer multi-stream state (the MTP drafter's input, `_mtp_hidden_buffer` [rows, hc*H]) as hc slots after the\n"
    "# final state; needs the MTP drafter configured (the buffer exists only then)\n"
    "MTP = os.environ.get('ARC3_S2_TAPS_MTP', '') == '1'\n")
rep("             prefix=PREFIX)\n", "             prefix=PREFIX, mtp=MTP, mtp_streams=0)\n")
rep("        k = len(TAPS)\n        slot['h'][:k, :n].copy_(model._dflash_tap_buffer[:, :n], non_blocking=True)\n"
    "        slot['h'][k, :n].copy_(state.hidden_states[:n], non_blocking=True)\n",
    "        k = len(TAPS)\n        slot['h'][:k, :n].copy_(model._dflash_tap_buffer[:, :n], non_blocking=True)\n"
    "        slot['h'][k, :n].copy_(state.hidden_states[:n], non_blocking=True)\n"
    "        if MTP:                                                       # hc streams of the MTP input, after the final state\n"
    "            mb = model._mtp_hidden_buffer\n"
    "            hc = mb.shape[-1] // slot['h'].shape[-1]\n"
    "            slot['h'][k + 1:k + 1 + hc, :n].copy_(mb[:n].view(n, hc, -1).transpose(0, 1), non_blocking=True)\n")
rep("                if not _W:\n                    _W.append(_Writer(len(TAPS) + 1, int(buf.shape[-1]), int(buf.shape[1]), buf.device))\n",
    "                if not _W:\n"
    "                    extra = 0\n"
    "                    if MTP:\n"
    "                        mb = getattr(model, '_mtp_hidden_buffer', None)\n"
    "                        if mb is None:\n"
    "                            raise RuntimeError('ARC3_S2_TAPS_MTP=1 needs the MTP drafter (no _mtp_hidden_buffer)')\n"
    "                        extra = int(mb.shape[-1]) // int(buf.shape[-1])\n"
    "                        STATS['mtp_streams'] = extra\n"
    "                    _W.append(_Writer(len(TAPS) + 1 + extra, int(buf.shape[-1]), int(buf.shape[1]), buf.device))\n")
open(dst, 'w').write(t)
print('patched copy written: %s (md5 %s)' % (dst, hashlib.md5(t.encode()).hexdigest()[:8]))
