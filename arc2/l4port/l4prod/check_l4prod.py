"""check_l4prod.py INSTANCE_DIR SERVER_LOG TRY_LOG [--verify] [--min-evicts N] (2026-10-31, ARC2 - main): the pre-stated pass line of
a production-path PP run (the reviewer's items, the user approved the plan). Numbers only; prints one 'PASS' or 'FAIL: ...' line first,
then the counts. Checks:
  harness   the try log's harness line: exit 0 and a score
  engine    no PersistentStateError / EngineDeadError in the server log (v2: EngineDeadError counted only before the first
            '[shutdown]' line: the stop's forced EngineCore kill raises one in the output handler); no '[l4port] evpp worker
            callable ... not ported' line
  receipts  '[l4port] evpp batch queue capped 3 -> 2' and '[l4port] evpp single-batch policy off under PP' present
  evictions >= N engine 'evict' records (default 1); each merged worker receipt synchronized; submits == commits (service)
  qsa       no 'frequency_snapshot_failed' service events (evictions ranked by QSA frequency, not the fallback)
  verify    with --verify: every eviction byte-verified (byte_verification_enabled, every copy equal) and allocator-checked
  preempt   the try log's metrics line: vllm:num_preemptions_total 0 (when present)
Also reported: the evpp TP-mismatch receipts (count, summed |diff| / hits)."""
import json
import os
import re
import sys

inst, slog, tlog = sys.argv[1:4]
VERIFY = '--verify' in sys.argv
MIN_EV = int(sys.argv[sys.argv.index('--min-evicts') + 1]) if '--min-evicts' in sys.argv else 1
fails, info = [], {}


def lines(path):
    try:
        with open(path, errors='replace') as f:
            return f.read().splitlines()
    except OSError:
        return []


T = lines(tlog)
S = lines(slog)
h = [l for l in T if re.search(r'\] \S+ exit \d+: ', l) and 'server wrapper' not in l]
if not h or not re.search(r' exit 0: score [0-9.]+', h[-1]):
    fails.append('harness line missing or not exit 0 with a score')
info['harness'] = h[-1][h[-1].index('] ') + 2:][:160] if h else None
SHUT = next((i for i, l in enumerate(S) if '[shutdown]' in l), len(S))   # v2 (2026-10-31): the stop's own EngineDeadError
info['shutdown_line'] = SHUT if SHUT < len(S) else None                     # (output handler at the forced EngineCore kill) is no failure
for pat, name in (('PersistentStateError', 'persistent_errors'), ('EngineDeadError', 'engine_dead'),
                  (r'\[l4port\] evpp worker callable .* not ported', 'unrouted')):
    n = sum(1 for l in (S[:SHUT] if name == 'engine_dead' else S) if re.search(pat, l))
    info[name] = n
    if n:
        fails.append('%s %d' % (name, n))
for pat, name in ((r'\[l4port\] evpp batch queue capped 3 -> 2', 'queue_cap'),
                  (r'\[l4port\] evpp single-batch policy off under PP', 'policy_off')):
    ok = any(re.search(pat, l) for l in S)
    info[name] = ok
    if not ok:
        fails.append('receipt missing: ' + name)
mm = [l for l in S if '[l4port] evpp snapshot TP mismatch' in l]
info['tp_mismatch_receipts'] = len(mm)
if mm:
    tot = hits = 0
    for l in mm:
        m = re.search(r"'hits': (\d+), 'abs_diff': (\d+)", l)
        if m:
            hits += int(m.group(1)); tot += int(m.group(2))
    info['tp_mismatch_abs_diff_over_hits'] = round(tot / hits, 6) if hits else None

ev, svc = [], {}
for line in lines(os.path.join(inst, 'engine-events.jsonl')):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get('event') == 'evict':
        ev.append(d.get('eviction') or {})
for line in lines(os.path.join(inst, 'service-events.jsonl')):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    k = d.get('event')
    svc[k] = svc.get(k, 0) + 1
info['evicts'] = len(ev)
info['service'] = {k: svc.get(k, 0) for k in ('evict_submit', 'evict_commit', 'frequency_snapshot_failed', 'close')}
if len(ev) < MIN_EV:
    fails.append('evictions %d < %d' % (len(ev), MIN_EV))
if svc.get('evict_submit', 0) != svc.get('evict_commit', 0):
    fails.append('evict submits %d != commits %d' % (svc.get('evict_submit', 0), svc.get('evict_commit', 0)))
if svc.get('frequency_snapshot_failed', 0):
    fails.append('frequency snapshots failed %d' % svc['frequency_snapshot_failed'])
unsync = verified = unverified = bad_bytes = no_alloc = copies = 0
for e in ev:
    wr = (e.get('worker_receipts') or [{}])[0] or {}
    if wr.get('worker_synchronized') is not True:
        unsync += 1
    cps = wr.get('copies') or []
    copies += len(cps)
    if wr.get('byte_verification_enabled'):
        verified += 1
        bad_bytes += sum(1 for c in cps if not (c.get('byte_verification') or {}).get('equal'))
    else:
        unverified += 1
    if e.get('allocator_reacquisition') is None:
        no_alloc += 1
info.update(unsynchronized=unsync, copies_per_evict=round(copies / len(ev), 1) if ev else None, byte_verified=verified,
            byte_mismatch_copies=bad_bytes, allocator_unchecked=no_alloc)
if unsync:
    fails.append('unsynchronized merged receipts %d' % unsync)
if VERIFY:
    if unverified:
        fails.append('evictions without byte verification %d' % unverified)
    if bad_bytes:
        fails.append('byte-mismatched copies %d' % bad_bytes)
    if no_alloc:
        fails.append('evictions without the allocator check %d' % no_alloc)
pm = [l for l in T if 'vllm:num_preemptions_total' in l]
if pm:
    vals = [float(x) for x in re.findall(r'num_preemptions_total\{[^}]*\} ([0-9.]+)', pm[-1])]
    info['preemptions'] = vals
    if any(v > 0 for v in vals):
        fails.append('preemptions %s' % vals)
print('PASS' if not fails else 'FAIL: ' + '; '.join(fails))
print(json.dumps(info, sort_keys=True))
