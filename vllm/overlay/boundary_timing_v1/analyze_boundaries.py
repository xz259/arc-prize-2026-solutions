"""Attribute the inclusive-minus-steady gap from a worker journal (+ optional core journal)."""
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else None
steps = []; events = []
for f in sorted(root.glob('*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        (steps if r['k'] == 'step' else events).append(r)
steps.sort(key=lambda r: r['t']); events.sort(key=lambda r: r['t'])
if N is None: N = max(r['n'] for r in steps)
full = [r for r in steps if r['p'] is not None and r['n'] == N and r['tok'] == N and r['new'] == 0 and r['fin'] == 0]
steady = statistics.median(r['p'] for r in full) if full else float('nan')
ev_by_kind = {}
for e in events: ev_by_kind.setdefault(e['k'], []).append(e)
# attribute each step's excess period to the boundary events that ended inside its period window
import bisect
ev_t = [e['t'] for e in events]
cats = {'settlement': 0.0, 'eviction': 0.0, 'admission': 0.0, 'prefill': 0.0, 'partial_batch': 0.0, 'other': 0.0}; counts = {k: 0 for k in cats}
for r in steps:
    if r['p'] is None: continue
    excess = r['p'] - steady
    if excess <= 0: continue
    lo, hi = r['t'] - r['p'], r['t']; i, j = bisect.bisect_left(ev_t, lo), bisect.bisect_right(ev_t, hi)
    kinds = {events[k]['k'] for k in range(i, j)}
    if 'worker_settle' in kinds or 'settle_core' in kinds: c = 'settlement'
    elif 'worker_compact' in kinds or 'evict_resident' in kinds: c = 'eviction'
    elif r['new'] > 0: c = 'admission'
    elif r['tok'] > r['n']: c = 'prefill'
    elif r['n'] < N: c = 'partial_batch'
    else: c = 'other'
    cats[c] += excess; counts[c] += 1
wall = steps[-1]['t'] - steps[0]['t'] if len(steps) > 1 else 0
print(f'steps {len(steps)} over {wall:.1f} s; N={N}; full-occupancy pure-decode steps {len(full)}; steady period median {steady*1e3:.2f} ms, p90 {statistics.quantiles([r["p"] for r in full], n=10)[-1]*1e3 if len(full) > 10 else float("nan"):.2f} ms')
tot = sum(cats.values())
print(f'excess over steady: {tot:.2f} s = {100*tot/wall:.1f}% of wall')
for k in cats: print(f'  {k:14s} {cats[k]:7.2f} s  {100*cats[k]/wall:5.1f}%  steps {counts[k]}')
for k, v in ev_by_kind.items():
    ds = sorted(e['d'] for e in v); print(f'event {k:16s} n={len(v):5d} mean {1e3*sum(ds)/len(ds):7.2f} ms  p50 {1e3*ds[len(ds)//2]:7.2f}  p90 {1e3*ds[int(0.9*len(ds))]:7.2f}  max {1e3*ds[-1]:8.2f}  total {sum(ds):.2f} s')
