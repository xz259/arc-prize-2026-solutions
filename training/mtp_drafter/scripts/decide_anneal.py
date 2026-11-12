"""decide_anneal.py (2026-10-14, Claude fork 5d20b23b): the pre-stated rule for the user's anneal at step 4,500 (SIDE_POD_LOG
10-14 05:3x). Reads the trainer's eval records (JSON lines with step and val_tau_srv) on stdin, fits log-linear (steps >=
750) and a saturating power law tau_inf - A s^-b (steps >= 375) to the stable-phase evals, takes the fit with the lower
rmse, and projects the gain from the last eval step S to the next milestone S + 2,250. ANNEAL if that gain is < 0.5%
(the plateau bar), else CONTINUE. usage: grep val_tau_srv log.jsonl | python decide_anneal.py [--next 2250] [--bar 0.5]"""
import argparse
import json
import sys

import numpy as np
from scipy.optimize import curve_fit

ap = argparse.ArgumentParser()
ap.add_argument('--next', type=int, default=2250)
ap.add_argument('--bar', type=float, default=0.5)
a = ap.parse_args()
pts = sorted({r['step']: r['val_tau_srv'] for r in map(json.loads, sys.stdin) if 'val_tau_srv' in r}.items())
s = np.array([p[0] for p in pts if p[0] >= 375], float)
t = np.array([p[1] for p in pts if p[0] >= 375], float)
S = s[-1]
print('evals: ' + ', '.join('%d %.4f' % p for p in pts))
fits = {}
m = s >= 750
c = np.polyfit(np.log2(s[m]), t[m], 1)
fits['log-linear'] = (np.sqrt(np.mean((t[m] - np.polyval(c, np.log2(s[m]))) ** 2)), lambda x: np.polyval(c, np.log2(x)))
f = lambda x, ti, A, b: ti - A * x ** (-b)
try:
    p, _ = curve_fit(f, s, t, p0=(t[-1] + 0.05, 5, 0.5), maxfev=50000)
    fits['power law (tau_inf %.4f, b %.3f)' % (p[0], p[2])] = (np.sqrt(np.mean((t - f(s, *p)) ** 2)), lambda x: f(x, *p))
except RuntimeError as e:
    print('power-law fit failed: %s' % e)
for name, (rmse, g) in fits.items():
    print('%-34s rmse %.4f  at %d %.4f  at %d %.4f  gain %+.2f%%' % (name, rmse, S, g(S), S + a.next, g(S + a.next),
                                                               100 * (g(S + a.next) / g(S) - 1)))
best = min(fits, key=lambda k: fits[k][0])
gain = 100 * (fits[best][1](S + a.next) / fits[best][1](S) - 1)
print('better fit: %s; projected gain %d -> %d: %+.2f%% -> %s' % (best, S, S + a.next, gain,
                                                                   'ANNEAL' if gain < a.bar else 'CONTINUE'))
