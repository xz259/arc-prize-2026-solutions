"""Figure 1 of the ARC-3 write-up: estimated validation score at each adopted configuration (rewritten 2026-11-12).

Source: inference_stack/experiments/score_extracts_20261028 (RUN_INDEX.tsv). Comparable runs only (about 4.9 min per game).
Estimated score on the 55 core games = (25 x public-25 official + 30 x copycat-30 estimate) / 55, as in the README.
Each point is the mean of all comparable runs of one stage; the thin bar spans their minimum and maximum.
Runs at other context lengths (p76 70K arm, p77 89K / 98K, k78 92K arm) and the rejected requant arm plf1b06 are left out. Exploratory runs of 10-21 to 10-23 (context search, precision tests, throttled host) and the broken kag71 / stopped ctx78 runs
are not stages and are omitted; the text reports the 61K vs 71K pair separately.
Run from this directory: python3 make_fig_score_over_time.py
"""
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SRC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'figures', 'arc3_validation_score_over_time.png')
BLUE, INK, MUTED, GRID, SURFACE = '#2a78d6', '#1f1e1c', '#6b6a66', '#e6e5e1', '#fcfcfb'

STAGES = [  # (label, runs)
    ('61.6K context, 36 lanes, FP16 state,\nhard restart at 200K', ['p18off01', 'p18on01']),
    ('Scaled FP16 state, no restarts,\nrefresh at 0.5M positions', ['hab70s16a01', 'hab70s16b01']),
    ('80K context, 200K refresh\nwith parking, rebase at 250K', ['p76s16a01']),  # p76s16b01 = the 70K arm (user 11-12); p77 = 89K vs 98K: other contexts
    ('Restart-value\nranking', ['n76s16a01', 'n76s16b01', 'k78s16a01', 'k81s16a02', 'k81s16b02']),  # k78s16b01 = 92K arm
    ('Stochastic rounding\nof the state', ['r16h01', 'plf1a06', 'plf2a01']),  # v10 82kr1632; plf1b06 (rejected expert requant) left out
    ('Exact KV set\n(final stack)', ['plf2b01']),  # ring 1,280 + exact 352
]

idx = {}
for line in open(os.path.join(SRC, 'RUN_INDEX.tsv')).read().splitlines()[1:]:
    r = line.split('\t')
    if r[2]:
        idx[r[0]] = (datetime.strptime('2026-' + r[1], '%Y-%m-%d %H:%M'), (25 * float(r[7]) + 30 * float(r[11])) / 55)

pts = []
for lab, rs in STAGES:
    ests = [idx[r][1] for r in rs]
    when = min(idx[r][0] for r in rs)
    pts.append((when, sum(ests) / len(ests), min(ests), max(ests), lab, len(rs)))

fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.vlines([p[0] for p in pts], [p[2] for p in pts], [p[3] for p in pts], color=BLUE, alpha=0.25, lw=3, zorder=1)
ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=2, zorder=2)
ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
offs = [(10, -26), (-14, 14), (-6, -30), (6, 16), (-6, -34), (-12, 16)]
for (when, m, lo, hi, lab, n), (dx, dy) in zip(pts, offs):
    ax.annotate('%s\n%.1f' % (lab, m), (when, m), xytext=(dx, dy), textcoords='offset points', fontsize=8, color=INK,
                ha='left' if dx > 0 else 'right', va='bottom' if dy > 0 else 'top', linespacing=1.25)

ax.set_ylim(50, 75)
ax.set_xlim(datetime(2026, 10, 16, 0), datetime(2026, 10, 29, 12))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax.get_xticklabels(), fontsize=7.5)
plt.setp(ax.get_yticklabels(), fontsize=8)
ax.set_ylabel('Estimated score, 55 validation games', fontsize=8.5, color=INK)
ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(GRID)
ax.tick_params(color=GRID, labelcolor=MUTED)
fig.tight_layout()
fig.savefig(OUT, facecolor=SURFACE)
for p in pts:
    print('%s  mean %.1f  range %.1f-%.1f  n=%d  %s' % (p[0].strftime('%m-%d'), p[1], p[2], p[3], p[5], p[4].replace('\n', ' ')))
