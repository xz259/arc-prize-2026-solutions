"""compare_tpident.py DIR (2026-10-31, ARC2 - main): joins l4port_tpident.py's records of the TP ranks of each pipeline stage by
(call, kind, seq) and counts hash mismatches per kind. Prints 'IDENTICAL' or 'DIFFER: ...' first, then the counts. Numbers only."""
import glob
import json
import os
import sys

recs = {}
for f in glob.glob(os.path.join(sys.argv[1], "tpident_pp*_tp*_*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        recs.setdefault((r["pp"], r["call"], r["kind"].split(":")[0] if r["kind"].startswith("pp_out") else r["kind"],
                         r["kind"], r.get("seq", 0)), {})[r["tp"]] = (tuple(r["shape"]), tuple(r["h"]))
stats = {}
for (pp, call, family, kind, seq), by_tp in recs.items():
    s = stats.setdefault("pp%d %s" % (pp, family), dict(compared=0, mismatched=0, unpaired=0, first_mismatch_call=None))
    if len(by_tp) < 2:
        s["unpaired"] += 1
        continue
    s["compared"] += 1
    if len(set(by_tp.values())) > 1:
        s["mismatched"] += 1
        if s["first_mismatch_call"] is None or call < s["first_mismatch_call"]:
            s["first_mismatch_call"] = call
bad = {k: v["mismatched"] for k, v in stats.items() if v["mismatched"]}
print("IDENTICAL" if stats and not bad else ("DIFFER: %s" % bad if stats else "NO RECORDS"))
print(json.dumps(stats, sort_keys=True))
