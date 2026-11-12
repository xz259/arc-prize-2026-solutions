#!/bin/bash
# nll_panel_i.sh (2026-10-11, two-instance generation): nll_panel.sh (ddc70c44) against one instance: PORT and IDENTITY (the
# instance's server-identity.json) are required; the probe runs through probe_instance.py, which points minimal_inference's
# IDENTITY at the instance file before the probe imports it (PORT reaches it through ARC3_PORT).
# Cold NLL on the stage-0 panel against the running server (production chain step 1, probe_multimodal_likelihood_stage0.py)
# and comparison with the production fingerprint (v5.31 + mtp0x: 0.35065279765643587). MTP0 servers only.
# usage: nll_panel.sh <label>   (EXPECT overrides the fingerprint)
# Milestone 2 (2026-10-01 06:30): the probe pins the panel's `model` (path, config hash, quantization, shards) to the
# running server's. For another checkpoint (the winners' format), a derived copy of the stage-0 panel with only `model`
# replaced is scored (streams, token ids and images unchanged; the probe still checks that the server renders exactly
# the frozen token ids), without the stage-0 reference (its panel_sha256 cannot match; pair results with nll_compare.py).
set -u
X=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
L=${1:?label}; R=/workspace/arc3-inference; O0=$R/runs/stage0-reference-20260919-v1; PY=$R/venv-duck/bin/python
EXPECT=${EXPECT:-0.35065279765643587}; OUT=$X/nll-$L; PORT=${PORT:?port}; IDENTITY=${IDENTITY:?instance identity}
PANEL=$O0/nll-panel-cold/panel.json; REF="--reference $O0/nll-panel-cold/result.json"
DERIVED=$(python3 - "$PANEL" "$IDENTITY" "$X/nll-panels" <<'PYEOF'
import hashlib, json, os, sys
panel_path, identity_path, root = sys.argv[1:]
raw = open(panel_path, 'rb').read(); panel = json.loads(raw); model = json.load(open(identity_path))['model']
if panel['model'] == model:
    sys.exit(0)
key = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()[:16]
d = os.path.join(root, key); p = os.path.join(d, 'panel.json')
if not os.path.exists(p):
    os.makedirs(d, exist_ok=True)
    derived = dict(panel, model=model)
    assert derived['streams'] == panel['streams']
    open(p, 'wb').write((json.dumps(derived, indent=2) + '\n').encode())
    json.dump(dict(source=panel_path, source_sha256=hashlib.sha256(raw).hexdigest(), changed=['model'],
                   source_model=panel['model'], model=model), open(os.path.join(d, 'DERIVED.json'), 'w'), indent=1)
print(p)
PYEOF
)
[ -n "$DERIVED" ] && { PANEL=$DERIVED; REF=; }
echo "[$(date -u +%H:%M:%S)] cold NLL panel on $L, port $PORT (panel $PANEL${REF:+, stage-0 reference})"
cd $R && ARC3_ROOT=$R ARC3_PORT=$PORT PYTHONPATH=$R $PY $X/probe_instance.py $IDENTITY $O0/probe_multimodal_likelihood_stage0.py --output $OUT --panel $PANEL \
    $REF --cold-image-cache > $X/nll-$L.log 2>&1
python3 -c "
import json; d=json.load(open('$OUT/result.json')); v=d.get('mean_nll')
print('NLL panel $L', repr(v), 'FAILED (see nll-$L.log)' if v is None else 'IDENTICAL' if v==$EXPECT else 'DIFFERS by %+.3e' % (v-$EXPECT), 'repeat_max_diff', d.get('repeat_max_logprob_difference'))"
