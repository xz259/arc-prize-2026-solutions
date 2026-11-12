#!/bin/bash
# sp_fullrun_replay.sh (2026-10-22, MTP refit port by Claude session 099768dc; from sidepod_20261012/scripts/sp_fullrun_replay.sh 479939ce):
# the full run's replay side on the main pod 2a159825c76b: fin73/s2 at ctx78 arm b's serving config (19 + 2 x 120,556, age cap 160K; the
# ctx78 numerics) with dflashtaps (MTP option) + replayfx, launched by /workspace/kvring_dev/mtp73/mtp_replay_start.sh MODE=train
# (mtp_replay_ps.sh -> lc_fin.sh, the streaming arm p6_replay_arm_i.sh) on GPU 0, port 1377, in the foreground. Segments of the MTP's 4
# pre-final-mixer streams land in /dev/shm/mtpfull_<RUN>/features (start gate SHM_GB, default 40). Afterwards, as fr1: the driver summary;
# DONE_<label> once every session was processed and at most 2% failed (failed sessions listed in logs/replay_failed_<label>.json); the
# launcher has already stopped its server. Single replay only (the 2-GPU layout: replay GPU 0, trainer GPU 1).
set -u
[ "$(hostname)" = 2a159825c76b ] || { echo "not the main pod"; exit 9; }
S2=/workspace/kvring_dev/fin73/s2
SP=/mnt/shared/arc3-inference/mtp_refit_20261022
K=/workspace/kvring_dev/mtp73
RUN=${RUN:?run}
SHM=/dev/shm/mtpfull_$RUN
FEAT=$SHM/features
SESS=${SESS:-$SP/jobs/valnext55/train_sessions.json}
LAB=mtpfull_$RUN${ATTEMPT:+_r$ATTEMPT}
PROG=$SHM/progress; WINF=$SHM/windows.json
[ -n "${INST:-}" ] && { echo "the refit runs one replay (no INST layout)"; exit 1; }
cd $S2 || exit 1
[ "$(df --output=fstype . | tail -n 1 | tr -d ' ')" = overlay ] || { echo "not the private stage-2 copy"; exit 1; }
grep -q "ARC3_S2_TAPS_MTP" stage2_v1/dflashtaps.py || { echo "dflashtaps without the MTP option"; exit 1; }
grep -q "shm-limit-gb" replay_driver.py || { echo "no streaming replay driver"; exit 1; }
[ -e client_mtp/$LAB ] && { echo "exists: client_mtp/$LAB"; exit 1; }
[ -s $SESS ] || { echo "no session list $SESS"; exit 1; }
md5sum stage2_v1/dflashtaps.py stage2_v1/replayfx.py replay_driver.py lc_fin.sh $K/mtp_replay_ps.sh $K/mtp_replay_start.sh $K/p6_replay_arm_i.sh | cut -c1-8,33-
echo "[$(date -u +%T)] full-run replay $RUN (label $LAB): $(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" $SESS) sessions -> $FEAT, concurrency ${CONC:-4}, start gate ${SHM_GB:-40} GB"
if [ -n "${DRY:-}" ]; then
  DRYRUN=1 MODE=train PROG=$PROG WINF=$WINF SHM_GB=${SHM_GB:-40} bash $K/mtp_replay_start.sh $LAB 0 1377 $SESS $FEAT ${SECS:-604800} ${CONC:-4} 2>&1 | tail -n 30
  exit 0
fi
mkdir -p $FEAT
[ -s $PROG ] || echo 0 > $PROG
[ -s $WINF ] || echo '{"chunk_first_step": []}' > $WINF
export MTP_RUN=$RUN
MODE=train PROG=$PROG WINF=$WINF SHM_GB=${SHM_GB:-40} bash $K/mtp_replay_start.sh $LAB 0 1377 $SESS $FEAT ${SECS:-604800} ${CONC:-4}
RC=$?
echo "[$(date -u +%T)] mtp_replay_start rc $RC; $(grep -E ' DONE |lc_run3i rc' /workspace/kvring_dev/fin73/runs/$LAB/status.txt 2>/dev/null | tail -n 2 | tr '\n' ' ')"
[ -s client_mtp/$LAB/driver.jsonl ] || { echo "[$(date -u +%T)] no driver output: no DONE_$LAB"; exit 1; }
ALLOK=$(python3 - client_mtp/$LAB/driver.jsonl $SESS <<'PYEOF'
import collections, json, sys
c = collections.Counter(); tok = 0; ok = set()
for line in open(sys.argv[1]):
    d = json.loads(line)
    if 'status' in d and 'session_id' in d:
        c[d['status'].split(':')[0][:40]] += 1; tok += d.get('stream_tokens') or 0
        if d['status'].startswith('ok'):
            ok.add(d['session_id'])
want = {s['session_id'] for s in json.load(open(sys.argv[2]))}
seen = set()
for line in open(sys.argv[1]):
    d = json.loads(line)
    if 'status' in d and 'session_id' in d:
        seen.add(d['session_id'])
bad = len((want & seen) - ok)
print(dict(c), 'stream tokens', tok, 'ok', len(ok & want), 'of', len(want), 'not ok', bad, 'unprocessed', len(want - seen), file=sys.stderr)
print(1 if want <= seen and bad <= 0.02 * len(want) else 0)       # all processed, at most 2% failed (e.g. no recorded turn)
PYEOF
)
mkdir -p $SP/logs
cp client_mtp/$LAB/driver.jsonl $SP/logs/driver_$LAB.jsonl.tmp && mv $SP/logs/driver_$LAB.jsonl.tmp $SP/logs/driver_$LAB.jsonl
python3 - client_mtp/$LAB/driver.jsonl $SESS $SP/logs/replay_failed_$LAB.json <<'PYEOF'
import json, sys
st, tok = {}, {}
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if 'status' in d and 'session_id' in d:
        st[d['session_id']] = d['status']; tok[d['session_id']] = d.get('stream_tokens') or 0
want = [s['session_id'] for s in json.load(open(sys.argv[2]))]
bad = [dict(session_id=s, status=st[s], stream_tokens=tok[s]) for s in want if s in st and not st[s].startswith('ok')]
json.dump(dict(label=sys.argv[3], failed=bad, failed_tokens=sum(b['stream_tokens'] for b in bad), sessions=len(want),
               unprocessed=sum(1 for s in want if s not in st)), open(sys.argv[3], 'w'), indent=1)
print('NOTE: %d of %d sessions failed and are skipped (user rule 10-13), %d stream tokens recorded for them -> %s' % (
    len(bad), len(want), sum(b['stream_tokens'] for b in bad), sys.argv[3]))
PYEOF
if [ "$ALLOK" = 1 ]; then echo done > $FEAT/DONE_$LAB; echo "[$(date -u +%T)] all sessions processed: DONE_$LAB written"; else echo "[$(date -u +%T)] sessions unprocessed or > 2% failed: no DONE_$LAB (resume the replay with the remaining sessions)"; fi
echo "[$(date -u +%T)] replay side ended"
