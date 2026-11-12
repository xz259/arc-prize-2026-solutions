#!/bin/bash
# CPU test of sp_fullrun_ctl.sh (status, resume with DRY=1) and sp_fullrun_health.sh (ONCE=1) on a fake run in a temporary
# root (SP_ROOT): the real session list, a stopped trainer with a state sidecar whose stream state consumed every segment
# of the first 10 sessions, a driver log with those 10 ok and the 11th failed. Expected resume: 2,469 sessions to replay,
# 10 done, 1 failed and skipped; on 3 trainer GPUs the accumulation drops from 3 to 1. Nothing outside the temporary
# directory is written; no process is started except the scripts themselves.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); T=$(mktemp -d); PY=${PY:-python3}
SP=/mnt/shared/arc3-inference/sidepod_20261012; RUN=ctltest_$$
mkdir -p $T/mtp_ft/$RUN $T/mtp_ft/${RUN}_ctl $T/logs $T/jobs/valrbp50v
cp $SP/jobs/valrbp50v/train_sessions.json $T/jobs/valrbp50v/
ln -s $D $T/scripts
$PY - $T $RUN /workspace/arc3-inference/dflash_streams_20261011/sessions.jsonl <<'PYEOF'
import json, math, os, sys
T, run, streams = sys.argv[1:4]
sess = json.load(open(T + '/jobs/valrbp50v/train_sessions.json'))
n = {}
for l in open(streams):
    d = json.loads(l); n[d['session_id']] = d['n']
cons = ['%s_s%04d' % (s['session_id'], k) for s in sess[:10] for k in range(math.ceil(n[s['session_id']] / 65536))]
out = T + '/mtp_ft/' + run
open(out + '/state_a.pt', 'wb').write(b'x' * 1000)
json.dump(dict(step=300, name='state_a.pt', bytes=1000, stream=dict(consumed=cons, trained_w={})), open(out + '/state_a.pt.json', 'w'))
json.dump(dict(status='stopped', step=300, state=out + '/state_a.pt'), open(out + '/trainer_status.json', 'w'))
with open(T + '/logs/driver_mtpfull_%s.jsonl' % run, 'w') as f:
    for s in sess[:10]:
        f.write(json.dumps(dict(session_id=s['session_id'], status='ok', stream_tokens=n[s['session_id']])) + '\n')
    f.write(json.dumps(dict(session_id=sess[10]['session_id'], status='error: no recorded turn', stream_tokens=77)) + '\n')
env = dict(RUN=run, ATTEMPT=1, LR='5e-5', TRAIN='all', OPT='adamw', NK=4, SW='1,0.69,0.40,0.18', LOSS='served_kl', CLOSS='lk',
           ACCEPT='""', WINDOW=16384, ACCUM=3, NRANKS=1, TRAIN_GPUS=1, ROUTER='1.0', MOE='loop', CKPTSTEPS='--ckpt-steps',
           STEPS=21571, WARMUP=1078, DECAY=0.2, DECAY_START=-1, CKPT=300, KEEP=3000, EVAL=500, POOL=8, EXTRA='""',
           SESS=T + '/jobs/valrbp50v/train_sessions.json', REPLAY_LAB='mtpfull_' + run, RESUME='', TRAINER_ONLY='', STARTED='test')
txt = ''.join('%s=%s\n' % kv for kv in env.items())
open(T + '/mtp_ft/%s_ctl/latest.env' % run, 'w').write(txt); open(T + '/mtp_ft/%s_ctl/run_a1.env' % run, 'w').write(txt)
PYEOF
echo "== status"; SP_ROOT=$T LOC_ROOT=$T bash $D/sp_fullrun_ctl.sh status $RUN | sed 's/^/  /'
echo "== resume (DRY) on 3 GPUs"; OUT=$(SP_ROOT=$T LOC_ROOT=$T DRY=1 bash $D/sp_fullrun_ctl.sh resume $RUN 1,2,3); echo "$OUT" | sed 's/^/  /' | cut -c1-260
echo "$OUT" | grep -q "resume list: 2469 of 2480 sessions to replay" && echo "$OUT" | grep -q "ACCUM=1 TRAIN_GPUS=1,2,3" \
  && echo "$OUT" | grep -q "ATTEMPT=2 RESUME=$T/mtp_ft/$RUN/state_a.pt" && echo "$OUT" | grep -q "STEPS=21571 WARMUP=1078" || { echo "resume plan wrong"; exit 1; }
$PY -c "import json,sys; s=json.load(open(sys.argv[1])); assert s['failed_skipped']==1 and s['done']==10, s; print('  summary ok: done', s['done'], 'failed', s['failed_skipped'])" $T/mtp_ft/${RUN}_ctl/resume_a2.summary.json
echo "== resume (DRY) with DECAY_NOW and a router override"; OUT2=$(SP_ROOT=$T LOC_ROOT=$T DRY=1 DECAY_NOW=1 NEW_ROUTER=0.15 bash $D/sp_fullrun_ctl.sh resume $RUN 2>&1)
echo "$OUT2" | grep -E "override|decay now" | sed 's/^/  /'
echo "$OUT2" | grep -q "DECAY_START=301" && echo "$OUT2" | grep -q "STEPS=4614" && echo "$OUT2" | grep -q "ROUTER=0.15" || { echo "decay-now plan wrong"; exit 1; }
echo "== resume (DRY) with the experts' escape hatch, then the revert from a given state (10-13)"
OUT3=$(SP_ROOT=$T LOC_ROOT=$T DRY=1 NEW_UNFREEZE=1 NEW_EXPERT_WARMUP=300 bash $D/sp_fullrun_ctl.sh resume $RUN 2>&1)
echo "$OUT3" | grep -E "override|would keep" | sed 's/^/  /' | cut -c1-200
echo "$OUT3" | grep -q "would keep $T/mtp_ft/$RUN/state_a.pt as $T/mtp_ft/$RUN/state_frozen_step300.pt" \
  && echo "$OUT3" | grep -q "UNFREEZE=1 EXPERT_LR=0.5 EXPERT_WARMUP=300" || { echo "unfreeze plan wrong"; exit 1; }
OUT4=$(SP_ROOT=$T LOC_ROOT=$T DRY=1 FROM=$T/mtp_ft/$RUN/state_a.pt NEW_UNFREEZE=0 bash $D/sp_fullrun_ctl.sh resume $RUN 2>&1)
echo "$OUT4" | grep -q "from $T/mtp_ft/$RUN/state_a.pt (step 300)" && echo "$OUT4" | grep -q "EXPW= UNFREEZE=0" \
  && ! echo "$OUT4" | grep -q "would keep" || { echo "$OUT4"; echo "revert plan wrong"; exit 1; }
echo "  escape hatch: pre-unfreeze state kept, warmup override passed; revert: FROM state, unfreeze off"
echo "== resume (DRY) into the 4-GPU two-replay layout (10-13: 4 windows per step, 2 replays + 2 trainers)"
sed -i 's/^ACCUM=3$/ACCUM=4/' $T/mtp_ft/${RUN}_ctl/latest.env
OUT5=$(SP_ROOT=$T LOC_ROOT=$T DRY=1 NEW_REPLAY_GPUS=0,1 bash $D/sp_fullrun_ctl.sh resume $RUN 2,3 2>&1)
echo "$OUT5" | grep -E "override|attempt" | sed 's/^/  /' | cut -c1-200
echo "$OUT5" | grep -q "ACCUM=2 TRAIN_GPUS=2,3" && echo "$OUT5" | grep -q "REPLAY_GPUS=0,1" || { echo "$OUT5"; echo "two-replay plan wrong"; exit 1; }
echo "== health (ONCE), with two replay instances' driver logs"
mkdir -p $T/s2/client_mtp/mtpfull_${RUN}_i0 $T/s2/client_mtp/mtpfull_${RUN}_i1
printf '{"session_id": "a", "status": "ok"}\n{"session_id": "b", "status": "error: x"}\n' > $T/s2/client_mtp/mtpfull_${RUN}_i0/driver.jsonl
printf '{"session_id": "c", "status": "ok"}\n{"session_id": "d", "status": "ok"}\n' > $T/s2/client_mtp/mtpfull_${RUN}_i1/driver.jsonl
H=$(S2_ROOT=$T/s2 SP_ROOT=$T LOC_ROOT=$T RUN=$RUN ATTEMPT=1 ONCE=1 bash $D/sp_fullrun_health.sh | tail -n 1)
echo "$H" | cut -c1-300 | sed 's/^/  /'
echo "$H" | grep -q '"replay_ok": 3, "replay_failed": 1' && [ -s $T/logs/driver_mtpfull_${RUN}_i0.jsonl ] && [ -s $T/logs/driver_mtpfull_${RUN}_i1.jsonl ] \
  || { echo "health did not sum / mirror both instances"; exit 1; }
echo "== newest state skips an abandoned stretch; anneal from a milestone (FROM + DECAY_NOW, 10-14)"
$PY - $T/mtp_ft/$RUN <<'PYEOF'
import json, sys
out = sys.argv[1]
open(out + '/state_b.pt', 'wb').write(b'y' * 900)
json.dump(dict(step=900, name='state_b.pt', bytes=900, time='2026-10-13T10:00:00Z', stream=dict(consumed=[], trained_w={})),
          open(out + '/state_b.pt.json', 'w'))                                   # an abandoned stretch's higher-step state
d = json.load(open(out + '/state_a.pt.json')); d['time'] = '2026-10-13T13:00:00Z'; json.dump(d, open(out + '/state_a.pt.json', 'w'))
PYEOF
E=$T/mtp_ft/${RUN}_ctl/latest.env
sed -i "s/^STARTED=.*/STARTED=2026-10-13T12:00:00Z/" $E
N1=$(SP_ROOT=$T LOC_ROOT=$T bash $D/sp_fullrun_ctl.sh status $RUN | grep "newest state")
echo "  $N1"; echo "$N1" | grep -q "300 $T/mtp_ft/$RUN/state_a.pt" || { echo "newest state took the abandoned stretch"; exit 1; }
sed -i "s/^STARTED=.*/STARTED=2026-10-13T14:00:00Z/; s#^RESUME=.*#RESUME=$T/mtp_ft/$RUN/state_a.pt#" $E
N2=$(SP_ROOT=$T LOC_ROOT=$T bash $D/sp_fullrun_ctl.sh status $RUN | grep "newest state")
echo "  $N2 (nothing written yet: the state the attempt started from)"; echo "$N2" | grep -q "300 $T/mtp_ft/$RUN/state_a.pt" || { echo "no fallback"; exit 1; }
OUT6=$(SP_ROOT=$T LOC_ROOT=$T DRY=1 FROM=$T/mtp_ft/$RUN/state_b.pt DECAY_NOW=1 COOLDOWN=500 bash $D/sp_fullrun_ctl.sh resume $RUN 2>&1)
echo "$OUT6" | grep -E "from |decay now" | sed 's/^/  /' | cut -c1-160
echo "$OUT6" | grep -q "DECAY_START=901" && echo "$OUT6" | grep -q "STEPS=1400" || { echo "$OUT6"; echo "anneal-from-milestone plan wrong"; exit 1; }
echo "ctl test ok; scratch left in $T"
