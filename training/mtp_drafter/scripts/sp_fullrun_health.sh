#!/bin/bash
# 2026-10-22 MTP refit port (Claude session 099768dc; from 16d6d4b0): host 2a159825c76b, SP = mtp_refit_20261022, S2 = fin73/s2.
# sp_fullrun_health.sh (2026-10-13, Claude fork 5d20b23b; side pod only): the MTP full run's on-pod health log (Phase 6
# supervision guide: works without the Mac or the network). Once a minute one JSON line to the container disk
# (/workspace/mtpfull_state_<RUN>/health.jsonl) and, best effort, to the volume (mtp_ft/<RUN>_ctl/health.jsonl): trainer
# step / loss / lr, trainer status, newest state sidecars (volume and local), stream directory size and segment count,
# the cgroup's unreclaimable memory (anon + shmem) and oom_kill count, container-disk free space, GPU memory / use, the
# replay driver's processed / failed sessions (both instances' in the two-replay layout), and the run's live processes (by MTP_RUN, never by a name pattern).
# Every 5 minutes it mirrors the trainer's local log and the replay's driver log to the volume. Read-only otherwise;
# exits 10 minutes after the trainer reports stopped / finished / stream_ended with no trainer process left.
# usage: RUN=fr1 ATTEMPT=1 bash start_detached.sh logs/fullrun_health_fr1_a1.log sp_fullrun_health.sh
set -u
[ "$(hostname)" = 2a159825c76b ] || { echo "not the main pod"; exit 9; }
SP=${SP_ROOT:-/mnt/shared/arc3-inference/mtp_refit_20261022}          # SP_ROOT: tests only
S2=${S2_ROOT:-/workspace/kvring_dev/fin73/s2}   # S2_ROOT: tests only
RUN=${RUN:?run}; ATTEMPT=${ATTEMPT:-1}
OUT=$SP/mtp_ft/$RUN; CTL=$SP/mtp_ft/${RUN}_ctl; LOC=${LOC_ROOT:-/workspace}/mtpfull_state_$RUN; FEAT=/dev/shm/mtpfull_$RUN/features
. $CTL/run_a$ATTEMPT.env
mkdir -p $LOC
LAB=$REPLAY_LAB
idle=0; m=0
while true; do
  m=$((m + 1))
  PROCS=$(for p in /proc/[0-9]*; do
    pid=${p#/proc/}; [ "$pid" = $$ ] && continue
    tr '\0' '\n' 2>/dev/null < $p/environ | grep -qx "MTP_RUN=$RUN" || continue
    c=$(tr '\0' ' ' 2>/dev/null < $p/cmdline)
    case "$c" in
      (*train_mtp_sp.py*) echo trainer;; (*replay_driver*) echo driver;; (*vllm.entrypoints*) echo server;;
      (*sp_fullrun_health*) ;; (*) echo other;;
    esac
  done | sort | uniq -c | awk '{printf "%s%s:%s", (NR>1?",":""), $2, $1}')
  LINE=$(python3 - "$OUT" "$LOC" "$FEAT" "$S2/client_mtp/$LAB" "$PROCS" "$ATTEMPT" <<'PYEOF'
import glob, json, os, subprocess, sys, time
out, loc, feat, drv, procs, att = sys.argv[1:7]
r = dict(t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), attempt=int(att), procs=procs)
try:
    with open(os.path.join(out, 'log.jsonl'), 'rb') as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 20000))
        rows = [json.loads(l) for l in f.read().decode(errors='ignore').splitlines()[1:] if '"loss"' in l]
    if rows:
        x = rows[-1]; r.update(step=x['step'], loss=round(x['loss'], 4), lr=x['lr'], sec=x['sec'],
                               wait_s=x.get('stream_wait_s'), consumed=x.get('stream_consumed'),
                               load_entropy=x.get('load_entropy'), load_max=x.get('load_max'),
                               load_dead=max((y.get('load_dead') or 0) for y in rows[-10:]))
except Exception as e:
    r['log_error'] = str(e)[:80]
try:
    r['status'] = json.load(open(os.path.join(out, 'trainer_status.json'))).get('status')
except Exception:
    r['status'] = None
def newest(pattern):
    best = None
    for p in glob.glob(pattern):
        try:
            d = json.load(open(p))
            if best is None or d['step'] > best[0]:
                best = (d['step'], os.path.basename(p)[:-5])
        except Exception:
            pass
    return best
r['state_volume'] = newest(os.path.join(out, 'state_*.pt.json'))
r['state_local'] = newest(os.path.join(loc, 'state_scratch.pt.json'))
try:
    segs = [n for n in os.listdir(feat) if n.endswith('.safetensors')]
    r['shm_segments'] = len(segs)
    r['shm_gb'] = round(sum(os.path.getsize(os.path.join(feat, n)) for n in segs) / 1e9, 1)
except Exception:
    r['shm_segments'] = None
try:
    st = dict(l.split() for l in open('/sys/fs/cgroup/memory.stat'))
    r['anon_gib'] = round(int(st['anon']) / 2**30, 1); r['shmem_gib'] = round(int(st['shmem']) / 2**30, 1)
    r['file_gib'] = round(int(st['file']) / 2**30, 1)
    ev = dict(l.split() for l in open('/sys/fs/cgroup/memory.events'))
    r['oom_kill'] = int(ev.get('oom_kill', 0))
except Exception:
    pass
s = os.statvfs('/workspace'); r['disk_free_gb'] = round(s.f_bavail * s.f_frsize / 1e9, 1)
try:
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=30).stdout
    r['gpu'] = ';'.join(l.replace(' ', '') for l in q.strip().splitlines())
except Exception:
    pass
try:                                   # the replay's driver log, or both instances' (two-replay layout: <label>_i<N>)
    ok = bad = 0; logs = [p for p in [drv + '/driver.jsonl'] + sorted(glob.glob(drv + '_i[0-9]/driver.jsonl')) if os.path.exists(p)]
    for p in logs:
        for l in open(p):
            try:
                d = json.loads(l)
            except ValueError:
                continue
            if 'status' in d and 'session_id' in d:
                ok += d['status'].startswith('ok'); bad += not d['status'].startswith('ok')
    if logs:
        r['replay_ok'], r['replay_failed'] = ok, bad
except Exception:
    pass
print(json.dumps(r))
PYEOF
)
  echo "$LINE" >> $LOC/health.jsonl
  echo "$LINE" >> $CTL/health.jsonl 2>/dev/null || true
  if [ $((m % 5)) = 1 ]; then
    [ -s $LOC/trainer_a$ATTEMPT.log ] && cp $LOC/trainer_a$ATTEMPT.log $SP/logs/mtp_ft_${RUN}_trainer_a$ATTEMPT.log.tmp 2>/dev/null \
      && mv $SP/logs/mtp_ft_${RUN}_trainer_a$ATTEMPT.log.tmp $SP/logs/mtp_ft_${RUN}_trainer_a$ATTEMPT.log 2>/dev/null
    for dd in $S2/client_mtp/$LAB $S2/client_mtp/${LAB}_i[0-9]; do     # one replay, or each instance
      L1=$(basename $dd)
      [ -s $dd/driver.jsonl ] && cp $dd/driver.jsonl $SP/logs/driver_$L1.jsonl.tmp 2>/dev/null \
        && mv $SP/logs/driver_$L1.jsonl.tmp $SP/logs/driver_$L1.jsonl 2>/dev/null
    done
  fi
  case "$LINE" in
    *'"status": "stopped"'*|*'"status": "finished"'*|*'"status": "stream_ended"'*)
      case "$LINE" in *trainer:*) idle=0;; *) idle=$((idle + 1));; esac;;
    *) idle=0;;
  esac
  [ -n "${ONCE:-}" ] && { echo "$LINE"; exit 0; }            # ONCE=1: one line (test)
  if [ -e $CTL/run_a$((ATTEMPT + 1)).env ]; then              # a later attempt has its own logger (10-13)
    echo "[$(date -u +%T)] attempt $((ATTEMPT + 1)) started; health logger of attempt $ATTEMPT exits"; exit 0
  fi
  if [ $idle -ge 10 ]; then
    cp $LOC/trainer_a$ATTEMPT.log $SP/logs/mtp_ft_${RUN}_trainer_a$ATTEMPT.log 2>/dev/null
    echo "[$(date -u +%T)] trainer ended (status line: $LINE); health logger exits"; exit 0
  fi
  sleep 60
done
