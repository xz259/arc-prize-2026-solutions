#!/bin/bash
# sp_fullrun_check.sh (2026-10-13, Claude fork 5d20b23b; side pod only): one poll for the local full-run watcher
# (watch_fullrun.sh). Prints the newest health line ("HEALTH {...}") and "FAIL <what>" for each failure signature
# (Phase 6 supervision guide, thresholds from the measured step time):
#   traceback / out of memory / NaN in the trainer's local log; trainer status not running while no "stopped" or end;
#   no trainer process while the status says running (grace 20 min after attempt 1's start, 10 min after a restart);
#   no step change for STALL_MIN minutes once the attempt has logged a step (default 10; user 10-13: catch failures
#   within ~15 min; a 4-window step is ~9 s, ~18 s beside a milestone branch, and the log flushes every 10 steps), and no
#   step at all 30 min after the start; the replay's driver or server process missing while its DONE file is absent;
#   no new state on the volume for STATE_MIN minutes (default 90; CKPT 225 steps is ~35 min on one trainer);
#   cgroup oom_kill above its value at the attempt's start; /dev/shm features > 160 GB; anon + shmem > 320 GiB;
#   container disk free < 20 GB; health line older than 5 minutes; replay failures above 2% of the processed sessions.
# Read-only. usage: bash sp_fullrun_check.sh RUN
set -u
RUN=${1:?run}
SP=${SP_ROOT:-/mnt/shared/arc3-inference/sidepod_20261012}          # SP_ROOT: tests only
CTL=$SP/mtp_ft/${RUN}_ctl; LOC=${LOC_ROOT:-/workspace}/mtpfull_state_$RUN
. $CTL/latest.env
H=$(tail -n 1 $LOC/health.jsonl 2>/dev/null)
echo "HEALTH $H"
grep -E "Traceback|OutOfMemoryError|CUDA out of memory|nan loss|loss.: NaN|STATE COPY FAILED|state copies failed" \
  $LOC/trainer_a$ATTEMPT.log 2>/dev/null | head -n 3 | sed 's/^/FAIL trainer log: /' | cut -c1-200
python3 - "$H" $LOC/health.jsonl "$STARTED" ${STALL_MIN:-10} ${STATE_MIN:-90} "${TRAIN:-all}" "$ATTEMPT" \
  /dev/shm/mtpfull_$RUN/features/DONE_$REPLAY_LAB <<'PYEOF'
import calendar, json, os, sys, time
line, path, started, stall_min, state_min = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
router_trained = sys.argv[6] in ('all', 'noexperts', 'experts')     # nonexpert: the router is frozen
grace = 1200 if sys.argv[7] == '1' else 600                         # attempt 1 scores the baseline first
replay_done = os.path.exists(sys.argv[8])
ts = lambda s: calendar.timegm(time.strptime(s, '%Y-%m-%dT%H:%M:%SZ'))
now = time.time()
try:
    h = json.loads(line)
except ValueError:
    print('FAIL no health line'); sys.exit()
rows = []
for l in open(path):
    try:
        rows.append(json.loads(l))
    except ValueError:
        pass
age = now - ts(h['t'])
if age > 300:
    print('FAIL health line %.0f min old' % (age / 60))
since_start = now - ts(started) if started[:2] == '20' else 1e9
st = h.get('status')
if st in ('stopped', 'finished', 'stream_ended'):
    print('EVENT trainer %s at step %s' % (st, h.get('step')))
elif since_start > grace and 'trainer:' not in (h.get('procs') or ''):
    print('FAIL no trainer process (status %s)' % st)
t0 = ts(started) if started[:2] == '20' else 0
recent = [r for r in rows if r.get('step') is not None and ts(r['t']) >= t0]      # this attempt's rows
moved = len({r['step'] for r in recent}) > 1
if st == 'running' and moved:
    last_step = recent[-1]['step']
    first_same = next((r for r in reversed(recent) if r['step'] != last_step), None)
    t_change = ts(first_same['t']) if first_same else ts(recent[0]['t'])
    if now - t_change > stall_min * 60:
        print('FAIL no step change for %.0f min (step %s, stream wait %s s)' % ((now - t_change) / 60, last_step, h.get('wait_s')))
elif st == 'running' and since_start > 1800:
    print('FAIL no training step %.0f min after the start' % (since_start / 60))
procs = h.get('procs') or ''
if st == 'running' and not replay_done and since_start > grace:   # the replay feeds the trainer until its DONE file
    for kind in ('driver', 'server'):
        if kind + ':' not in procs:
            print('FAIL replay %s not running (no DONE file yet)' % kind)
sv = h.get('state_volume')
sv_rows = [r for r in rows if r.get('state_volume')]
if st == 'running' and since_start > 3600:
    if sv_rows:
        first = next((r for r in sv_rows if r['state_volume'] == sv), None)
        if first and now - ts(first['t']) > state_min * 60:
            print('FAIL no new state on the volume for %.0f min (newest %s)' % ((now - ts(first['t'])) / 60, sv))
    elif since_start > state_min * 60 + 3600:
        print('FAIL no state on the volume %.0f min after the start' % (since_start / 60))
oom0 = next((r.get('oom_kill') for r in rows if r.get('oom_kill') is not None), None)
if oom0 is not None and h.get('oom_kill', oom0) > oom0:
    print('FAIL cgroup oom_kill %s -> %s' % (oom0, h['oom_kill']))
if (h.get('shm_gb') or 0) > 160:
    print('FAIL /dev/shm features %.0f GB' % h['shm_gb'])
if (h.get('anon_gib') or 0) + (h.get('shmem_gib') or 0) > 320:
    print('FAIL unreclaimable memory %.0f GiB' % (h['anon_gib'] + h['shmem_gib']))
if (h.get('disk_free_gb') or 1e9) < 20:
    print('FAIL container disk free %.0f GB' % h['disk_free_gb'])
e = [r['load_entropy'] for r in rows if r.get('load_entropy') is not None]           # router guard (IE 10-13)
if len(e) > 10 and e[-1] < 0.9 * sum(e[:5]) / 5:
    print('EVENT router: load entropy %.3f, > 10%% below the start (%.3f): halve the router LR at the next resume' % (e[-1], sum(e[:5]) / 5))
if router_trained and (h.get('load_dead') or 0) > 0:      # a frozen router's rare experts are the production tail, not a collapse
    print('EVENT router: %d experts got no tokens in a recent step' % h['load_dead'])
ok, bad = h.get('replay_ok') or 0, h.get('replay_failed') or 0
if ok + bad >= 50 and bad > 0.02 * (ok + bad):
    print('FAIL replay failures %d of %d processed sessions' % (bad, ok + bad))
PYEOF
