"""kaggle_run.py CONFIG.json (2026-10-24, Claude main session 099768dc; the Kaggle port, KAGGLE_PORT_PLAN_2026-10-24.md section 2): the
guard-free launcher of the ARC3 stack, for the Kaggle notebook and its pod rehearsal. Standard library only (it runs under the notebook's
Python). The user 10-24: "It should run without any guards because a block fails the submission and we get zero"; games not started in
the first ~15 min seem to time out. So:
  - the server starts as the pod starts it (k_runtime_python.sh -> k_launch_scaling_s2i.py: launch_scaling_s2i.py's composition with its
    checks logging) from the environment in KDEF.json (mk_kdef.py: the p77 pair's live server and client, every arm difference explained
    by a profile knob, the 80K profile checked against p76's 80K launch), rendered for the chosen profile, label, GPU and port;
  - the client (run_v6, the client venv) starts at the notebook start + client_start_s or as soon as the server answers, whichever is
    first, with the v7.9 Kaggle switches: no blocking model query, the transport waits for the server's first /health, one warmup RESET,
    the game source and order, FIFO or restart_value;
  - every pod stop gate (pool short, < 512 MiB free, preemptions, engine errors, refusals, the frozen and receipt checks) is a log line;
    a status line every status_every_s;
  - the clock runs from the notebook start: gameplay ends at notebook_start + budget_s - teardown_margin_s; then the client's own
    teardown, SIGTERM / SIGKILL if it overruns, the server's stop, and kaggle_run.json in the run directory.
Config (JSON; every key optional): label, profile (KDEF's tiers: 80k lutc, 81kg4 g4, 82kp16 lutc + prune16, 81kg4p16), gpu, port, kdef, notebook_start (epoch), budget_s,
teardown_margin_s, client_start_s, game_source (offline | gateway), games (a list, '@all' or '@<file>'), environments_dir,
use_restart_value, rv_constants ('p=..,stuck=..'), shuffle, shuffle_seed, server_wait_s, engine_audit, relaunch_max, status_every_s,
secs_override (gameplay seconds), trace_dir + filetrace_py (the file tracer: every server and client process records the files it reads),
arc_wheels_dir (a directory first on the client's PYTHONPATH: the competition's arc_agi when it differs), extra_server_env,
extra_client_env, drop_server_env / drop_client_env (variables removed after the render), diagnostics (default true; false = the
competition rerun's set: no step timer, canary, PLE / fqfast / schedfast verifies, engine and service audits), keepalive_s (default
420: gateway runs only, one keep-alive RESET per period while the server loads; v7.9b K5), kaggle_client (default true: the v7.9 switches), memstats_triggers (default true: as lc_fin.sh), dry_run (print the
server environment and the client command against a capture, dry_capture = cap_env.py's JSON of the same label / profile, and stop),
nll_panel (default false: after the server is healthy and before the client starts, nll_panel_i.sh's cold NLL panel against
nll_expect, logged only; skipped once the client runs)."""
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

S2 = '/workspace/kvring_dev/fin74/s2'
ROOT = '/workspace/arc3-inference'
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = dict(label='k80a01', profile='80k', gpu='0', port=1377, kdef=os.path.join(HERE, 'kaggle_defs', 'KDEF.json'), notebook_start=None,
           budget_s=32400, teardown_margin_s=900, client_start_s=600, game_source='offline', games='@all',
           environments_dir=ROOT + '/pub25cc30/environment_files', use_restart_value=False, rv_constants='', shuffle=False, shuffle_seed=0,
           server_wait_s=7200, engine_audit=True, relaunch_max=1, status_every_s=600, secs_override=None, trace_dir='', filetrace_py='',
           arc_wheels_dir='', extra_server_env={}, extra_client_env={}, kaggle_client=True, memstats_triggers=True, dry_run=False,
           dry_capture='', nll_panel=False, nll_expect='0.3422915321304067', drop_server_env=[], drop_client_env=[],
           diagnostics=True, keepalive_s=420)
BASE_KEYS = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TERM', 'USER', 'LOGNAME', 'SHELL', 'TMPDIR', 'TZ', 'HOSTNAME')
T_LOG = [None]


def log(msg):
    line = '[%s kaggle_run] %s' % (time.strftime('%H:%M:%S', time.gmtime()), msg)
    print(line, flush=True)
    if T_LOG[0]:
        try:
            with open(T_LOG[0], 'a') as fh:
                fh.write(line + '\n')
        except OSError:
            pass


def get(url, timeout=5):
    try:
        with urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as exc:  # noqa: BLE001
        return None, repr(exc)[:200]


def render(D, part, prof, label, gpu, port):
    vals = dict(prof, GPU=str(gpu), PORT=str(port))
    out = {}
    for k, v in D['env'][part].items():
        kn = D['knobs'][part].get(k)
        out[k] = str(vals[kn]) if kn not in (None, 'LABEL') else v.replace(D['label_token'], label)
    return out


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w') as fh:
        fh.write(text)
    os.replace(path + '.tmp', path)


def alive(pid):
    try:
        st = open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()[0]
        return st != 'Z'
    except (OSError, IndexError):
        return False


def session_pids(sid):
    out = []
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                f = open('/proc/%s/stat' % p).read().rsplit(')', 1)[1].split()
                if int(f[3]) == sid and f[0] != 'Z':
                    out.append(int(p))
            except (OSError, IndexError, ValueError):
                pass
    return sorted(out)


def tree_cpu_s(sid):
    """CPU seconds (utime + stime + the reaped children's) of the processes in session sid (IE's CPU review, M2)."""
    tot = 0
    for pid in session_pids(sid):
        try:
            f = open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()
            tot += sum(int(x) for x in f[11:15])
        except (OSError, ValueError, IndexError):
            pass
    return tot / float(os.sysconf('SC_CLK_TCK'))


def cpu_pressure_avg60():
    try:
        for l in open('/proc/pressure/cpu'):
            if l.startswith('some'):
                return float(l.split('avg60=')[1].split()[0])
    except (OSError, IndexError, ValueError):
        pass
    return None


def maps_snapshot(pids, d, tag):
    os.makedirs(d, exist_ok=True)
    n = 0
    for pid in pids:
        try:
            paths = sorted({l.split(None, 5)[5].strip() for l in open('/proc/%d/maps' % pid) if len(l.split(None, 5)) == 6
                            and l.split(None, 5)[5].startswith('/')})
            write(os.path.join(d, 'maps-%s-%d.txt' % (tag, pid)), '\n'.join(paths) + '\n')
            n += 1
        except OSError:
            pass
    return n


def metrics(port):
    st, body = get('http://127.0.0.1:%d/metrics' % port)
    if st != 200:
        return {}
    m = {}
    for line in body.decode('utf-8', 'replace').splitlines():
        if line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if name in ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:num_preemptions_total', 'vllm:generation_tokens_total',
                    'vllm:prompt_tokens_total', 'vllm:kv_cache_usage_perc'):
            try:
                m[name] = m.get(name, 0.0) + float(line.rsplit(' ', 1)[1])
            except ValueError:
                pass
        if 'num_gpu_blocks="' in line and 'blocks' not in m:
            try:
                m['blocks'] = int(line.split('num_gpu_blocks="', 1)[1].split('"', 1)[0])
            except ValueError:
                pass
    return m


def gpu_line(gpu):
    try:
        out = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        used, tot = [int(x) for x in out.split(',')]
        return used, tot
    except Exception:  # noqa: BLE001
        return None, None


def mem_avail_gb():
    try:
        for l in open('/proc/meminfo'):
            if l.startswith('MemAvailable:'):
                return round(int(l.split()[1]) / 1048576, 1)
    except OSError:
        pass
    return None


def disk_free_gb(p):
    try:
        st = os.statvfs(p)
        return round(st.f_bavail * st.f_frsize / 1e9, 1)
    except OSError:
        return None


class Levels:
    """Counts '"level_completed": true' in the client's per-game action logs, reading only new bytes."""

    def __init__(self, root):
        self.root, self.off, self.n = root, {}, 0

    def count(self):
        for f in glob.glob(os.path.join(self.root, 'prolong', 'game-*', 'log.jsonl')):
            try:
                with open(f, 'rb') as fh:
                    fh.seek(self.off.get(f, 0))
                    b = fh.read()
                    self.off[f] = fh.tell()
                self.n += b.count(b'"level_completed": true')
            except OSError:
                pass
        return self.n


class ErrLines:
    def __init__(self):
        self.off, self.n = 0, 0

    def count(self, path):
        try:
            with open(path, 'rb') as fh:
                fh.seek(self.off)
                b = fh.read()
                self.off = fh.tell()
            self.n += sum(b.count(x) for x in (b'Traceback', b'EngineDeadError', b'CUDA out of memory'))
        except OSError:
            pass
        return self.n


def dry(cfg, D, senv_def, cenv_def, prof, label, port):
    """dry_run: the rendered server / client environments against a live capture (cap_env.py) of the same label and profile."""
    cap = json.load(open(cfg['dry_capture']))
    for part, mine in (('server_serve', senv_def), ('client_run_v6', cenv_def)):
        theirs = cap[part]
        diff = sorted(k for k in mine if theirs.get(k) != mine[k])
        notmine = sorted(k for k in theirs if k not in mine)
        print('%s: %d rendered keys, %d differ from the capture %s; capture-only keys (dropped by the definition): %s' % (
            part, len(mine), len(diff), [(k, mine[k][:60], str(theirs.get(k))[:60]) for k in diff[:6]], notmine))
    argv = [ROOT + '/venv-duck/bin/python', '-m', 'v62.run_v6', '--games', '<games>', '--seconds', '<secs>', '--output',
            os.path.join(S2, 'client_mtp', label), '--base-url', 'http://127.0.0.1:%d' % port, '--context', str(prof['CTX']),
            '--basis', str(prof['BASIS']), '--protect', str(prof['TAIL']), '--concurrency', '0', '--label', label, '--environments-dir',
            cfg['environments_dir']]
    ca = list(cap['client_argv'])
    for i, a in enumerate(ca):
        if i > 0 and ca[i - 1] in ('--games', '--seconds'):
            ca[i] = '<games>' if ca[i - 1] == '--games' else '<secs>'
    print('client argv equal to the capture (games / seconds aside): %s; client cwd %s (capture %s, definition %s)' % (
        argv == ca, S2 + '/harness_v6', cap.get('client_cwd'), D.get('client_cwd')))
    if argv != ca:
        print('  mine  ', argv)
        print('  theirs', ca)


def main():
    cfg = dict(CFG)
    if len(sys.argv) > 1:
        cfg.update(json.load(open(sys.argv[1])))
    t_nb = float(cfg['notebook_start'] or time.time())
    G = t_nb + cfg['budget_s'] - cfg['teardown_margin_s']           # gameplay ends here (the client's deadline)
    label, gpu, port = cfg['label'], str(cfg['gpu']), int(cfg['port'])
    D = json.load(open(cfg['kdef']))
    prof = D['profiles'][cfg['profile']]
    senv_def = render(D, 'server_serve', prof, label, gpu, port)
    cenv_def = render(D, 'client_run_v6', prof, label, gpu, port)
    RUN = senv_def['FIN_RUN']
    if cfg['dry_run']:
        return dry(cfg, D, senv_def, cenv_def, prof, label, port)
    os.makedirs(RUN, exist_ok=True)
    T_LOG[0] = os.path.join(RUN, 'kaggle_run.log')
    rec = dict(config=cfg, profile=prof, notebook_start=t_nb, gameplay_end=G, events={})
    ev = lambda k: rec['events'].setdefault(k, round(time.time() - t_nb, 1))   # noqa: E731
    log('label %s, profile %s (context %s x %s sessions, pool lanes %s, KV %s GiB), GPU %s, port %d; notebook start %s, gameplay end %s '
        '(%.0f s from now)' % (label, cfg['profile'], prof['CTX'], prof['NL'], prof['NP'], prof['KVG'], gpu, port,
                               time.strftime('%H:%M:%S', time.gmtime(t_nb)), time.strftime('%H:%M:%S', time.gmtime(G)), G - time.time()))
    # the run directory's side files (p77_ps.sh and lc_fin.sh write them before the launch)
    for d in ('steptimer', 'torch', 'memstats', 'canary'):
        os.makedirs(os.path.join(RUN, d), exist_ok=True)
    write(senv_def['ARC3_S2_DRAFTQSA_SWITCH'], 'on\n')
    write(senv_def['ARC3_S2_PRESTAGE_SWITCH'], 'on\n')
    base = {k: os.environ[k] for k in BASE_KEYS if k in os.environ}
    trace = bool(cfg['trace_dir'])
    senv = dict(base, **senv_def)
    if os.environ.get('LD_LIBRARY_PATH'):       # runtime_python.sh puts the runtime's CUDA libraries first and keeps these after them
        senv['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
    if not cfg['engine_audit']:
        senv['ARC3_ENGINE_AUDIT'] = '0'
    if cfg['use_restart_value']:
        senv['ARC3_V72_PARK_RANK'] = 'restart'
    if trace:
        senv.update(ARC3_FILETRACE_DIR=cfg['trace_dir'], ARC3_PY_PREPEND=cfg['filetrace_py'])
    if not cfg['diagnostics']:                   # the user 10-24 (via IE): off in a competition rerun; tests keep them on (IE's verified set)
        for k in ('ARC3_S2_STEPTIMER_DIR', 'ARC3_S2_STEPTIMER_CPU', 'LC_STEPTIMER_DIR', 'ARC3_S2_CANARY_DIR', 'ARC3_S2_CANARY_MALLOCSTATS',
                  'ARC3_S2_MEMSTATS_DIR'):         # memstats: off unless its dir is set (memstats.py); no triggers below
            senv.pop(k, None)
        senv.update(ARC3_S2_PLECONV_VERIFY='0', ARC3_S2_FQFAST_VERIFY='0', ARC3_S2_SCHEDFAST_VERIFY='0', ARC3_ENGINE_AUDIT='0',
                    ARC3_K_SERVICE_AUDIT='0',
                    # v9 (the user 10-27: no throughput-costing diagnostic in a competition rerun; ARC3 variables, outside vLLM's
                    # compile hash): the allocator-retry watcher, the eviction batch-count trace, the staging timings
                    ARC3_S2_MEMDIAG_RETRY='0', ARC3_EVICTION_GRAPH_TRACE='0', ARC3_EVICTION_PROFILE='0')
    senv.update({k: str(v) for k, v in cfg['extra_server_env'].items()})
    for k in cfg['drop_server_env']:             # e.g. the read-only diagnostics for a final definition (IE's lever ledger C4 / C5 / C8)
        senv.pop(k, None)
    for k in ('PROFILE', 'PROFILE_STEPS', 'ARC3_PROFILE_DIR', 'ARC3_PROFILE_STEPS'):
        senv.pop(k, None)
    stop_flag = threading.Event()

    def on_signal(signum, frame):
        log('signal %d: stopping (client first, then the server)' % signum)
        stop_flag.set()
    for sg in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sg, on_signal)
        except ValueError:
            pass

    srv = dict(pid=None, log_dir=None, launches=0)

    def launch():
        srv['launches'] += 1
        u, t = gpu_line(gpu)
        log('server launch %d (GPU %s memory in use %s of %s MiB)' % (srv['launches'], gpu, u, t))
        try:
            p = subprocess.run(['bash', S2 + '/k_runtime_python.sh', S2 + '/k_launch_scaling_s2i.py', 'start'], cwd=S2, env=senv,
                               capture_output=True, text=True, timeout=900)
        except Exception as exc:  # noqa: BLE001
            log('server launch failed to run: %r' % exc)
            return
        for line in (p.stderr or '').splitlines():
            if line.strip():
                log('  ' + line[:300])
        try:
            j = re.search(r'\{\s*"pid":\s*(\d+),\s*"log_dir":\s*"([^"]+)"', p.stdout or '')
            srv['pid'], srv['log_dir'] = int(j.group(1)), j.group(2)
            write(os.path.join(S2, 'instances', label, 'last_logdir'), srv['log_dir'] + '\n')
            log('server root PID %d, log dir %s (launcher rc %d)' % (srv['pid'], srv['log_dir'], p.returncode))
        except Exception as exc:  # noqa: BLE001
            log('server launch: no launch record (rc %s): %r; stdout %r' % (p.returncode, exc, (p.stdout or '')[-300:]))

    def stop_server():
        if srv['pid'] is None:
            return
        try:
            p = subprocess.run(['bash', S2 + '/k_runtime_python.sh', S2 + '/k_launch_scaling_s2i.py', 'stop'], cwd=S2, env=senv,
                               capture_output=True, text=True, timeout=180)
            log('server stop rc %d: %s' % (p.returncode, ' '.join((p.stdout + p.stderr).split())[-300:]))
        except Exception as exc:  # noqa: BLE001
            log('server stop failed: %r' % exc)
        if alive(srv['pid']):
            for pid in session_pids(srv['pid']):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            log('server session killed')

    launch()
    ev('server_launched')
    # client environment (run_v62i_arm.sh's, the capture) with the Kaggle switches
    cenv = dict(base, **cenv_def)
    if cfg['kaggle_client'] and cfg['game_source'] == 'gateway' and cfg['keepalive_s']:   # v7.9b K5: the gateway closes a scorecard idle 15 min
        cenv['ARC3_KAGGLE_KEEPALIVE_S'] = str(cfg['keepalive_s'])
    if cfg['kaggle_client']:
        cenv.update(ARC3_KAGGLE='1', ARC3_KAGGLE_SERVER_WAIT_S=str(cfg['server_wait_s']), ARC3_WARMUP_ACTION_GAMES='1',
                    ARC3_HTTP_RETRY_INITIAL_SECONDS='900', ARC3_KAGGLE_SHUFFLE='1' if cfg['shuffle'] else '0',
                    ARC3_KAGGLE_SHUFFLE_SEED=str(int(cfg['shuffle_seed'])))
    cenv['ARC3_V72_PARK_RANK'] = 'restart' if cfg['use_restart_value'] else 'fifo'
    if cfg['use_restart_value'] and cfg['rv_constants']:
        cenv['ARC3_V72_RV'] = cfg['rv_constants']
    pp = cenv.get('PYTHONPATH', '')
    if cfg['arc_wheels_dir']:
        pp = cfg['arc_wheels_dir'] + ':' + pp
    if trace:
        pp = cfg['filetrace_py'] + ':' + pp
        cenv['ARC3_FILETRACE_DIR'] = cfg['trace_dir']
    cenv['PYTHONPATH'] = pp
    for k in ('ARC_BASE_URL', 'ARC_API_KEY', 'KAGGLE_IS_COMPETITION_RERUN'):   # the gateway's, from the notebook
        if k in os.environ:
            cenv[k] = os.environ[k]
    if not cfg['diagnostics']:                   # the same keys reach the client harmlessly; dropped for consistency
        for k in ('ARC3_S2_STEPTIMER_DIR', 'ARC3_S2_STEPTIMER_CPU', 'LC_STEPTIMER_DIR', 'ARC3_S2_CANARY_DIR', 'ARC3_S2_CANARY_MALLOCSTATS',
                  'ARC3_S2_MEMSTATS_DIR'):
            cenv.pop(k, None)
        cenv.update(ARC3_S2_PLECONV_VERIFY='0', ARC3_S2_FQFAST_VERIFY='0', ARC3_S2_SCHEDFAST_VERIFY='0')
    cenv.update({k: str(v) for k, v in cfg['extra_client_env'].items()})
    for k in cfg['drop_client_env']:
        cenv.pop(k, None)
    OUT = os.path.join(S2, 'client_mtp', label)
    games = cfg['games']
    games_arg = ','.join(games) if isinstance(games, list) else str(games)
    cli = dict(p=None)
    client_at = t_nb + float(cfg['client_start_s'])
    lv, errs = Levels(OUT), ErrLines()
    trig = dict(t0=None, nm=3300)                 # lc_fin.sh's memstats readings: 'start light', then 'm55 light' and every hour

    def memstats_start():                         # once the server is up and the client runs (a reading before a process starts is not run)
        if cfg['memstats_triggers'] and cfg['diagnostics'] and trig['t0'] is None and cli['p'] is not None and t_up_box[0] is not None:
            write(os.path.join(senv_def['ARC3_S2_MEMSTATS_DIR'], 'trigger'), 'start light\n')
            trig['t0'] = time.time()
    t_up_box = [None]

    def start_client(why):
        secs = int(cfg['secs_override'] or max(60, G - 60 - time.time()))
        argv = [ROOT + '/venv-duck/bin/python', '-m', 'v62.run_v6', '--games', games_arg, '--seconds', str(secs), '--output', OUT,
                '--base-url', 'http://127.0.0.1:%d' % port, '--context', str(prof['CTX']), '--basis', str(prof['BASIS']),
                '--protect', str(prof['TAIL']), '--concurrency', '0', '--label', label, '--environments-dir', cfg['environments_dir']]
        if cfg['game_source'] == 'gateway':
            argv += ['--game-source', 'gateway']
        fh = open(os.path.join(RUN, 'client.log'), 'a')
        cli['p'] = subprocess.Popen(argv, cwd=S2 + '/harness_v6', env=cenv, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        ev('client_started')
        log('client started (%s): PID %d, %d s of gameplay, games %s, source %s, %s, shuffle %s' % (
            why, cli['p'].pid, secs, games_arg[:80], cfg['game_source'], 'restart_value' if cfg['use_restart_value'] else 'FIFO',
            ('seed %d' % cfg['shuffle_seed']) if cfg['shuffle'] else 'off'))
        memstats_start()

    # ---------------------------------------------------------------- the server's start; the client at its time or the server's
    t_up = None
    last = 0
    while not stop_flag.is_set():
        st, _ = get('http://127.0.0.1:%d/health' % port)
        if st == 200:
            t_up = t_up_box[0] = time.time()
            ev('server_healthy')
            log('server healthy, %.0f s after the notebook start' % (t_up - t_nb))
            break
        if cli['p'] is None and time.time() >= client_at:
            start_client('the start time')
        if srv['pid'] is not None and not alive(srv['pid']):
            tail = ''
            try:
                tail = ' | '.join(open(os.path.join(srv['log_dir'], 'server.log'), errors='replace').read().splitlines()[-6:])[-600:]
            except OSError:
                pass
            log('the server process %d exited before it was healthy: %s' % (srv['pid'], tail))
            if srv['launches'] <= cfg['relaunch_max']:
                stop_server()
                srv['pid'] = None
                launch()
            else:
                log('no relaunch left; waiting (the client waits for the server, then its requests fail)')
                srv['pid'] = None
        if time.time() - last >= 60:
            last = time.time()
            line = ''
            try:
                line = open(os.path.join(srv['log_dir'], 'server.log'), errors='replace').read().splitlines()[-1][:160]
            except (OSError, IndexError, TypeError):
                pass
            u, t = gpu_line(gpu)
            log('waiting for the server: %.0f s since the notebook start, GPU %s / %s MiB, RAM available %s GB; %s' % (
                time.time() - t_nb, u, t, mem_avail_gb(), line))
        if time.time() > G:
            log('the gameplay end passed before the server was healthy')
            break
        time.sleep(5)
    if t_up is not None:
        m = metrics(port)
        blocks = m.get('blocks')
        page = int(senv_def.get('PAGE_TOKENS') or senv_def.get('ARC3_S2_POOLPRESS_PAGE') or 3104)   # v8: the profile's block (g4 3,552)
        need = int(prof['NP']) * (math.ceil(int(prof['CTX']) / page) + 3) + 1
        need -= int(((D.get('tiers') or {}).get('rows') or {}).get(cfg['profile'], {}).get('trim') or 0)   # v9: a tier's trimmed spares
        if blocks:
            write(senv_def['ARC3_S2_POOLPRESS_POOL_FILE'], '%d\n' % blocks)
        u, t = gpu_line(gpu)
        log('pool %s pages, need %d%s; GPU memory in use idle %s of %s MiB' % (blocks, need, '' if (blocks or 0) >= need else ' (SHORT: logged only)', u, t))
        rec['pool'] = dict(blocks=blocks, need=need)
        try:   # the serve process's environment against the definition (log only)
            env = dict(kv.split('=', 1) for kv in open('/proc/%d/environ' % srv['pid'], 'rb').read().decode('utf-8', 'replace').split('\0') if '=' in kv)
            exp = dict(senv_def)
            diff = sorted(k for k in set(exp) | set(env) if k in exp and env.get(k) != exp[k])
            missing = sorted(k for k in exp if k not in env)
            rec['env_check'] = dict(compared=len(exp), differ=[(k, (exp.get(k) or '')[:120], (env.get(k) or '')[:120]) for k in diff],
                                    missing=missing, extra=sorted(set(env) - set(exp))[:400])
            log('serve environment: %d definition keys, %d differ (%s), %d missing' % (len(exp), len(diff), ', '.join(diff[:8]), len(missing)))
        except Exception as exc:  # noqa: BLE001
            log('serve environment unreadable: %r' % exc)
        if trace and srv['pid']:
            log('maps snapshot (server up): %d processes' % maps_snapshot(session_pids(srv['pid']), cfg['trace_dir'], 'up'))
    if t_up is not None and cfg['nll_panel'] and cli['p'] is None and not stop_flag.is_set():   # logged only
        log('NLL panel (expect %s)' % cfg['nll_expect'])
        try:
            p = subprocess.run(['bash', S2 + '/nll_panel_i.sh', label], cwd=S2, capture_output=True, text=True, timeout=900,
                               env=dict(base, EXPECT=str(cfg['nll_expect']), PORT=str(port), PYTHONDONTWRITEBYTECODE='1',
                                        IDENTITY=os.path.join(S2, 'instances', label, 'server-identity.json')))
            rec['nll'] = (p.stdout or '').strip().splitlines()[-1:] or [(p.stderr or '')[-300:]]
            log('NLL panel: %s' % rec['nll'][0][:300])
        except Exception as exc:  # noqa: BLE001
            log('NLL panel failed: %r' % exc)
    if cli['p'] is None and not stop_flag.is_set():
        start_client('the server is healthy')
    memstats_start()
    # ---------------------------------------------------------------- gameplay: status lines, triggers, end
    t_stat, gen0, tok_t0 = 0, None, None
    cpu_prev = None                               # (time, server tree CPU s, client tree CPU s): cores busy per status line
    snaps = [t + time.time() for t in (300, 1200)] if trace else []
    while not stop_flag.is_set():
        p = cli['p']
        if p is not None and p.poll() is not None:
            ev('client_exited')
            log('client exited rc %s' % p.returncode)
            break
        now = time.time()
        if snaps and now >= snaps[0]:
            snaps.pop(0)
            n1 = maps_snapshot(session_pids(srv['pid']), cfg['trace_dir'], 's%d' % int(now - t_nb)) if srv['pid'] else 0
            n2 = maps_snapshot(session_pids(p.pid), cfg['trace_dir'], 'c%d' % int(now - t_nb)) if p is not None else 0
            log('maps snapshot: %d server and %d client processes' % (n1, n2))
        if trig['t0'] is not None and now - trig['t0'] >= trig['nm']:
            write(os.path.join(senv_def['ARC3_S2_MEMSTATS_DIR'], 'trigger'), 'm%d light\n' % (trig['nm'] // 60))
            trig['nm'] += 3600
        if now - t_stat >= cfg['status_every_s']:
            t_stat = now
            m = metrics(port)
            gen = m.get('vllm:generation_tokens_total')
            tps = ''
            if gen is not None:
                if gen0 is not None and now > tok_t0:
                    tps = '%.0f tok/s' % ((gen - gen0) / (now - tok_t0))
                gen0, tok_t0 = gen, now
            u, t = gpu_line(gpu)
            e = errs.count(os.path.join(srv['log_dir'], 'server.log')) if srv['log_dir'] else None
            cs, cc = (tree_cpu_s(srv['pid']) if srv['pid'] else 0.0), (tree_cpu_s(p.pid) if p is not None else 0.0)
            cores = ('%.2f / %.2f' % ((cs - cpu_prev[1]) / (now - cpu_prev[0]), (cc - cpu_prev[2]) / (now - cpu_prev[0]))
                     if cpu_prev and now > cpu_prev[0] else '-')
            cpu_prev = (now, cs, cc)
            log('status %.2f h: server %s, client %s; requests running %s waiting %s, %s, preemptions %s; levels %d; GPU %s/%s MiB '
                '(free %s); RAM avail %s GB; disk free run %s GB, /kaggle/working %s GB; server error lines %s; CPU cores server / client %s, '
                'load1 %.1f, cpu pressure avg60 %s' % (
                    (now - t_nb) / 3600, 'up' if srv['pid'] and alive(srv['pid']) else 'DOWN', 'up' if p is not None and p.poll() is None else '-',
                    m.get('vllm:num_requests_running'), m.get('vllm:num_requests_waiting'), tps or '-', m.get('vllm:num_preemptions_total'),
                    lv.count(), u, t, (t - u) if u is not None else None, mem_avail_gb(), disk_free_gb(RUN), disk_free_gb('/kaggle/working'), e,
                    cores, os.getloadavg()[0], cpu_pressure_avg60()))
        if now > G + 240 and p is not None and p.poll() is None:
            log('the client is still running 240 s after the gameplay end: SIGTERM (its teardown)')
            ev('client_sigterm')
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass
            for _ in range(180):
                if p.poll() is not None:
                    break
                time.sleep(1)
            if p.poll() is None:
                log('the client did not exit 180 s after SIGTERM: SIGKILL')
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
            break
        time.sleep(10)
    if stop_flag.is_set() and cli['p'] is not None and cli['p'].poll() is None:
        try:
            os.killpg(cli['p'].pid, signal.SIGTERM)
            cli['p'].wait(timeout=180)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(cli['p'].pid, signal.SIGKILL)
            except OSError:
                pass
    rec['levels'] = lv.count()
    try:
        rec['score'] = json.load(open(os.path.join(OUT, 'score.json'))).get('score')
    except Exception:  # noqa: BLE001
        rec['score'] = None
    if trace and srv['pid'] and alive(srv['pid']):
        log('maps snapshot (end): %d processes' % maps_snapshot(session_pids(srv['pid']), cfg['trace_dir'], 'end'))
    stop_server()
    ev('server_stopped')
    write(os.path.join(RUN, 'kaggle_run.json'), json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n')
    log('done: levels %s, score %s, %.0f s after the notebook start (record %s)' % (rec['levels'], rec['score'], time.time() - t_nb,
                                                                                    os.path.join(RUN, 'kaggle_run.json')))


if __name__ == '__main__':
    main()
