"""kaggle_run_arc2.py CONFIG.json (2026-10-31, ARC2 - main; the user: "a provisional version for kaggle submission tomorrow afternoon"):
the ARC-2 Kaggle driver for 4x L4. One server on all four GPUs (TP2 x PP2): the profile rendered from the ARC-2 KDEF exactly as
kaggle_run.py renders it, then arc2_server_l4.py's L4 overrides (L4ENV, EXTRA, DROP) and the shipped sizing (cfg l4_sets), started
through k_launch_scaling_l4.py; then the ARC-2 harness (harness_v2m, arc2.runner) over the challenges file until the deadline;
submission.json kept valid throughout (placeholders for every task first, the harness's own file merged over them every minute and
at the end); then the server's stop. Every check logs and continues: nothing here blocks a submission (a block scores zero).
cfg dry_run: the rendered server environment and both command lines, no launch (the notebook validation and the pod rehearsal).
kaggle_run_arc2_v2f.py (2026-11-01, ARC2 - main; package v2f, the user's lane item "the driver's capture-vs-lanes log line"):
kaggle_run_arc2.py 1003e255 plus one log line after the KV sizing: the lanes, the decode microbatches they make under PP (the
balancer's ceil / floor split) in requests and tokens (MTP K + 1 per request), and the capture size each one replays (padding), or a
WARNING when a microbatch exceeds the largest capture size (it then runs without a CUDA graph). Logs only."""
import glob
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time

S2 = "/workspace/kvring_dev/fin74/s2"
ROOT = "/workspace/arc3-inference"
KR_PATH = "/workspace/kvring_dev/kaggle/kaggle_run.py"
L4P = "/mnt/shared/arc2/l4port/l4prod"
CFG = dict(label="ka2t01", profile="g60f8inp128", kdef="/mnt/shared/arc2/l4port/sidepod/KDEF_sp20.json", gpus="0,1,2,3", port=1390,
           notebook_start=None, budget_s=11.5 * 3600, teardown_margin_s=150, server_wait_s=4800, relaunch_max=1, harness_grace_s=45,
           relaunch_mid_max=1, competition=False, l4_sets={}, drop_parts=["ctxfree"], extra_server_env={}, extra_harness_env={},
           challenges="", solutions="", ids="", limit=0, harness_dir="/mnt/shared/arc2/harness_v2m",
           tokenizer="/workspace/models/Qwen3.8-Flash-Next-W4A16-intel/tokenizer.json", lanes="auto", lanes_cap=34,
           hard_lanes="half", harness_args=[], submission_path="/kaggle/working/submission.json",
           runs_dir="/mnt/shared/arc2/runs", status_every_s=600, merge_every_s=60, trace_dir="", filetrace_py="",
           dry_run=False)
# 2026-10-31 (the user, via the helper session: "Use 80K threshold and 200K cap for the next package"): phase-3 thresholds 60000 ->
# 80000 and the pass budget 160000 -> 200000 (the wrap-up tiers count down from the budget, so they start at 160K on their own).
# Max passes 3 -> 4 (the user, via the helper: "if a game goes 4 times without two valid solutions ... just submit whatever it has").
# harness_v2m (2026-10-31, the user: "third pass on identical answers > additional pass > review"): --phase3-priority third,more,review;
# --phase3-rule valid (the user: "We run until a puzzle gets two valid answers then it either gets additional pass if identical and
# review pass otherwise"; "No review will be performed on puzzles without two distinct answers"); --phase3-length mean (the user:
# "Use average": the mean of passes 1 and 2 against the 80K thresholds; "identical" = the attempt-1 test outputs); --rank-length mean
# (the user: the easy / hard lane order reads the same mean).
HARNESS_ARGS = ["--policy", "rank", "--passes", "2", "--valid-target", "2", "--max-passes", "4",
                "--phase3-priority", "third,more,review", "--phase3-rule", "valid", "--phase3-length", "mean",
                "--rank-length", "mean",
                "--extra-pass-identical-long", "80000", "--review-long", "80000",
                "--no-call", "keep_going", "--idle-limit", "6", "--repeat-limit", "3", "--observations", "turns",
                "--reply-caps", "v2c", "--cue-from", "120000", "--cap-ext", "4096", "--marker", "v2e", "--recall-at", "120000",
                "--limit-at", "100000", "--wrap-at", "120000", "--idle-skip-status", "--notice", "0", "--temperature", "0.6",
                "--admit-below", "0.85", "--admit-forecast", "--evict-fraction", "0.2", "--note-guard", "--prompt", "v2j",
                "--cap-close", "32000", "--budget", "200000"]
DIAG_KEYS = ("ARC3_S2_STEPTIMER_DIR", "ARC3_S2_STEPTIMER_CPU", "LC_STEPTIMER_DIR", "ARC3_S2_CANARY_DIR", "ARC3_S2_CANARY_MALLOCSTATS",
             "ARC3_S2_MEMSTATS_DIR")
DIAG_OFF = dict(ARC3_S2_PLECONV_VERIFY="0", ARC3_S2_FQFAST_VERIFY="0", ARC3_S2_SCHEDFAST_VERIFY="0", ARC3_ENGINE_AUDIT="0",
                ARC3_K_SERVICE_AUDIT="0", ARC3_S2_MEMDIAG_RETRY="0", ARC3_EVICTION_GRAPH_TRACE="0", ARC3_EVICTION_PROFILE="0")
T_LOG = [None]


def log(msg):
    line = "[%s kaggle_run_arc2] %s" % (time.strftime("%H:%M:%S", time.gmtime()), msg)
    print(line, flush=True)
    if T_LOG[0]:
        try:
            with open(T_LOG[0], "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh)
    os.replace(path + ".tmp", path)


def valid_grid(g):
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30 and len(r) == len(g[0])
                                                              and all(isinstance(v, int) and 0 <= v <= 9 for v in r) for r in g))


def placeholders(challenges):
    """{task_id: [{attempt_1: [[0]], attempt_2: [[0]]} per test input]} for every task of the challenges file (Kaggle's format)."""
    try:
        tasks = json.load(open(challenges))
    except Exception as exc:  # noqa: BLE001  (a log line, never a stop)
        log("challenges %s unreadable: %r" % (challenges, exc))
        return {}
    return {tid: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in (t.get("test") or [{}])] for tid, t in tasks.items()}


def merged(base, harness_file):
    """base with each task the harness file holds in the right shape (one slot per test input, both attempts valid grids)."""
    out, n = dict(base), 0
    try:
        sub = json.load(open(harness_file))
    except Exception:  # noqa: BLE001
        return out, 0
    for tid, entry in (sub or {}).items():
        if tid in base and isinstance(entry, list) and len(entry) == len(base[tid]) and all(
                isinstance(s, dict) and valid_grid(s.get("attempt_1")) and valid_grid(s.get("attempt_2")) for s in entry):
            out[tid] = entry
            n += 1
    return out, n


def main():
    cfg = dict(CFG)
    if len(sys.argv) > 1:
        cfg.update(json.load(open(sys.argv[1])))
    t_nb = float(cfg["notebook_start"] or time.time())
    G = t_nb + float(cfg["budget_s"]) - float(cfg["teardown_margin_s"])   # the harness's deadline (its attempts end here)
    label, gpus, port = cfg["label"], str(cfg["gpus"]), int(cfg["port"])
    kr = load("kaggle_run", KR_PATH)
    srvmod = load("arc2_server_l4", os.path.join(L4P, "arc2_server_l4.py"))
    D = json.load(open(cfg["kdef"]))
    prof = D["profiles"][cfg["profile"]]
    senv_def = kr.render(D, "server_serve", prof, label, gpus, port)
    drop = set(srvmod.DROP) | set(cfg["drop_parts"])
    l4 = dict(srvmod.L4ENV, ARC3_STAGE2=",".join(p for p in senv_def["ARC3_STAGE2"].split(",") if p not in drop))
    l4.update({k: str(v) for k, v in cfg["l4_sets"].items()})
    if "ARC3_S2_CAPTURE" in l4 and "ARC3_STAGE4_GRAPH_SIZE" not in l4:   # vLLM: max_cudagraph_capture_size == the largest size
        l4["ARC3_STAGE4_GRAPH_SIZE"] = str(max(int(x) for x in l4["ARC3_S2_CAPTURE"].split(",") if x))
    base = {k: os.environ[k] for k in kr.BASE_KEYS if k in os.environ}
    senv = dict(base, **senv_def)
    if os.environ.get("LD_LIBRARY_PATH"):
        senv["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    if cfg["competition"]:                    # kaggle_run.py's diagnostics=False set (the user 10-24 / 10-27)
        for k in DIAG_KEYS:
            senv.pop(k, None)
        senv.update(DIAG_OFF)
    senv.update(srvmod.EXTRA)
    senv.update(l4)
    senv.update({k: str(v) for k, v in cfg["extra_server_env"].items()})
    trace = bool(cfg["trace_dir"])
    if trace:                                 # the pod rehearsal's file trace: the tracer first on the server processes' path
        senv.update(ARC3_FILETRACE_DIR=cfg["trace_dir"], ARC3_PY_PREPEND=cfg["filetrace_py"] + ":" + senv["ARC3_PY_PREPEND"])
    for k in ("PROFILE", "PROFILE_STEPS", "ARC3_PROFILE_DIR", "ARC3_PROFILE_STEPS"):
        senv.pop(k, None)
    RUN = senv_def["FIN_RUN"]
    HOUT = os.path.join(cfg["runs_dir"], label)
    sub_path = cfg["submission_path"]
    henv = dict(base, ARC2_ARC3_HARNESS_ROOT=S2 + "/harness_v6", LOCAL_ANALYZER_TEMPERATURE="0.6", PYTHONDONTWRITEBYTECODE="1")
    if trace:
        henv.update(ARC3_FILETRACE_DIR=cfg["trace_dir"], PYTHONPATH=cfg["filetrace_py"])
    henv.update({k: str(v) for k, v in cfg["extra_harness_env"].items()})

    def harness_argv(lanes, fcp, deadline_s):
        hard = lanes // 2 if cfg["hard_lanes"] == "half" else int(cfg["hard_lanes"])
        argv = [ROOT + "/venv-duck/bin/python", "-m", "arc2.runner", "--out", HOUT, "--base-url", "http://127.0.0.1:%d" % port,
                "--tokenizer", cfg["tokenizer"], "--profile", cfg["profile"], "--kdef", cfg["kdef"],
                "--challenges", cfg["challenges"], "--solutions", cfg["solutions"], "--lanes", str(lanes), "--hard-lanes", str(hard),
                "--fc-pool", str(fcp), "--deadline-s", str(int(deadline_s))]
        if cfg["ids"]:
            argv += ["--ids", str(cfg["ids"])]
        if int(cfg["limit"] or 0):
            argv += ["--limit", str(int(cfg["limit"]))]
        return argv + list(cfg["harness_args"] or HARNESS_ARGS)

    if cfg["dry_run"]:
        print(json.dumps(dict(server_env=senv, l4=l4, launcher=["bash", S2 + "/k_runtime_python.sh", S2 + "/k_launch_scaling_l4.py",
                                                                 "start"], harness=harness_argv(34, 755, 3600), harness_env=henv,
                              harness_cwd=cfg["harness_dir"], run=RUN, harness_out=HOUT), indent=1, sort_keys=True))
        return 0
    os.makedirs(RUN, exist_ok=True)
    T_LOG[0] = os.path.join(RUN, "kaggle_run_arc2.log")
    rec = dict(config=cfg, profile=prof, l4=l4, notebook_start=t_nb, harness_deadline=G, events={})
    ev = lambda k: rec["events"].setdefault(k, round(time.time() - t_nb, 1))   # noqa: E731
    base_sub = placeholders(cfg["challenges"])
    write_json(sub_path, base_sub)
    log("placeholder submission: %d tasks -> %s" % (len(base_sub), sub_path))
    log("label %s, profile %s (context %s), GPUs %s, port %d; L4 sizing %s; harness deadline %s (%.0f s from now)" % (
        label, cfg["profile"], prof.get("CTX"), gpus, port, {k: l4[k] for k in sorted(l4) if k.startswith(("ARC3_STAGE4_", "ARC3_S2_CAPTURE",
                                                                                                               "ARC3_L4PORT_PP_SPLIT"))},
        time.strftime("%H:%M:%S", time.gmtime(G)), G - time.time()))
    for d in ("steptimer", "torch", "memstats", "canary"):
        os.makedirs(os.path.join(RUN, d), exist_ok=True)
    kr.write(senv_def["ARC3_S2_DRAFTQSA_SWITCH"], "on\n")
    kr.write(senv_def["ARC3_S2_PRESTAGE_SWITCH"], "on\n")
    stop_flag = threading.Event()

    def on_signal(signum, frame):
        log("signal %d: stopping (the harness first, then the server)" % signum)
        stop_flag.set()
    for sg in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sg, on_signal)
        except ValueError:
            pass
    srv = dict(pid=None, log_dir=None, launches=0)

    def launch():                             # a relaunch keeps the label, as kaggle_run.py's does
        srv["launches"] += 1
        log("server launch %d (GPU memory in use %s MiB)" % (srv["launches"], [kr.gpu_line(g)[0] for g in gpus.split(",")]))
        try:
            p = subprocess.run(["bash", S2 + "/k_runtime_python.sh", S2 + "/k_launch_scaling_l4.py", "start"], cwd=S2, env=senv,
                               capture_output=True, text=True, timeout=900)
        except Exception as exc:  # noqa: BLE001
            log("server launch failed to run: %r" % exc)
            return
        for line in (p.stderr or "").splitlines():
            if line.strip():
                log("  " + line[:300])
        m = re.search(r'\{\s*"pid":\s*(\d+),\s*"log_dir":\s*"([^"]+)"', p.stdout or "")
        if not m:
            log("server launch: no launch record (rc %s); stdout %r" % (p.returncode, (p.stdout or "")[-300:]))
            return
        srv["pid"], srv["log_dir"] = int(m.group(1)), m.group(2)
        kr.write(os.path.join(S2, "instances", label, "last_logdir"), srv["log_dir"] + "\n")
        log("server root PID %d, log dir %s (launcher rc %d)" % (srv["pid"], srv["log_dir"], p.returncode))

    def stop_server():
        if srv["pid"] is None:
            return
        try:
            p = subprocess.run(["bash", S2 + "/k_runtime_python.sh", S2 + "/k_launch_scaling_l4.py", "stop"], cwd=S2,
                               env=senv, capture_output=True, text=True, timeout=60)
            log("server stop rc %d: %s" % (p.returncode, " ".join(((p.stdout or "") + (p.stderr or "")).split())[-300:]))
        except Exception as exc:  # noqa: BLE001
            log("server stop failed: %r" % exc)
        if kr.alive(srv["pid"]):
            for pid in kr.session_pids(srv["pid"]):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            log("server session killed")

    def server_log():
        return os.path.join(srv["log_dir"], "server.log") if srv["log_dir"] else ""

    launch()
    ev("server_launched")
    t_up, last = None, 0.0
    while not stop_flag.is_set():
        st, _ = kr.get("http://127.0.0.1:%d/health" % port)
        if st == 200:
            t_up = time.time()
            ev("server_healthy")
            log("server healthy, %.0f s after the notebook start" % (t_up - t_nb))
            break
        if srv["pid"] is not None and not kr.alive(srv["pid"]):
            tail = ""
            try:
                tail = " | ".join(open(server_log(), errors="replace").read().splitlines()[-6:])[-600:]
            except OSError:
                pass
            log("the server process %d exited before it was healthy: %s" % (srv["pid"], tail))
            if srv["launches"] <= int(cfg["relaunch_max"]):
                stop_server()
                srv["pid"] = None
                launch()
            else:
                log("no relaunch left: the placeholder submission stays")
                srv["pid"] = None
                break
        if srv["pid"] is None and srv["launches"] > int(cfg["relaunch_max"]):
            break
        if time.time() - last >= 120:
            last = time.time()
            line = ""
            try:
                line = open(server_log(), errors="replace").read().splitlines()[-1][:160]
            except (OSError, IndexError):
                pass
            log("waiting for the server: %.0f s since the notebook start, GPU MiB %s, RAM available %s GB; %s" % (
                time.time() - t_nb, [kr.gpu_line(g)[0] for g in gpus.split(",")], kr.mem_avail_gb(), line))
        if time.time() - t_nb > float(cfg["server_wait_s"]) or time.time() > G - 300:
            log("the server was not healthy within its wait: the placeholder submission stays")
            break
        time.sleep(5)
    hp = None
    if t_up is not None:
        SL = server_log()
        txt = ""
        try:
            txt = open(SL, errors="replace").read()
        except OSError:
            pass
        conc = [float(x) for x in re.findall(r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x", txt)]
        kvt = [int(x.replace(",", "")) for x in re.findall(r"GPU KV cache size: ([0-9,]+) tokens", txt)]
        cap = min(int(cfg["lanes_cap"]), int(senv.get("ARC3_PERSISTENT_MAX_SESSIONS") or 999),   # more lanes than sessions: refusals;
                  int(senv.get("ARC3_STAGE4_MAX_SEQS") or 999))                                  # than sequences: lanes wait
        lanes = int(cfg["lanes"]) if cfg["lanes"] != "auto" else (min(cap, int(math.floor(conc[-1]))) if conc else cap // 2)
        fcp = round(755 * kvt[0] / 2148573) if kvt else 755
        rec["sizing"] = dict(concurrency=conc[-1] if conc else None, kv_tokens=kvt[0] if kvt else None, lanes=lanes, fc_pool=fcp)
        log("KV sizing: concurrency %s, KV tokens %s -> %d lanes (cap %d), fc-pool %d; GPU MiB in use %s" % (
            rec["sizing"]["concurrency"], rec["sizing"]["kv_tokens"], lanes, cap, fcp, [kr.gpu_line(g)[0] for g in gpus.split(",")]))
        try:                                  # v2f: capture vs lanes (logs only)
            pp = 2 if l4.get("ARC3_L4PORT_PP_SPLIT") else 1
            k = int(senv.get("ARC3_S2_MTP_TOKENS") or l4.get("ARC3_S2_MTP_TOKENS") or 0)
            caps = sorted(int(x) for x in str(l4.get("ARC3_S2_CAPTURE", "")).split(",") if x.strip())
            reqs = sorted({-(-lanes // pp), lanes // pp} - {0}, reverse=True)
            toks = [r * (k + 1) for r in reqs]
            graphs = [min([c for c in caps if c >= t] or [None]) if caps else None for t in toks]
            rec["capture_vs_lanes"] = dict(lanes=lanes, pp=pp, mtp_k=k, microbatch_requests=reqs, microbatch_tokens=toks, capture=caps,
                                           graphs=graphs)
            bad = [t for t, g in zip(toks, graphs) if g is None]
            log("%scapture vs lanes: %d lanes, PP %d -> decode microbatches %s requests x %d tokens = %s tokens; capture %s -> graphs %s"
                " (padding %s)%s" % ("WARNING " if bad else "", lanes, pp, " / ".join(map(str, reqs)), k + 1,
                                     " / ".join(map(str, toks)), ",".join(map(str, caps)) or "-",
                                     " / ".join("eager" if g is None else str(g) for g in graphs),
                                     " / ".join("-" if g is None else str(g - t) for t, g in zip(toks, graphs)),
                                     ": microbatches above the largest capture size run without a CUDA graph" if bad else ""))
        except Exception as exc:  # noqa: BLE001
            log("capture vs lanes: not computed (%s)" % exc)
        try:                                  # host RAM at health (Kaggle has 188 GiB; the PLE table is one ~100 GB process)
            rss = []
            for pid in kr.session_pids(srv["pid"]):
                for l in open("/proc/%d/status" % pid):
                    if l.startswith("VmRSS:"):
                        rss.append(int(l.split()[1]) // 1024)
            log("host RAM at health: server session %d processes, RSS %.1f GB (largest %s MB), MemAvailable %s GB" % (
                len(rss), sum(rss) / 1024, sorted(rss)[-3:], kr.mem_avail_gb()))
        except Exception as exc:  # noqa: BLE001
            log("host RAM unreadable: %r" % exc)
        try:   # the serve process's environment against the definition plus the L4 overrides (log only)
            env = dict(kv.split("=", 1) for kv in open("/proc/%d/environ" % srv["pid"], "rb").read().decode("utf-8", "replace").split("\0")
                       if "=" in kv)
            diff = sorted(k for k in senv if k.startswith(("ARC3_", "VLLM_", "NCCL_")) and env.get(k) != senv[k])
            log("serve environment: %d keys compared, %d differ%s" % (len(senv), len(diff), (" (%s)" % ", ".join(diff[:8])) if diff else ""))
        except Exception as exc:  # noqa: BLE001
            log("serve environment unreadable: %r" % exc)
        if trace and srv["pid"]:
            log("maps snapshot (server up): %d processes" % kr.maps_snapshot(kr.session_pids(srv["pid"]), cfg["trace_dir"], "up"))
        deadline = G - time.time()
        if deadline > 120 and not stop_flag.is_set():
            argv = harness_argv(lanes, fcp, deadline)
            os.makedirs(cfg["runs_dir"], exist_ok=True)
            fh = open(HOUT + ".out", "a")
            hp = subprocess.Popen(argv, cwd=cfg["harness_dir"], env=henv, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            ev("harness_started")
            log("harness started: PID %d, %d lanes, deadline %.0f s, %s" % (hp.pid, lanes, deadline, " ".join(argv[1:])[:900]))
        else:
            log("no time left for the harness (%.0f s)" % deadline)
    t_stat = t_merge = 0.0
    gen0 = tok_t0 = None
    snaps = [t + time.time() for t in (300, 1200)] if trace else []
    nmerged = 0
    paused = [0.0]                            # when the harness was SIGSTOPped for a mid-run relaunch (0: running)

    def resume(why):
        try:
            os.killpg(hp.pid, signal.SIGCONT)
        except OSError as exc:
            log("harness resume failed: %r" % exc)
        log("harness resumed (SIGCONT) after %.0f s: %s" % (time.time() - paused[0], why))
        paused[0] = 0.0
    while hp is not None and not stop_flag.is_set():
        now = time.time()
        if hp.poll() is not None:
            ev("harness_exited")
            log("harness exited rc %s" % hp.returncode)
            break
        if paused[0]:                         # the harness is stopped while the relaunched server loads
            st, _ = kr.get("http://127.0.0.1:%d/health" % port)
            why = ("the relaunched server is healthy" if st == 200 else "the relaunched server died" if srv["pid"] is None or
                   not kr.alive(srv["pid"]) else "30 min before the harness deadline" if now > G - 1800 else "")
            if why:
                resume(why)
        elif srv["pid"] is not None and not kr.alive(srv["pid"]) and now < G - 1800:
            # the server died mid-run: one relaunch (cfg relaunch_mid_max), the harness paused (SIGSTOP of its process group) until
            # the new server is healthy, so its refusal budget (8 zero-token starts per task, up to 60 s apart: ~8 min) cannot
            # run out during a Kaggle-length load (the reviewer, 10-31); its in-flight passes end as errors after the resume.
            # With < 30 min left a relaunch cannot pay off.
            tail = ""
            try:
                tail = " | ".join(open(server_log(), errors="replace").read().splitlines()[-6:])[-600:]
            except OSError:
                pass
            mid = rec.setdefault("mid_relaunches", 0)
            log("the server process %d DIED mid-run (%.2f h): %s" % (srv["pid"], (now - t_nb) / 3600, tail))
            stop_server()
            srv["pid"] = None
            if mid < int(cfg.get("relaunch_mid_max", 1)):
                rec["mid_relaunches"] = mid + 1
                ev("server_mid_relaunch")
                try:
                    os.killpg(hp.pid, signal.SIGSTOP)
                    paused[0] = time.time()
                    log("harness paused (SIGSTOP) for the relaunch")
                except OSError as exc:
                    log("harness pause failed: %r" % exc)
                launch()
            else:
                log("no mid-run relaunch left: the harness runs on against a dead server; the submission keeps what it has")
        if now - t_merge >= float(cfg["merge_every_s"]):
            t_merge = now
            sub, nmerged = merged(base_sub, os.path.join(HOUT, "submission.json"))
            write_json(sub_path, sub)
        if snaps and now >= snaps[0]:
            snaps.pop(0)
            n1 = kr.maps_snapshot(kr.session_pids(srv["pid"]), cfg["trace_dir"], "s%d" % int(now - t_nb)) if srv["pid"] else 0
            n2 = kr.maps_snapshot(kr.session_pids(hp.pid), cfg["trace_dir"], "h%d" % int(now - t_nb))
            log("maps snapshot: %d server and %d harness processes" % (n1, n2))
        if now - t_stat >= float(cfg["status_every_s"]):
            t_stat = now
            m = kr.metrics(port)
            gen = m.get("vllm:generation_tokens_total")
            tps = "-"
            if gen is not None:
                if gen0 is not None and now > tok_t0:
                    tps = "%.0f tok/s" % ((gen - gen0) / (now - tok_t0))
                gen0, tok_t0 = gen, now
            prog = ""
            try:
                prog = [l for l in open(HOUT + ".out", errors="replace").read().splitlines() if "] done " in l][-1][:220]
            except (OSError, IndexError):
                pass
            log("status %.2f h: server %s; running %s waiting %s, %s, KV %s, preemptions %s; GPU MiB %s; RAM avail %s GB; "
                "submission %d / %d tasks from the harness; %s" % (
                    (now - t_nb) / 3600, "up" if srv["pid"] and kr.alive(srv["pid"]) else "DOWN", m.get("vllm:num_requests_running"),
                    m.get("vllm:num_requests_waiting"), tps, m.get("vllm:kv_cache_usage_perc"), m.get("vllm:num_preemptions_total"),
                    [kr.gpu_line(g)[0] for g in gpus.split(",")], kr.mem_avail_gb(), nmerged, len(base_sub), prog))
        if now > G + float(cfg["harness_grace_s"]):    # the user (10-31): the harness runs to 150 s before the end, as ARC-3's
            if paused[0]:                                # gameplay: so the grace is short and the teardown fits before the hard stop
                resume("the harness deadline passed")
            log("the harness still runs %.0f s after its deadline: SIGTERM, then SIGKILL after 30 s" % float(cfg["harness_grace_s"]))
            ev("harness_sigterm")
            try:
                os.killpg(hp.pid, signal.SIGTERM)
            except OSError:
                pass
            for _ in range(30):
                if hp.poll() is not None:
                    break
                time.sleep(1)
            if hp.poll() is None:
                try:
                    os.killpg(hp.pid, signal.SIGKILL)
                except OSError:
                    pass
            break
        time.sleep(5)
    if hp is not None and hp.poll() is None and paused[0]:
        resume("the driver is stopping")      # a stopped process acts on SIGTERM only once continued
    if stop_flag.is_set() and hp is not None and hp.poll() is None:
        try:
            os.killpg(hp.pid, signal.SIGTERM)
            hp.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(hp.pid, signal.SIGKILL)
            except OSError:
                pass
    sub, nmerged = merged(base_sub, os.path.join(HOUT, "submission.json"))
    write_json(sub_path, sub)
    rec["submission"] = dict(tasks=len(sub), from_harness=nmerged, path=sub_path)
    log("submission written: %d tasks, %d from the harness -> %s" % (len(sub), nmerged, sub_path))
    try:
        sc = json.load(open(os.path.join(HOUT, "score.json")))
        rec["score"] = {k: sc.get(k) for k in ("score", "model_only_score", "outputs_right", "outputs", "wall_seconds", "tok_per_s")}
        log("harness score.json: %s" % rec["score"])
    except Exception:  # noqa: BLE001
        rec["score"] = None
    if trace and srv["pid"] and kr.alive(srv["pid"]):
        log("maps snapshot (end): %d processes" % kr.maps_snapshot(kr.session_pids(srv["pid"]), cfg["trace_dir"], "end"))
    st, body = kr.get("http://127.0.0.1:%d/metrics" % port, timeout=20)
    if st == 200:                             # the raw preemption lines (the pod's pass-line check reads this format)
        log("final metrics: %s" % " ".join(l for l in body.decode("utf-8", "replace").splitlines()
                                           if l.startswith(("vllm:num_preemptions_total", "vllm:generation_tokens_total"))))
    stop_server()
    ev("server_stopped")
    kr.write(os.path.join(RUN, "kaggle_run_arc2.json"), json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    log("done, %.0f s after the notebook start (record %s)" % (time.time() - t_nb, os.path.join(RUN, "kaggle_run_arc2.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
