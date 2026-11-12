"""arc2_server_l4.py (2026-10-31, ARC2 - main; the user: "focus on getting the harness to work with this profile on 4xL4"):
arc2_server_s2j.py for TP2 x PP2 on four GPUs: the profile rendered from KDEF as on Kaggle, then the L4 overrides (L4ENV:
every GPU, the parallel layout, the L4 port hooks through bootstrap_l4prod, the PLE copy handoff, MTP0X / glue fusion / the
private MoE adapter off until ported, NCCL P2P off) and the stage-2 parts without the four that the PP hooks replace or that
are TP=1-only (DROP), started through k_launch_scaling_l4.py. --set K=V (repeatable) overrides any key last (sizing: ARC3_STAGE4_UTIL,
ARC3_STAGE4_KV_GIB, ARC3_STAGE4_MAX_SEQS, ARC3_S2_CAPTURE, ARC3_STAGE2_BATCHED_TOKENS). NLL gate off by default (no L4
fingerprint yet). The original docstring follows.

Start one ARC-3 production server (server only) for ARC-2 runs on a pod, then gate it.

    python3 arc2_server.py --label arc2a --gpu 0 --port 8100            # run detached (start_detached.sh)

The server is started exactly as the Kaggle driver starts it (kaggle_run.py 36db5677 with KDEF.json c66c4a3f): the
profile's environment rendered from KDEF for this label, GPU and port, the driver's base keys and LD_LIBRARY_PATH,
the switch files, then `k_runtime_python.sh k_launch_scaling_s2j.py start` from the production tree. Extra server keys
follow the v12 rehearsal (k12t01): the ple19 checkpoint and ARC3_S2_CKPTFETCH_FILES=8. The ARC-3 client is not started.
After /health answers, the cold NLL panel runs against this instance and must reproduce the profile's fingerprint.
Writes <FIN_RUN>/arc2_launch.json (environment, launch record, identities) and ends with SERVER READY or STOP lines.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

KR = "/workspace/kvring_dev/kaggle/kaggle_run.py"
NLL = {"82kr1632": "0.3471216700030355", "80kr512": "0.3456764103217549", "82kx1280": "0.35258596732945363"}
EXTRA = {"ARC3_MODEL": "/workspace/models/Qwen3.8-Flash-Next-W4A16-intel-ple19", "ARC3_S2_CKPTFETCH_FILES": "8"}
L4ENV = {"ARC3_S2_GPU": "0,1,2,3", "ARC3_L4_TP": "2", "ARC3_L4_PP": "2", "ARC3_PY_PREPEND": "/mnt/shared/arc2/l4port/bootstrap_l4prod",
         "ARC3_PLE_HANDOFF_IMPL": "copy", "ARC3_MTP0X": "0", "ARC3_GLUE_FUSION": "0", "ARC3_STAGE4_PRIVATE_MOE": "0",
         "ARC3_L4PORT_QSA_TP": "1", "ARC3_L4PORT_HC_SHARD": "col", "ARC3_L4PORT_PLE_PP": "1", "ARC3_L4PORT_PP_SKIPMIXER": "1",
         "ARC3_L4PORT_PP_MTP": "1", "ARC3_L4PORT_PP_WARMCAP": "4", "ARC3_L4PORT_PP_BCASTPAD": "1", "ARC3_L4PORT_PP_DRAFTBC": "1",
         "ARC3_L4PORT_PP_EMBHOST": "1", "ARC3_L4PORT_VPTOPK": "20", "ARC3_L4PORT_DHEAD_TP": "1", "ARC3_L4PORT_DHEAD_FREE": "1",
         "ARC3_S2_B1INC_PLE": "bf16", "NCCL_P2P_DISABLE": "1", "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
         "ARC3_STAGE4_KV_GIB": "0", "ARC3_STAGE4_UTIL": "0.89", "ARC3_STAGE4_MAX_SEQS": "22"}
DROP = ("embedhost", "drafthead", "splitfix", "topkshare")   # PP: ppembhost / dheadtp replace the first two; TP=1-only / no source


def load_kr():
    spec = importlib.util.spec_from_file_location("kaggle_run", KR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--profile", default="82kr1632")
    ap.add_argument("--kdef", default="/workspace/kvring_dev/kaggle/kaggle_defs/KDEF.json")
    ap.add_argument("--competition", action="store_true", help="the competition set: diagnostics off (default: test mode, on)")
    ap.add_argument("--health-timeout-s", type=float, default=2700)
    ap.add_argument("--nll", action="store_true", help="run the cold NLL panel (off by default on the L4)")
    ap.add_argument("--expect-nll", default="", help="the fingerprint to reproduce (default: the profile's own in NLL); "
                    "a new profile with 82kr1632's KV format and ring is expected to reproduce 82kr1632's on the short panel")
    ap.add_argument("--stop", action="store_true", help="stop this label's server through the launcher's own stop action")
    ap.add_argument("--set", action="append", default=[], help="K=V applied last (repeatable)")
    ap.add_argument("--keep-parts", default="", help="comma list: parts of DROP to keep")
    ap.add_argument("--drop-parts", default="", help="comma list: further production parts to drop (bisection)")
    a = ap.parse_args()
    kr = load_kr()
    D = json.load(open(a.kdef))
    prof = D["profiles"][a.profile]
    senv_def = kr.render(D, "server_serve", prof, a.label, a.gpu, a.port)
    keep = set(x for x in a.keep_parts.split(",") if x)
    drop = (set(DROP) - keep) | set(x for x in a.drop_parts.split(",") if x)
    l4 = dict(L4ENV, ARC3_STAGE2=",".join(p for p in senv_def["ARC3_STAGE2"].split(",") if p not in drop))
    for kv in a.set:
        k, v = kv.split("=", 1)
        l4[k] = v
    if "ARC3_S2_CAPTURE" in l4 and "ARC3_STAGE4_GRAPH_SIZE" not in l4:   # vLLM: max_cudagraph_capture_size == the largest size
        l4["ARC3_STAGE4_GRAPH_SIZE"] = str(max(int(x) for x in l4["ARC3_S2_CAPTURE"].split(",") if x))
    if a.stop:                                # the launcher's stop: this instance's recorded server only
        base = {k: os.environ[k] for k in kr.BASE_KEYS if k in os.environ}
        senv = dict(base, **senv_def)
        if os.environ.get("LD_LIBRARY_PATH"):
            senv["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        senv.update(EXTRA)
        senv.update(l4)
        p = subprocess.run(["bash", kr.S2 + "/k_runtime_python.sh", kr.S2 + "/k_launch_scaling_l4.py", "stop"], cwd=kr.S2,
                           env=senv, capture_output=True, text=True, timeout=600)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}] stop {a.label}: rc {p.returncode}; "
              f"{((p.stdout or '') + (p.stderr or '')).strip()[-600:]}", flush=True)
        return p.returncode
    run = senv_def["FIN_RUN"]
    if os.path.exists(run):
        print(f"STOP: {run} exists; use a fresh label", flush=True)
        return 1
    os.makedirs(run)
    logf = open(os.path.join(run, "arc2_server.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    for d in ("steptimer", "torch", "memstats", "canary"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    kr.write(senv_def["ARC3_S2_DRAFTQSA_SWITCH"], "on\n")
    kr.write(senv_def["ARC3_S2_PRESTAGE_SWITCH"], "on\n")
    base = {k: os.environ[k] for k in kr.BASE_KEYS if k in os.environ}
    senv = dict(base, **senv_def)
    if os.environ.get("LD_LIBRARY_PATH"):
        senv["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    if a.competition:                         # kaggle_run.py's diagnostics=False set
        for k in ("ARC3_S2_STEPTIMER_DIR", "ARC3_S2_STEPTIMER_CPU", "LC_STEPTIMER_DIR", "ARC3_S2_CANARY_DIR",
                  "ARC3_S2_CANARY_MALLOCSTATS", "ARC3_S2_MEMSTATS_DIR"):
            senv.pop(k, None)
        senv.update(ARC3_S2_PLECONV_VERIFY="0", ARC3_S2_FQFAST_VERIFY="0", ARC3_S2_SCHEDFAST_VERIFY="0",
                    ARC3_ENGINE_AUDIT="0", ARC3_K_SERVICE_AUDIT="0", ARC3_S2_MEMDIAG_RETRY="0",
                    ARC3_EVICTION_GRAPH_TRACE="0", ARC3_EVICTION_PROFILE="0")
    senv.update(EXTRA)
    senv.update(l4)
    for k in ("PROFILE", "PROFILE_STEPS", "ARC3_PROFILE_DIR", "ARC3_PROFILE_STEPS"):
        senv.pop(k, None)
    gpu_uuid = subprocess.run(["nvidia-smi", "-i", str(a.gpu).split(",")[0], "--query-gpu=uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")).stdout.strip()
    rec = dict(l4=l4, label=a.label, gpu=a.gpu, gpu_uuid=gpu_uuid, port=a.port, profile=a.profile, profile_values=prof,
               competition=a.competition, extra=EXTRA, host=os.uname().nodename,
               boot_id=open("/proc/sys/kernel/random/boot_id").read().strip(), kdef=a.kdef, kaggle_run=KR,
               env=senv, start_epoch=time.time())
    used, tot = kr.gpu_line(a.gpu.split(",")[0])
    log(f"l4 overrides {l4}")
    log(f"label {a.label}, profile {a.profile} (context {prof['CTX']}, sessions {prof['NL']}, pool lanes {prof['NP']}, "
        f"KV {prof['KVG']} GiB), GPU {a.gpu} {gpu_uuid} (in use {used} of {tot} MiB), port {a.port}, "
        f"{'competition' if a.competition else 'test'} diagnostics")
    p = subprocess.run(["bash", kr.S2 + "/k_runtime_python.sh", kr.S2 + "/k_launch_scaling_l4.py", "start"], cwd=kr.S2, env=senv,
                       capture_output=True, text=True, timeout=900)
    for line in (p.stderr or "").splitlines():
        if line.strip():
            log("  " + line[:300])
    m = re.search(r'\{\s*"pid":\s*(\d+),\s*"log_dir":\s*"([^"]+)"', p.stdout or "")
    if not m:
        log(f"STOP: no launch record (rc {p.returncode}); stdout {(p.stdout or '')[-400:]!r}")
        json.dump(rec, open(os.path.join(run, "arc2_launch.json"), "w"), indent=1, default=str)
        return 2
    pid, log_dir = int(m.group(1)), m.group(2)
    kr.write(os.path.join(kr.S2, "instances", a.label, "last_logdir"), log_dir + "\n")
    rec.update(pid=pid, log_dir=log_dir, launcher_rc=p.returncode)
    json.dump(rec, open(os.path.join(run, "arc2_launch.json"), "w"), indent=1, default=str)
    log(f"server root PID {pid}, log dir {log_dir} (launcher rc {p.returncode})")
    t0, last = time.time(), 0.0
    while True:
        st, _ = kr.get(f"http://127.0.0.1:{a.port}/health")
        if st == 200:
            log(f"server healthy after {time.time() - t0:.0f} s")
            break
        if not kr.alive(pid):
            tail = ""
            try:
                tail = " | ".join(open(os.path.join(log_dir, "server.log"), errors="replace").read().splitlines()[-6:])[-800:]
            except OSError:
                pass
            log(f"STOP: server process {pid} exited before it was healthy: {tail}")
            return 3
        if time.time() - t0 > a.health_timeout_s:
            log(f"STOP: OVERDUE: not healthy after {a.health_timeout_s:.0f} s")
            return 4
        if time.time() - last >= 120:
            last = time.time()
            used, tot = kr.gpu_line(a.gpu.split(",")[0])
            log(f"waiting for health: {time.time() - t0:.0f} s, GPU {a.gpu} memory {used} of {tot} MiB")
        time.sleep(5)
    rec["healthy_after_s"] = round(time.time() - t0, 1)
    if a.nll:
        expect = a.expect_nll or NLL.get(a.profile)
        p = subprocess.run(["bash", kr.S2 + "/nll_panel_i.sh", a.label], cwd=kr.S2, capture_output=True, text=True, timeout=1800,
                           env=dict(base, EXPECT=str(expect), PORT=str(a.port), PYTHONDONTWRITEBYTECODE="1",
                                    IDENTITY=os.path.join(kr.S2, "instances", a.label, "server-identity.json")))
        out = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
        rec["nll"] = out[-3:]
        log(f"NLL panel rc {p.returncode} (expect {expect}): {' | '.join(out[-3:])[:600]}")
        if not any(l.startswith("NLL panel") and "IDENTICAL" in l for l in out):
            log("STOP: NLL fingerprint not reproduced (see the panel output above)")
            json.dump(rec, open(os.path.join(run, "arc2_launch.json"), "w"), indent=1, default=str)
            return 5
    json.dump(rec, open(os.path.join(run, "arc2_launch.json"), "w"), indent=1, default=str)
    log(f"SERVER READY label {a.label} GPU {a.gpu} port {a.port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
