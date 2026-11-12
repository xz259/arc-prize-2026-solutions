"""kaggle_arc3_notebook.py (2026-10-24, Claude main session 099768dc): the ARC3 Kaggle notebook (KAGGLE_PORT_PLAN_2026-10-24.md section 2).
Paste it into a Kaggle notebook (RTX PRO 6000, internet off) whose inputs are the three Kaggle Models (the target
qwen3.8-flash-next-w4a16-intel-ple19, arc3-bundle, arc3-code) and the competition. The "# %%" lines split it into cells.
In a competition rerun (KAGGLE_IS_COMPETITION_RERUN) it plays the gateway for 9 h; otherwise it runs the configured test (default: the
core 55 games offline for 4.5 h, in submission behaviour) and writes the placeholder submission.parquet. Nothing here stops the run:
every check logs and the run continues (the user 10-24: a block fails the submission and scores zero).
v10 (2026-10-28, Claude main session 099768dc), for arc3-code v10 (the engine fork): its definition has two profiles, 82kr1632 and
80kr512, selected by name. The user (10-28): ship v10 with the default 82kr1632; the g4 KV and expert pruning are retired.
v11 (2026-10-28, Claude main session 099768dc), for arc3-code v11: its definition adds the profile 82kx1280 (82kr1632 with a 1,280-row
ring plus the 352-slot exact KV set and the promptmove rerope fix). The user (10-28): idea 2 adopted behind a flag; the default stays
82kr1632. Later on 10-28 (the user): the CKPTFETCH toggle (default on, as shipped) for the startup timing with the prefetch on and off.
v12 (2026-10-28, Claude main session 099768dc), for arc3-code v12: the parallel weight read (the user: "add the parallel PLE read to the
next build"): CKPTFETCH_FILES shards read at once, one stream each (Kaggle's v11b starts read the PLE shards one at a time, about 53 s
per shard); TEARDOWN_MARGIN_S 150 (the user, 10-28; was 300)."""

# %% [1] Configuration
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

NOTEBOOK_START = time.time()
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in ("1", "true")

# --- the two submissions differ here (the user 10-24)
PROFILE = "82kr1632"            # arc3-code v10's profiles, by name: "82kr1632" = lutc 82,528 x 26 + 2 (KVG 17.63), BF16 ring 1,632 rows
                                # (the user 10-28: the shipped default) | "80kr512" = v9's 80k (80,204 x 27 + 2, KVG 18.31) with a
                                # 512-row ring; both with scaled16 + stochastic rounding and the two-kernel ring reader |
                                # "82kx1280" (arc3-code v11; the user 10-28: idea 2, behind this flag) = 82kr1632 with a 1,280-row ring
                                # plus 352 exact slots (64 pinned + 288 promoted by selection rate) and the promptmove rerope fix
FALLBACK_PROFILE = "80kr512"    # used, and logged, when PROFILE is unknown or not in the mounted code's definition (the user's choice);
                                # with arc3-code v9 or older mounted, 80k
PROFILE = str(PROFILE).strip()
USE_RESTART_VALUE = True      # True: restart_value (the default from 10-25: the 9-h FIFO vs restart_value A/B n76, +3.5 est. points);
                                # False: FIFO within each parked band
RV_BY_PROFILE = {               # restart_value's constants per profile: the 10-25 refit (n76 FIFO + restart_value pooled; agent.py's parse
                                # and range check OK; left capped at 1.00); lane_toks per tier by the 10-22 rule (the throughput model's
                                # per-lane exponent): 55 x (29 / sessions)^(1/3)
    "80k": "p=0.48,stuck=0.36,next=0.81,decay=0.81,left=1.00,level_tokens=57661,lane_toks=55",     # v9 only (mounted by mistake)
    "82kr1632": "p=0.48,stuck=0.36,next=0.81,decay=0.81,left=1.00,level_tokens=57661,lane_toks=56",   # arc3-code v10 (its KDEF's
    "80kr512": "p=0.48,stuck=0.36,next=0.81,decay=0.81,left=1.00,level_tokens=57661,lane_toks=55",    # LANE_TOKS per profile)
    "82kx1280": "p=0.48,stuck=0.36,next=0.81,decay=0.81,left=1.00,level_tokens=57661,lane_toks=56",   # arc3-code v11 (82kr1632's lanes)
}
NLL_BY_PROFILE = {"80k": "0.3422915321304067",       # the pod's NLL panel per profile (test runs log the comparison); 80k: v9 only
                  "82kr1632": "0.3471216700030355", "80kr512": "0.3456764103217549",   # v10 (IE's hand-over note, 10-28)
                  "82kx1280": "0.35258596732945363"}  # v11 (the 10-28 A/B arm plf2b01's panel, repeat difference 0)
PROFILE_ASKED = PROFILE
if PROFILE not in RV_BY_PROFILE:  # a name the shipped definition lacks would stop kaggle_run.py (a zero): the fallback instead, logged below
    PROFILE = FALLBACK_PROFILE if FALLBACK_PROFILE in RV_BY_PROFILE else "82kr1632"
RV_CONSTANTS = RV_BY_PROFILE[PROFILE]   # "" = the code's built-in constants (the 10-20 fit; not for a submission)
SHUFFLE_GAMES = False           # False: the games in the order served (offline: the list's order); True: a seeded shuffle
SHUFFLE_SEED = 0
STATEFUL_KERNEL = True          # True (the user 10-25, adopted after the k81s A/B: 69.5 vs 67.7 est., within noise; the kernel healthy):
                                # each conversation's Python calls share one persistent kernel, so variables persist between calls as in
                                # a notebook (ARC3_V80_STATEFUL=1 with its matching prompt lines; needs arc3-code v5); False: a fresh
                                # process per call
CKPTFETCH = True                # True (as shipped): our startup prefetch of the weight shards (ckptfetch, 8 threads, one shard ahead);
                                # False: ARC3_S2_CKPTFETCH_THREADS=0, the loader reads the shards itself. The user 10-15: ckptfetch must be
                                # shown faster than the default loader on Kaggle; compare the server-up time of one run each way
CKPTFETCH_FILES = 8             # arc3-code v12 (the user 10-28: the parallel PLE read): with CKPTFETCH on, this many weight shards are
                                # read at once, one stream each, ahead of the loader (ARC3_S2_CKPTFETCH_FILES); 1 = v11b's one shard
                                # ahead with 8 readers on it. arc3-code v11b or older ignores it

# --- test runs (interactive sessions / "Save & Run All"); a competition rerun always plays the gateway for 9 h
TEST_GAMES = "core55"           # "core55": the core 55 offline from the arc3-core55 dataset (the user 10-25: Save & Run All = the 4.5-h stress
                                # test on the public 25 + copycat 30; scores readable) | "core55-sim": the same games through a local
                                # competition-mode arcade (the gateway code path; baselines hidden, so the local score reads 0) |
                                # "public25": the competition's environment files offline | "gateway": the real gateway
TEST_BUDGET_S = 4.5 * 3600      # the 4.5-h stack test
BUDGET_S = 9 * 3600 if TRUE_SUBMISSION else TEST_BUDGET_S
TEARDOWN_MARGIN_S = 150         # gameplay ends 150 s before the budget (the user 10-28; was 300; measured over 24 pod runs ending at the
                                # gameplay end G: the client exits G-12..G+28 s, the server stops 2-7 s later, kaggle_run done <= G+35 s)
HARD_STOP_S = 60                # if kaggle_run.py still runs this long before the budget's end (G + 90 s): SIGKILL to the client, the
                                # server and kaggle_run.py, and the notebook ends (kaggle_run's own SIGTERM at G + 240 s comes later)
CLIENT_START_S = 600            # the games start 10 min after the notebook start, or as soon as the server answers
ARC_WHEEL_POLICY = "competition-if-different"   # the client's arc_agi: the competition's wheel when its version differs from ours
OUTPUTS_TO_WORKING = not TRUE_SUBMISSION        # test runs keep the run's logs in /kaggle/working/arc3 (about 4 GB per 4.5 h)
USER = "xiaoz259"
LABEL = (("ks" if TRUE_SUBMISSION else "kt") + PROFILE.replace("k", "", 1) + ("r" if USE_RESTART_VALUE else "f")
         + time.strftime("%m%d%H%M", time.gmtime(NOTEBOOK_START)))    # e.g. kt82r1632r10281200 (82kr1632, the default), kt80r512r10281200,
                                                                       # kt82x1280r10281200 (82kx1280)
W = Path("/kaggle/working")
W.mkdir(parents=True, exist_ok=True)
REPORT = dict(label=LABEL, true_submission=TRUE_SUBMISSION, profile=PROFILE, fallback_profile=FALLBACK_PROFILE, use_restart_value=USE_RESTART_VALUE,
              rv_constants=RV_CONSTANTS if USE_RESTART_VALUE else None, shuffle=SHUFFLE_GAMES, stateful_kernel=STATEFUL_KERNEL,
              shuffle_seed=SHUFFLE_SEED, ckptfetch=CKPTFETCH, ckptfetch_files=CKPTFETCH_FILES, budget_s=BUDGET_S,
              notebook_start=NOTEBOOK_START)


def say(msg):
    print("[%s +%4.0fs] %s" % (time.strftime("%H:%M:%S", time.gmtime()), time.time() - NOTEBOOK_START, msg), flush=True)


def sh(cmd, timeout=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def rv_ok(rv):
    """agent.py's own parse and range rules for ARC3_V72_RV (v7.2b): a string that fails them raises at the client's import, so a
    submission would score zero; a failing string falls back to the code's constants (logged)."""
    v = dict(p=0.486, stuck=0.213, next=0.792, decay=0.734, left=0.79, level_tokens=62910.0, lane_toks=55.4)
    try:
        for kv in [x for x in rv.split(",") if x.strip()]:
            k, eq, val = kv.partition("=")
            if not eq or k.strip() not in v:
                return False
            v[k.strip()] = float(val)
    except ValueError:
        return False
    return (0 < v["p"] <= 1 and 0 < v["left"] <= 1 and min(v["stuck"], v["next"], v["decay"]) >= 0 and v["level_tokens"] > 0
            and v["lane_toks"] > 0)


if USE_RESTART_VALUE and RV_CONSTANTS and not rv_ok(RV_CONSTANTS):
    say("RV_CONSTANTS %r FAILS agent.py's parse / range rules: falling back to the code's constants" % RV_CONSTANTS)
    RV_CONSTANTS = ""
    REPORT["rv_constants"] = "(invalid; the code's)"
if PROFILE != PROFILE_ASKED:
    say("PROFILE %r is unknown (%s): using %s" % (PROFILE_ASKED, ", ".join(sorted(RV_BY_PROFILE)), PROFILE))
    REPORT["profile_asked"] = PROFILE_ASKED


def _proc_start(pid):
    """A process's start (epoch s): /proc/stat btime + /proc/<pid>/stat starttime / CLK_TCK; None if unreadable."""
    try:
        btime = next(float(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
        start_ticks = float(open("/proc/%s/stat" % pid).read().rsplit(")", 1)[1].split()[19])
        return btime + start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001
        return None


CLOCK = dict(first_cell=NOTEBOOK_START, kernel=_proc_start("self"), pid1=_proc_start(1))   # 10-25: how long before the first cell
try:                                                       # the session's container ran (input preparation); log only
    CLOCK["host_boot"] = time.time() - float(open("/proc/uptime").read().split()[0])
    CLOCK["kaggle_input_mtime"] = os.stat("/kaggle/input").st_mtime
except Exception:  # noqa: BLE001
    pass
REPORT["clock"] = CLOCK
say("clock: first cell %s; before it: this kernel %s s, the container's PID 1 %s s, the host's boot %s s, /kaggle/input's mtime %s s" % (
    time.strftime("%H:%M:%S", time.gmtime(NOTEBOOK_START)),
    *[("%.0f" % (NOTEBOOK_START - CLOCK[k])) if CLOCK.get(k) else "?" for k in ("kernel", "pid1", "host_boot", "kaggle_input_mtime")]))
say("ARC3 notebook: %s, label %s, profile %s, %s, shuffle %s, stateful kernel %s, ckptfetch %s, budget %.1f h" % (
    "COMPETITION RERUN" if TRUE_SUBMISSION else "test run (%s)" % TEST_GAMES, LABEL, PROFILE,
    ("restart_value (%s)" % (RV_CONSTANTS or "the code's constants")) if USE_RESTART_VALUE else "FIFO",
    ("seed %d" % SHUFFLE_SEED) if SHUFFLE_GAMES else "off", "on" if STATEFUL_KERNEL else "off", ("on, %d shards at once" % CKPTFETCH_FILES) if CKPTFETCH else "off (THREADS=0)",
    BUDGET_S / 3600))

# %% [2] Inputs: the three models, their mounts
def find_model(slug, marker):
    """The model's version directory: /kaggle/input/models/<user>/<slug>/<framework>/<variation>/<version> (the highest version holding
    the marker file); else any directory under /kaggle/input whose path names the slug and holds the marker."""
    best = None
    for c in glob.glob("/kaggle/input/models/*/%s/*/*/*" % slug):
        v = c.rstrip("/").rsplit("/", 1)[-1]
        if os.path.exists(os.path.join(c, marker)) and (best is None or (v.isdigit() and (not best[0].isdigit() or int(v) > int(best[0])))):
            best = (v, c)
    if best:
        return best[1]
    hits = sorted((os.path.dirname(p) for p in glob.glob("/kaggle/input/**/" + marker, recursive=True) if slug in p), key=len)
    return hits[0] if hits else None


TARGET = find_model("qwen3.8-flash-next-w4a16-intel-ple19", "ARC3_TARGET_VIEW.json")
BUNDLE = find_model("arc3-bundle", "MANIFEST.json")
CODE = find_model("arc3-code", "CODE_MANIFEST.json")
CORE55 = next((os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True))
               if "arc3-core55" in p), None)   # the dataset (test runs only)
COMP = next((p for p in glob.glob("/kaggle/input/competitions/*") if os.path.isdir(p)), None) or next(
    (p for p in glob.glob("/kaggle/input/arc-prize-2026*") if os.path.isdir(p)), None)
REPORT["inputs"] = dict(target=TARGET, bundle=BUNDLE, code=CODE, core55=CORE55, competition=COMP)
say("inputs: %s" % REPORT["inputs"])
CODE_INFO = {}                  # which code this run has (10-25): the manifest's build, the archive's sha256 checked, the override md5s
try:
    import hashlib
    _cm = json.load(open(os.path.join(CODE, "CODE_MANIFEST.json")))
    _h = hashlib.sha256()
    with open(os.path.join(CODE, "code.tar.zst"), "rb") as _fh:
        for _b in iter(lambda: _fh.read(64 << 20), b""):
            _h.update(_b)
    CODE_INFO = dict(built=_cm.get("built"), members=_cm.get("members"), sha256=((_cm.get("archive") or {}).get("sha256") or "")[:16],
                     sha256_ok=_h.hexdigest() == (_cm.get("archive") or {}).get("sha256"),
                     overrides={k: v.get("md5") for k, v in (_cm.get("overrides") or {}).items()})
except Exception as exc:  # noqa: BLE001  (a log line, never a stop)
    CODE_INFO = dict(error=repr(exc))
REPORT["code_info"] = CODE_INFO
say("code: built %s, %s members, sha256 %s (archive %s), overrides %s" % (
    CODE_INFO.get("built"), CODE_INFO.get("members"), CODE_INFO.get("sha256"),
    "matches the manifest" if CODE_INFO.get("sha256_ok") else "MISMATCH or unread: %s" % CODE_INFO.get("error", "sha256 differs"),
    {k.rsplit("/", 1)[-1]: v for k, v in (CODE_INFO.get("overrides") or {}).items()}))
REPORT["mounts"] = [l.split()[:3] for l in open("/proc/mounts") if "/kaggle/input" in l]
for m in REPORT["mounts"]:
    say("mount %s" % " ".join(m))

# %% [3] Extract the runtime, venvs, games and code at / (the pod's absolute paths), in parallel
EXTRACT = {}


def untar_zst(archive, dest="/"):
    """tar -x of a zstd-compressed tar under dest: (rc, the tail of tar's messages, how). The zstd CLI when the image has one, else
    Python's zstandard module streamed into tar, across zstd frames (10-25: Kaggle's image has no zstd binary; zstandard 0.25.0 is
    installed). tar's messages go to a temp file, so a chatty tar cannot block the pipe; a truncated stream shows as tar's nonzero rc."""
    if shutil.which("zstd"):
        p = subprocess.run("zstd -dc '%s' | tar -C '%s' -xf -" % (archive, dest), shell=True, capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()[-400:], "zstd CLI"
    import tempfile
    import zstandard
    err = ""
    with tempfile.TemporaryFile() as msg:
        p = subprocess.Popen(["tar", "-C", dest, "-xf", "-"], stdin=subprocess.PIPE, stdout=msg, stderr=subprocess.STDOUT)
        try:
            with open(archive, "rb") as fh:
                rd = zstandard.ZstdDecompressor().stream_reader(fh, read_size=1 << 23, read_across_frames=True)
                while True:
                    b = rd.read(1 << 23)
                    if not b:
                        break
                    p.stdin.write(b)
        except (BrokenPipeError, OSError, zstandard.ZstdError) as exc:
            err = " " + repr(exc)
        finally:
            try:
                p.stdin.close()
            except OSError:
                pass
        rc = p.wait()
        msg.seek(0)
        out = msg.read().decode(errors="replace")
    return (rc or (1 if err else 0)), (out + err).strip()[-400:], "python zstandard " + zstandard.__version__


def extract(name, archive):
    t0 = time.time()
    rc, out, how = untar_zst(archive)
    EXTRACT[name] = dict(rc=rc, seconds=round(time.time() - t0, 1), out=out[-400:], how=how)
    say("extracted %s: rc %d, %.0f s (%s) %s" % (name, rc, time.time() - t0, how, out[-200:]))


for base in ("/workspace", "/mnt/shared"):
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as exc:
        say("cannot create %s: %r (the stack needs the pod's paths)" % (base, exc))
jobs = []
GAMES_SRC = None if TRUE_SUBMISSION else ((CORE55, "core55") if CORE55 else (BUNDLE, "games"))   # one writer per file
for name, d in (("runtime", BUNDLE), ("venvs", BUNDLE), ("code", CODE)) + ((GAMES_SRC[::-1],) if GAMES_SRC else ()):
    a = os.path.join(d or "/nonexistent", name + ".tar.zst")
    if os.path.exists(a):
        jobs.append(threading.Thread(target=extract, args=(name, a)))
    else:
        say("MISSING archive %s" % a)
for j in jobs:
    j.start()
for j in jobs:
    j.join()
REPORT["extract"] = EXTRACT
for _p, _want in (CODE_INFO.get("overrides") or {}).items():   # the final code's K5 files as extracted (10-25), a log line
    try:
        import hashlib
        _got = hashlib.md5(open(_p, "rb").read()).hexdigest()[:8]
    except OSError as exc:
        _got = repr(exc)
    say("code file %s: md5 %s (manifest %s)%s" % (_p, _got, _want, "" if _got == _want else " MISMATCH"))

# %% [4] Links: the runtime links the code archive leaves out, the models, the outputs
def link(target, path):
    try:
        if os.path.islink(path):
            os.unlink(path)
        elif os.path.exists(path):
            say("link %s: a real path exists; kept" % path)
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.symlink(target, path)
    except OSError as exc:
        say("link %s -> %s failed: %r" % (path, target, exc))


A = "/mnt/shared/arc3-inference"
for n in ("runtime", "venv-runtime", "venv-duck", "python", "pub25cc30"):
    link(A + "/" + n, "/workspace/arc3-inference/" + n)
os.makedirs("/workspace/models", exist_ok=True)
link(TARGET or "/nonexistent", "/workspace/models/Qwen3.8-Flash-Next-W4A16-intel")
link(os.path.join(BUNDLE or "/nonexistent", "drafter"), "/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-fr1cfinal")
S2 = "/workspace/kvring_dev/fin74/s2"
OUTS = {"/workspace/kvring_dev/fin74/runs": "runs", S2 + "/instances": "instances", S2 + "/receipts": "receipts",
        S2 + "/client_mtp": "client_mtp", "/workspace/arc3-inference/logs": "server_logs"}
for p, n in OUTS.items():
    if OUTPUTS_TO_WORKING:
        os.makedirs(W / "arc3" / n, exist_ok=True)
        link(str(W / "arc3" / n), p)
    else:
        os.makedirs(p, exist_ok=True)
checks = {p: os.path.exists(p) for p in ("/workspace/arc3-inference/runtime_python.sh", S2 + "/k_launch_scaling_s2i.py", S2 + "/harness_v6/v62/run_v6.py",
                                        "/workspace/arc3-inference/venv-duck/bin/python", "/workspace/arc3-inference/venv-runtime/bin/python",
                                        "/workspace/arc3-inference/runtime/extraction.json", "/workspace/models/Qwen3.8-Flash-Next-W4A16-intel/config.json",
                                        "/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-fr1cfinal/config.json",
                                        "/workspace/kvring_dev/kaggle/kaggle_run.py")}
REPORT["paths"] = checks
say("paths: %s" % ", ".join("%s %s" % (os.path.basename(k), "ok" if v else "MISSING") for k, v in checks.items()))

# %% [5] The client's arc_agi: the competition's wheel when its version differs from the venv's (dfranzen installs it at runtime)
ARC_WHEELS_DIR = ""
VENV_PY = "/workspace/arc3-inference/venv-duck/bin/python"
wheels = sorted(glob.glob((COMP or "/nonexistent") + "/arc_agi_3_wheels/*.whl"))
ours = sorted(os.path.basename(p) for p in glob.glob("/workspace/arc3-inference/venv-duck/lib/python3.12/site-packages/arc*-*.dist-info"))
theirs = sorted(os.path.basename(p) for p in wheels if os.path.basename(p).startswith(("arc_agi-", "arcengine-")))
say("arc wheels: ours %s, the competition's %s" % (ours, theirs))
want = [p for p in wheels if os.path.basename(p).startswith(("arc_agi-", "arcengine-"))]
if ARC_WHEEL_POLICY == "competition-if-different" and want and \
        sorted(x.split("-")[0] + "-" + x.split("-")[1] for x in theirs) != sorted(x.replace(".dist-info", "") for x in ours):
    d = "/tmp/arc3_arcwheels"
    shutil.rmtree(d, ignore_errors=True)
    rc, out = sh("%s -m pip install --quiet --no-deps --no-index --target %s %s" % (sys.executable, d, " ".join(want)), timeout=300)
    rc2, out2 = sh("%s -c 'import sys; sys.path.insert(0, \"%s\"); import arc_agi, arcengine; print(arc_agi.__file__)'" % (VENV_PY, d), timeout=120)
    if rc == 0 and rc2 == 0 and d in out2:
        ARC_WHEELS_DIR = d
        say("the client uses the competition's arc_agi / arcengine (%s)" % out2[-120:])
    else:
        say("the competition's wheels did not install / import (rc %s / %s: %s %s); the client keeps its own" % (rc, rc2, out[-200:], out2[-200:]))
REPORT["arc_wheels"] = dict(ours=ours, competition=theirs, used=ARC_WHEELS_DIR or "venv")

# %% [6] Diagnostics (logged only)
def free_gb(p):
    try:
        u = shutil.disk_usage(p)
        return round(u.free / 1e9, 1)
    except OSError:
        return None


REPORT["host"] = dict(gpu=sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")[1],
                      cpus=len(os.sched_getaffinity(0)), mem_avail_gb=round(int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1048576, 1),
                      disk_free_gb={p: free_gb(p) for p in ("/", "/tmp", "/kaggle/working", "/dev/shm")})
say("host: %s" % REPORT["host"])

# %% [7] Run: the server, then the games (kaggle_run.py: every check logs; status lines every 10 min)
SIM = None
if TRUE_SUBMISSION or TEST_GAMES == "gateway":
    os.environ.setdefault("ARC_API_KEY", "test-key-123")          # the gateway's key, as dfranzen's notebook sets it
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    game_source, games, env_dir = "gateway", "", ""
elif TEST_GAMES == "core55-sim":                                 # taaf/competition_arcade.py: arc_agi.server in competition mode
    SIM_PORT = 8011
    sim_code = ("import time\nfrom taaf.competition_arcade import CompetitionArcadeServer\n"
                "ids = open('/workspace/kvring_dev/fin74/games_a.txt').read().strip().split(',')\n"
                "srv = CompetitionArcadeServer(game_ids=ids, environments_dir='/workspace/arc3-inference/pub25cc30/environment_files', port=%d).start()\n"
                "print('SIM', srv.base_url, len(srv.exposed_game_ids), flush=True)\nwhile True:\n    time.sleep(3600)\n" % SIM_PORT)
    sim_pp = ":".join(x for x in (ARC_WHEELS_DIR, S2 + "/harness_v6/src", S2 + "/harness_v6") if x)
    SIM = subprocess.Popen([VENV_PY, "-c", sim_code], env=dict(os.environ, PYTHONPATH=sim_pp, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg"),
                           stdout=open(W / "simulator.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        if "SIM " in (W / "simulator.log").read_text() or SIM.poll() is not None:
            break
        time.sleep(1)
    say("simulator: %s" % (W / "simulator.log").read_text()[-300:].strip())
    os.environ["ARC_BASE_URL"] = "http://127.0.0.1:%d/" % SIM_PORT
    os.environ["ARC_API_KEY"] = "test-key-123"                   # the simulator's own test key (competition_arcade.DEFAULT_API_KEY)
    game_source, games, env_dir = "gateway", "", ""
elif TEST_GAMES == "public25":
    game_source, games, env_dir = "offline", "@all", os.path.join(COMP or "/nonexistent", "environment_files")
else:
    game_source, games, env_dir = "offline", "@/workspace/kvring_dev/fin74/games_a.txt", "/workspace/arc3-inference/pub25cc30/environment_files"
KDEF_PATH = "/workspace/kvring_dev/kaggle/kaggle_defs/KDEF.json"
try:                                   # 10-25: the mounted code's definition must have PROFILE; another arc3-code version lacks it and
    _profiles = sorted(json.load(open(KDEF_PATH))["profiles"])          # kaggle_run.py would stop on it (a zero): the fallback, logged
    say("code definition: profiles %s (code built %s)" % (_profiles, CODE_INFO.get("built")))
    _fb = next((p for p in (FALLBACK_PROFILE, "80k") if p in _profiles and p in RV_BY_PROFILE), None)
    if PROFILE not in _profiles and _fb:
        say("PROFILE %r is NOT in the mounted code's definition: using %s (attach arc3-code v12)" % (PROFILE, _fb))
        REPORT["profile_fallback"] = dict(asked=PROFILE, used=_fb, profiles=_profiles)
        PROFILE, RV_CONSTANTS = _fb, RV_BY_PROFILE[_fb]
except Exception as exc:  # noqa: BLE001
    say("the code definition %s is unreadable (%r): kaggle_run.py reads it itself" % (KDEF_PATH, exc))
KERNEL_EXT = "/workspace/kvring_dev/fin74/s2/harness_v6/v62/kernel_ext.py"   # 10-25: the stateful kernel (arc3-code v5 and later)
if STATEFUL_KERNEL and not os.path.exists(KERNEL_EXT):
    say("STATEFUL_KERNEL is set but the mounted code has no kernel_ext (arc3-code v4 or older): every Python call runs in a fresh "
        "process (attach arc3-code v5)")
    REPORT["stateful_fallback"] = True
CFG = dict(label=LABEL, profile=PROFILE, gpu="0", port=1377, notebook_start=NOTEBOOK_START, budget_s=BUDGET_S, teardown_margin_s=TEARDOWN_MARGIN_S,
           client_start_s=CLIENT_START_S, game_source=game_source, games=games, environments_dir=env_dir, use_restart_value=USE_RESTART_VALUE,
           rv_constants=RV_CONSTANTS, shuffle=SHUFFLE_GAMES, shuffle_seed=SHUFFLE_SEED, server_wait_s=7200, relaunch_max=1,
           diagnostics=not TRUE_SUBMISSION,   # the user 10-24: the read-only diagnostics off in every scored submission, on in test runs
           engine_audit=not TRUE_SUBMISSION,
           status_every_s=600, arc_wheels_dir=ARC_WHEELS_DIR, kdef=KDEF_PATH,
           extra_client_env={"ARC3_V80_STATEFUL": "1" if STATEFUL_KERNEL else "0"},   # the stateful kernel (kernel_ext; v5)
           extra_server_env=(({"ARC3_S2_CKPTFETCH_FILES": str(CKPTFETCH_FILES)} if CKPTFETCH_FILES > 1 else {}) if CKPTFETCH
                             else {"ARC3_S2_CKPTFETCH_THREADS": "0"}),  # v12: shards at once; ckptfetch off: the loader's own reads
           nll_panel=not TRUE_SUBMISSION,   # test runs: the NLL fingerprint on Kaggle's GPU (log only; only if the server is up before the client starts)
           nll_expect=NLL_BY_PROFILE.get(PROFILE, NLL_BY_PROFILE["80k"]))   # the pod's panel for the tier that runs
REPORT["tier_run"] = dict(profile=PROFILE)
REPORT["run_config"] = CFG
cfg_path = W / ("kaggle_run_%s.json" % LABEL)
cfg_path.write_text(json.dumps(CFG, indent=1))
(W / ("notebook_report_%s.json" % LABEL)).write_text(json.dumps(REPORT, indent=1, default=str))
say("kaggle_run.py %s" % cfg_path)
proc = subprocess.Popen([sys.executable, "-u", "/workspace/kvring_dev/kaggle/kaggle_run.py", str(cfg_path)], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
RUN_PIDS = {}


def _reader():                         # kaggle_run.py's lines, read in a thread: the hard stop below never waits on the pipe
    for line in proc.stdout:
        print(line, end="", flush=True)
        for key, tag in (("server", "server root PID "), ("client", "client started (")):
            if tag in line and "PID " in line:
                try:
                    RUN_PIDS[key] = int(line.split("PID ", 1)[1].split(",")[0].split()[0])
                except (IndexError, ValueError):
                    pass


_rd = threading.Thread(target=_reader, daemon=True)
_rd.start()
T_HARD = NOTEBOOK_START + BUDGET_S - HARD_STOP_S
while proc.poll() is None and time.time() < T_HARD:
    time.sleep(2)
if proc.poll() is None:
    say("HARD STOP %d s before the budget's end: kaggle_run.py still running; SIGKILL to the client %s, the server %s and kaggle_run.py" % (
        HARD_STOP_S, RUN_PIDS.get("client"), RUN_PIDS.get("server")))
    REPORT["hard_stop_s"] = time.time() - NOTEBOOK_START
    for _pg in (RUN_PIDS.get("client"), RUN_PIDS.get("server")):
        if _pg:
            try:
                os.killpg(_pg, 9)
            except OSError:
                pass
    try:
        proc.kill()
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass
_rd.join(timeout=5)
REPORT["kaggle_run_rc"] = proc.poll()
say("kaggle_run.py rc %s" % REPORT["kaggle_run_rc"])

# %% [8] End: the placeholder submission for runs that do not score through the gateway; the report
if SIM is not None:
    try:
        os.killpg(SIM.pid, 15)
    except OSError:
        pass
if not TRUE_SUBMISSION:
    try:
        import pandas as pd
        pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(W / "submission.parquet", index=False)
        say("wrote the placeholder submission.parquet")
    except Exception as exc:  # noqa: BLE001
        say("submission.parquet not written: %r" % exc)
try:
    REPORT["kaggle_run"] = json.load(open("/workspace/kvring_dev/fin74/runs/%s/kaggle_run.json" % LABEL))
except Exception as exc:  # noqa: BLE001
    REPORT["kaggle_run"] = repr(exc)
(W / ("notebook_report_%s.json" % LABEL)).write_text(json.dumps(REPORT, indent=1, default=str))
say("done, %.2f h after the notebook start" % ((time.time() - NOTEBOOK_START) / 3600))
