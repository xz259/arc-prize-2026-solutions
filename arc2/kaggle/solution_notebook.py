"""kaggle_arc2_notebook.py (2026-10-31, ARC2 - main; the user: "a provisional version for kaggle submission tomorrow afternoon"): the
ARC-2 Kaggle notebook for 4x L4 (internet off). Paste it into a notebook whose inputs are the three Kaggle Models (the target
qwen3.8-flash-next-w4a16-intel-ple19, arc3-bundle, arc2-code-v2j) and the competition; the "# %%" lines split it into cells. Modeled on
the ARC-3 notebook (inference_stack/kaggle_upload/kaggle_arc3_notebook_v16.py: inputs, extraction, links, logs, hard stop).
In a competition rerun (KAGGLE_IS_COMPETITION_RERUN) it solves the test challenges for BUDGET_S; otherwise the configured test
(default: the evaluation set with its solutions, so the harness's score.json is readable). Either way /kaggle/working/submission.json
holds every task of the challenges file from the first minutes on (placeholders, then the harness's answers merged in).
Nothing here stops the run: every check logs and the run continues (a block fails the submission and scores zero).
The server: one TP2 x PP2 server on the four L4s (profile g60f8inp128: 60K context, MTP K 4, prune 128, FP8 tier A_in), the L4
production path of arc2_stack/l4port (L4PORT_STATE_2026-10-31.md); the harness: harness_v2p (P3_ORDER, PASS2_FRAME; harness_v2n the fallback) (v2m's passes until two valid answers, at most 4;
then one extra pass for identical answers or one review for distinct ones when the mean of passes 1-2 reaches 80K; that order of
priority after the second passes; rank policy, KV admission)."""

# %% [1] Configuration
import glob
import hashlib
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

PROFILE = "g60f8inp128"         # v2h: KDEF_sp20's 60K profile: context 60,240, MTP K 4, prune 128, FP8 tier A_in (its expert set:
                                # L4_SIZING). v2g's g605f8inp128 (context 60,496: the 19th attention page full) ended most turns past an
                                # eviction with the engine's finish reason 'error' (HTTP 500) on the pods; this profile evicts cleanly
L4_LANES = 35                   # v2h (the user, 11-01: "Default to 35 lanes"): KV 4.70 GiB = 35.00x: v2f's sizing, with its compile
                                # caches in the package (the gate ka2r07). 36 = KV 4.85 GiB = 36.13x: on Kaggle it had 12 allocator retries
                                # on GPU 0 in the opening prefill burst (Kaggle's CUDA graphs take 0.63 GiB per worker against the pods'
                                # 0.43), and at this profile's context the package has no 36-lane compile cache (Kaggle compiles it)
L4_SIZING = {                   # v2d (2026-10-31): v2's sizing (split 26/22, the microbatch balancer, batched tokens 2048, the user's
    # train120 + eval120 expert set) plus the PLE host-poll TP handoff (bootstrap_l4prod5: the offload process off GPUs 0-1), the
    # capture-list trim without the 60-75 sizes (the user, 10-31 21:2x), the indexer dedupe (the QSA indexer's compressed keys split
    # by block parity over the TP ranks: x1.098 KV tokens; exact selection), explicit KV 4.60 GiB: 34.26x -> 34 lanes (max sequences
    # = sessions = 34; 4.70 gives 35.00x but the 35th lane is capped, so the 0.1 GiB stays as headroom for the +191 MiB transients;
    # the 85 capture size covers 2 microbatches of 17 requests x 5 tokens). Equal to the traced rehearsal's l4_sets (ka2r05).
    # v2f (2026-11-01, the user's calls): (1) the Kaggle decode-gap fix: bootstrap v13 + ARC3_L4PORT_PP_DEFER=1 (stage 0's PP token /
    # count / draft receives posted at the consuming step; on Kaggle's driver the early receive spun through the next forward and
    # stalled its CUDA graph 45-81 ms per step: ~515 vs ~806 tok/s; the pods with one hardware queue: 527 vs 911, with the fix 904);
    # (2) 35 lanes: KV 4.70 GiB = 35.00x, max sequences = sessions = 35, the 90 capture size for the 18-request microbatch; headroom
    # from expandable segments (peaks -76 to -197 MiB) and the QSA logits workspace at 16 MiB (-76 / -120 MiB, exact); (3) the
    # determinism fixes: marlindet's canonical expert order under EP and the sorted QSA top-k (bitwise reruns on the PP path).
    # v2g (2026-11-01): the 36-lane sizing (KV 4.85 GiB; the pods' 45-min soak pvf3 ran it at this profile); L4_FALLBACK_35 below
    # gives 35 lanes, the default since v2h (L4_LANES).
    "ARC3_STAGE4_UTIL": "0.89", "ARC3_STAGE4_MAX_SEQS": "36", "ARC3_S2_CAPTURE": "1,2,4,5,8,10,16,20,40,80,85,90",
    "ARC3_L4PORT_EVPP": "1", "ARC3_L4PORT_PP_SPLIT": "26,22", "ARC3_STAGE4_KV_GIB": "4.85", "ARC3_L4PORT_PP_BALANCE": "1",
    "ARC3_STAGE2_BATCHED_TOKENS": "2048",
    "ARC3_S2_EPRUNE_FILE": "/mnt/shared/arc2/l4port/sidepod/prune_sets_hope/eprune_freq_k128.json",
    "ARC3_PY_PREPEND": "/mnt/shared/arc2/l4port/bootstrap_l4prod8", "ARC3_PLE_HANDOFF_IMPL": "hostpoll", "ARC3_L4PORT_PLE_HPTP": "1",
    "ARC3_L4PORT_IDX_SHARD": "1", "ARC3_L4PORT_IDX_SHARD_MERGE": "keys", "ARC3_PERSISTENT_MAX_SESSIONS": "36",
    "ARC3_L4PORT_PP_DEFER": "1", "ARC3_S2_ALLOC_CONF": "expandable_segments:True", "ARC3_L4PORT_QSA_WS_MIB": "16",
    "ARC3_L4PORT_MARLINDET_EP": "1", "ARC3_L4PORT_QSA_SORT": "1",
}
L4_FALLBACK_35 = {"ARC3_STAGE4_MAX_SEQS": "35", "ARC3_PERSISTENT_MAX_SESSIONS": "35", "ARC3_STAGE4_KV_GIB": "4.70"}   # L4_LANES = 35
assert L4_LANES in (35, 36), "L4_LANES: 36 or 35 (the package has these two sizings' compile caches)"
if L4_LANES == 35:
    L4_SIZING = dict(L4_SIZING, **L4_FALLBACK_35)
IDXSHARD_V3 = True              # v2j (11-01): the L4 bootstrap v14 (bootstrap_l4prod9): the indexer dedupe's sharded QSA selection
                                # behind production's eviction wrapper (l4prod/l4port_idxshard_v3.py). v2d-v2i's shard skipped it, so a
                                # session's selection after its first eviction ran without the remap to resident indices or the
                                # frequency counts (passes degraded after their first eviction). False = v2i's bootstrap_l4prod8
L4_SIZING = dict(L4_SIZING, ARC3_PY_PREPEND="/mnt/shared/arc2/l4port/bootstrap_l4prod9" if IDXSHARD_V3
                 else "/mnt/shared/arc2/l4port/bootstrap_l4prod8")
LANES = "auto"                  # "auto": floor of vLLM's maximum concurrency at the profile's context (every lane fits at full length),
LANES_CAP = L4_LANES            # capped at the server's max sequences
CODE_MODEL = "arc2-code-v2j"    # the code package's Kaggle Model slug: package v2j as its own model, the version in the name (the
                                # user, 11-01); arc2-code-v2i holds v2i, arc2-code-v2h v2h, arc2-code-ver v2g, arc2-code v2d / v2e
TEST_TASKS = "eval"             # test runs: "eval" = the competition's evaluation challenges with their solutions (a local score) |
                                # "test" = the test challenges (no score) | "eval8" = the first 8 evaluation tasks (a short smoke)
TEST_BUDGET_S = 2.0 * 3600      # a test run's budget from the first cell
KAGGLE_LIMIT_S = 12 * 3600      # the competition's run limit (the user, 10-31: "The budget is 12h. 11.5 is excluding the server start up
                                # and teardown")
SAFETY_MARGIN_S = 0             # the user (10-31): "I think 12 is ok. 150s teardown is more than enough" (the L4 rehearsal ka2r05: the
                                # driver ended ~100 s before its budget's end, the harness 45 s past its deadline, then SIGTERM)
BUDGET_S = (KAGGLE_LIMIT_S - SAFETY_MARGIN_S) if TRUE_SUBMISSION else TEST_BUDGET_S   # from the first cell: 12 h; the server's start
                                # (10-30 min on Kaggle) and the 150-s teardown come out of it, the rest is solving time
TEARDOWN_MARGIN_S = 150         # the harness's attempts end this long before the budget's end (the user 10-31: 2.5 min, as ARC-3's
                                # gameplay); then the submission's last merge (seconds) and the server's stop
HARD_STOP_S = 60                # if the driver still runs this long before the budget's end: SIGKILL to it, the harness and the server
SERVER_WAIT_S = (BUDGET_S - 3600) if TRUE_SUBMISSION else 4800   # the server must be healthy within this long after the first cell,
                                # else the placeholders stay (a rerun waits as long as a start can still pay off: until 1 h before the end)
CKPTFETCH = False               # False (the first Kaggle starts; the reviewer, 10-31): ARC3_S2_CKPTFETCH_THREADS=0, vLLM's own loader, each
                                # worker reading only the tensors it maps | True (the A/B arm, the pod's setting): ckptfetch's prefetch, 8
                                # files at once, where each of the four workers prefetches the whole 73.6 GB target checkpoint (the pod's
                                # receipts) into a page cache of ~60 GB on Kaggle (188 GiB less ~127 GB of server memory): re-reads likely.
                                # The user's rule (10-15): ckptfetch must be shown faster on Kaggle. The PLE worker's 105 GB read is
                                # unaffected. On a FUSE input mount it is off whatever is set here (the ARC-3 v14 guard, below)
CKPTFETCH_FILES = 4             # ARC-3 v12 (the parallel shard read): with CKPTFETCH on, this many weight shards are read at once, one
                                # stream each, ahead of the loader (ARC3_S2_CKPTFETCH_FILES); 1 = one shard ahead. Inert while CKPTFETCH is off
PLE_COPY_THREADS = 16           # ARC-3 v13 (v2d: ckptfetch v14 in this package): the PLE offload worker copies each checkpoint tensor's
                                # rows into its ~105 GB host table on this many threads, so the page faults of the mmap'd shard overlap
                                # (ARC3_S2_PLECOPY_THREADS; active with CKPTFETCH off too). The same bytes; 1 = the single-threaded copy
SHARD_FAULT_THREADS = 16        # ARC-3 v14: while a server worker loads a weight shard, this many threads fault its bytes in through the
                                # page cache (ARC3_S2_FAULTIN_THREADS). It rides on ckptfetch's loader, so it is inert while CKPTFETCH is
                                # off (the v2d default: under PP each of the four workers would prefetch the whole checkpoint)
PASS2_FRAME = "off"             # v2h (the user, 11-01: "Just add it as a flag"; an A/B flag): "colour" runs pass 2 of every task on a copy
                                # with the colours 1-9 permuted (0 kept); its answers are mapped back before they are scored and voted
                                # on (harness v2o; the pods' 6-h A/B arm g6b). "colour_symmetry" also rotates or reflects the copy.
                                # "off" (shipped): pass 2 on the puzzle itself
P3_ORDER = "more1,third,review,more0"   # v2j (the user, 11-01 18:1x: the swap): phase 3's classes by points per M tokens over six
                                # eval runs: a further pass of a task with one valid pass (0.66), the extra pass of identical answers
                                # (0.40), the review of distinct answers (0.21), a further pass of a task with none (-1.62). v2i
                                # shipped "third,more1,review,more0"; "third,more,review" = v2h's order. Harness v2p reads it
OUTPUTS_TO_WORKING = not TRUE_SUBMISSION   # test runs keep the harness's run directory and the server logs in /kaggle/working/arc2
LABEL = ("ka2s" if TRUE_SUBMISSION else "ka2t") + time.strftime("%m%d%H%M", time.gmtime(NOTEBOOK_START))
W = Path("/kaggle/working")
W.mkdir(parents=True, exist_ok=True)
SUBMISSION = str(W / "submission.json")
REPORT = dict(label=LABEL, true_submission=TRUE_SUBMISSION, profile=PROFILE, l4_sizing=L4_SIZING, lanes=LANES, code_model=CODE_MODEL,
              test_tasks=None if TRUE_SUBMISSION else TEST_TASKS, budget_s=BUDGET_S, notebook_start=NOTEBOOK_START)


def say(msg):
    print("[%s +%4.0fs] %s" % (time.strftime("%H:%M:%S", time.gmtime()), time.time() - NOTEBOOK_START, msg), flush=True)


def sh(cmd, timeout=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


say("ARC2 notebook: %s, label %s, profile %s, sizing %s, lanes %s, budget %.2f h" % (
    "COMPETITION RERUN" if TRUE_SUBMISSION else "test run (%s)" % TEST_TASKS, LABEL, PROFILE, L4_SIZING, LANES, BUDGET_S / 3600))

# %% [2] Inputs: the three models, the competition
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
CODE = find_model(CODE_MODEL, "CODE_MANIFEST.json")
COMP = next((os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True))), None)
REPORT["inputs"] = dict(target=TARGET, bundle=BUNDLE, code=CODE, competition=COMP)
say("inputs: %s" % REPORT["inputs"])
CODE_INFO = {}
try:
    _cm = json.load(open(os.path.join(CODE, "CODE_MANIFEST.json")))
    _h = hashlib.sha256()
    with open(os.path.join(CODE, "code.tar.zst"), "rb") as _fh:
        for _b in iter(lambda: _fh.read(64 << 20), b""):
            _h.update(_b)
    CODE_INFO = dict(built=_cm.get("built"), members=_cm.get("members"), sha256=((_cm.get("archive") or {}).get("sha256") or "")[:16],
                     sha256_ok=_h.hexdigest() == (_cm.get("archive") or {}).get("sha256"), runtime_patch=_cm.get("runtime_patch"))
except Exception as exc:  # noqa: BLE001  (a log line, never a stop)
    CODE_INFO = dict(error=repr(exc))
REPORT["code_info"] = CODE_INFO
say("code: built %s, %s members, sha256 %s (archive %s)" % (
    CODE_INFO.get("built"), CODE_INFO.get("members"), CODE_INFO.get("sha256"),
    "matches the manifest" if CODE_INFO.get("sha256_ok") else "MISMATCH or unread: %s" % CODE_INFO.get("error", "sha256 differs")))
if TRUE_SUBMISSION or TEST_TASKS == "test":
    CHALLENGES, SOLUTIONS, LIMIT = os.path.join(COMP or "/nonexistent", "arc-agi_test_challenges.json"), "", 0
else:
    CHALLENGES = os.path.join(COMP or "/nonexistent", "arc-agi_evaluation_challenges.json")
    SOLUTIONS = os.path.join(COMP or "/nonexistent", "arc-agi_evaluation_solutions.json")
    LIMIT = 8 if TEST_TASKS == "eval8" else 0
    if not os.path.exists(SOLUTIONS):
        SOLUTIONS = ""
try:                            # placeholders for every task right away (the reviewer, 10-31: insurance before anything else can fail)
    _tasks0 = json.load(open(CHALLENGES))
    _n = len(_tasks0)
    with open(SUBMISSION + ".tmp", "w") as _fh:
        json.dump({_tid: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in (_t.get("test") or [{}])] for _tid, _t in _tasks0.items()}, _fh)
    os.replace(SUBMISSION + ".tmp", SUBMISSION)
except Exception as exc:  # noqa: BLE001
    _n = repr(exc)
REPORT["tasks"] = dict(challenges=CHALLENGES, solutions=SOLUTIONS, limit=LIMIT, n=_n)
say("tasks: %s (%s tasks; placeholder submission.json written), solutions %s, limit %s" % (CHALLENGES, _n, SOLUTIONS or "none",
                                                                                         LIMIT or "none"))
try:
    for m in [l.split()[:3] for l in open("/proc/mounts") if "/kaggle/input" in l]:
        say("mount %s" % " ".join(m))
except OSError:
    pass


def _fs_type(path, mounts="/proc/mounts"):
    """The filesystem type of the mount holding path (the longest matching mount point), or None (ARC-3 v14's FUSE guard)."""
    try:
        real, best = os.path.realpath(path), (-1, None)
        for line in open(mounts):
            f = line.split()
            if len(f) < 3:
                continue
            mp = f[1].replace("\\040", " ")
            if (real == mp or real.startswith(mp.rstrip("/") + "/")) and len(mp) > best[0]:
                best = (len(mp), f[2])
        return best[1]
    except Exception:  # noqa: BLE001  (a log line, never a stop)
        return None


INPUT_FS = {name: _fs_type(path) for name, path in (("target", TARGET), ("bundle", BUNDLE)) if path}
REPORT["input_fs"] = INPUT_FS
_fuse = sorted(n for n, t in INPUT_FS.items() if t and t.lower().startswith("fuse"))
if _fuse and CKPTFETCH:          # on FUSE the read-ahead does not stay in the page cache, so it would read every shard twice
    CKPTFETCH = False
    REPORT["ckptfetch_guard"] = "off: FUSE mount for %s" % ", ".join(_fuse)
REPORT["ckptfetch"] = CKPTFETCH
REPORT["loading"] = dict(ckptfetch=CKPTFETCH, ckptfetch_files=CKPTFETCH_FILES, ple_copy_threads=PLE_COPY_THREADS,
                         shard_fault_threads=SHARD_FAULT_THREADS)
say("input filesystems %s: ckptfetch %s; PLE copy %s; shard fault-in %s" % (
    INPUT_FS, ("on, %d shards at once" % CKPTFETCH_FILES) if CKPTFETCH else ("OFF (FUSE for %s)" % ", ".join(_fuse) if _fuse else "off (as set)"),
    ("%d threads" % PLE_COPY_THREADS) if PLE_COPY_THREADS > 1 else "off",
    ("%d threads" % SHARD_FAULT_THREADS) if SHARD_FAULT_THREADS > 1 and CKPTFETCH else "off (needs ckptfetch)"))

# %% [3] Extract the runtime and venvs (in parallel), then the code, at / (the pod's absolute paths)
EXTRACT = {}


def untar_zst(archive, dest="/"):
    """tar -x of a zstd-compressed tar under dest: (rc, the tail of tar's messages, how). The zstd CLI when the image has one, else
    Python's zstandard module streamed into tar, across zstd frames (Kaggle's image has no zstd binary; zstandard is installed)."""
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
for name, d in (("runtime", BUNDLE), ("venvs", BUNDLE)):
    a = os.path.join(d or "/nonexistent", name + ".tar.zst")
    if os.path.exists(a):
        jobs.append(threading.Thread(target=extract, args=(name, a)))
    else:
        say("MISSING archive %s" % a)
for j in jobs:
    j.start()
for j in jobs:
    j.join()
_a = os.path.join(CODE or "/nonexistent", "code.tar.zst")     # after the runtime: the runtime patch below copies over its files
if os.path.exists(_a):
    try:                                                      # the main thread: an exception here would end the notebook
        extract("code", _a)
    except Exception as exc:  # noqa: BLE001
        EXTRACT["code"] = dict(error=repr(exc))
        say("extracting the code FAILED: %r" % exc)
else:
    say("MISSING archive %s" % _a)
REPORT["extract"] = EXTRACT

# %% [4] Links: the runtime, venvs and models at the pod's paths; the L4 port's runtime patch; the outputs
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
for n in ("runtime", "venv-runtime", "venv-duck", "python"):
    link(A + "/" + n, "/workspace/arc3-inference/" + n)
os.makedirs("/workspace/models", exist_ok=True)
link(TARGET or "/nonexistent", "/workspace/models/Qwen3.8-Flash-Next-W4A16-intel")          # the harness's tokenizer
link(TARGET or "/nonexistent", "/workspace/models/Qwen3.8-Flash-Next-W4A16-intel-ple19")    # the server's model path (the pod's)
link(os.path.join(BUNDLE or "/nonexistent", "drafter"), "/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-fr1cfinal")
# the L4 port's runtime patch (the pod's private runtime differs from the shared one in these vLLM files: l4port_patch_01 / 02c / 04
# / 05): each file's original is checked against the manifest's sha256 (a log line), then the patched copy replaces it
PATCH = dict(applied=0, original_differs=[], missing=[])
_rp = "/mnt/shared/arc2/kaggle/runtime_patch"
try:
    for rel, d in sorted(json.load(open(os.path.join(_rp, "RUNTIME_PATCH.json")))["files"].items()):
        dst, src = os.path.join(A, "runtime", rel), os.path.join(_rp, "files", rel)
        try:
            if hashlib.sha256(open(dst, "rb").read()).hexdigest() != d["original_sha256"]:
                PATCH["original_differs"].append(rel)
            shutil.copy2(src, dst + ".l4tmp")
            os.replace(dst + ".l4tmp", dst)
            ok = hashlib.sha256(open(dst, "rb").read()).hexdigest() == d["patched_sha256"]
            PATCH["applied"] += int(ok)
            if not ok:
                PATCH["missing"].append(rel + " (patched sha256 differs)")
        except OSError as exc:
            PATCH["missing"].append("%s (%r)" % (rel, exc))
except Exception as exc:  # noqa: BLE001
    PATCH["error"] = repr(exc)
REPORT["runtime_patch"] = PATCH
say("runtime patch: %s" % PATCH)
S2 = "/workspace/kvring_dev/fin74/s2"
OUTS = {"/mnt/shared/arc2/runs": "runs", "/workspace/kvring_dev/fin74/runs": "fin_runs", S2 + "/instances": "instances",
        S2 + "/receipts": "receipts", "/workspace/arc3-inference/logs": "server_logs"}
for p, n in OUTS.items():
    try:
        if OUTPUTS_TO_WORKING:
            os.makedirs(W / "arc2" / n, exist_ok=True)
            link(str(W / "arc2" / n), p)
        else:
            os.makedirs(p, exist_ok=True)
    except OSError as exc:
        say("output dir %s: %r" % (p, exc))
checks = {p: os.path.exists(p) for p in ("/workspace/arc3-inference/runtime_python.sh", S2 + "/k_launch_scaling_l4.py",
                                        S2 + "/k_runtime_python.sh", S2 + "/harness_v6/v62/transport.py",
                                        "/workspace/arc3-inference/venv-duck/bin/python", "/workspace/arc3-inference/venv-runtime/bin/python",
                                        "/workspace/models/Qwen3.8-Flash-Next-W4A16-intel-ple19/config.json",
                                        "/workspace/models/Qwen3.8-Flash-Next-W4A16-intel/tokenizer.json",
                                        "/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-fr1cfinal/config.json",
                                        "/workspace/kvring_dev/kaggle/kaggle_run.py", "/mnt/shared/arc2/kaggle/kaggle_run_arc2_v2f.py",
                                        "/mnt/shared/arc2/l4port/l4prod/arc2_server_l4.py", "/mnt/shared/arc2/l4port/sidepod/KDEF_sp20.json",
                                        "/mnt/shared/arc2/harness_v2n/arc2/runner.py", "/mnt/shared/arc2/harness_v2p/arc2/runner.py",
                                        "/mnt/shared/arc2/l4port/bootstrap_l4prod9/sitecustomize.py",
                                        "/mnt/shared/arc2/l4port/l4prod/l4port_idxshard_v3.py",
                                        CHALLENGES)}
REPORT["paths"] = checks
say("paths: %s" % ", ".join("%s %s" % (os.path.basename(k), "ok" if v else "MISSING") for k, v in checks.items()))

# %% [5] Diagnostics (logged only)
def free_gb(p):
    try:
        return round(shutil.disk_usage(p).free / 1e9, 1)
    except OSError:
        return None


try:
    REPORT["host"] = dict(gpu=sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", timeout=60)[1],
                          cpus=len(os.sched_getaffinity(0)), mem_avail_gb=round(int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1048576, 1),
                          disk_free_gb={p: free_gb(p) for p in ("/", "/tmp", "/kaggle/working", "/dev/shm")})
except Exception as exc:  # noqa: BLE001  (a log line, never a stop)
    REPORT["host"] = repr(exc)
say("host: %s" % REPORT["host"])

# %% [6] Run: the server on the four GPUs, then the harness (kaggle_run_arc2.py: every check logs; status lines every 10 min)
CFG = dict(label=LABEL, profile=PROFILE, kdef="/mnt/shared/arc2/l4port/sidepod/KDEF_sp20.json", gpus="0,1,2,3", port=1390,
           notebook_start=NOTEBOOK_START, budget_s=BUDGET_S, teardown_margin_s=TEARDOWN_MARGIN_S, server_wait_s=SERVER_WAIT_S,
           relaunch_max=1, relaunch_mid_max=1, competition=TRUE_SUBMISSION, l4_sets=L4_SIZING, drop_parts=["ctxfree"],
           extra_server_env=dict((({"ARC3_S2_CKPTFETCH_FILES": str(CKPTFETCH_FILES)} if CKPTFETCH_FILES > 1 else {}) if CKPTFETCH
                                  else {"ARC3_S2_CKPTFETCH_THREADS": "0"}),          # ckptfetch off: the loader's own reads
                                 **({"ARC3_S2_PLECOPY_THREADS": str(PLE_COPY_THREADS)} if PLE_COPY_THREADS > 1 else {}),       # v13
                                 **({"ARC3_S2_FAULTIN_THREADS": str(SHARD_FAULT_THREADS)} if SHARD_FAULT_THREADS > 1 else {})),  # v14
           challenges=CHALLENGES, solutions=SOLUTIONS, limit=LIMIT, lanes=LANES, lanes_cap=LANES_CAP,
           harness_dir="/mnt/shared/arc2/harness_v2n",      # v2g: harness v2n (the reviewer's input 22: answers saved by
                                                           # passes open at the stop are recorded; the review pair)
           submission_path=SUBMISSION, runs_dir="/mnt/shared/arc2/runs", status_every_s=600, merge_every_s=60)
try:                                    # v2i: harness v2p with the driver's own HARNESS_ARGS (read from its source), its phase-3
    import ast                          # order replaced by P3_ORDER, and --pass2-frame when PASS2_FRAME is set
    _drv = ast.parse(open("/mnt/shared/arc2/kaggle/kaggle_run_arc2_v2f.py").read())
    _ha = [ast.literal_eval(n.value) for n in _drv.body if isinstance(n, ast.Assign) and len(n.targets) == 1
           and getattr(n.targets[0], "id", "") == "HARNESS_ARGS"]
    assert len(_ha) == 1 and "--phase3-priority" in _ha[0] and "--pass2-frame" not in _ha[0], len(_ha)
    assert sorted(P3_ORDER.split(",")) in (["more", "review", "third"], ["more0", "more1", "review", "third"]), P3_ORDER
    _args = list(_ha[0])
    _args[_args.index("--phase3-priority") + 1] = P3_ORDER
    if PASS2_FRAME != "off":
        if PASS2_FRAME in ("colour", "colour_symmetry"):
            _args += ["--pass2-frame", PASS2_FRAME]
        else:
            say("PASS2_FRAME %r not applied (colour or colour_symmetry); pass 2 runs on the puzzle itself" % PASS2_FRAME)
    CFG.update(harness_dir="/mnt/shared/arc2/harness_v2p", harness_args=_args)
    say("harness v2p: phase-3 order %s, PASS2_FRAME %s, %d harness arguments (the driver's %d)" % (
        P3_ORDER, PASS2_FRAME, len(_args), len(_ha[0])))
except Exception as exc:  # noqa: BLE001  (a log line, never a stop: the run keeps v2h's harness v2n and the driver's arguments)
    say("harness v2p not configured: %r; the run keeps harness v2n with the driver's arguments (order third,more,review)" % exc)
REPORT["harness"] = dict(dir=CFG["harness_dir"], p3_order=P3_ORDER, pass2_frame=PASS2_FRAME, args=CFG.get("harness_args"))
REPORT["idxshard_v3"] = dict(on=IDXSHARD_V3, py_prepend=L4_SIZING["ARC3_PY_PREPEND"])
REPORT["run_config"] = CFG
cfg_path = W / ("kaggle_run_arc2_%s.json" % LABEL)
RUN_PIDS = {}
proc = None
try:
    cfg_path.write_text(json.dumps(CFG, indent=1))
    (W / ("notebook_report_%s.json" % LABEL)).write_text(json.dumps(REPORT, indent=1, default=str))
    say("kaggle_run_arc2_v2f.py %s" % cfg_path)
    proc = subprocess.Popen([sys.executable, "-u", "/mnt/shared/arc2/kaggle/kaggle_run_arc2_v2f.py", str(cfg_path)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
except Exception as exc:  # noqa: BLE001  (the placeholders stay; cell 7 still runs)
    say("the driver did not start: %r" % exc)


def _reader():                         # the driver's lines, read in a thread: the hard stop below never waits on the pipe
    for line in proc.stdout:
        print(line, end="", flush=True)
        for key, tag in (("server", "server root PID "), ("harness", "harness started: PID ")):
            if tag in line:
                try:
                    RUN_PIDS[key] = int(line.split(tag, 1)[1].split(",")[0].split()[0])
                except (IndexError, ValueError):
                    pass


_rd = threading.Thread(target=_reader, daemon=True)
if proc is not None:
    _rd.start()
T_HARD = NOTEBOOK_START + BUDGET_S - HARD_STOP_S
while proc is not None and proc.poll() is None and time.time() < T_HARD:
    time.sleep(2)
if proc is not None and proc.poll() is None:
    say("HARD STOP %d s before the budget's end: the driver still runs; SIGKILL to the harness %s, the server %s and the driver" % (
        HARD_STOP_S, RUN_PIDS.get("harness"), RUN_PIDS.get("server")))
    REPORT["hard_stop_s"] = time.time() - NOTEBOOK_START
    for _pg in (RUN_PIDS.get("harness"), RUN_PIDS.get("server"), proc.pid):
        if _pg:
            try:
                os.killpg(_pg, 9)
            except OSError:
                pass
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass
if proc is not None:
    _rd.join(timeout=5)
REPORT["driver_rc"] = proc.poll() if proc is not None else "not started"
say("kaggle_run_arc2.py rc %s" % REPORT["driver_rc"])

# %% [7] End: the submission check (every task of the challenges file, both attempts; placeholders where missing); the report
try:
    _tasks = json.load(open(CHALLENGES))
except Exception as exc:  # noqa: BLE001
    _tasks = {}
    say("challenges unreadable at the end: %r" % exc)
try:
    _sub = json.load(open(SUBMISSION))
except Exception as exc:  # noqa: BLE001
    _sub = {}
    say("submission.json unreadable (%r): writing placeholders" % exc)
_fixed = 0
for _tid, _t in _tasks.items():
    _e = _sub.get(_tid)
    if not (isinstance(_e, list) and len(_e) == len(_t.get("test") or [])):
        _sub[_tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in (_t.get("test") or [{}])]
        _fixed += 1
with open(SUBMISSION + ".tmp", "w") as _fh:
    json.dump(_sub, _fh)
os.replace(SUBMISSION + ".tmp", SUBMISSION)
REPORT["submission"] = dict(tasks=len(_sub), placeholders_added_at_end=_fixed)
say("submission.json: %d tasks (%d filled with placeholders at the end)" % (len(_sub), _fixed))
try:
    REPORT["driver"] = json.load(open(glob.glob("/workspace/kvring_dev/fin74/runs/%s/kaggle_run_arc2.json" % LABEL)[0]))
except Exception as exc:  # noqa: BLE001
    REPORT["driver"] = repr(exc)
(W / ("notebook_report_%s.json" % LABEL)).write_text(json.dumps(REPORT, indent=1, default=str))
say("done, %.2f h after the notebook start" % ((time.time() - NOTEBOOK_START) / 3600))
