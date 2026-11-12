"""ARC-2 runner: `--lanes` attempts at a time, each up to `--budget` generated tokens. Pass 1 of every task runs first,
FIFO (v0 is exactly this with --passes 1). With --passes N every task then gets N independent passes, later passes by
band (slots.band: no answer, no verified answer, disagreeing verified answers, a single verified answer, agreement);
--max-passes fills idle lanes with further passes; --judge reviews a task's candidates (judge.py). The submission takes
the two best grids per test input across passes (slots.py). Writes per-pass logs (tasks/ID/pN/), results.jsonl (one
row per attempt), submission.json (rewritten after every attempt, Kaggle format), and at the end score.json and
summary.md.

  python -m arc2.runner --out RUN_DIR --base-url http://127.0.0.1:8000 --tokenizer MODEL_DIR/tokenizer.json
  python -m arc2.runner --out RUN_DIR --fake            # dry run, no server
"""
from __future__ import annotations

import argparse
import collections
import math
import json
import os
import queue
import signal
import random
import statistics
import sys
import threading
import time
import traceback
import types
import urllib.request
import zlib
from pathlib import Path

from . import tasks as T
from . import slots as S
from .agent import AttemptResult, earlier_answers, run_attempt
from .judge import Candidate, run_judge
from .kernel import Kernel
from .program import program_source, verify
from .shared import CAP_STATE
from . import review as RV
from . import represent as RP
from .prompts import load_texts



def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--challenges", default=str(T.EVAL_CHALLENGES))
    p.add_argument("--solutions", default=str(T.EVAL_SOLUTIONS), help="'' for none (hidden-set style run)")
    p.add_argument("--ids", default="", help="comma list or @file of task ids (default: all, sorted)")
    p.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--lanes", type=int, default=None, help="concurrent attempts (default: the profile's pool lanes NP, else 27)")
    p.add_argument("--profile", default="", help="server profile in --kdef (e.g. 82kr1632): sets --basis, --protect, default lanes")
    p.add_argument("--kdef", default="/workspace/kvring_dev/kaggle/kaggle_defs/KDEF.json")
    p.add_argument("--deadline-s", type=float, default=0.0,
                   help="run wall-clock limit in seconds (0 = none): no new attempts after it, running attempts end at it")
    p.add_argument("--budget", type=int, default=120_000, help="generated tokens per attempt")
    p.add_argument("--notice", type=int, default=8_000, help="tokens left when the harness says so")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="", help="served model name (default: the server's first /v1/models entry)")
    p.add_argument("--tokenizer", default="", help="tokenizer.json of the served model")
    p.add_argument("--fake", action="store_true", help="scripted replies, no server")
    p.add_argument("--protect", type=int, default=16_000, help="protected recent tokens (server eviction)")
    p.add_argument("--basis", type=int, default=64_204, help="eviction basis tokens (82kr1632: 64,204)")
    p.add_argument("--evict-fraction", type=float, default=0.2,
                   help="history eviction per step; 0.2 = ARC-3 production (ARC3_V6_EVICT_FRACTION); runs to 2026-10-29 "
                        "used 0.1")
    p.add_argument("--temperature", type=float, default=float(os.environ.get("LOCAL_ANALYZER_TEMPERATURE", "0.6")),
                   help="sampling temperature (default 0.6 = ARC-3's shipped KDEF value; 0.6 vs 1.0 is a Phase B ablation)")
    p.add_argument("--kernel-timeout", type=float, default=60.0)
    p.add_argument("--kernel-mem-mb", type=int, default=4096)
    p.add_argument("--tool-chars", type=int, default=12_000, help="tool output cap (head and tail kept)")
    p.add_argument("--no-pin", action="store_true", help="do not pin the opener's grids on the server")
    p.add_argument("--grid-format", choices=T.GRID_FORMATS, default="letters",
                   help="letters: ARC-3's colour symbols (default, the drafter's training form); digits: 0-9")
    p.add_argument("--reply-caps", choices=("off", "arc3", "custom", "v2", "v2c"), default="off",
                   help="off (run 1): no per-reply cue, close or repetition cue; arc3: ARC-3's 12K / 16K caps")
    p.add_argument("--request-tokens", type=int, default=8000, help="tokens per request before a length pause")
    p.add_argument("--observations", choices=("opener", "turns"), default="opener",
                   help="opener (v0, run 1): the opener is the only observation, so nothing is evictable; turns: a short "
                        "harness marker after each tool-result append is an observation, so finished turns become evictable")
    p.add_argument("--policy", choices=("bands", "cycle", "rank"), default="bands",
                   help="bands: --passes for every task (pass 1 for all first), later passes by band. cycle (the user, "
                        "2026-10-29): one pass per turn, then the task goes to the back of the queue until it has "
                        "--valid-target valid passes (slots.pass_valid) or the deadline ends the run")
    p.add_argument("--valid-target", type=int, default=2, help="cycle / rank: valid passes that complete a task")
    p.add_argument("--extra-pass-identical-long", type=int, default=0,
                   help="phase 3 (the user, 2026-10-30): a task whose passes 1 and 2 saved identical verified attempt-1 "
                        "answers, with pass 1 at least this many tokens, gets one more pass, ordered and admitted with the "
                        "tasks still short of valid passes (0 = off)")
    p.add_argument("--review-long", type=int, default=0,
                   help="the reviewer pass (the user, 2026-10-30): a task whose pass 1 took at least this many tokens and "
                        "whose passes 1 and 2 did not save identical verified attempt-1 answers gets one review session "
                        "(review.py) in the phase-3 rounds; its answers enter by P3: the votes keep slot 1, the reviewer's "
                        "answer takes slot 2 (0 = off; rank policy only)")
    p.add_argument("--review-budget", type=int, default=60_000)
    p.add_argument("--review-cue-from", type=int, default=45_000,
                   help="a review's in-reply status notes start here (the reviewer experiment's value)")
    p.add_argument("--review-wrap-at", type=int, default=45_000)
    p.add_argument("--review-limit-at", type=int, default=30_000)
    p.add_argument("--phase3-order", choices=("rank", "reviews_first"), default="rank",
                   help="rounds from pass 3 on: 'rank' orders reviews and passes alike by the rank key; 'reviews_first' "
                        "starts a round's open reviews before its passes (the reviewer session's item 2c; the user's call)")
    p.add_argument("--phase3-priority", default="rank",
                   help="v2m (the user, 2026-10-31): 'rank' (default) leaves --phase3-order in charge; or a comma order of the "
                        "classes third (the extra pass of identical verified answers), more (a further pass of a task short of "
                        "--valid-target valid passes) and review: from the round after two passes on, a round starts only its "
                        "highest non-empty class, by the rank key within it (the user's order: third,more,review). v2p (the "
                        "user, 2026-11-01, by gain per token): more may be split into more1 (a task with at least one valid "
                        "pass) and more0 (none), as an order of third, more1, review and more0 (third,more1,review,more0)")
    p.add_argument("--pass2-frame", choices=("off", "colour", "colour_symmetry"), default="off",
                   help="v2o (the reviewer session's input 23, item 2; the user, 2026-11-01: \"add it as a flag\"): pass 2 of a "
                        "task runs on a re-presented copy, its colours 1-9 permuted (0 kept) and, with colour_symmetry, one symmetry "
                        "of the square applied; the copy is an ordinary puzzle in its own kernel, and its answers are mapped back to "
                        "the original frame when the pass is recorded. The reviewer sees such a pass's mapped-back grids, its "
                        "explanation and its verified status, with the fact that it worked on a copy, and not its program (it runs "
                        "only on the copy). off (default): unchanged")
    p.add_argument("--pass2-frame-tasks", choices=("all", "odd", "even"), default="all",
                   help="v2o: which tasks get --pass2-frame: all, or the half whose crc32(task id) is odd / even (a crossover "
                        "A/B: one run each way)")
    p.add_argument("--pass2-frame-seed", type=int, default=2, help="v2o: the frames' seed (represent.task_key)")
    p.add_argument("--phase3-rule", choices=("v2l", "valid"), default="v2l",
                   help="v2m: 'v2l' (default) = passes 1 and 2: identical verified attempt-1 answers earn the third pass, any "
                        "other pair a review; 'valid' (the user, 2026-10-31: \"We run until a puzzle gets two valid answers then "
                        "it either gets additional pass if identical and review pass otherwise\"; \"No review will be performed "
                        "on puzzles without two distinct answers\") = the task's first two valid passes: identical attempt-1 "
                        "answers (verified or not) earn one extra pass, distinct ones one review; fewer than two valid: neither")
    p.add_argument("--phase3-length", choices=("first", "max", "mean"), default="first",
                   help="v2m: the length the phase-3 thresholds read: pass 1's tokens (default, v2l), or the max or the mean "
                        "of passes 1 and 2 (the user, 2026-10-31: the mean)")
    p.add_argument("--rank-length", choices=("first", "max", "mean"), default="first",
                   help="v2m: the length in the rank key (easy lanes shortest first, hard lanes longest): pass 1's tokens "
                        "(default, v2l), or the max or the mean of passes 1 and 2 when both ran (the user, 2026-10-31: the "
                        "thresholds' mean)")
    p.add_argument("--no-phase3-guard", action="store_true",
                   help="start phase-3 passes and reviews even when they cannot finish before the deadline at the lanes' "
                        "current speed (by default they do not start then, leaving the end of the run to the passes of "
                        "tasks short of valid passes)")
    p.add_argument("--fc-review-tokens", type=float, default=42_000.0,
                   help="the expected length of a review for admission and the deadline guard (rev12a / rev12b: 42.0K)")
    p.add_argument("--max-refusals", type=int, default=8,
                   help="refused starts in a row of one task's pass (or review) before it is recorded as a failed attempt, "
                        "so that a start that can never succeed does not hold a lane forever")
    p.add_argument("--admit-below", type=float, default=0.0,
                   help="rank: a lane starts its next pass only while the server's KV-pool use is below this fraction "
                        "(e.g. 0.85; 0 = off)")
    p.add_argument("--note-guard", action="store_true",
                   help="v2j: a reply that ends right after an injected note is not a finish (a short continuation follows)")
    p.add_argument("--admit-forecast", action="store_true",
                   help="rank: length-aware admission (Forecast) on top of --admit-below")
    p.add_argument("--fc-shadow", action="store_true",
                   help="forecast: compute and record each decision (sched.fc_projected / fc_would_block) without enforcing it")
    p.add_argument("--fc-z", type=float, default=0.0,
                   help="forecast: quantile of a pass's length as a z score (0 = the median; the replay: caution only idles lanes)")
    p.add_argument("--fc-horizon-s", type=float, default=1200.0, help="forecast horizon (s; replay: 20 min)")
    p.add_argument("--fc-margin", type=int, default=0, help="forecast: pages kept free below the usable pool")
    p.add_argument("--fc-pool", type=int, default=755, help="forecast: pool pages (82kr1632 / 82kx1280: 755)")
    p.add_argument("--fc-reserve", type=int, default=8, help="forecast: the server's pool-pressure reserve (pages)")
    p.add_argument("--fc-sigma-task", type=float, default=0.37,
                   help="forecast: sd of log length given the task's other passes (replay fit, 951 passes)")
    p.add_argument("--fc-sigma-prior", type=float, default=0.88, help="forecast: sd of log length of a first pass (replay fit)")
    p.add_argument("--fc-prior-tokens", type=float, default=42000.0, help="forecast: median pass length (replay fit)")
    p.add_argument("--fc-rho", type=float, default=0.86, help="forecast: resident tokens per generated token (replay fit)")
    p.add_argument("--fc-opener", type=float, default=10100.0, help="forecast: resident tokens at a pass's start (replay fit)")
    p.add_argument("--hard-lanes", type=int, default=0,
                   help="rank: after the first passes, these lanes take the open task ranked hardest (no valid answer, "
                        "then the most pass-1 tokens) instead of the easiest")
    p.add_argument("--wrap-at", type=int, default=120_000,
                   help="v2c: from this many tokens spent, status notes and turn markers carry the wrap-up reminder")
    p.add_argument("--limit-at", type=int, default=100_000,
                   help="--marker v2d: harness messages name the budget from this many tokens spent (before: the tokens "
                        "spent only)")
    p.add_argument("--cue-from", type=int, default=0,
                   help="--reply-caps v2c: in-reply status notes only from this many tokens of the pass (v2d: 120,000)")
    p.add_argument("--recall-at", type=int, default=120_000,
                   help="--marker v2e: once, the first harness message from this many tokens spent reminds of history()")
    p.add_argument("--marker", choices=("v1", "v2c", "v2d", "v2e"), default="v1",
                   help="v2c: the turn marker carries the tokens spent (and the wrap-up reminder from --wrap-at)")
    p.add_argument("--passes", type=int, default=1,
                   help="independent passes every task gets: pass 1 for all tasks first (FIFO), then later passes by band")
    p.add_argument("--max-passes", type=int, default=0,
                   help="tail fill: idle lanes run further passes, up to this many per task, on tasks below band 4 "
                        "(0 = --passes, no fill)")
    p.add_argument("--fill-agreed", action="store_true", help="tail fill also takes band-4 tasks (verified agreement)")
    p.add_argument("--band-order", default="0,1,2,3,4",
                   help="priority of bands for passes after the first, first listed first (reviewer 10-29: 0,1,3,2,4)")
    p.add_argument("--confirm-order", choices=("fifo", "first_allpass_desc"), default="fifo",
                   help="order within band 3 (one verified answer): FIFO, or by the verified pass's tokens to its first "
                        "all-pass check, most first (reviewer 10-29; AUC 0.75 for a wrong all-pass answer in run 1)")
    p.add_argument("--corrections", type=int, default=0,
                   help="times an unverified attempt-1 answer is answered with the correction text before it may finish")
    p.add_argument("--judge", choices=("off", "3plus", "2plus"), default="off",
                   help="review a task's candidates with a judge session once its planned passes are done: 3plus when "
                        "some test input has 3 or more distinct grids, 2plus at 2 or more")
    p.add_argument("--judge-budget", type=int, default=32_000)
    p.add_argument("--judge-notice", type=int, default=4_000)
    p.add_argument("--cap-cue", type=int, default=24_000, help="--reply-caps custom: in-reply cue at this many tokens")
    p.add_argument("--cap-close", type=int, default=32_000, help="--reply-caps custom: reply closed at this many tokens")
    p.add_argument("--cap-ext", type=int, default=4_096, help="--reply-caps custom / v2: extension for an open tool call")
    p.add_argument("--status-every-tokens", type=int, default=None,
                   help="--reply-caps v2: a status note every this many tokens of a reply (default 12,000); v2c: every this "
                        "many tokens of the pass (default 10,000); 0 = none")
    p.add_argument("--prompt", choices=("v1", "v2", "v2d", "v2e", "v2e_budget", "v2j", "v2j_budget"), default="v1",
                   help="v2: the system prompt of the user's v2 design; v2d: the same without the token budget; v2e: v2d "
                        "saying the preloaded names are used inside python; v2e_budget: v2e with the budget sentence")
    p.add_argument("--no-call", choices=("nudge_once", "keep_going"), default="nudge_once",
                   help="a reply without a tool call: nudge once, then end (v1); keep_going: answer it until a valid "
                        "solution is saved (v2)")
    p.add_argument("--idle-limit", type=int, default=6, help="keep_going: replies without a tool call in a row that end the pass")
    p.add_argument("--repeat-limit", type=int, default=0, help="repetition closes that end a pass ('degenerate'; 0 = never)")
    p.add_argument("--invalid-recover", type=int, default=4,
                   help="v2q: after this many replies in a row whose tool calls all failed (nothing ran), a recovery note "
                        "follows the tool results (0 = never)")
    p.add_argument("--invalid-limit", type=int, default=8,
                   help="v2q: this many such replies in a row end the attempt ('stalled'; its saved answers are kept, the task "
                        "takes its next pass from the queue; 0 = never)")
    p.add_argument("--status-text", choices=("v2", "v2b"), default="v2",
                   help="v2b: the status note says it does not end the reply (run03a: 84%% of v2 notes ended it)")
    p.add_argument("--idle-skip-status", action="store_true",
                   help="a no-call reply that carried a status note does not count toward --idle-limit (v2b)")
    p.add_argument("--image-cell", type=int, default=0,
                   help="show each grid also as an image (an aid beside the text), cell size in px (16 = one vision "
                        "patch per cell; 0 = text only); needs --prompt v2")
    p.add_argument("--status-every", type=float, default=60.0)
    p.add_argument("--timeline-every", type=float, default=60.0, help="seconds between timeline.jsonl rows")
    a = p.parse_args(argv)
    a.profile_values = None
    if a.profile:
        prof = json.load(open(a.kdef))["profiles"][a.profile]
        a.basis, a.protect = int(prof["BASIS"]), int(prof["TAIL"])
        a.profile_values = {k: prof[k] for k in ("CTX", "BASIS", "TAIL", "NL", "NP", "KVG", "KVRING") if k in prof}
        if a.lanes is None:
            a.lanes = int(prof["NP"])
    if a.lanes is None:
        a.lanes = 27
    return a


def select_tasks(args):
    ids = None
    if args.ids:
        raw = open(args.ids[1:]).read() if args.ids.startswith("@") else args.ids
        ids = [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]
    tasks = T.load_tasks(args.challenges, args.solutions or None, ids)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    return tasks


def fetch(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except Exception as exc:                      # noqa: BLE001 - metrics are optional
        return f"# fetch failed: {exc!r}\n"


def kv_usage(text):
    """The server's KV-pool use (vllm:kv_cache_usage_perc, 0-1), or None."""
    for line in text.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc"):
            try:
                return float(line.rsplit(" ", 1)[1])
            except ValueError:
                return None
    return None


class StartRefused(Exception):
    """The server refused a pass's session start (HTTP 429, its resident-session limit; or a capacity refusal) before any
    token was generated. Not a pass: the lane gives the task back and retries (2026-10-29: run08a lost 93 tasks when such
    refusals were recorded as finished passes)."""


def start_refused(res):
    """An attempt that generated no token and did not finish is a refused start, not a pass: the server's 429 or capacity
    refusal and, from 2026-10-30 (the reviewer session's item 1a), any other such end, e.g. a connection error while the
    server restarts, so that an outage cannot burn a task's passes in seconds. The lanes back off when refusals repeat,
    and --max-refusals bounds the retries per task."""
    return res.generated_tokens == 0 and res.end != "finished"


def close_open(items):
    """Close every open session and kernel (best effort); returns how many sessions were closed. Used on SIGTERM so a
    stopped run does not leave resident sessions on the server (2026-10-29: a killed runner's sessions stayed open).
    v2n: an item is (session, kernel) or (session, kernel, context)."""
    n = 0
    for item in list(items.values()):
        session, kernel = item[0], item[1]
        try:
            session.close()
            n += 1
        except Exception:                     # noqa: BLE001
            pass
        try:
            kernel.close()
        except Exception:                     # noqa: BLE001
            pass
    return n


def frame_back(res, key):
    """v2o: an attempt run in key's frame (--pass2-frame): its saved answers, earlier answers and fallback back in the original
    frame (in place); returns res."""
    res.submissions = {a: [RP.from_frame(g, key) for g in outs] for a, outs in (res.submissions or {}).items()}
    res.earlier = [dict(e, outputs=[RP.from_frame(g, key) for g in e["outputs"]]) if isinstance(e, dict) and "outputs" in e
                   else e for e in (res.earlier or [])]
    if res.fallback and res.fallback.get("outputs") is not None:
        res.fallback = dict(res.fallback, outputs=[RP.from_frame(g, key) for g in res.fallback["outputs"]])
    return res


def provisional_row(task, kernel, ctx, tokens=None):
    """v2n (the reviewer session's input 22, item 1; the user, 2026-11-01: "fold it in"): the row of a pass still open when
    the run is stopped (the driver's SIGTERM, harness_grace_s after the deadline): the answers its kernel saved, as the
    attempt's end would have recorded them, with end 'killed' and provisional True. Before v2n a pass in flight at the stop
    lost every answer it had saved (HARNESS_DESIGN section 8: the answers a deadline-cut pass saved count). ctx: dict(kind
    'solve', pass_no, sched) or dict(kind 'review', sched, labels, previous). None when the pass saved nothing."""
    subs = kernel.submissions()
    if not subs:
        return None
    key = ctx.get("frame")                              # v2o: a re-presented pass's answers come back to the original frame
    earlier = earlier_answers(kernel)
    if key is not None:
        back = frame_back(types.SimpleNamespace(submissions=subs, earlier=earlier, fallback=None), key)
        subs, earlier = back.submissions, back.earlier
    submits = [dict(tokens=None, call=r.get("call"), attempt=r["attempt"], fn=r.get("fn"), train_passed=r.get("train_passed"),
                    train_total=r.get("train_total"), verified_by=r.get("verified_by"))
               for r in kernel.records if r["kind"] == "submit"]
    res = types.SimpleNamespace(submits=submits)
    row = dict(AttemptResult(task.task_id).to_json(), end="killed", generated_tokens=int(tokens or 0), submits=submits,
               submissions=subs, earlier=earlier, kernel_restarts=kernel.restarts, provisional=True,
               kind=ctx["kind"], programs={} if key is not None else attempt_programs(kernel, res), sched=ctx.get("sched"))
    if key is not None:
        row.update(frame=dict(key, mode=ctx.get("frame_mode")), programs_framed=attempt_programs(kernel, res))
    if ctx["kind"] == "solve":
        attempts = {int(k): v for k, v in subs.items()}
        row.update(pass_no=ctx["pass_no"], answer_source="model",
                   score=T.score_task(task, T.submission_entry(task, attempts)),
                   train_complete_submit=any(x.get("train_passed") == x.get("train_total") for x in submits))
    else:
        row.update(labels=ctx.get("labels"), previous=ctx.get("previous"))
    return row


class Forecast:
    """Length-aware admission (the user, 2026-10-29): a pass starts only if the pool forecast over the next horizon stays
    below the usable pool minus a margin. Each running pass is projected from its task's finished pass lengths (mean log;
    the population prior for a first pass) at a cautious quantile exp(mu + z * sigma); a pass already past that length is
    taken to run to 1.5x its tokens so far. A pass expected to end before a horizon point frees its pages there; the others
    grow at the current per-lane speed, up to the cap. The new pass is projected the same way. Rough by design: every
    admission re-forecasts from the server's actual pool use and the passes' actual progress (calibration:
    analysis/admission_replay.py)."""

    def __init__(self, *, pool, reserve, margin, horizon_s, z, sig_task, sig_prior, prior_tokens, rho, opener, cap_pages,
                 page=3184, fixed=4, speed=60.0):
        self.pool, self.reserve, self.margin, self.horizon_s, self.z = pool, reserve, margin, horizon_s, z
        self.sig_task, self.sig_prior, self.mu_prior = sig_task, sig_prior, math.log(prior_tokens)
        self.rho, self.opener, self.cap_pages, self.page, self.fixed, self.speed = rho, opener, cap_pages, page, fixed, speed
        self.blocks = 0
        self.shadow_blocks = 0

    def pages(self, resident):
        return min(self.cap_pages, math.ceil(resident / self.page) + self.fixed)

    def mu_sig(self, lengths):
        lengths = [x for x in lengths if x > 0]
        if not lengths:
            return self.mu_prior, self.sig_prior
        return sum(math.log(x) for x in lengths) / len(lengths), self.sig_task

    def length(self, mu, sig, g):
        L = math.exp(mu + self.z * sig)
        return max(L, 1.5 * g) if g >= L else L

    def projected(self, used_pages, live, new, pending=()):
        """The worst forecast pool use at 1/4, 1/2, 3/4 and 1 of the horizon. live: [{g, res, mu, sig}]; new: (mu, sig);
        pending: (mu, sig) of hard passes waiting with priority, counted as if they started now (the user, 2026-10-29: a
        two-sided queue whose hard side has priority and waits for room)."""
        worst = used_pages
        v = self.speed
        for f in (0.25, 0.5, 0.75, 1.0):
            tau = f * self.horizon_s
            tot = used_pages
            for p in live:
                cur = self.pages(p["res"])
                if self.length(p["mu"], p["sig"], p["g"]) - p["g"] <= v * tau:
                    tot -= cur
                else:
                    tot += self.pages(p["res"] + self.rho * v * tau) - cur
            for mu, sig in [new] + list(pending):
                if self.length(mu, sig, 0.0) > v * tau:
                    tot += self.pages(self.opener + self.rho * v * tau)
            worst = max(worst, tot)
        return worst

    def admits(self, used_pages, live, new, pending=()):
        return self.projected(used_pages, live, new, pending) <= self.pool - self.reserve - self.margin


class PoolGauge:
    """Admission at pass start (the user, 2026-10-29: 32 lanes with admission): a lane starts its next pass only while the
    pool's use is below `below`. One /metrics read serves all lanes for `max_age_s`; an unreadable gauge admits."""

    def __init__(self, url, below, max_age_s=2.0, fetcher=None):
        self.url, self.below, self.max_age_s = url, below, max_age_s
        self.fetcher = fetcher or (lambda: fetch(url, timeout=5))
        self.lock = threading.Lock()
        self.value, self.at = None, -1e9
        self.waits = 0
        self.wait_s = 0.0

    def read(self):
        with self.lock:
            if time.monotonic() - self.at >= self.max_age_s:
                try:
                    self.value = kv_usage(self.fetcher())
                except Exception:                     # noqa: BLE001 - an unreadable gauge admits
                    self.value = None
                self.at = time.monotonic()
            return self.value

    def admits(self):
        v = self.read()
        return v is None or v < self.below

    def wait(self, deadline=None, step_s=2.0):
        """Block until admitted (or the deadline); returns the seconds waited."""
        t0 = time.monotonic()
        counted = False
        while not self.admits():
            if deadline is not None and time.monotonic() >= deadline:
                break
            if not counted:
                with self.lock:
                    self.waits += 1
                counted = True
            time.sleep(step_s)
        if not counted:
            return 0.0
        dt = time.monotonic() - t0
        with self.lock:
            self.wait_s += dt
        return dt


def spec_counters(text):
    """Sum vLLM speculative-decoding counters from /metrics text (whole server, any traffic)."""
    out = {}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_") and not line.startswith("#"):
            name = line.split("{")[0].split(" ")[0]
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def generation_counter(text):
    """The server's cumulative generated tokens (vLLM's generation_tokens counter), or None when absent."""
    total, seen = 0.0, False
    for line in text.splitlines():
        if line.startswith("vllm:generation_tokens_total") or line.startswith("vllm:generation_tokens "):
            try:
                total += float(line.rsplit(" ", 1)[1])
                seen = True
            except ValueError:
                pass
    return total if seen else None


def timeline_stats(path, lanes):
    """Throughput over time from timeline.jsonl: peak and mean 10-minute server rates, and the share of the run with
    fewer busy lanes than configured (the tail)."""
    try:
        rows = [json.loads(l) for l in open(path)]
    except OSError:
        return {}
    pts = [(r["t"], r["server_generation_tokens"]) for r in rows if r.get("server_generation_tokens") is not None]
    out = dict(rows=len(rows))
    busy = [r for r in rows if r["running"] > 0]
    if busy:
        out["share_below_full_lanes"] = round(sum(r["running"] < lanes for r in busy) / len(busy), 3)
    if len(pts) >= 2:
        rates = []
        for i in range(len(pts)):
            j = next((k for k in range(i, len(pts)) if pts[k][0] - pts[i][0] >= 600), None)
            if j is not None:
                rates.append((pts[j][1] - pts[i][1]) / (pts[j][0] - pts[i][0]))
        span = pts[-1][0] - pts[0][0]
        out["mean_rate"] = round((pts[-1][1] - pts[0][1]) / span, 1) if span > 0 else None
        if rates:
            out["peak_10min_rate"] = round(max(rates), 1)
            if out.get("mean_rate"):
                out["mean_over_peak"] = round(out["mean_rate"] / out["peak_10min_rate"], 3)
    return out


class Session:
    """Factory for real or fake sessions with the run's fixed settings."""

    def __init__(self, args):
        self.args = args
        self.shared = None
        if args.fake:
            return
        from .shared import configure_transport, load_transport
        self.transport = load_transport()
        self.caps = configure_transport(self.transport, texts=load_texts(), reply_caps=args.reply_caps,
                                        request_tokens=args.request_tokens,
                                        custom=(args.cap_cue, args.cap_close, args.cap_ext),
                                        v2=dict(close=args.cap_close, ext=args.cap_ext,
                                                status_every=args.status_every_tokens if args.status_every_tokens
                                                is not None else (10_000 if args.reply_caps == "v2c" else 12_000),
                                                status_key="status_cue_v2b" if args.status_text == "v2b" else "status_cue",
                                                wrap_at=args.wrap_at, cue_from=args.cue_from))
        self.shared = dict(self.transport.ARC2_SHARED, **self.caps)
        from tokenizers import Tokenizer
        if not args.tokenizer:
            raise SystemExit("--tokenizer is required with a real server")
        tok = Tokenizer.from_file(args.tokenizer)
        self.im_end = tok.encode("<|im_end|>", add_special_tokens=False).ids
        self.decode = lambda ids: tok.decode(ids, skip_special_tokens=False)    # noqa: E731
        self.model = args.model or json.loads(fetch(args.base_url.rstrip("/") + "/v1/models"))["data"][0]["id"]
        self.history_eviction = dict(fraction=args.evict_fraction, policy="qsa_frequency",
                                     protect_recent_tokens=args.protect, target_basis_tokens=args.basis, unit="split")
        self.sampling = dict(temperature=args.temperature, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0,
                             presence_penalty=0.0, frequency_penalty=0.0, max_tokens=12000)
        self.template_kwargs = dict(enable_thinking=True, reasoning_effort="xhigh", preserve_thinking=True)

    def make(self, log):
        if self.args.fake:
            from .fake import FakeSession
            return FakeSession(log=log)
        return self.transport.PersistentSession(
            base_url=self.args.base_url, model=self.model, tools=[TOOL], template_kwargs=self.template_kwargs,
            sampling=self.sampling, history_eviction=self.history_eviction, im_end_ids=self.im_end,
            decode=self.decode, log=log)

    def settings(self):
        if self.args.fake:
            return dict(fake=True)
        return dict(model=self.model, shared=self.shared, history_eviction=self.history_eviction, sampling=self.sampling,
                    template_kwargs=self.template_kwargs)


TOOL = {"type": "function", "function": {
    "name": "python",
    "description": "Run Python code in this puzzle's persistent session (numpy as np; train, test, check, submit, "
                   "note, history, source and show are preloaded).",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source to run."}},
                   "required": ["code"]}}}


def main(argv=None):
    args = parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "tasks").mkdir()
    tasks = select_tasks(args)
    texts = load_texts()
    factory = Session(args)
    base = args.base_url.rstrip("/")
    max_passes = max(args.passes, args.max_passes or args.passes)
    settings = dict(argv=sys.argv, args=vars(args), n_tasks=len(tasks), task_ids=[t.task_id for t in tasks],
                    start_epoch=time.time(), session=factory.settings(), tool=TOOL, max_passes=max_passes)
    T.write_json(out / "settings.json", settings)
    if not args.fake:
        (out / "metrics_before.txt").write_text(fetch(base + "/metrics"))

    lock = threading.Lock()
    open_items = {}                                           # open sessions and kernels, closed on SIGTERM
    flushed = set()                                           # v2n: (task, pass_no | 'review') recorded provisionally at SIGTERM
    live = {}                                                 # Forecast: running passes' progress, keyed (task, pass)
    pending_hard = {}                                         # Forecast: hard lane -> (mu, sig) of its waiting pass
    refusals = [0]

    def on_term(signum, frame):
        """v2n: first the answers the open passes saved (provisional_row), then close the sessions. The lock is taken with
        a timeout: a lane holds it only briefly, and if this (main) thread was interrupted holding it, the flush goes on
        without it (the run ends here)."""
        got = lock.acquire(timeout=10)
        saved = 0
        try:
            for (tid, which, _), item in list(open_items.items()):
                if len(item) < 3 or (tid, which) in flushed:
                    continue
                try:
                    row = provisional_row(st[tid]["task"], item[1], item[2], (live.get((tid, which)) or {}).get("g"))
                    if row is not None:
                        flushed.add((tid, which))
                        _record(tid, row)
                        saved += 1
                except Exception as exc:          # noqa: BLE001 - the stop goes on
                    sys.stderr.write(f"provisional row {tid} {which}: {exc!r}\n")
        finally:
            if got:
                lock.release()
        n = close_open(open_items)
        (out / "terminated.txt").write_text(f"SIGTERM at {time.time():.0f}; recorded the saved answers of {saved} open "
                                            f"passes; closed {n} open sessions\n")
        raise SystemExit(143)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, on_term)
    order = [t.task_id for t in tasks]
    pos = {tid: i for i, tid in enumerate(order)}
    st = {t.task_id: dict(task=t, rows=[], started=0, running=0, judge=None, judged=None, judges=0, review=None,
                          review_row=None, refused=0, review_refused=0) for t in tasks}
    judge_q = []
    queue = collections.deque(order)                       # cycle: tasks waiting for a pass, front first
    entries = {t.task_id: T.placeholder_entry(t) for t in tasks}
    results = []
    running = {}
    t_start = time.monotonic()
    deadline = (t_start + args.deadline_s) if args.deadline_s > 0 else None
    kernel_kwargs = dict(timeout_s=args.kernel_timeout, mem_mb=args.kernel_mem_mb, output_chars=args.tool_chars)

    def task_log(task_id, part):
        path = out / "tasks" / task_id / part / "log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        flock = threading.Lock()

        def log(kind, payload):
            line = json.dumps(dict(kind=kind, task_id=task_id, epoch=time.time(), payload=payload),
                              ensure_ascii=False, default=str)
            with flock, open(path, "a") as f:
                f.write(line + "\n")
        return log

    def judge_order(tid):
        """The judge's ranking as indices into the task's current full answers (None without a judge)."""
        s = st[tid]
        if not s["judge"]:
            return None
        fulls = S.full_answers(s["rows"])
        return [k for o in s["judge"] for k, f in enumerate(fulls) if f["outputs"] == o]

    def task_entry(tid):
        s = st[tid]
        outs = S.slot_outputs(s["rows"], len(s["task"].test), judge_order(tid))
        if s["review_row"] is not None:                     # the reviewer pass: P3 (the user, 2026-10-30)
            outs = S.reviewed_outputs(outs, s["review_row"].get("submissions"))
        return T.submission_entry(s["task"], outs)

    band_rank = {int(b): i for i, b in enumerate(args.band_order.split(","))}

    def first_allpass(tid):
        """Tokens to the first all-pass check in the task's verified passes (the most, for band 3's order)."""
        vals = [r.get("first_full_pass_tokens") for r in st[tid]["rows"]
                if any(v for _, _, v in S.pass_answers(r)) and r.get("first_full_pass_tokens") is not None]
        return max(vals) if vals else 0

    def key(tid):
        s = st[tid]
        b = S.band(s["rows"])
        intra = -first_allpass(tid) if (b == 3 and args.confirm_order == "first_allpass_desc") else 0
        return (s["running"] > 0, band_rank.get(b, len(band_rank)), s["started"], intra, pos[tid])

    def pick_work():
        """Under the lock: ('judge', id), ('solve', id) or None."""
        if judge_q:
            return "judge", judge_q.pop(0)
        for tid in order:                                   # pass 1 of every task first, FIFO
            if st[tid]["started"] == 0:
                return "solve", tid
        need = [tid for tid in order if st[tid]["started"] < args.passes]
        if not need:                                        # tail fill
            need = [tid for tid in order if st[tid]["started"] < max_passes
                    and (args.fill_agreed or S.band(st[tid]["rows"]) < 4)]
        return ("solve", min(need, key=key)) if need else None

    def maybe_judge(tid):
        """Under the lock, after a pass ends: queue a review when the task's planned passes are done and its
        candidate set is new and large enough."""
        s = st[tid]
        if args.judge == "off" or s["running"] or s["started"] < args.passes or tid in judge_q:
            return
        widest = max((len(S.ranked_grids(s["rows"], i)) for i in range(len(s["task"].test))), default=0)
        fulls = [f["outputs"] for f in S.full_answers(s["rows"])]
        if widest >= (3 if args.judge == "3plus" else 2) and len(fulls) >= 2 and fulls != s["judged"]:
            judge_q.append(tid)

    def _record(tid, row):
        """record()'s body; the caller holds the lock (on_term: or the run is ending)."""
        if row["kind"] == "solve":
            st[tid]["rows"].append(row)
            st[tid]["refused"] = 0
        elif row["kind"] == "review":
            st[tid]["review_row"] = row
            st[tid]["review_refused"] = 0
        results.append(row)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        entries[tid] = task_entry(tid)
        T.write_json(out / "submission.json", entries)

    def record(tid, row):
        with lock:
            if (tid, row.get("pass_no") if row["kind"] == "solve" else row["kind"]) in flushed:
                return                                      # v2n: already recorded provisionally at SIGTERM
            _record(tid, row)

    def frame_of(tid, pass_no):
        """v2o: the frame of this pass under --pass2-frame (a key), or None."""
        if args.pass2_frame == "off" or pass_no != 2:
            return None
        if args.pass2_frame_tasks != "all" and (zlib.crc32(tid.encode()) % 2 == 1) != (args.pass2_frame_tasks == "odd"):
            return None
        return RP.pass_key(tid, args.pass2_frame_seed, symmetry=args.pass2_frame == "colour_symmetry")

    def run_solve(tid, pass_no, sched):
        task = st[tid]["task"]
        frame = frame_of(tid, pass_no)
        shown = RP.framed_task(task, frame) if frame is not None else task       # v2o: the copy the pass works on
        log = task_log(tid, f"p{pass_no}")
        if frame is not None:
            log("frame", dict(mode=args.pass2_frame, symmetry=frame["symmetry"], colour_map=frame["colour_map"]))
        session = factory.make(log)
        kernel = Kernel(shown.sandbox_payload(args.grid_format), log=log, **kernel_kwargs)
        key = (tid, pass_no, id(session))
        open_items[key] = (session, kernel, dict(kind="solve", pass_no=pass_no, sched=sched, frame=frame,
                                                 frame_mode=args.pass2_frame if frame is not None else None))
        programs = {}

        def progress(g, res):
            e = live.get((tid, pass_no))
            if e is not None:
                e["g"] = float(g)
                e["res"] = float(res) if res else (fc.opener + fc.rho * g if fc is not None else 0.0)
        try:
            kernel.start()
            v2 = args.prompt in ("v2", "v2d", "v2e", "v2e_budget", "v2j", "v2j_budget")
            v2d = args.marker in ("v2d", "v2e")
            v2e = args.marker == "v2e"
            res = run_attempt(shown, session, kernel, texts, budget=args.budget, notice=args.notice, log=log,
                              fmt=args.grid_format, pin=not args.no_pin, deadline=deadline,
                              observe_turns=args.observations == "turns", corrections=args.corrections,
                              no_call=args.no_call, idle_limit=args.idle_limit, repeat_limit=args.repeat_limit,
                              invalid_recover=args.invalid_recover, invalid_limit=args.invalid_limit,
                              cap_state=CAP_STATE, prompt_key={"v1": "system", "v2": "system_v2", "v2d": "system_v2d",
                                                               "v2e": "system_v2e",
                                                               "v2e_budget": "system_v2e_budget", "v2j": "system_v2j",
                                                               "v2j_budget": "system_v2j_budget"}[args.prompt],
                              reply_cap=args.cap_close if args.reply_caps in ("custom", "v2", "v2c") else None,
                              recovery_budget_key="recovery_budget_v2d" if v2d else (
                                  "recovery_budget_v2" if v2 else "recovery_budget"),
                              image_cell=args.image_cell or None, idle_skip_status=args.idle_skip_status,
                              marker_key={"v1": "turn_marker", "v2c": "turn_marker_v2c", "v2d": "turn_marker_v2d",
                                          "v2e": "turn_marker_v2d"}[args.marker],
                              wrap_at=args.wrap_at if args.marker in ("v2c", "v2d", "v2e") else None,
                              recall_at=args.recall_at if v2e else None,
                              limit_at=args.limit_at if v2d else None,
                              text_keys=dict(dict(keep_going="keep_going_v2d", recovery_repeat="recovery_repeat_v2d",
                                                  nudge_format="nudge_format_v2d"),
                                             **(dict(unknown_tool="unknown_tool_v2e") if v2e else {})) if v2d else None,
                              progress=progress if fc is not None else None,
                              note_guard=args.note_guard)
            programs = attempt_programs(kernel, res)
        finally:
            try:
                session.close()
            except Exception as exc:              # noqa: BLE001
                log("session-close-error", dict(error=repr(exc)))
            kernel.close()
            open_items.pop(key, None)
            live.pop((tid, pass_no), None)
        programs_framed = None
        if frame is not None:                               # v2o: back to the original frame before anything reads it
            frame_back(res, frame)
            programs_framed, programs = programs, {}
        if start_refused(res):
            log("start-refused", dict(error=res.error, end=res.end, refused_before=st[tid]["refused"]))
            if st[tid]["refused"] < args.max_refusals:
                raise StartRefused(res.error)
        attempts = {int(k): v for k, v in res.submissions.items()}
        source = "model" if attempts else ("fallback" if res.fallback else "none")
        if not attempts and res.fallback:
            attempts = {1: res.fallback["outputs"]}
        score = T.score_task(task, T.submission_entry(task, attempts))          # this pass alone
        row = dict(res.to_json(), kind="solve", pass_no=pass_no, answer_source=source, score=score, programs=programs,
                   train_complete_submit=any(s.get("train_passed") == s.get("train_total") for s in res.submits),
                   sched=sched)
        if frame is not None:
            row.update(frame=dict(frame, mode=args.pass2_frame), programs_framed=programs_framed)
        (out / "tasks" / tid / f"p{pass_no}" / "attempt.json").write_text(json.dumps(row, default=str))
        record(tid, row)

    def run_review_job(tid, sched):
        """The reviewer pass (the user, 2026-10-30; JUDGE_EXPERIMENT_2026-10-29.md): a fresh session sees the puzzle and the
        answers of passes 1 and 2 and saves its own attempts; the row (kind 'review') is not a pass, and task_entry folds
        its answers in by P3. Settings as in the experiment (review_offline.py run): 60K budget, status notes and the
        wrap-up reminder from 45K, the budget named from 30K; the texts of the solver base (v2j's, the user)."""
        task = st[tid]["task"]
        with lock:                                      # v2n (input 22, item 2): the pair review_open judged, under 'valid'
            pair = phase3_pair(tid) if args.phase3_rule == "valid" else None
            p1, p2 = pair if pair is not None else (st[tid]["rows"][0], st[tid]["rows"][1])
        previous, labels = RV.previous_from_rows(p1, p2, lambda k: out / "tasks" / tid / f"p{k}" / "log.jsonl",
                                                 seed=f"{tid}:review")
        previous = [dict(x, kind_text=RV.answer_kind(texts, x["verified"], x["fn"], x["passed"], x["total"], x.get("frame")))
                    for x in previous]
        log = task_log(tid, "review")
        session = factory.make(log)
        kernel = Kernel(RV.review_payload(task, previous, args.grid_format), log=log, **kernel_kwargs)
        key = (tid, "review", id(session))
        open_items[key] = (session, kernel, dict(kind="review", sched=sched, labels=labels, previous=[
            {k: x[k] for k in ("label", "attempt", "verified", "fn", "pass_no")} for x in previous]))
        programs = {}

        def progress(g, res):
            e = live.get((tid, "review"))
            if e is not None:
                e["g"] = float(g)
                e["res"] = float(res) if res else (fc.opener + fc.rho * g if fc is not None else 0.0)
        CAP_STATE.cue_from, CAP_STATE.wrap_at = args.review_cue_from, args.review_wrap_at
        try:
            res, kernel = RV.run_review(task, previous, session, texts, budget=args.review_budget, fmt=args.grid_format,
                                        log=log, kernel=kernel, deadline=deadline, cap_state=CAP_STATE,
                                        no_call="keep_going", idle_limit=args.idle_limit, repeat_limit=args.repeat_limit,
                                        invalid_recover=args.invalid_recover, invalid_limit=args.invalid_limit,
                                        observe_turns=args.observations == "turns", reply_cap=args.cap_close,
                                        recovery_budget_key="recovery_budget_v2d", idle_skip_status=args.idle_skip_status,
                                        marker_key="turn_marker_v2d", wrap_at=args.review_wrap_at, recall_at=None,
                                        limit_at=args.review_limit_at,
                                        text_keys=dict(keep_going="keep_going_v2d", recovery_repeat="recovery_repeat_v2d",
                                                       nudge_format="nudge_format_v2d", unknown_tool="unknown_tool_v2e"),
                                        note_guard=args.note_guard, progress=progress if fc is not None else None)
            programs = attempt_programs(kernel, res)
        finally:
            CAP_STATE.cue_from = CAP_STATE.wrap_at = None
            try:
                session.close()
            except Exception as exc:              # noqa: BLE001
                log("session-close-error", dict(error=repr(exc)))
            kernel.close()
            open_items.pop(key, None)
            live.pop((tid, "review"), None)
        if start_refused(res):
            log("start-refused", dict(error=res.error, end=res.end, refused_before=st[tid]["review_refused"]))
            if st[tid]["review_refused"] < args.max_refusals:
                raise StartRefused(res.error)
        row = dict(res.to_json(), kind="review", labels=labels, programs=programs, sched=sched,
                   previous=[{k: x[k] for k in ("label", "attempt", "verified", "fn", "pass_no")} for x in previous])
        (out / "tasks" / tid / "review" / "attempt.json").write_text(json.dumps(row, default=str))
        record(tid, row)

    def run_judge_job(tid):
        with lock:
            s = st[tid]
            task, rows = s["task"], list(s["rows"])
            s["judges"] += 1
            n = s["judges"]
        fulls = S.full_answers(rows)
        cands = [judge_candidate(task, f, rows, args.grid_format, kernel_kwargs) for f in fulls]
        perm = list(range(len(cands)))
        random.Random(f"{tid}:judge{n}").shuffle(perm)
        log = task_log(tid, f"judge{n}")
        session = factory.make(log)
        try:
            res, shown_order = run_judge(task, [cands[i] for i in perm], session, texts, budget=args.judge_budget,
                                         notice=args.judge_notice, fmt=args.grid_format, log=log, deadline=deadline,
                                         kernel_kwargs=kernel_kwargs, observe_turns=args.observations == "turns")
        finally:
            try:
                session.close()
            except Exception as exc:              # noqa: BLE001
                log("session-close-error", dict(error=repr(exc)))
        ranked = [perm[i] for i in shown_order] if shown_order else None
        with lock:
            s["judged"] = [f["outputs"] for f in fulls]
            if ranked:
                s["judge"] = [fulls[i]["outputs"] for i in ranked]
        row = dict(res.to_json(), kind="judge", judge_no=n, n_candidates=len(cands), order=ranked,
                   candidate_kinds=[("grids" if not c.source else "code_all" if c.passed == c.total else "code_part")
                                    for c in cands])
        record(tid, row)

    def lane(i):
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return                                  # no new attempts after the run's deadline
            with lock:
                work = pick_work()
                if work is not None:
                    kind, tid = work
                    if kind == "solve":
                        sched = dict(band=S.band(st[tid]["rows"]), first_allpass=first_allpass(tid),
                                     running_passes=st[tid]["running"], t=round(time.monotonic() - t_start, 1))
                        st[tid]["started"] += 1
                        st[tid]["running"] += 1
                        pass_no = st[tid]["started"]
                    running[i] = (tid, kind)
                busy = bool(running) or bool(judge_q)
            if work is None:
                if not busy:
                    return
                time.sleep(1.0)                         # a running attempt may still create work (a later pass, a review)
                continue
            try:
                if kind == "solve":
                    run_solve(tid, pass_no, sched)
                else:
                    run_judge_job(tid)
            except Exception:                     # noqa: BLE001 - one task's failure never stops the lane
                with open(out / "lane_errors.log", "a") as f:
                    f.write(f"{time.time():.0f} lane {i} task {tid} {kind}\n{traceback.format_exc()}\n")
            finally:
                with lock:
                    running.pop(i, None)
                    if kind == "solve":
                        st[tid]["running"] -= 1
                        maybe_judge(tid)

    def n_valid(tid):
        return sum(S.pass_valid(r) for r in st[tid]["rows"])

    def phase3_pair(tid):
        """v2m: the task's first two valid passes (rule 'valid'), or None."""
        rows = [r for r in st[tid]["rows"] if S.pass_valid(r)]
        return (rows[0], rows[1]) if len(rows) >= 2 else None

    def same_a1(r1, r2):
        """v2m: the two passes' final attempt-1 answers are identical on every test input, verified or not (the user,
        2026-10-31: "Identical means identical solutions ... If not both are code, then we go by the test outputs"; for two
        verified functions equal test outputs is their rule too: verified_by means every demonstration passes and the
        submitted grids are the function's outputs)."""
        g = [{int(k): v for k, v in (r.get("submissions") or {}).items()}.get(1) for r in (r1, r2)]
        return g[0] is not None and g[0] == g[1]

    def p3_length(tid):
        """v2m: the length the phase-3 thresholds read (--phase3-length): pass 1's tokens ('first', v2l), or the max or the
        mean of passes 1 and 2 (the user, 2026-10-31: "Use average"). v2n (input 22, item 2): under --phase3-rule valid, the
        pair phase3_pair returns, the one review_open and extra_pass judge."""
        pair = phase3_pair(tid) if args.phase3_rule == "valid" else None
        t = [r["generated_tokens"] for r in (pair if pair is not None else st[tid]["rows"][:2])]
        if args.phase3_length == "first":
            return t[0]
        return max(t) if args.phase3_length == "max" else sum(t) / len(t)

    def extra_pass(tid):
        """The user's phase 3 (2026-10-30): identical verified attempt-1 answers in passes 1 and 2 after a long pass 1 earn
        one more pass (a blind third pass whose answer the votes place in slot 2), scheduled with the open tasks."""
        if not args.extra_pass_identical_long:
            return False
        s = st[tid]
        if args.phase3_rule == "valid":                               # v2m: one extra pass on identical valid answers
            if s.get("extra_started") or (args.max_passes and s["started"] >= args.max_passes):
                return False
            two = phase3_pair(tid)
            return two is not None and p3_length(tid) >= args.extra_pass_identical_long and same_a1(*two)
        if len(s["rows"]) < 2 or s["started"] >= 3:
            return False
        p1, p2 = s["rows"][0], s["rows"][1]
        if p3_length(tid) < args.extra_pass_identical_long:
            return False

        def a1(r):
            subs = {int(k): v for k, v in (r.get("submissions") or {}).items()}
            recs = [x for x in r.get("submits") or [] if x.get("attempt") == 1]
            return subs.get(1), bool(recs and recs[-1].get("verified_by"))
        (g1, v1), (g2, v2) = a1(p1), a1(p2)
        return bool(v1 and v2 and g1 == g2)

    def lane_cycle(i):
        """One pass per turn; afterwards the task returns to the back of the queue unless it is complete."""
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return
            with lock:
                tid = queue.popleft() if queue else None
                if tid is not None:
                    sched = dict(valid_before=n_valid(tid), passes_before=st[tid]["started"], queued=len(queue),
                                 t=round(time.monotonic() - t_start, 1))
                    st[tid]["started"] += 1
                    st[tid]["running"] += 1
                    pass_no = st[tid]["started"]
                    running[i] = (tid, "solve")
                busy = bool(running)
            if tid is None:
                if not busy:
                    return
                time.sleep(1.0)                         # a running pass may put its task back in the queue
                continue
            refused = False
            try:
                run_solve(tid, pass_no, sched)
            except StartRefused:                  # not a pass: give the task back and retry after a pause
                refused = True
                with lock:
                    st[tid]["started"] -= 1
                    st[tid]["refused"] += 1
                    refusals[0] += 1
            except Exception:                     # noqa: BLE001 - one task's failure never stops the lane
                with open(out / "lane_errors.log", "a") as f:
                    f.write(f"{time.time():.0f} lane {i} task {tid} solve\n{traceback.format_exc()}\n")
            finally:
                with lock:
                    running.pop(i, None)
                    st[tid]["running"] -= 1
                    capped = args.max_passes and st[tid]["started"] >= args.max_passes
                    if (n_valid(tid) < args.valid_target and not capped) or extra_pass(tid):
                        queue.append(tid)
            if refused:
                time.sleep(5.0)

    def rank_key(tid):
        """The user's order after the first pass (2026-10-29): tasks with a valid answer before tasks without one, each by
        pass-1 tokens, fewest first. Easy lanes take the lowest key, hard lanes (--hard-lanes) the highest."""
        rows = st[tid]["rows"]
        t = [r["generated_tokens"] for r in rows[:1 if args.rank_length == "first" else 2]]
        length = (max(t) if args.rank_length == "max" else sum(t) / len(t)) if t else 0     # v2m: --rank-length
        return (n_valid(tid) == 0, length, pos[tid])

    gauge = (PoolGauge(base + "/metrics", args.admit_below if args.admit_below > 0 else 1.01)
             if ((args.admit_below > 0 or args.admit_forecast) and not args.fake) else None)
    cap_pages = math.ceil(int((args.profile_values or {}).get("CTX") or 82528) / 3184) + 4
    fc = Forecast(pool=args.fc_pool, reserve=args.fc_reserve, margin=args.fc_margin, horizon_s=args.fc_horizon_s,
                  z=args.fc_z, sig_task=args.fc_sigma_task, sig_prior=args.fc_sigma_prior, prior_tokens=args.fc_prior_tokens,
                  rho=args.fc_rho, opener=args.fc_opener, cap_pages=cap_pages) if args.admit_forecast else None

    def task_mu_sig(tid):
        """The task's finished passes; a first pass uses the eval prior until 20 passes of this run have finished, then
        this run's own mean log length (the user, 2026-10-29: the average length; adapts to a harder hidden set)."""
        mine = [r["generated_tokens"] for r in st[tid]["rows"] if r["generated_tokens"] > 0]
        if mine:
            return fc.mu_sig(mine)
        done = [r["generated_tokens"] for s in st.values() for r in s["rows"] if r["generated_tokens"] > 0]
        if len(done) >= 20:
            return sum(math.log(x) for x in done) / len(done), fc.sig_prior
        return fc.mu_prior, fc.sig_prior

    def identical_verified(r1, r2):
        def a1(r):
            subs = {int(k): v for k, v in (r.get("submissions") or {}).items()}
            recs = [x for x in r.get("submits") or [] if x.get("attempt") == 1]
            return subs.get(1), bool(recs and recs[-1].get("verified_by"))
        (g1, v1), (g2, v2) = a1(r1), a1(r2)
        return bool(v1 and v2 and g1 is not None and g1 == g2)

    def review_open(tid):
        """The reviewer pass (the user, 2026-10-30): pass 1 at least --review-long tokens and passes 1 and 2 without
        identical verified attempt-1 answers; once per task."""
        s = st[tid]
        if not args.review_long or s["review"] is not None or len(s["rows"]) < 2:
            return False
        if args.phase3_rule == "valid":                               # v2m: two valid passes with distinct answers
            two = phase3_pair(tid)
            return two is not None and p3_length(tid) >= args.review_long and not same_a1(*two)
        p1, p2 = s["rows"][0], s["rows"][1]
        return p3_length(tid) >= args.review_long and not identical_verified(p1, p2)

    def expected_tokens(tid, kind):
        if kind == "review":
            return args.fc_review_tokens
        toks = [r["generated_tokens"] for r in st[tid]["rows"] if r["generated_tokens"] > 0]
        return math.exp(sum(math.log(x) for x in toks) / len(toks)) if toks else args.fc_prior_tokens

    def fits_before_deadline(tid, kind):
        """The reviewer session's item 2d: a phase-3 pass (identical + long) or a review starts only if it can finish
        before the deadline at the lanes' current speed; passes of tasks short of valid passes may run to the end."""
        if args.no_phase3_guard or deadline is None:
            return True
        if kind == "solve" and not (extra_pass(tid) and n_valid(tid) >= args.valid_target):
            return True
        speed = fc.speed if fc is not None else 60.0
        return (deadline - time.monotonic()) * speed >= expected_tokens(tid, kind)

    PRIO = [] if args.phase3_priority == "rank" else [c.strip() for c in args.phase3_priority.split(",") if c.strip()]
    if PRIO and sorted(PRIO) not in (["more", "review", "third"], ["more0", "more1", "review", "third"]):
        raise SystemExit("--phase3-priority: 'rank', an order of third, more and review, or (v2p) of third, more1, review and "
                         "more0, got %r" % args.phase3_priority)

    def lane_rank(i):
        """First passes FIFO on every lane; afterwards, round by round, easy lanes take the open task with the lowest
        rank_key and the first --hard-lanes lanes the highest (the user, 2026-10-29). With --admit-below, a lane waits
        for pool room before it picks a task. From pass 3 on, a task's open review (--review-long) is its round-3 item
        and comes before its own further passes; --phase3-order reviews_first starts a round's reviews before its passes.
        Refused starts back off (5 s doubling to 60 s while they repeat on this lane)."""
        hard = i < args.hard_lanes
        lane_refused = 0
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return
            waited = gauge.wait(deadline) if gauge is not None else 0.0
            if deadline is not None and time.monotonic() >= deadline:
                return
            with lock:
                def open_(tid):
                    s = st[tid]
                    capped = args.max_passes and s["started"] >= args.max_passes
                    return (n_valid(tid) < args.valid_target and not capped) or extra_pass(tid)

                def item(tid):
                    if (PRIO and n_valid(tid) < args.valid_target and st[tid]["started"] <= 2 and open_(tid)
                            and fits_before_deadline(tid, "solve")):
                        return "solve"                     # v2m: class 'more' before the task's own review (pass 3 only)
                    if review_open(tid) and fits_before_deadline(tid, "review"):
                        return "review"
                    if open_(tid) and fits_before_deadline(tid, "solve"):
                        return "solve"
                    return None

                items = {t: item(t) for t in order if st[t]["running"] == 0}
                idle = [t for t, it in items.items() if it is not None]

                def round_of(t):
                    return min(st[t]["started"], 2) if items[t] == "review" else st[t]["started"]

                def p3_class(t):                           # v2m: third (n_valid >= target: only the extra pass is open)
                    if items[t] == "review":
                        return "review"
                    nv = n_valid(t)
                    if nv >= args.valid_target:
                        return "third"
                    if "more" in PRIO:
                        return "more"
                    return "more1" if nv >= 1 else "more0"   # v2p: one valid pass or more, or none
                tid, kind = None, None
                if idle:
                    k = min(round_of(t) for t in idle)          # rounds: pass k of every open task before pass k + 1
                    rnd = [t for t in idle if round_of(t) == k]
                    if PRIO and k >= 2:                    # v2m: the round's highest non-empty class
                        rnd = next((sub for sub in ([t for t in rnd if p3_class(t) == c] for c in PRIO) if sub), rnd)
                    elif k > 0 and args.phase3_order == "reviews_first" and any(items[t] == "review" for t in rnd):
                        rnd = [t for t in rnd if items[t] == "review"]
                    tid = rnd[0] if k == 0 else (max if hard else min)(rnd, key=rank_key)   # first passes FIFO
                    kind = items[tid]
                blocked = False
                fc_projected = None
                if tid is not None and fc is not None and gauge is not None and gauge.value is not None:
                    new = (math.log(args.fc_review_tokens), fc.sig_task) if kind == "review" else task_mu_sig(tid)
                    others = [v for j, v in pending_hard.items() if j != i] if not hard else []
                    fc_projected = fc.projected(gauge.value * args.fc_pool, list(live.values()), new, others)
                    ok = fc_projected <= fc.pool - fc.reserve - fc.margin
                    if ok:
                        pending_hard.pop(i, None)
                    if not ok and args.fc_shadow:
                        fc.shadow_blocks += 1
                    elif not ok:                           # first passes too, at the eval / run prior (the user's OK)
                        fc.blocks += 1
                        if hard and k > 0:                 # rounds only: first passes are FIFO on every lane
                            pending_hard[i] = new          # priority: easy lanes leave room for it
                        tid, blocked = None, True
                if tid is not None:
                    sched = dict(valid_before=n_valid(tid), passes_before=st[tid]["started"], lane="hard" if hard else "easy",
                                 t=round(time.monotonic() - t_start, 1), admit_wait_s=round(waited, 1),
                                 kv_at_start=gauge.value if gauge is not None else None,
                                 fc_projected=fc_projected, item=kind)
                    st[tid]["running"] += 1
                    running[i] = (tid, kind)
                    if kind == "review":
                        st[tid]["review"] = "running"
                        if fc is not None:
                            live[(tid, "review")] = dict(g=0.0, res=fc.opener, mu=math.log(args.fc_review_tokens),
                                                         sig=fc.sig_task)
                    else:
                        if n_valid(tid) >= args.valid_target:         # v2m: the task's extra pass (class 'third')
                            st[tid]["extra_started"] = True
                        st[tid]["started"] += 1
                        pass_no = st[tid]["started"]
                        if fc is not None:
                            mu, sig = task_mu_sig(tid)
                            live[(tid, pass_no)] = dict(g=0.0, res=fc.opener, mu=mu, sig=sig)
                busy = bool(running)
            if tid is None:
                if not busy and not blocked:
                    return
                time.sleep(5.0 if blocked else 1.0)
                continue
            refused = False
            try:
                if kind == "review":
                    run_review_job(tid, sched)
                else:
                    run_solve(tid, pass_no, sched)
            except StartRefused:                  # not a pass: give the task back and retry after a pause
                refused = True
                with lock:
                    if kind == "review":
                        st[tid]["review"] = None
                        st[tid]["review_refused"] += 1
                    else:
                        st[tid]["started"] -= 1
                        st[tid]["refused"] += 1
                    refusals[0] += 1
            except Exception:                     # noqa: BLE001 - one task's failure never stops the lane
                with open(out / "lane_errors.log", "a") as f:
                    f.write(f"{time.time():.0f} lane {i} task {tid} {kind}\n{traceback.format_exc()}\n")
            finally:
                with lock:
                    running.pop(i, None)
                    st[tid]["running"] -= 1
                    if kind == "review" and st[tid]["review"] == "running":
                        st[tid]["review"] = "done"        # recorded, or failed with an error: not retried
            if refused:
                lane_refused += 1
                time.sleep(min(60.0, 5.0 * 2 ** (lane_refused - 1)))
            else:
                lane_refused = 0

    target = {"cycle": lane_cycle, "rank": lane_rank}.get(args.policy, lane)
    threads = [threading.Thread(target=target, args=(i,), daemon=True) for i in range(max(1, args.lanes))]
    for th in threads:
        th.start()
    last = 0.0
    tick = 0.0
    prev_gen = None                                           # (time, server generation counter) for Forecast's speed
    while any(th.is_alive() for th in threads):
        time.sleep(1.0)
        if time.monotonic() - tick >= args.timeline_every:      # busy lanes and the server's generated-token counter
            tick = time.monotonic()
            gen_counter = None if args.fake else generation_counter(fetch(base + "/metrics", timeout=5))
            if fc is not None and gen_counter is not None:
                if prev_gen is not None and tick > prev_gen[0] and gen_counter >= prev_gen[1]:
                    rate = (gen_counter - prev_gen[1]) / (tick - prev_gen[0]) / max(1, len(running))
                    if rate > 0:
                        fc.speed = 0.5 * fc.speed + 0.5 * rate
                prev_gen = (tick, gen_counter)
            with lock:
                row = dict(t=round(tick - t_start, 1), running=len(running), waiting=sum(s["started"] == 0 for s in st.values()),
                           done=sum(1 for s in st.values() if s["rows"]), attempts=len(results), queued=len(queue),
                           complete=sum(n_valid(t) >= args.valid_target for t in order),
                           passes_started=sum(s["started"] for s in st.values()),
                           finished_tokens=sum(r["generated_tokens"] for r in results),
                           server_generation_tokens=gen_counter,
                           kv_usage=gauge.value if gauge is not None else None,
                           admit_waits=gauge.waits if gauge is not None else None,
                           admit_wait_s=round(gauge.wait_s, 1) if gauge is not None else None,
                           start_refusals=refusals[0],
                           fc_blocks=fc.blocks if fc is not None else None,
                           fc_shadow_blocks=fc.shadow_blocks if fc is not None else None,
                           fc_pending_hard=len(pending_hard),
                           fc_speed=round(fc.speed, 1) if fc is not None else None)
            with open(out / "timeline.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        if time.monotonic() - last >= args.status_every:
            last = time.monotonic()
            with lock:
                done = sum(1 for s in st.values() if s["rows"])
                gen = sum(r["generated_tokens"] for r in results)
                solved = sum(T.score_task(st[t]["task"], entries[t]) or 0 for t in order)
                started = sum(s["started"] for s in st.values())
                waiting = sum(s["started"] == 0 for s in st.values())
                complete = sum(n_valid(t) >= args.valid_target for t in order)
            el = time.monotonic() - t_start
            print(f"[{time.strftime('%H:%M:%S')}] done {done}/{len(tasks)} running {len(running)} waiting {waiting} | "
                  f"passes started {started}, attempts ended {len(results)}, tasks with {args.valid_target} valid passes "
                  f"{complete} | generated {gen:,} (finished attempts) | "
                  f"{gen / max(el, 1):.0f} tok/s | score {solved:.2f}", flush=True)
    for th in threads:
        th.join()
    wall = time.monotonic() - t_start
    if not args.fake:
        (out / "metrics_after.txt").write_text(fetch(base + "/metrics"))
    summarize(out, tasks, st, results, entries, wall, args)


def attempt_programs(kernel, res):
    """{slot: {fn, source, passed, total}} for the recorded answers that came from a program (the verifying function
    for verified grids), rebuilt from the kernel's replay library at the attempt's end."""
    lib = {e["name"]: e for e in kernel.library}
    out = {}
    for a in (1, 2):
        recs = [s for s in res.submits if s["attempt"] == a]
        if not recs:
            continue
        last = recs[-1]
        name = last.get("verified_by") or (last["fn"] if last["fn"] != "grids" else None)
        src = program_source(lib, name) if name else None
        if src:
            passed = last["train_total"] if last.get("verified_by") else last.get("train_passed")
            out[str(a)] = dict(fn=name, source=src, passed=passed, total=last.get("train_total"))
    return out


def judge_candidate(task, full, rows, fmt, kernel_kwargs):
    """A judge Candidate for one full answer: shown with a program only when a program recorded for it re-runs to
    exactly these grids (program.verify); the program with the most demonstration passes is preferred."""
    progs = []
    for o in full["origins"]:
        p, a = o.split(":")                            # origins index the rows list (slots.full_answers), not pass_no
        if not a.startswith("a"):
            continue
        row = rows[int(p[1:]) - 1]
        prog = (row.get("programs") or {}).get(a[1:])
        if prog and prog.get("source"):
            progs.append(prog)
    progs.sort(key=lambda g: -(g.get("passed") or 0))
    for prog in progs:
        v = verify(task.sandbox_payload(fmt), prog["source"], prog["fn"], timeout_s=kernel_kwargs.get("timeout_s", 60.0))
        if not v.get("error") and v["outputs"] == full["outputs"]:
            return Candidate(outputs=full["outputs"], source=prog["source"], passed=v["passed"], total=v["total"],
                             origins=list(full["origins"]))
    return Candidate(outputs=full["outputs"], origins=list(full["origins"]))


def summarize(out, tasks, st, results, entries, wall, args):
    total, per_task = T.score_submission(tasks, entries)
    hits, n_out = T.score_outputs(tasks, entries)
    solves = [r for r in results if r["kind"] == "solve"]
    judges = [r for r in results if r["kind"] == "judge"]
    reviews = [r for r in results if r["kind"] == "review"]
    plain = {t.task_id: T.submission_entry(t, S.slot_outputs(st[t.task_id]["rows"], len(t.test))) for t in tasks}
    plain_total, _ = T.score_submission(tasks, plain)          # the votes alone, without the reviews' P3 slot 2
    plain_hits, _ = T.score_outputs(tasks, plain)
    toks = sorted(r["generated_tokens"] for r in solves) or [0]
    ends = {}
    for r in solves:
        ends[r["end"]] = ends.get(r["end"], 0) + 1
    by_pass = {}
    for r in solves:
        b = by_pass.setdefault(r["pass_no"], dict(attempts=0, tokens=0, ends={}))
        b["attempts"] += 1
        b["tokens"] += r["generated_tokens"]
        b["ends"][r["end"]] = b["ends"].get(r["end"], 0) + 1
    upto = {}
    for k in sorted(by_pass):                          # the slot rule over passes 1..k only
        sc = 0.0
        for t in tasks:
            rows = [r for r in st[t.task_id]["rows"] if r["pass_no"] <= k]
            sc += T.score_task(t, T.submission_entry(t, S.slot_outputs(rows, len(t.test)))) or 0
        upto[k] = round(sc, 2)
    bands, valid_hist, pass_hist = {}, {}, {}
    for t in tasks:
        rows = st[t.task_id]["rows"]
        b = S.band(rows)
        bands[b] = bands.get(b, 0) + 1
        v = sum(S.pass_valid(r) for r in rows)
        valid_hist[v] = valid_hist.get(v, 0) + 1
        pass_hist[len(rows)] = pass_hist.get(len(rows), 0) + 1
    model_only = sum((r["score"] or 0) for r in solves if r["pass_no"] == 1 and r["answer_source"] == "model")
    overfit = sorted({r["task_id"] for r in solves if r["train_complete_submit"] and (r["score"] or 0) < 1})
    spec = {}
    before, after = out / "metrics_before.txt", out / "metrics_after.txt"
    if before.exists() and after.exists():
        b, a = spec_counters(before.read_text()), spec_counters(after.read_text())
        spec = {k: a.get(k, 0) - b.get(k, 0) for k in a}
        drafts = spec.get("vllm:spec_decode_num_drafts_total") or spec.get("vllm:spec_decode_num_drafts")
        acc = spec.get("vllm:spec_decode_num_accepted_tokens_total") or spec.get("vllm:spec_decode_num_accepted_tokens")
        if drafts:
            spec["mean_acceptance_length"] = 1 + acc / drafts
    timeline = timeline_stats(out / "timeline.jsonl", args.lanes)
    attempted = sum(1 for t in tasks if st[t.task_id]["rows"])
    all_tokens = sum(r["generated_tokens"] for r in results)
    score = dict(tasks=len(tasks), attempted=attempted, not_started=len(tasks) - attempted, deadline_s=args.deadline_s,
                 profile=args.profile, profile_values=args.profile_values, lanes=args.lanes, budget=args.budget,
                 passes=args.passes, max_passes=max(args.passes, args.max_passes or args.passes), judge=args.judge,
                 corrections=args.corrections, score=total, score_by_passes=upto, model_only_score=model_only,
                 outputs_right=hits, outputs=n_out, competition_pct=round(100 * hits / n_out, 2) if n_out else None,
                 timeline=timeline, per_task=per_task, ends=ends, by_pass=by_pass, bands=bands, overfit_tasks=overfit,
                 policy=args.policy, valid_target=args.valid_target, tasks_by_valid_passes=valid_hist,
                 tasks_by_passes=pass_hist,
                 reviews=dict(n=len(reviews), tokens=sum(r["generated_tokens"] for r in reviews),
                              ends={e: sum(1 for r in reviews if r["end"] == e) for e in sorted({r["end"] for r in reviews})},
                              score_without_reviews=plain_total, outputs_right_without_reviews=plain_hits),
                 judges=dict(n=len(judges), picked=sum(1 for r in judges if r.get("order")),
                             tokens=sum(r["generated_tokens"] for r in judges)),
                 generated_tokens=dict(total=sum(toks), median=statistics.median(toks), p90=toks[int(0.9 * (len(toks) - 1))],
                                       max=toks[-1], all_sessions=all_tokens),
                 wall_seconds=round(wall, 1), tok_per_s=round(all_tokens / max(wall, 1), 1), spec_decode=spec)
    T.write_json(out / "score.json", score)
    lines = [f"# ARC-2 run {out.name}", "",
             f"- Competition metric: **{hits}/{n_out} test outputs ({score['competition_pct']}%)**; per-task score {total:.2f}.",
             f"- Tasks {len(tasks)}, attempted {attempted} (not started: {len(tasks) - attempted}); score **{total:.2f}**; "
             f"by passes {upto}. Lanes {args.lanes}, budget {args.budget:,} per attempt, passes {args.passes} "
             f"(max {score['max_passes']}), judge {args.judge}, corrections {args.corrections}, deadline "
             f"{args.deadline_s:.0f} s, profile {args.profile or '-'}.",
             f"- Ends: {ends}. Bands at the end: {dict(sorted(bands.items()))}. Judge sessions: {score['judges']}.",
             f"- Policy {args.policy}: tasks by valid passes {dict(sorted(valid_hist.items()))}; by passes "
             f"{dict(sorted(pass_hist.items()))}.",
             f"- Generated tokens per solve attempt: median {score['generated_tokens']['median']:,.0f}, "
             f"p90 {score['generated_tokens']['p90']:,}, max {score['generated_tokens']['max']:,}; all sessions "
             f"{all_tokens:,} in {wall / 3600:.2f} h ({score['tok_per_s']:.0f} tok/s).",
             f"- Train-complete submissions that were wrong on test (overfit), tasks: {len(overfit)}.",
             f"- Spec decode: {spec.get('mean_acceptance_length', 'n/a')}.",
             f"- Timeline: peak 10-min {timeline.get('peak_10min_rate', 'n/a')} tok/s, mean {timeline.get('mean_rate', 'n/a')} "
             f"(mean/peak {timeline.get('mean_over_peak', 'n/a')}); share of busy time below full lanes "
             f"{timeline.get('share_below_full_lanes', 'n/a')}.", "",
             "| task | passes | ends | tokens | band | score |", "|---|---|---|---|---|---|"]
    for t in tasks:
        rows = sorted(st[t.task_id]["rows"], key=lambda r: r["pass_no"])
        lines.append(f"| {t.task_id} | {len(rows)} | {', '.join(r['end'] for r in rows) or '-'} | "
                     f"{sum(r['generated_tokens'] for r in rows):,} | {S.band(rows)} | "
                     f"{per_task[t.task_id] if per_task[t.task_id] is not None else '-'} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]), flush=True)


if __name__ == "__main__":
    main()
