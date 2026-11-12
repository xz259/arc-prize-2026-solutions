"""l4port_ppbalance.py (2026-10-31, ARC2 - main; the user: "increase lanes as many as possible, increase throughput, reduce harness
friction"): balance the two pipeline microbatches under persistent-session traffic. Env ARC3_L4PORT_PP_BALANCE=1 (default off);
installed by l4port_evpp.py v5 at the end of EngineCore.__init__ when PP > 1.

Why (l4n3d's step timer, 2.4 h of harness traffic on the L4): 19% of the decode steps took over twice the median period, 40% of the
span, and 94% of them scheduled 0-15 tokens: near-empty microbatches. Under V2 + PP + async scheduling a request decodes only every
PP steps (the scheduler's next_decode_eligible_step), so it keeps one microbatch parity; a request that resumes after a tool call
joins whichever parity is scheduled next and nothing moves requests back, so the parities drift apart until one microbatch is
nearly empty and the pipeline idles half the time. The decode bench never shows it: its lanes never pause.

What: each schedule() call schedules at most ceil(active / PP) running requests, where active = the running requests in flight or
with tokens to schedule. A request held back by the cap at step s is still eligible at s + 1 (it is not in flight), so it is
scheduled into the other microbatch: the parities converge to equal sizes. The cap sits in production's own guard of the running
loop (engine_patch's _arc3_persistent_skip_inflight, the global of its compiled schedule) and counts only requests the stock loop
would schedule now (past the PP cadence check, with tokens to schedule); new requests from the waiting queue are not capped.
Receipts: one line at install, then every 20,000 steps: steps, steps where the cap held a request back, requests held back.
v2 (2026-10-31 13:3x; l4v1b's two starts failed at install, 'carries no _arc3_persistent_skip_inflight guard'): in test mode the
step timer wraps Scheduler.schedule in a closure (its __wrapped__ is engine_patch's first compile, no longer called) and stallguard
swaps its own compile into that closure cell, so the guard lives in the globals of a function inside the wrapper. The install now
walks closures first, then __wrapped__ (the order stallguard's _find_compiled uses), to the function whose globals hold the guard;
the receipt names it. The schedule wrapper still goes outermost."""
import math
import os

ENV = "ARC3_L4PORT_PP_BALANCE"
GUARD = "_arc3_persistent_skip_inflight"
STATE = dict(cap=None, n=0, sched=None, steps=0, capped_steps=0, held=0, capped_now=False)


def _pending(r):
    return r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens


def _find_guarded(fn, depth=0):
    """The schedule function that reads production's guard from its globals: fn itself, else (depth first) a callable in its
    closure cells, else its __wrapped__. Closures come first: a wrapper's __wrapped__ can be a stale function it no longer calls."""
    g = getattr(fn, "__globals__", None)
    if g is not None and GUARD in g and GUARD in getattr(getattr(fn, "__code__", None), "co_names", ()):
        return fn
    if depth >= 4:
        return None
    nxt = []
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            nxt.append(cell.cell_contents)
        except ValueError:
            continue
    nxt.append(getattr(fn, "__wrapped__", None))
    for c in nxt:
        if callable(c):
            f = _find_guarded(c, depth + 1)
            if f is not None:
                return f
    return None


def install(core, pp, say=print):
    if os.environ.get(ENV, "0") != "1" or pp <= 1:
        return False
    sched = core.scheduler
    cls = type(sched)
    if getattr(cls, "_l4port_ppbalance", False):
        return True
    fn = cls.schedule
    target = _find_guarded(fn)
    if target is None:
        raise RuntimeError("L4 port ppbalance: no function under Scheduler.schedule reads the %s guard (production's patch)" % GUARD)
    g = target.__globals__
    orig_guard = g[GUARD]

    def guard(request):
        if orig_guard(request):
            return True
        s, cap = STATE["sched"], STATE["cap"]
        if s is None or cap is None:
            return False
        if s.current_step < getattr(request, "next_decode_eligible_step", 0) or _pending(request) <= 0:
            return False                       # the stock loop skips it (cadence) or has nothing to schedule: not counted
        if STATE["n"] >= cap:
            STATE["held"] += 1
            STATE["capped_now"] = True
            return True                        # held back: eligible next step, in the other microbatch
        STATE["n"] += 1
        return False

    def schedule(self, *a, **k):
        active = sum(1 for r in self.running if getattr(r, "num_in_flight_tokens", 0) > 0 or _pending(r) > 0)
        STATE.update(sched=self, cap=max(1, math.ceil(active / pp)) if active else None, n=0, capped_now=False)
        try:
            return fn(self, *a, **k)
        finally:
            STATE["sched"] = None
            STATE["steps"] += 1
            STATE["capped_steps"] += int(STATE["capped_now"])
            if STATE["steps"] % 20000 == 0:
                say("[l4port] ppbalance: %d steps, %d capped, %d requests held back" % (STATE["steps"], STATE["capped_steps"], STATE["held"]))

    g[GUARD] = guard
    schedule.__wrapped__ = fn
    cls.schedule = schedule
    cls._l4port_ppbalance = True
    where = getattr(getattr(target, "__code__", None), "co_filename", "?").rsplit("/", 1)[-1]
    say("[l4port] ppbalance installed: at most ceil(active / %d) running requests per step (the microbatch parities balanced); "
        "guard in %s%s" % (pp, where, "" if target is fn else ", inside %s.%s" % (getattr(fn, "__module__", "?"), getattr(fn, "__qualname__", "?"))))
    return True
