"""stage2_v1 'poolpress' part (2026-10-14, Claude fork 12a92b45; THROUGHPUT_LEVERS §7 design (a) revised in
POOLPRESS_DESIGN_2026-10-14.md): KV overcommit by N' = N + 2 / N + 3 lanes with pool-pressure eviction at turn starts.

The pool is sized for N full lanes but N' sessions run. At every turn start the session service estimates the pool's page
use over all live sessions (the launch's own formula: 1 + sum(ceil((resident + padding) / 3,104) + 3), running turns projected
forward by a growth allowance G) and, when the projection plus this turn's prompt would pass POOL - RESERVE, lowers THIS
session's budget to the dynamic cap C = ((POOL - RESERVE - 1) / N' - 3) x 3,104 if the session is above it (fullest-first by
construction; lanes below the cap are untouched). The budget goes through padbudget's base (session._arc3_padbudget_base) so
padbudget's own wrapper, the frequency-snapshot condition, the planner's trigger, the 413 check and the max_tokens clamps all
see it for this turn; the base is restored when the turn returns. Nothing changes in the planner or the engine: the normal
eviction transaction removes max(quantum, capacity_needed) by QSA frequency. A pool-room clamp of max_tokens guards the
interval between decisions. Requires padbudget (same pins); registered after it so this wrapper is the outer one.
Knobs: ARC3_S2_POOLPRESS_POOL (pages; required, the launch's block count), ARC3_S2_POOLPRESS_LANES (N'; required),
ARC3_S2_POOLPRESS_RESERVE (8), ARC3_S2_POOLPRESS_GROWTH (2048 tokens per running turn), ARC3_S2_POOLPRESS_INCOMING (2048:
the prompt estimate before prepare()), ARC3_S2_POOLPRESS_SWITCH=<file> ('off' disables the cut, estimates still run).
Errors leave production's behaviour (counted).

v2 (2026-10-14, with the stallguard engine part; STALLGUARD_DESIGN_2026-10-14.md): the cap is floored to whole pages,
C = (floor((POOL - RESERVE - 1) / N') - 3) x 3,104, and applied as the padbudget base itself (min(base, C)), so a capped lane's
resident + padding stays within C and N' capped lanes provably fit (v1 kept a fractional page and added the padding back);
an emergency cap one page lower applies at every turn start while the engine reports an open stall or fewer free pages than the
reserve (ARC3_S2_POOLPRESS_SG_STATE, the stallguard state file); the max_tokens clamp is off by default
(ARC3_S2_POOLPRESS_CLAMP=1 restores it). ARC3_S2_POOLPRESS_FLOOR=0 restores v1's cap and budget."""
import dataclasses
import json
import os
import threading
import time

PINS = {'session_service': '7ce0eaef072e45354cfae8476719313a401387ea379b3fe9c3ca9c1ab40193e8',
        'persistent_endpoint': 'ceead0163e99ffcdf3f376affee74a597a343ef06781a737e0b09bf9a56b0001'}
PAGE, FIXED, SHARED = 3104, 3, 1
# gdn32 (2026-10-22, Claude session 099768dc): the attention block in tokens follows the GDN page (an FP32 state doubles it to 6,208);
# ARC3_S2_POOLPRESS_PAGE carries the engine's block size (unset = 3,104, the 16-bit stack, unchanged). FIXED / SHARED count pages.
PAGE = int(os.environ.get('ARC3_S2_POOLPRESS_PAGE', '') or PAGE)
POOL = int(os.environ.get('ARC3_S2_POOLPRESS_POOL', '0') or 0)
POOL_FILE = os.environ.get('ARC3_S2_POOLPRESS_POOL_FILE')      # written by the launcher after it reads the engine's block count


def _pool():
    global POOL
    if not POOL and POOL_FILE:
        try:
            POOL = int(open(POOL_FILE).read().strip() or 0)
            STATS['pool'] = POOL
        except (OSError, ValueError):
            POOL = 0
    return POOL
LANES = int(os.environ.get('ARC3_S2_POOLPRESS_LANES', '0') or 0)
RESERVE = int(os.environ.get('ARC3_S2_POOLPRESS_RESERVE', '8') or 8)
GROWTH = int(os.environ.get('ARC3_S2_POOLPRESS_GROWTH', '2048') or 2048)
INCOMING = int(os.environ.get('ARC3_S2_POOLPRESS_INCOMING', '2048') or 2048)
FLOOR = os.environ.get('ARC3_S2_POOLPRESS_FLOOR', '1') != '0'
CLAMP = os.environ.get('ARC3_S2_POOLPRESS_CLAMP', '0') == '1'
SG_STATE = os.environ.get('ARC3_S2_POOLPRESS_SG_STATE', '/dev/shm/arc3_stallguard.json')
_SG = dict(t=0.0, v=None)
STATS = dict(turns=0, pressure_turns=0, cuts=0, cut_tokens=0, max_pages_now=0, max_pages_projected=0, min_free_projected=None,
             clamps=0, clamp_tokens=0, switch_off_turns=0, errors=0, last_error=None, pool=POOL, lanes=LANES, reserve=RESERVE,
             growth=GROWTH, cap=None, last_cuts=[], emergency_turns=0, floor=FLOOR, clamp=CLAMP)     # last_cuts: the newest 20 (time, pages_now, pages_projected, cap, cut)
_SWITCH = dict(path=os.environ.get('ARC3_S2_POOLPRESS_SWITCH'), t=0.0, on=True)
STATS_FILE = os.environ.get('ARC3_S2_POOLPRESS_STATS')      # the API server writes STATS here every 20 s (receipts come from other processes)
_WRITER = dict(started=False)


def _start_writer():
    if _WRITER['started'] or not STATS_FILE:
        return
    _WRITER['started'] = True

    def run():
        while True:
            try:
                tmp = STATS_FILE + '.tmp'
                with open(tmp, 'w') as fh:
                    json.dump(dict(STATS, pid=os.getpid(), written=time.time()), fh)
                os.replace(tmp, STATS_FILE)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(20)
    threading.Thread(target=run, name='arc3-s2-poolpress-stats', daemon=True).start()


def _switch_on():
    s = _SWITCH
    if s['path'] is None:
        return True
    now = time.monotonic()
    if now - s['t'] > 1.0:
        s['t'] = now
        try:
            s['on'] = open(s['path']).read().strip().lower() != 'off'
        except OSError:
            s['on'] = True
    return s['on']


def lane_pages(resident, padding=0):
    r = max(0, int(resident) + int(padding))
    return (-(-r // PAGE) if r > 0 else 0) + FIXED


def dynamic_cap(pool=None, lanes=None, reserve=None, emergency=False):
    pool = _pool() if pool is None else pool; lanes = LANES if lanes is None else lanes; reserve = RESERVE if reserve is None else reserve
    if FLOOR:
        per_lane = (pool - reserve - SHARED) // max(1, lanes) - FIXED - (1 if emergency else 0)
        return int(max(1, per_lane) * PAGE)
    per_lane = (pool - reserve - SHARED) / max(1, lanes) - FIXED
    return int(max(1.0, per_lane) * PAGE)


def _sg_state():
    """The stallguard engine part's state (open stalls, free blocks), read at most every 0.2 s; None when absent or stale."""
    now = time.monotonic()
    if now - _SG['t'] > 0.2:
        _SG['t'] = now
        try:
            v = json.load(open(SG_STATE))
            _SG['v'] = v if time.time() - float(v.get('t', 0)) < 5.0 else None
        except (OSError, ValueError):
            _SG['v'] = None
    return _SG['v']


def _padding(session):
    try:
        from . import padbudget
        return padbudget.group_padding(session.residency.evicted)
    except Exception:  # noqa: BLE001
        return 0


def pool_pages(sessions, growth=None):
    """(pages_now, pages_projected) over the live sessions of a SessionManager."""
    growth = GROWTH if growth is None else growth
    now = proj = SHARED
    for s in sessions:
        if getattr(s, 'state', '') in ('closed', 'error'):
            continue
        res = getattr(getattr(s, 'residency', None), 'resident_tokens', 0) or 0
        pad = _padding(s)
        now += lane_pages(res, pad)
        proj += lane_pages(res + (growth if getattr(s, 'state', '') == 'running' else 0), pad)
    return now, proj


def decide(sessions, session, pool=None, lanes=None, reserve=None, incoming=None):
    """the cut for `session` at its turn start: (cap_tokens or None, info)."""
    pool = _pool() if pool is None else pool; lanes = LANES if lanes is None else lanes
    reserve = RESERVE if reserve is None else reserve; incoming = INCOMING if incoming is None else incoming
    now, proj = pool_pages(sessions)
    res = getattr(session.residency, 'resident_tokens', 0) or 0; pad = _padding(session)
    prompt_pages = lane_pages(res + incoming, pad) - lane_pages(res, pad)
    free_after = pool - reserve - (proj + prompt_pages)
    info = dict(pages_now=now, pages_projected=proj, free_after=free_after, pressure=free_after < 0, cap=None, cut=0)
    STATS['max_pages_now'] = max(STATS['max_pages_now'], now)
    STATS['max_pages_projected'] = max(STATS['max_pages_projected'], proj)
    STATS['min_free_projected'] = free_after if STATS['min_free_projected'] is None else min(STATS['min_free_projected'], free_after)
    sg = _sg_state()
    emergency = bool(sg) and (int(sg.get('open_stalls', 0)) > 0 or int(sg.get('free_blocks', 1 << 30)) < reserve)
    info['emergency'] = emergency
    if emergency:
        info['pressure'] = True
        STATS['emergency_turns'] += 1
    if free_after >= 0 and not emergency:
        return None, info
    cap = dynamic_cap(pool, lanes, reserve, emergency=emergency); info['cap'] = cap; STATS['cap'] = cap
    if res + incoming <= cap:
        return None, info
    info['cut'] = res + incoming - cap
    return cap, info


def patch_session_service(mod):
    cls = mod.SessionManager
    original = cls._run_turn

    async def _run_turn(self, session, payload, *, initial):
        saved = None
        try:
            if _pool() and LANES and getattr(session, 'residency', None) is not None:
                STATS['turns'] += 1
                _start_writer()
                if getattr(self.backend, '_arc3_poolpress_manager', None) is not self:
                    self.backend._arc3_poolpress_manager = self      # the turn clamp needs the other sessions
                cap, info = decide(list(self.sessions.values()), session)
                if info['pressure']:
                    STATS['pressure_turns'] += 1
                if cap is not None:
                    if not _switch_on():
                        STATS['switch_off_turns'] += 1
                    else:
                        from . import padbudget
                        base = padbudget._base(session)        # padbudget's cached base (initialised here if needed)
                        pad = _padding(session)
                        want = min(base, cap) if FLOOR else min(base, cap + pad)   # v2: padbudget's base = the cap (resident + padding <= cap)
                        if want < base:
                            saved = base
                            session._arc3_padbudget_base = want
                            STATS['cuts'] += 1
                            STATS['cut_tokens'] += info['cut']
                            STATS['last_cuts'] = (STATS['last_cuts'] + [(round(time.time()), info['pages_now'], info['pages_projected'], cap, info['cut'])])[-20:]
                            try:
                                self.audit('pool_pressure', session, operation_id=session.current_operation,
                                           pages_now=info['pages_now'], pages_projected=info['pages_projected'],
                                           free_after=info['free_after'], cap=cap, budget=want, cut=info['cut'])
                            except Exception:  # noqa: BLE001
                                pass
        except Exception as exc:  # noqa: BLE001  production's budget stands
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
        try:
            return await original(self, session, payload, initial=initial)
        finally:
            if saved is not None:
                session._arc3_padbudget_base = saved

    _run_turn.__wrapped__ = original
    cls._run_turn = _run_turn


def patch_persistent_endpoint(mod):
    cls = mod.VllmBackend
    original = cls.turn

    async def turn(self, session, turn_input, *, initial):
        try:
            sampling = getattr(turn_input, 'sampling', None)
            manager = getattr(self, '_arc3_poolpress_manager', None)
            if _pool() and LANES and isinstance(sampling, dict) and type(sampling.get('max_tokens')) is int and manager is not None:
                others = [s for s in manager.sessions.values() if s is not session]
                now, proj = pool_pages(others)
                res = session.residency.resident_tokens; pad = _padding(session)
                room_pages = _pool() - RESERVE - proj - lane_pages(res, pad)
                room_tokens = room_pages * PAGE - ((res + pad) % PAGE) if room_pages > 0 else 0
                if sampling['max_tokens'] > max(1, room_tokens):
                    new = max(1, room_tokens)
                    STATS['clamps'] += 1
                    STATS['clamp_tokens'] += sampling['max_tokens'] - new
                    sampling['max_tokens'] = new
        except Exception as exc:  # noqa: BLE001  the session service's clamp stands
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
        return await original(self, session, turn_input, initial=initial)

    turn.__wrapped__ = original
    cls.turn = turn


PATCHERS = {'session_service': patch_session_service}
if CLAMP:
    PATCHERS['persistent_endpoint'] = patch_persistent_endpoint
