"""stage2_v1 'stallguard' part (2026-10-14, Claude fork 12a92b45 / opus-5-5; STALLGUARD_DESIGN_2026-10-14.md, IE's two-tier design):
KV overcommit without preemptions. Engine core only.

Tier 1, stall instead of preempt: production compiles Scheduler.schedule from the pinned stock source with one guard
(engine_patch._patched_schedule_ast). At the first Scheduler construction (after engine_patch.install()), this part compiles the
same function with two more edits in the running loop and installs it instead:
  - in the allocation loop, right after 'if new_blocks is not None: break', a request whose blocks cannot be allocated is marked
    stalled and the loop exits with new_blocks None: no victim is chosen, nothing is freed or preempted;
  - after the loop, 'if new_blocks is None: break' becomes: a stalled request is skipped ('req_index += 1; continue'), so the
    lanes that still have room in their last page keep decoding. The request keeps its pages, worker state, ctxtail store,
    drafts and GDN state and is retried next step (MRv2 keeps unscheduled running requests; production already skips
    persistent requests with a step in flight the same way).
Tier 2, breaker: after the stock update_from_output, a persistent request stalled for >= ARC3_S2_STALLGUARD_BREAK_S (1.0 s wall
time: with every lane stalled the engine runs empty steps quickly) and settled (no output placeholders, nothing in flight, the
AR frontier: len(all) - computed == 1, prefill done) has its turn ended: status FINISHED_LENGTH_CAPPED, then the patched
_handle_stopped_request (engine_patch's frontier checks, persist's stop fixup: the same state a length stop at the end of its last
step would have), removal from running, and an EngineCoreOutput with no new tokens and finish_reason LENGTH (the stock pattern
for error finishes). The session pauses holding its pages; the harness continues a length stop; the continuation's turn start is
where the pool-pressure cut acts.
Never preempt: _preempt_request raises for a persistent request (a safety net; Tier 1 never calls it).
Admission reserve: not here; vLLM's own --watermark keeps a share of blocks free from waiting (resume) admissions, and
scheduler_reserve_full_isl (default on) admits a resume only if its whole prompt fits.
State for the session service: ARC3_S2_STALLGUARD_STATE (default /dev/shm/arc3_stallguard.json): open stalls, the oldest stall's
age, free and total blocks, breaker stops; written when it changes (at most every 0.2 s).
ARC3_S2_STALLGUARD=0 installs nothing new (stock preemption).
v3 (2026-10-15 04:3x; IE review, user-approved forcing arms): no exception leaves stallguard on any known path.
  - The breaker is atomic: the pre-fixup checks (_precheck; v3.1: not engine_patch.assert_settled, the post-fixup invariant
    that rejects a lane holding its drafts) run before any state changes. A failure in the stop path restores the status,
    removes a partial waiting-queue entry and leaves the request stalled, marked so it is not broken again. A finished result
    (a stop that did not pause) is freed as the stock update_from_output does. Every failure is counted and logged, never raised.
  - reset_prefix_cache(reset_running_requests=True) with a persistent request running refuses (returns False, counted) instead of
    reaching the preemption guard. The guard itself stays a raise. It is unreachable: the pinned scheduler has two
    _preempt_request call sites, the running loop (transformed: the stall check precedes any victim choice) and
    reset_prefix_cache (refused upstream). A refusal after a caller's pop has no safe generic form (orphaned request, a
    `while self.running` loop, or a worker that drops the state of a request the scheduler keeps).
  - Test-only knobs (default off): ARC3_S2_STALLGUARD_INJECT=EVERY_S:HOLD_S refuses allocate_slots for one chosen running
    persistent decode-phase request for HOLD_S seconds every EVERY_S seconds (rotating over lanes; the first event
    ARC3_S2_STALLGUARD_INJECT_START_S after the first allocation), so the normal stall path runs and the breaker fires with
    certainty. ARC3_S2_STALLGUARD_AUDIT=<jsonl> records each breaker stop (state before / after the stop path, invariant checks)
    and the continuation's first scheduled step (committed-prefix digest, blocks), and keeps ARC3_S2_STALLGUARD_BROKEN
    (default /dev/shm/arc3_sg_broken.json: request id -> first block id per KV group) for a worker-side dump hook."""
import ast
import json
import os
import time

PINS = {'vllm.v1.core.sched.scheduler': 'c710f49e41e974e5b7b8f1cd2f5fb9722523f4da0165252e16351199ebd03124'}
ENABLED = os.environ.get('ARC3_S2_STALLGUARD', '1') != '0'
T_BREAK = float(os.environ.get('ARC3_S2_STALLGUARD_BREAK_S', '1.0') or 1.0)
T_WARN = float(os.environ.get('ARC3_S2_STALLGUARD_WARN_S', '30') or 30)
STATE_FILE = os.environ.get('ARC3_S2_STALLGUARD_STATE', '/dev/shm/arc3_stallguard.json')
INJECT = tuple(float(x) for x in os.environ['ARC3_S2_STALLGUARD_INJECT'].split(':')) if os.environ.get('ARC3_S2_STALLGUARD_INJECT') else None
INJECT_START = float(os.environ.get('ARC3_S2_STALLGUARD_INJECT_START_S', '0') or 0) or (INJECT[0] if INJECT else 0.0)
INJECT_RECENT = 8
AUDIT = os.environ.get('ARC3_S2_STALLGUARD_AUDIT', '')
BROKEN_FILE = os.environ.get('ARC3_S2_STALLGUARD_BROKEN', '/dev/shm/arc3_sg_broken.json')
STATS = dict(installed=False, stall_episodes=0, stalled_steps=0, stall_seconds=0.0, max_stall_s=0.0, breaker_stops=0,
             breaker_unsettled_skips=0, long_stall_warnings=0, preempt_refused=0, open_stalls=0, min_free_blocks=None,
             errors=0, last_error=None, breaker_precheck_failed=0, breaker_stop_failed=0, breaker_emit_failed=0, breaker_finished=0,
             breaker_nobreak_skips=0, reset_refused=0, guard_errors=0, inject_events=0, inject_refusals=0, audit_breaks=0,
             audit_resumes=0, audit_violations=0, audit_evicted_between=0)
_OPEN = {}           # request id -> stall start (monotonic)
_BROKEN = {}         # request id -> break record, until the continuation's first scheduled step (audit only)
_OPEN_INFO = {}      # request id -> dict(t0 wall time, blocks: first block id per KV group), for the state file's 'open' list
_LAST = {}           # audit: request id -> (window start, speculative) of its last scheduled step, from the scheduler output
_LASTTOK = {}        # audit: request id -> the last token id emitted to the client
_STATE = dict(last=None, t=0.0)
COMPILE_TAG = ':arc3-persistent+stallguard'


def _anchor(text):
    return ast.dump(ast.parse(text).body[0], include_attributes=False)


def transform_schedule(fn):
    """Edit engine_patch's patched schedule FunctionDef in place (the running loop only); returns it."""
    loop_test = ast.dump(ast.parse('req_index < len(self.running) and token_budget > 0', mode='eval').body, include_attributes=False)
    ok_break = _anchor('if new_blocks is not None:\n    break\n')
    none_break = _anchor('if new_blocks is None:\n    break\n')
    stall_stmt = ast.parse('if _arc3_sg_stall(self, request):\n    break\n').body[0]
    skip_stmt = ast.parse('if new_blocks is None:\n    if _arc3_sg_take_stalled(request):\n        req_index += 1\n        continue\n'
                          '    break\n').body[0]
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.While) and ast.dump(n.test, include_attributes=False) == loop_test]
    assert len(loops) == 1, ('stallguard: running loop', len(loops))
    run = loops[0]
    inner, inserted = 0, 0
    for node in ast.walk(run):
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
            body = node.body
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Attribute) and n.attr == 'allocate_slots']
            if not calls:
                continue
            inner += 1
            for i, st in enumerate(body):
                if ast.dump(st, include_attributes=False) == ok_break:
                    body.insert(i + 1, stall_stmt)
                    inserted += 1
                    break
    assert inner == 1 and inserted == 1, ('stallguard: allocation loop', inner, inserted)
    replaced = 0
    for i, st in enumerate(run.body):
        if ast.dump(st, include_attributes=False) == none_break:
            run.body[i] = skip_stmt
            replaced += 1
    assert replaced == 1, ('stallguard: running-loop break', replaced)
    return ast.fix_missing_locations(fn)


def _persistent(request):
    import engine_patch
    return engine_patch.is_persistent(request)


def _first_blocks(scheduler, rid):
    try:
        return [int(g[0]) if len(g) else -1 for g in scheduler.kv_cache_manager.get_block_ids(rid)]
    except Exception:  # noqa: BLE001
        return None


def sg_stall(scheduler, request):
    """Called where the stock loop would pick a preemption victim: stall this request instead (any request: with the
    guard on nothing is preempted)."""
    rid = request.request_id
    if rid not in _OPEN:
        _OPEN[rid] = time.monotonic()
        STATS['stall_episodes'] += 1
        _OPEN_INFO[rid] = dict(rid=rid, t0=time.time(), blocks=_first_blocks(scheduler, rid))
    request._arc3_sg_now = True
    STATS['stalled_steps'] += 1
    return True


def sg_take_stalled(request):
    if getattr(request, '_arc3_sg_now', False):
        request._arc3_sg_now = False
        return True
    return False


def _settled(req):
    return (req.num_output_placeholders == 0 and getattr(req, 'num_in_flight_tokens', 0) == 0
            and len(req._all_token_ids) - req.num_computed_tokens == 1
            and req.num_computed_tokens >= req.num_prompt_tokens and not getattr(req, 'is_prefill_chunk', False))


def _log(msg):
    try:
        import logging
        logging.getLogger('vllm').warning('ARC3 STALLGUARD %s', msg)
    except Exception:  # noqa: BLE001
        pass


def _fail(req, key, exc):
    """Count and log a failure on a request; the request stays stalled and is not broken again."""
    STATS[key] += 1
    STATS['errors'] += 1
    STATS['last_error'] = ('%s %s: %r' % (key, getattr(req, 'request_id', '?'), exc))[:300]
    try:
        req._arc3_sg_nobreak = True
    except Exception:  # noqa: BLE001
        pass
    _log(STATS['last_error'])


def _digest(ids):
    import array
    import hashlib
    return hashlib.sha256(array.array('q', ids).tobytes()).hexdigest()[:16]


def _snapshot(scheduler, req):
    ids = list(req._all_token_ids)
    try:
        groups = scheduler.kv_cache_manager.get_block_ids(req.request_id)
        nblocks = [len(g) for g in groups]
        first = [int(g[0]) if len(g) else -1 for g in groups]
    except Exception:  # noqa: BLE001
        nblocks = first = None
    comp = int(req.num_computed_tokens)
    return dict(status=str(getattr(req.status, 'name', req.status)), all=len(ids), computed=comp, prompt=int(req.num_prompt_tokens),
                placeholders=int(req.num_output_placeholders), spec=len(req.spec_token_ids or []), prefix_digest=_digest(ids[:comp]),
                frontier=(ids[comp] if comp < len(ids) else None), blocks=nblocks, first_block=first,
                evict_state=id(getattr(req, '_arc3_eviction_state', None)))


def _audit_write(rec):
    try:
        with open(AUDIT, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    except Exception as exc:  # noqa: BLE001
        STATS['errors'] += 1
        STATS['last_error'] = ('audit write: %r' % exc)[:300]


def _broken_write():
    try:
        tmp = BROKEN_FILE + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(dict(broken=[dict(rid=rid, t=r['t'], blocks=r['first_block'], n=r['computed'], start=r.get('start'),
                                        m_audit=r.get('m_audit'), frontier=r.get('frontier'), prefix_digest=r['prefix_digest'])
                                   for rid, r in _BROKEN.items()]), fh)
        os.replace(tmp, BROKEN_FILE)
    except Exception as exc:  # noqa: BLE001
        STATS['errors'] += 1
        STATS['last_error'] = ('broken file: %r' % exc)[:300]


def _undo_enqueue(scheduler, req, n_wait):
    for qname in ('waiting', 'skipped_waiting'):
        q = getattr(scheduler, qname, None)
        try:
            if q is not None and req in q:
                q.remove_requests([req])
        except Exception:  # noqa: BLE001
            pass
    if n_wait is not None:
        try:
            scheduler.num_waiting_for_streaming_input = n_wait
        except Exception:  # noqa: BLE001
            pass


def _precheck(req):
    """What the stop path needs before persist's fixup: nothing in flight or staged, the processed prefix inside the stream."""
    for name in ('num_in_flight_tokens', 'num_output_placeholders', 'num_stale_output_tokens'):
        v = getattr(req, name, 0)
        if v:
            raise RuntimeError('stallguard: %s=%s at the break' % (name, v))
    if not 0 <= req.num_computed_tokens <= len(req._all_token_ids):
        raise RuntimeError('stallguard: computed %d outside the stream (%d)' % (req.num_computed_tokens, len(req._all_token_ids)))


def _break_turn(scheduler, req, outs):
    """End a settled stalled persistent turn at its frontier; True when ended. Atomic and non-raising (v3)."""
    import engine_patch
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
    from vllm.v1.request import RequestStatus
    try:                       # v3.1 (ffinj 05:5x): the PRE-fixup conditions only. engine_patch.assert_settled is the post-fixup
        _precheck(req)         # boundary invariant (no speculative tokens): a stalled running lane under MTP always holds its next
    except Exception as exc:   # window's drafts, which persist's stop fixup drops inside _handle_stopped_request
        _fail(req, 'breaker_precheck_failed', exc)
        return False
    pre = _snapshot(scheduler, req) if AUDIT else None
    prev = req.status
    n_wait = getattr(scheduler, 'num_waiting_for_streaming_input', None)
    try:
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        finish_reason = req.get_finished_reason()
        finished = scheduler._handle_stopped_request(req)
    except Exception as exc:  # noqa: BLE001
        req.status = prev
        _undo_enqueue(scheduler, req, n_wait)
        _fail(req, 'breaker_stop_failed', exc)
        return False
    try:
        if finished:                      # not a pause: free it as the stock update_from_output does (the session ends)
            scheduler._free_request(req)
            STATS['breaker_finished'] += 1
            _log('breaker stop of %s finished the request (freed; the session ends)' % req.request_id)
        scheduler.running = [r for r in scheduler.running if r is not req]
        eco = outs.get(req.client_index)
        out = EngineCoreOutput(request_id=req.request_id, new_token_ids=[], finish_reason=finish_reason,
                               events=req.take_events(), trace_headers=req.trace_headers)
        if eco is None:
            outs[req.client_index] = EngineCoreOutputs(outputs=[out])
        else:
            eco.outputs.append(out)
    except Exception as exc:  # noqa: BLE001
        scheduler.running = [r for r in scheduler.running if r is not req]
        _fail(req, 'breaker_emit_failed', exc)
        return True
    STATS['breaker_stops'] += 1
    if AUDIT:
        try:
            post = _snapshot(scheduler, req)
            start, spec = _LAST.get(req.request_id, (None, False))
            m_audit = (pre['computed'] - start) if (spec and start is not None) else None
            kept = getattr(req, '_arc3_s2_kept', None)
            k = _num_spec(scheduler)
            last = _LASTTOK.get(req.request_id)
            pre.update(start=start, spec=bool(spec), m_audit=m_audit, kept_persist=kept, last_emitted=last)
            ok = dict(frontier=pre['computed'] == pre['all'] - 1, spec_empty=post['spec'] == 0, paused=post['status'] == 'WAITING_FOR_STREAMING_REQ',
                      history_kept=post['all'] == pre['all'] and post['prefix_digest'] == pre['prefix_digest'],
                      blocks_kept=post['blocks'] == pre['blocks'], not_running=req not in scheduler.running,
                      m_matches_persist=(m_audit == kept) if (m_audit is not None or kept is not None) else True,
                      m_in_window=(m_audit is None or k is None or 1 <= m_audit <= k + 1),
                      frontier_emitted=(last is None or pre['frontier'] == last))
            STATS['audit_breaks'] += 1
            STATS['audit_violations'] += sum(1 for v in ok.values() if not v)
            _audit_write(dict(kind='break', rid=req.request_id, t=time.time(), pre=pre, post=post, ok=ok))
            if not finished:
                _BROKEN[req.request_id] = dict(pre, t=time.time())
                _broken_write()
        except Exception as exc:  # noqa: BLE001
            STATS['errors'] += 1
            STATS['last_error'] = ('audit break: %r' % exc)[:300]
    return True


def _audit_resume(scheduler, req, scheduled, spec_first=0):
    b = _BROKEN.pop(req.request_id)
    now = _snapshot(scheduler, req)
    evicted = now['evict_state'] != b['evict_state']
    ids = list(req._all_token_ids)
    sp = getattr(req, 'sampling_params', None)
    passed = ((getattr(sp, 'extra_args', None) or {}).get('arc3_s2_kept')) if sp is not None else None
    start_cont = _LAST.get(req.request_id, (None, False))[0]
    ok = dict(prefix_kept=evicted or _digest(ids[:b['computed']]) == b['prefix_digest'], no_loss=evicted or len(ids) >= b['computed'],
              kept_passed=(passed == b.get('m_audit')) if (passed is not None or b.get('m_audit') is not None) else True,
              starts_at_frontier=evicted or start_cont is None or start_cont == b['computed'],
              no_stale_drafts=spec_first == 0)     # v3.2 (IE): the continuation re-drafts from the frontier, never the dropped drafts
    now.update(kept_passed=passed, start=start_cont, spec_first=int(spec_first))
    STATS['audit_resumes'] += 1
    STATS['audit_evicted_between'] += int(evicted)
    STATS['audit_violations'] += sum(1 for v in ok.values() if not v)
    _audit_write(dict(kind='resume', rid=req.request_id, t=time.time(), dt=round(time.time() - b['t'], 3), scheduled=int(scheduled),
                      evicted_between=evicted, frontier_kept=(None if evicted or b['frontier'] is None or len(ids) <= b['computed']
                                                              else ids[b['computed']] == b['frontier']),
                      now=now, ok=ok))
    _broken_write()


def _write_state(scheduler, now):
    try:
        pool = scheduler.kv_cache_manager.block_pool
        free = pool.get_num_free_blocks()
        STATS['min_free_blocks'] = free if STATS['min_free_blocks'] is None else min(STATS['min_free_blocks'], free)
        oldest = round(now - min(_OPEN.values()), 3) if _OPEN else 0.0
        st = (len(_OPEN), STATS['breaker_stops'])
        if st == _STATE['last'] and now - _STATE['t'] < 1.0:      # on a change of open stalls / stops at once, else once a second
            return
        tmp = STATE_FILE + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(dict(open_stalls=len(_OPEN), oldest_stall_s=oldest, free_blocks=free, num_blocks=pool.num_gpu_blocks,
                           breaker_stops=STATS['breaker_stops'], stall_episodes=STATS['stall_episodes'], t=time.time(),
                           open=[_OPEN_INFO[r] for r in _OPEN if r in _OPEN_INFO and _OPEN_INFO[r].get('blocks') is not None],
                           stats={k: v for k, v in STATS.items() if k != 'last_error'}), fh)
        os.replace(tmp, STATE_FILE)
        _STATE['last'], _STATE['t'] = st, now
    except Exception as exc:  # noqa: BLE001  diagnostics only
        STATS['errors'] += 1
        STATS['last_error'] = repr(exc)[:200]


def _num_spec(scheduler):
    for get in (lambda: scheduler.vllm_config.num_speculative_tokens,
                lambda: scheduler.vllm_config.speculative_config.num_speculative_tokens):
        try:
            v = get()
            if v is not None:
                return int(v)
        except Exception:  # noqa: BLE001
            pass
    return None


def _note_step(scheduler, scheduler_output, outs):
    """Audit: this step's window start per scheduled request (the scheduler output's num_computed_tokens, i.e. what the worker was
    told) and the last token emitted to each client: sources independent of persist's own window fields."""
    spec = scheduler_output.scheduled_spec_decode_tokens or {}
    for nr in scheduler_output.scheduled_new_reqs or []:
        _LAST[nr.req_id] = (int(nr.num_computed_tokens), nr.req_id in spec)
    cr = scheduler_output.scheduled_cached_reqs
    if cr is not None:
        for rid, nct in zip(cr.req_ids, cr.num_computed_tokens):
            _LAST[rid] = (int(nct), rid in spec)
    for eco in (outs or {}).values():
        for o in (eco.outputs or []):
            if o.new_token_ids:
                _LASTTOK[o.request_id] = int(o.new_token_ids[-1])
    if len(_LAST) > 4 * max(8, len(scheduler.requests)):
        for rid in [r for r in _LAST if r not in scheduler.requests]:
            _LAST.pop(rid, None); _LASTTOK.pop(rid, None)


def _after_update(scheduler, scheduler_output, outs):
    now = time.monotonic()
    ran = scheduler_output.num_scheduled_tokens or {}
    if AUDIT:
        try:
            _note_step(scheduler, scheduler_output, outs)
        except Exception as exc:  # noqa: BLE001
            STATS['errors'] += 1
            STATS['last_error'] = ('audit note: %r' % exc)[:300]
    for rid in list(_OPEN):
        if rid in ran or rid not in scheduler.requests:          # it ran (stall over) or it is gone
            _OPEN_INFO.pop(rid, None)
            d = now - _OPEN.pop(rid)
            STATS['stall_seconds'] += d
            STATS['max_stall_s'] = max(STATS['max_stall_s'], d)
    if _BROKEN:
        for rid in list(_BROKEN):
            try:
                if rid in ran and rid in scheduler.requests:
                    _audit_resume(scheduler, scheduler.requests[rid], ran[rid],
                                  len((scheduler_output.scheduled_spec_decode_tokens or {}).get(rid, ())))
                elif rid not in scheduler.requests:
                    _BROKEN.pop(rid, None)
                    _broken_write()
            except Exception as exc:  # noqa: BLE001
                _BROKEN.pop(rid, None)
                STATS['errors'] += 1
                STATS['last_error'] = ('audit resume %s: %r' % (rid, exc))[:300]
    for req in list(scheduler.running):
        try:
            since = _OPEN.get(req.request_id)
            if since is None or req.request_id in ran:
                continue
            age = now - since
            if age >= T_WARN and not getattr(req, '_arc3_sg_warned', False):
                req._arc3_sg_warned = True
                STATS['long_stall_warnings'] += 1
                _log('request %s stalled for %.1f s' % (req.request_id, age))
            if age < T_BREAK or not _persistent(req):
                continue
            if getattr(req, '_arc3_sg_nobreak', False):
                STATS['breaker_nobreak_skips'] += 1
                continue
            if not _settled(req):
                STATS['breaker_unsettled_skips'] += 1
                continue
            if _break_turn(scheduler, req, outs):
                _OPEN_INFO.pop(req.request_id, None)
                d = now - _OPEN.pop(req.request_id)
                STATS['stall_seconds'] += d
                STATS['max_stall_s'] = max(STATS['max_stall_s'], d)
        except Exception as exc:  # noqa: BLE001
            _fail(req, 'guard_errors', exc)
    STATS['open_stalls'] = len(_OPEN)
    _write_state(scheduler, now)


def _install_inject(sched):
    """Test-only: refuse allocate_slots for one chosen running persistent decode-phase request for HOLD_S every EVERY_S (rotating)."""
    from vllm.v1.request import RequestStatus
    every, hold = INJECT
    mgr = sched.kv_cache_manager
    orig = mgr.allocate_slots
    st = dict(next=None, rid=None, until=0.0, recent=[])

    def allocate_slots(request, *args, **kwargs):
        try:
            now = time.monotonic()
            if st['next'] is None:
                st['next'] = now + INJECT_START
            rid = request.request_id
            if st['rid'] is not None and now >= st['until']:      # hold over (a broken target is paused and never allocates again)
                st['rid'] = None
            if (st['rid'] is None and now >= st['next'] and request.status == RequestStatus.RUNNING and _persistent(request)
                    and request.num_computed_tokens >= request.num_prompt_tokens and rid not in st['recent']):
                st.update(rid=rid, until=now + hold, next=now + every)
                st['recent'] = (st['recent'] + [rid])[-INJECT_RECENT:]
                STATS['inject_events'] += 1
            if st['rid'] == rid:
                if now < st['until'] and request.status == RequestStatus.RUNNING:
                    STATS['inject_refusals'] += 1
                    return None
                st['rid'] = None
        except Exception as exc:  # noqa: BLE001
            STATS['errors'] += 1
            STATS['last_error'] = ('inject: %r' % exc)[:300]
        return orig(request, *args, **kwargs)
    allocate_slots.__wrapped__ = orig
    mgr.allocate_slots = allocate_slots
    STATS['inject'] = '%g:%g' % INJECT


def _find_compiled(fn, depth=0):
    """engine_patch's compiled schedule: fn itself, or held in a wrapper's closure (the steptimer profiling part wraps
    Scheduler.schedule with closures over the original). Returns (function, closure cell or None)."""
    code = getattr(fn, '__code__', None)
    if code is not None and code.co_filename.endswith(':arc3-persistent'):
        return fn, None
    if depth >= 3:
        return None, None
    for cell in getattr(fn, '__closure__', None) or ():
        try:
            c = cell.cell_contents
        except ValueError:
            continue
        cc = getattr(c, '__code__', None)
        if cc is not None and cc.co_filename.endswith(':arc3-persistent'):
            return c, cell
        if callable(c) and getattr(c, '__closure__', None):
            f, inner = _find_compiled(c, depth + 1)
            if f is not None:
                return f, inner
    return None, None


def _install(sm):
    """At the first Scheduler construction: compile and install the stall-guarded schedule, the breaker and the preemption guard."""
    import engine_patch
    from pathlib import Path
    cls = sm.Scheduler
    cur = cls.schedule
    target, cell = _find_compiled(cur)
    if target is None:
        raise RuntimeError('stallguard: Scheduler.schedule is not engine_patch\'s compiled schedule (%r)' % getattr(getattr(cur, '__code__', None), 'co_filename', None))
    source = Path(sm.__file__).read_text()
    fn = transform_schedule(engine_patch._patched_schedule_ast(source))
    path = sm.__file__ + COMPILE_TAG
    code = compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec')
    ns = dict(sm.__dict__)
    ns['_arc3_persistent_skip_inflight'] = engine_patch.should_skip_inflight
    ns['_arc3_sg_stall'] = sg_stall
    ns['_arc3_sg_take_stalled'] = sg_take_stalled
    exec(code, ns)
    new = ns['schedule']
    new.__wrapped__ = target
    if cell is None:
        cls.schedule = new
    else:                                   # keep the wrapper (e.g. steptimer); swap the schedule it calls
        cell.cell_contents = new
        STATS['installed_inside'] = '%s.%s' % (getattr(cur, '__module__', '?'), getattr(cur, '__qualname__', '?'))
    orig_update = cls.update_from_output

    def update_from_output(self, scheduler_output, model_runner_output):
        outs = orig_update(self, scheduler_output, model_runner_output)
        _after_update(self, scheduler_output, outs)
        return outs
    update_from_output.__wrapped__ = orig_update
    cls.update_from_output = update_from_output
    orig_preempt = cls._preempt_request

    def _preempt_request(self, request, *args, **kwargs):
        if _persistent(request):                 # unreachable (see v3 in the docstring); a loud stop beats a silent corruption
            STATS['preempt_refused'] += 1
            raise RuntimeError('stallguard: refusing to preempt persistent request %s' % request.request_id)
        return orig_preempt(self, request, *args, **kwargs)
    _preempt_request.__wrapped__ = orig_preempt
    cls._preempt_request = _preempt_request
    orig_reset = cls.reset_prefix_cache

    def reset_prefix_cache(self, *args, **kwargs):
        reset_running = kwargs.get('reset_running_requests', args[0] if args else False)
        try:
            refuse = bool(reset_running) and any(_persistent(r) for r in self.running)
        except Exception:  # noqa: BLE001
            refuse = bool(reset_running)
        if refuse:
            STATS['reset_refused'] += 1
            _log('reset_prefix_cache(reset_running_requests=True) refused: persistent requests are running')
            return False
        return orig_reset(self, *args, **kwargs)
    reset_prefix_cache.__wrapped__ = orig_reset
    cls.reset_prefix_cache = reset_prefix_cache
    STATS['installed'] = True


def patch_scheduler(sm):
    if not ENABLED:
        return
    cls = sm.Scheduler
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        if not STATS['installed']:
            _install(sm)
        res = orig_init(self, *args, **kwargs)
        if INJECT:
            _install_inject(self)
        return res
    __init__.__wrapped__ = orig_init
    cls.__init__ = __init__


PATCHERS = {'vllm.v1.core.sched.scheduler': patch_scheduler}
