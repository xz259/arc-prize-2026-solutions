"""stage2_v1 'overlap' part: under the single-batch policy the engine core serves resume admissions, eviction begins
and frequency snapshots while the dispatched step runs, instead of between steps with the GPU idle (GAP_PLAN item 4,
2026-09-30).

With an active persistent owner, production's persistent_batch_policy makes EngineCore.step_with_batch_queue
schedule and dispatch a step and then block on that same step; the busy loop processes the input queue between steps.
Everything that arrives meanwhile therefore runs with the GPU idle: an arc3_evict_begin costs about 74 ms (g4d:
+72.7 ms per overlapping step, 2.3% of wall time) and a resume admission 10-13 ms (about 1% at one resume per second).

- The policy's step gains one statement before its block: while the dispatched step's output is not yet available,
  eligible items at the head of the input queue are handled in arrival order (_handle_client_request, unchanged);
  the wait itself is native (the executor's own FutureWrapper drain, called only once the next response has arrived,
  which a read-only check of the response ring's flags detects without consuming it).
- Eligible while a step is in flight: ADD (new sessions and resumes: X is paused, so not in the step; a resume
  arriving during step t is scheduled at t+1 exactly as before), arc3_qsa_frequency_snapshot_compact (read-only, paused
  owner, no work in flight), and arc3_evict_begin only right after the dispatch (the async eviction's owner-local
  design: a paused owner with private pages; its worker copies wait for the compute stream). A begin arriving later is
  left at the head, and the pre-schedule drain leaves begins and snapshots for the next dispatch, so a begin always
  overlaps a whole step. Everything else (aborts, commits, other utilities) stops the serving; order is preserved.
- Served utilities that call collective_rpc drain the pending step futures first (FutureWrapper, in order); the step's
  result is then already set and the native future.result() returns it.
ARC3_S2_OVERLAP_SWITCH=<file>: a file containing 'off' restores the native loop (checked at most once a second), for
within-run A/Bs."""
import ast
import hashlib
import inspect
import os
import textwrap
import time

PINS = {'vllm.v1.engine.core': 'f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad'}
INPUT_QUEUE_SHA = '2c3091b60af8807ca839147987a2aad56ce32cbcf60269c6c6af4ea8d157e778'
POLICY_VERSION = 'arc3-persistent-single-batch-v1'
BEGIN = 'arc3_evict_begin'
SNAPSHOT = 'arc3_qsa_frequency_snapshot_compact'
WAIT_LIMIT_S = float(os.environ.get('ARC3_S2_OVERLAP_WAIT_LIMIT', '2.0'))
STATS = dict(waits=0, served_add=0, served_begin=0, served_snapshot=0, served_noop=0, deferred_begin=0,
             deferred_snapshot=0, defer_limit=0, serve_ms=0.0, fallbacks=0, unsupported=0, switch_off_waits=0)
_SWITCH = dict(path=os.environ.get('ARC3_S2_OVERLAP_SWITCH'), t=0.0, on=True)
MAX_DEFER = 4
_DEFER = dict(item=None, count=0)
_TYPES = {}


def _on():
    s = _SWITCH
    if s['path'] is None:
        return True
    now = time.monotonic()
    if now - s['t'] > 1.0:
        s['t'] = now
        try:
            with open(s['path']) as f:
                s['on'] = f.read().strip() != 'off'
        except OSError:
            s['on'] = True
    return s['on']


def _kind(req):
    rtype, payload = req
    if rtype == _TYPES['ADD']:
        return 'add'
    if rtype == _TYPES['WAKEUP']:
        return 'noop'
    if rtype == _TYPES['UTILITY']:
        name = payload[2]                  # (client_idx, call_id, method_name, args)
        if name == BEGIN:
            return 'begin'
        if name == SNAPSHOT:
            return 'snapshot'
    return None


def _peek(q):
    with q.mutex:
        return q.queue[0] if q.queue else None


def defer_head(core):
    """Pre-schedule drain: leave a begin or snapshot at the head for the next dispatch (the policy is on). The same item
    is deferred at most MAX_DEFER times in a row (progress guarantee); after that the native drain handles it."""
    if not getattr(core, '_s2_policy_on', False) or not _on():
        return False
    head = _peek(core.input_queue)
    kind = _kind(head) if head is not None else None
    if kind in ('begin', 'snapshot'):
        if head is _DEFER['item']:
            _DEFER['count'] += 1
            if _DEFER['count'] > MAX_DEFER:
                STATS['defer_limit'] += 1
                return False
        else:
            _DEFER.update(item=head, count=1)
        STATS['deferred_' + kind] += 1
        return True
    return False


def _response_mq(ex):
    mqs = getattr(ex, 'response_mqs', None)
    if not mqs or len(mqs) != 1 or not hasattr(ex, 'futures_queue'):
        return None
    mq = mqs[0]
    if not getattr(mq, '_is_local_reader', False):
        return None
    return mq


def _ready(mq, fence):
    """The next response has been written and not read (the reader check of MessageQueue.acquire_read)."""
    with mq.buffer.get_metadata(mq.current_idx) as md:
        fence()
        return md[0] != 0 and md[mq.local_reader_rank + 1] == 0


def wait(core, batch_queue):
    """Serve eligible input items until the dispatched step's output is available (then future.result() is native)."""
    if not _on():
        STATS['switch_off_waits'] += 1
        return
    ex = core.model_executor
    mq = _response_mq(ex)
    if mq is None:
        STATS['unsupported'] += 1
        return
    from vllm.distributed.device_communicators.shm_broadcast import memory_fence
    STATS['waits'] += 1
    future = batch_queue[-1][0]
    fq, q = ex.futures_queue, core.input_queue
    post = True
    t0 = time.monotonic()
    while not future.done():
        head = _peek(q)
        kind = _kind(head) if head is not None else None
        servable = kind in ('add', 'snapshot', 'noop') or (kind == 'begin' and post)
        # The streak right after the dispatch is served whatever the replies (a deferred begin is always served, even
        # after an instantly answered step); after it, replies come first: once the step's output is in, it is processed.
        ready = False if (servable and post) else _ready(mq, memory_fence)
        if servable and not ready:
            req = q.get_nowait()            # the engine thread is the only consumer: this is the head just read
            t = time.perf_counter()
            core._handle_client_request(*req)
            STATS['serve_ms'] += (time.perf_counter() - t) * 1e3
            STATS['served_' + kind] += 1
            continue
        post = False
        if ready:
            if not fq:
                break
            fq.pop()._wait_for_response()   # the oldest pending future; its response has arrived
            continue
        if ex.is_failed or time.monotonic() - t0 > WAIT_LIMIT_S:
            STATS['fallbacks'] += 1         # the native blocking wait (timeouts, cancellation) takes over
            break
        os.sched_yield()


def extend_step(fn):
    """Add to the policy-transformed step_with_batch_queue AST: record the policy decision; serve before the block."""
    top = fn.body
    assert '_arc3_s2_overlap_wait' not in ast.unparse(fn), 'stage2 overlap: step already extended'
    policy = [i for i, n in enumerate(top) if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_arc3_single_persistent_batch']
    pops = [i for i, n in enumerate(top) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
            and ast.unparse(n.value) == 'batch_queue.pop()']
    assert len(policy) == 1 and len(pops) == 1 and policy[0] < pops[0], ('stage2 overlap: step anchors', policy, pops)
    top.insert(pops[0], ast.parse('if _arc3_single_persistent_batch and batch_queue:\n'
                                  '    _arc3_s2_overlap_wait(self, batch_queue)').body[0])
    top.insert(policy[0] + 1, ast.parse('self._s2_policy_on = _arc3_single_persistent_batch').body[0])
    return fn


def patch_input_queue(mod):
    cls = mod.EngineCoreProc
    src = textwrap.dedent(inspect.getsource(cls._process_input_queue))
    digest = hashlib.sha256(src.encode()).hexdigest()
    assert digest == INPUT_QUEUE_SHA, f'stage2 overlap: _process_input_queue changed ({digest})'
    tree = ast.parse(src)
    fn = tree.body[0]
    loops = [n for n in fn.body if isinstance(n, ast.While) and ast.unparse(n.test) == 'not self.input_queue.empty()']
    assert len(loops) == 1, 'stage2 overlap: input drain anchor'
    loops[0].body.insert(0, ast.parse('if _arc3_s2_overlap_defer(self):\n    break').body[0])
    scope = dict(mod.__dict__)
    scope['_arc3_s2_overlap_defer'] = defer_head
    exec(compile(ast.fix_missing_locations(tree), '<stage2-overlap-input-queue>', 'exec'), scope)
    patched = scope[fn.name]
    patched.__qualname__ = cls._process_input_queue.__qualname__
    cls._process_input_queue = patched


def patch_core(mod):
    from vllm.v1.engine import EngineCoreRequestType
    _TYPES.update(ADD=EngineCoreRequestType.ADD, UTILITY=EngineCoreRequestType.UTILITY,
                  WAKEUP=EngineCoreRequestType.WAKEUP)
    patch_input_queue(mod)
    mod._arc3_s2_overlap_wait = wait        # read by the policy-built step (its globals copy this module's)
    import persistent_batch_policy as policy
    assert policy.VERSION == POLICY_VERSION and policy.CORE_SHA256 == PINS['vllm.v1.engine.core'], \
        'stage2 overlap: batch policy version'
    original = policy.transform

    def transform(source):
        return extend_step(original(source))

    policy.transform = transform


PATCHERS = {'vllm.v1.engine.core': patch_core}
