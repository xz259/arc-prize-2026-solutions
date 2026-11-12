"""Optional CPU boundary timings; no token logging, GPU reads or synchronization.

Install after engine_eviction and eviction_runner. This observes the existing
methods; it does not change request state or scheduling. Nested phase times are
inclusive and must not be added together. Journal writes occur after the outer
measured interval and their CPU cost is intentionally not hidden in that time.
"""
from __future__ import annotations

import contextvars
import functools
import json
import os
import threading
import time
from dataclasses import dataclass, field

VERSION = "arc3-resume-profile-v1"
FLAG = "ARC3_RESUME_PROFILE"
PATH = "ARC3_RESUME_PROFILE_PATH"
_CURRENT = contextvars.ContextVar("arc3_resume_profile_event", default=None)
_INSTALLED = False
_STATS = {"attempted": 0, "written": 0, "dropped": 0}
_IO_LOCK = threading.Lock()


def _length(value):
    return len(value) if value is not None else 0


def _request_info(request):
    """Only lengths/scalars; never serialize token arrays or read GPU values."""
    state = getattr(request, "_arc3_eviction_state", None)
    params = getattr(request, "sampling_params", None)
    context = (getattr(params, "extra_args", None) or {}).get(
        "arc3_eviction_context", {})
    return {
        "request_id": getattr(request, "request_id", getattr(request, "req_id", None)),
        "turn": getattr(request, "_arc3_persistent_turn", None),
        "input_tokens": _length(getattr(request, "prompt_token_ids", None)),
        "resident_tokens": _length(getattr(request, "_all_token_ids", None)),
        "prefill_tokens": _length(getattr(request, "prefill_token_ids", None)),
        "computed_tokens": getattr(request, "num_computed_tokens", None),
        "logical_tokens": _length(state.logical_ids) if state is not None else
            _length(context.get("logical_token_ids")),
        "eviction_revision": getattr(state, "eviction_revision", context.get("eviction_revision")),
    }


@dataclass
class _Event:
    kind: str
    before: object
    wall_start: float = field(default_factory=time.time)
    mono_start: float = field(default_factory=time.monotonic)
    thread_id: int = field(default_factory=threading.get_ident)
    phases: dict = field(default_factory=dict)

    def phase(self, name, elapsed):
        phase = self.phases.setdefault(name, {"seconds": 0.0, "calls": 0})
        phase["seconds"] += elapsed
        phase["calls"] += 1

    def finish(self, after, error=None):
        mono_end = time.monotonic()
        return {"version": VERSION, "event": self.kind, "pid": os.getpid(),
                "thread_id": self.thread_id, "wall_start": self.wall_start,
                "wall_end": time.time(), "mono_start": self.mono_start,
                "mono_end": mono_end, "seconds": mono_end-self.mono_start,
                "before": self.before, "after": after, "phases": self.phases,
                "error_type": type(error).__name__ if error is not None else None}


def _write_locked(event):
    """Separate per-process files avoid multiprocess partial-line interleaving.

    Diagnostic I/O failure cannot turn a successful model request into failure.
    Missing/dropped events invalidate timing completeness; status() exposes it.
    """
    _STATS["attempted"] += 1
    try:
        path = f"{os.environ[PATH]}.{os.getpid()}.jsonl"
        blob = (json.dumps(event, separators=(",", ":"), allow_nan=False)+"\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(blob)
            while view:
                size = os.write(fd, view)
                if size <= 0:
                    raise OSError("Diagnostic journal write made no progress")
                view = view[size:]
        finally:
            os.close(fd)
        _STATS["written"] += 1
    except Exception as error:
        _STATS["dropped"] += 1
        if _STATS["dropped"] == 1:
            _STATS["first_error_type"] = type(error).__name__
            try:
                os.write(2, b"ARC3 resume profile journal failed; timing coverage is incomplete.\n")
            except OSError:
                pass


def _write(event):
    # Native preprocess_add_request runs on the core input thread, while
    # core_add runs on its scheduler thread; prevent partial-line overlap.
    with _IO_LOCK:
        _write_locked(event)


def status():
    return {"version": VERSION, "installed": _INSTALLED, "pid": os.getpid(),
            "journal": f"{os.environ.get(PATH, '')}.{os.getpid()}.jsonl",
            **_STATS}


def _phase(method, name):
    @functools.wraps(method)
    def timed(*args, **kwargs):
        event = _CURRENT.get()
        if event is None:
            return method(*args, **kwargs)
        start = time.monotonic()
        try:
            return method(*args, **kwargs)
        finally:
            event.phase(name, time.monotonic()-start)
    return timed


def _metadata(fn, args, kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as error:
        return {"profile_metadata_error": type(error).__name__}


def _outer(method, kind, select, before, after):
    nested = _phase(method, kind)
    @functools.wraps(method)
    def timed(*args, **kwargs):
        if not select(*args, **kwargs):
            return method(*args, **kwargs)
        parent = _CURRENT.get()
        if parent is not None:
            return nested(*args, **kwargs)
        event = _Event(kind, _metadata(before, args, kwargs))
        token = _CURRENT.set(event)
        error = None
        try:
            return method(*args, **kwargs)
        except BaseException as caught:
            error = caught
            raise
        finally:
            # Reset even when collecting metadata fails; observer failure must
            # never replace a native exception or alter the return value.
            _CURRENT.reset(token)
            try:
                record = event.finish(_metadata(after, args, kwargs), error)
                _write(record)
            except Exception:
                _STATS["dropped"] += 1
    return timed


def _async_outer(method, kind, select):
    @functools.wraps(method)
    async def timed(self, request, *args, **kwargs):
        if not select(request):
            return await method(self, request, *args, **kwargs)
        # Do not propagate a diagnostic ContextVar across await: only the
        # native ADD send is measured, not concurrently scheduled API tasks.
        event = _Event(kind, _metadata(_request_info, (request,), {}))
        error = None
        try:
            return await method(self, request, *args, **kwargs)
        except BaseException as caught:
            error = caught
            raise
        finally:
            try:
                _write(event.finish(_metadata(_request_info, (request,), {}), error))
            except Exception:
                _STATS["dropped"] += 1
    return timed


def install_targets(*, scheduler, core, engine_patch, engine_eviction,
                    runner, model_state, request_state, eviction_runner,
                    frontend=None, executor=None):
    """Dependency injection for CPU tests; caller already checked flag/order."""
    persistent = engine_patch.is_persistent
    core_select = lambda self, request, *a, **kw: persistent(request)
    info = lambda self, request, *a, **kw: _request_info(request)
    core_after = lambda self, request, *a, **kw: _request_info(
        self.scheduler.requests.get(request.request_id, request))
    sched_after = lambda self, request, *a, **kw: _request_info(
        self.requests.get(request.request_id, request))

    core.add_request = _outer(core.add_request, "core_add", core_select, info, core_after)
    if hasattr(core, "preprocess_add_request"):
        core.preprocess_add_request = _outer(core.preprocess_add_request,
            "core_preprocess_add", core_select, info, info)
    scheduler.add_request = _outer(scheduler.add_request, "scheduler_add", core_select, info, sched_after)
    scheduler._update_request_as_session = _phase(
        scheduler._update_request_as_session, "core_update_session")
    engine_patch.append_session = _phase(engine_patch.append_session, "core_append_session")
    engine_patch._audit = _phase(engine_patch._audit, "core_audit")
    engine_eviction.LogicalState.paused = _phase(engine_eviction.LogicalState.paused, "core_logical_paused")
    engine_eviction.LogicalState.context = _phase(engine_eviction.LogicalState.context, "core_logical_context")
    engine_eviction.publish_context = _phase(engine_eviction.publish_context, "core_publish_context")

    def batch_select(self, scheduler_output):
        return bool(scheduler_output.scheduled_new_reqs)

    def batch_before(self, scheduler_output):
        return [dict(_request_info(req), readded=req.req_id in self.req_states.req_id_to_index)
                for req in scheduler_output.scheduled_new_reqs]

    def batch_after(self, scheduler_output):
        return [dict(_request_info(req), slot=self.req_states.req_id_to_index.get(req.req_id))
                for req in scheduler_output.scheduled_new_reqs]

    runner.add_requests = _outer(runner.add_requests, "worker_add_batch", batch_select,
                                batch_before, batch_after)
    model_state.add_request = _phase(model_state.add_request, "worker_model_state_add")
    eviction_runner._stage_request = _phase(eviction_runner._stage_request, "worker_eviction_stage")
    request_state.add_request = _phase(request_state.add_request, "worker_token_state_add")
    request_state.apply_staged_writes = _phase(request_state.apply_staged_writes, "worker_token_apply")
    model_state.apply_staged_writes = _phase(model_state.apply_staged_writes, "worker_model_apply")
    if frontend is not None:
        frontend.add_request_async = _async_outer(frontend.add_request_async, "frontend_add_send", persistent)
    if executor is not None:
        def rpc_fields(pos, named):
            # Remaining positional slots in the pinned native signature:
            # timeout, args, kwargs, non_block, unique_reply_rank, aggregators.
            return (pos[1] if len(pos) > 1 else named.get("args", ()),
                    pos[3] if len(pos) > 3 else named.get("non_block", False))

        def select_rpc(self, method, *pos, **named):
            args, non_block = rpc_fields(pos, named)
            return (method == "execute_model" and non_block and bool(args)
                    and bool(getattr(args[0], "scheduled_new_reqs", ())))

        def rpc_info(self, method, *pos, **named):
            args, non_block = rpc_fields(pos, named)
            return {"method": method, "non_block": non_block,
                    "requests": [_request_info(req) for req in args[0].scheduled_new_reqs]}

        executor.collective_rpc = _outer(executor.collective_rpc, "executor_enqueue_new",
                                         select_rpc, rpc_info, rpc_info)


def install():
    """Install after all serving hooks, in API/core/worker plugin processes."""
    global _INSTALLED
    flag = os.environ.get(FLAG, "0")
    if flag not in ("0", "1"):
        raise ValueError(f"{FLAG} must be 0 or 1")
    if flag == "0":
        return {"version": VERSION, "installed": False}
    if _INSTALLED:
        return status()
    path = os.environ.get(PATH, "")
    if not os.path.isabs(path) or not os.path.isdir(os.path.dirname(path)):
        raise ValueError(f"{PATH} must be an absolute journal prefix in an existing directory")
    import engine_patch
    import engine_eviction
    import eviction_runner
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.models.qwen3_8_flash_next.nvidia.model_state import Qwen3_8FlashNextModelState
    if (getattr(Scheduler, "_arc3_eviction_patch", None) != engine_eviction.VERSION
            or not eviction_runner._INSTALLED):
        raise RuntimeError("Install resume profiling after the complete core and worker hooks")
    install_targets(scheduler=Scheduler, core=EngineCore, engine_patch=engine_patch,
        engine_eviction=engine_eviction, runner=GPUModelRunner,
        model_state=Qwen3_8FlashNextModelState, request_state=RequestState,
        eviction_runner=eviction_runner, frontend=AsyncMPClient, executor=MultiprocExecutor)
    _INSTALLED = True
    return status()
