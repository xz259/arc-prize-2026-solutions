"""Opt-in single-RPC eviction control, preserving the native global fence.

Install after engine_eviction.install(). The existing arc3_evict_resident
utility and all native pause/resume calls remain available. While a fused
transaction owns a lease, external pause/resume calls fail before mutation.
No GPU copies or allocator changes are implemented in this module.
"""
from __future__ import annotations

import contextvars
import functools
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass


VERSION = "arc3-fused-fenced-control-v1"
FLAG = "ARC3_EVICTION_FUSED_CONTROL"
METHOD = "arc3_evict_resident_transaction"
_LEASE = "_arc3_fused_eviction_lease"
_THREAD = "_arc3_fused_control_thread"
_INTERNAL = contextvars.ContextVar("arc3_fused_control_internal", default=None)


class ControlTransactionError(RuntimeError):
    pass


class ControlThreadError(ControlTransactionError):
    pass


class ControlCleanupError(ControlTransactionError):
    def __init__(self, primary, cleanup):
        self.primary = primary
        self.cleanup = cleanup
        super().__init__(f"Fused eviction cleanup failed: {cleanup!r}; "
                         f"original operation error: {primary!r}. "
                         "Control lease retained; explicit recovery is required.")


@dataclass(eq=False)
class _Lease:
    operation_id: str
    thread_id: int
    phase: str = "entering"


def enabled():
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ControlTransactionError(f"{FLAG} must be 0 or 1")
    return value == "1"


def _internal_call(core, lease, method, *args, **kwargs):
    if threading.get_ident() != lease.thread_id:
        raise ControlThreadError("Native pause/resume must execute on the initiating core thread")
    token = _INTERNAL.set((core, lease))
    try:
        return method(*args, **kwargs)
    finally:
        _INTERNAL.reset(token)


def _guard(method):
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        lease = getattr(self, _LEASE, None)
        internal = _INTERNAL.get()
        if lease is not None and not (
                internal is not None and internal[0] is self and internal[1] is lease
                and threading.get_ident() == lease.thread_id):
            raise ControlTransactionError(
                f"Native {method.__name__} rejected: fused eviction owns the pause lease")
        return method(self, *args, **kwargs)
    guarded._arc3_fused_control_guard = VERSION
    return guarded


def _classes(root):
    """Include actual EngineCoreProc overrides, not only inherited base methods."""
    result = []
    pending = [root]
    while pending:
        cls = pending.pop()
        if cls not in result:
            result.append(cls)
            pending.extend(cls.__subclasses__())
    return result


def _transaction(core, external_request_id, payload, *, wakeup_type):
    # This method is entered through native EngineCore utility dispatch. The
    # native idle callback must return to this same thread before touching KV.
    thread_id = threading.get_ident()
    previous_thread = getattr(core, _THREAD, thread_id)
    if previous_thread != thread_id:
        raise ControlThreadError("Fused utility invoked from a different core thread")
    if (not isinstance(external_request_id, str) or not external_request_id
            or not isinstance(payload, dict)
            or not isinstance(payload.get("operation_id"), str)
            or not payload["operation_id"]):
        raise ControlTransactionError("Fused eviction requires request and operation IDs")
    if getattr(core, _LEASE, None) is not None:
        raise ControlTransactionError("Another fused eviction owns the control lease")
    started = time.perf_counter()
    if core.is_scheduler_paused():
        raise ControlTransactionError("Scheduler was already paused outside this transaction")
    if not hasattr(core, "input_queue") or not callable(getattr(core.input_queue, "put_nowait", None)):
        raise ControlTransactionError("Fused control requires native EngineCoreProc input queue")

    lease = _Lease(payload["operation_id"], thread_id)
    setattr(core, _THREAD, thread_id)
    setattr(core, _LEASE, lease)
    result = Future()
    # Client-side cancellation must not cancel a mutation or its cleanup.
    result.set_running_or_notify_cancel()
    profile = {"status_seconds": time.perf_counter() - started,
               "drain_seconds": 0.0, "eviction_seconds": 0.0,
               "resume_seconds": 0.0, "wakeup_seconds": 0.0,
               "deferred_drain": 0, "callback_thread_verified": 0}
    drain_started = time.perf_counter()

    def fail_closed(error):
        # A foreign callback cannot safely mutate scheduling state. Keep the
        # pause and lease, expose failure, and require explicit core recovery.
        if not result.done():
            result.set_exception(error)

    def finish(primary=None, *, run_eviction=False):
        if threading.get_ident() != thread_id:
            fail_closed(ControlThreadError(
                "Drain callback ran outside the initiating core thread; "
                "pause/lease retained, no eviction or cross-thread resume attempted"))
            return
        profile["callback_thread_verified"] = 1
        if getattr(core, _LEASE, None) is not lease:
            fail_closed(ControlTransactionError(
                "Fused control lost its pause lease; refusing mutation or unrelated resume"))
            return
        ack = None
        if primary is None and run_eviction:
            lease.phase = "evicting"
            phase = time.perf_counter()
            try:
                if not core.is_scheduler_paused():
                    raise ControlTransactionError("Native drain completed without an owned pause")
                # Existing utility proves empty queue, every owner settled,
                # exclusive pages, exact ledgers and synchronized worker copy.
                ack = core.arc3_evict_resident(external_request_id, payload)
                if not isinstance(ack, dict):
                    raise ControlTransactionError("Existing eviction utility returned an invalid acknowledgement")
            except BaseException as error:
                primary = error
            finally:
                profile["eviction_seconds"] = time.perf_counter() - phase

        lease.phase = "resuming"
        cleanup_error = None
        try:
            # A pause can fail before setting its flag. Do not resume in that
            # case; if it failed after setting it, this lease still owns it.
            if core.is_scheduler_paused():
                phase = time.perf_counter()
                try:
                    _internal_call(core, lease, core.resume_scheduler)
                finally:
                    profile["resume_seconds"] = time.perf_counter() - phase
                if core.is_scheduler_paused():
                    raise ControlTransactionError("Owned resume left scheduler paused")
                phase = time.perf_counter()
                # Native _process_input_queue may be inside its idle while
                # body and about to block on get(), without rechecking work.
                # Resume must wake that loop even with no further API traffic.
                core.input_queue.put_nowait((wakeup_type, None))
                profile["wakeup_seconds"] = time.perf_counter() - phase
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            lease.phase = "cleanup_failed"
            fail_closed(ControlCleanupError(primary, cleanup_error))
            return
        if getattr(core, _LEASE, None) is not lease:
            fail_closed(ControlTransactionError("Pause lease changed during cleanup"))
            return
        delattr(core, _LEASE)
        profile["total_seconds"] = time.perf_counter() - started
        if primary is not None:
            result.set_exception(primary)
        else:
            # Do not insert timing fields into the signed/audited engine ack.
            result.set_result({"ack": ack, "control_profile": profile})

    def drained(done):
        profile["drain_seconds"] = time.perf_counter() - drain_started
        if threading.get_ident() != thread_id:
            finish()
            return
        try:
            done.result()
        except BaseException as error:
            finish(error)
        else:
            finish(run_eviction=True)

    try:
        lease.phase = "draining"
        pending = _internal_call(core, lease, core.pause_scheduler,
                                 mode="keep", clear_cache=False)
    except BaseException as error:
        profile["drain_seconds"] = time.perf_counter() - drain_started
        finish(error)
    else:
        if pending is None:
            profile["drain_seconds"] = time.perf_counter() - drain_started
            finish(run_eviction=True)
        elif isinstance(pending, Future):
            profile["deferred_drain"] = 1
            pending.add_done_callback(drained)
        else:
            finish(ControlTransactionError("Native pause returned an unsupported drain handle"))
    return result


def install(EngineCore=None, *, wakeup_type=None):
    """Install after normal engine_eviction.install; default flag leaves baseline untouched.

    ``wakeup_type`` is injectable for CPU tests only. Production resolves the
    real native EngineCoreRequestType.WAKEUP; it is not a model input token.
    """
    if not enabled():
        return {"installed": False, "enabled": False, "version": VERSION}
    if EngineCore is None:
        from vllm.v1.engine.core import EngineCore
    if wakeup_type is None:
        from vllm.v1.engine import EngineCoreRequestType
        wakeup_type = EngineCoreRequestType.WAKEUP
    if not callable(getattr(EngineCore, "arc3_evict_resident", None)):
        raise ControlTransactionError("Install normal engine eviction before fused control")
    existing = EngineCore.__dict__.get(METHOD)
    if existing is not None:
        if getattr(existing, "_arc3_fused_control_version", None) != VERSION:
            raise ControlTransactionError("Refusing to replace an unknown core utility")
        return {"installed": True, "enabled": True, "version": VERSION, "method": METHOD}
    for cls in _classes(EngineCore):
        for name in ("pause_scheduler", "resume_scheduler"):
            method = cls.__dict__.get(name)
            if method is not None and getattr(method, "_arc3_fused_control_guard", None) != VERSION:
                setattr(cls, name, _guard(method))

    def utility(self, external_request_id, payload):
        return _transaction(self, external_request_id, payload, wakeup_type=wakeup_type)
    utility.__name__ = METHOD
    utility._arc3_fused_control_version = VERSION
    setattr(EngineCore, METHOD, utility)
    return {"installed": True, "enabled": True, "version": VERSION, "method": METHOD}
