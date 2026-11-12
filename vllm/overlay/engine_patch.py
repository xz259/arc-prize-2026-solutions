"""Opt-in, source-pinned persistent MTP0 streaming-input control.

Importing this module does nothing. ``install()`` must run in every engine
process (and in the API process for empty-delta validation). No runtime files
are modified. Ordinary requests and unmarked streaming inputs retain stock
behavior. Mark each persistent chunk with SamplingParams.extra_args containing
``arc3_persistent_state=True``; use detokenize=False and DELTA output.

This phase-two copy also installs the isolated eager eviction hooks. Marked
requests still have at most one model step in flight; only the additional
``arc3_managed_eviction`` marker permits explicit resident compaction. The
phase-one checkpoint and the serving runtime files remain unchanged.
"""
from __future__ import annotations

import ast
import contextvars
import copy
import functools
import hashlib
import inspect
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PATCH_VERSION = "arc3-persistent-mtp0-v1"
MARKER = "arc3_persistent_state"
SOURCE_HASHES = {
    "v1/core/sched/scheduler.py": "c710f49e41e974e5b7b8f1cd2f5fb9722523f4da0165252e16351199ebd03124",
    "v1/core/sched/async_scheduler.py": "e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508",
    "v1/request.py": "0287844f70eeaeb077d714e833a4b449a15e045a6516f8530182e357a5bec82f",
    "v1/engine/input_processor.py": "f9a7946a16acc2374ff2bdfc22f212cb43461d9ef4d99c5e19a536339f11212f",
    "v1/worker/gpu_model_runner.py": "ccd19277d570ea6f7611f72e27ef1f69dfc8835ae3eb1be30d4948401b8a46ee",
    "v1/worker/gpu/model_runner.py": "f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8",
    "v1/worker/gpu/model_states/mamba_hybrid.py": "2e65f1da0a440f11a784bd36afe9206a7a2f674d522c047b838ad604e4c6acc0",
    "models/qwen3_8_flash_next/nvidia/model_state.py": "f127380fcd884c1fb7b010a2c32065d55c319f65a1268c18ef21376e39ba8425",
}
_INSTALL_RECEIPT: dict[str, Any] | None = None
_ALLOW_EMPTY_DELTA = contextvars.ContextVar("arc3_allow_empty_persistent_delta", default=False)


class PersistentStateError(RuntimeError):
    """Unsupported configuration or a violated append-only state invariant."""


def marked_params(params: Any) -> bool:
    args = getattr(params, "extra_args", None)
    return isinstance(args, dict) and args.get(MARKER) is True


def is_persistent(request: Any) -> bool:
    return bool(getattr(request, "resumable", False) and marked_params(request.sampling_params))


def validate_config(config: Any) -> dict[str, Any]:
    """Reject unsupported modes before any persistent request is admitted."""
    if config.speculative_config is not None or config.num_speculative_tokens != 0:
        raise PersistentStateError("Persistent phase 1 requires MTP0 without a draft model")
    if config.cache_config.enable_prefix_caching is not False:
        raise PersistentStateError("Persistent phase 1 requires prefix caching disabled")
    parallel = config.parallel_config
    if parallel.pipeline_parallel_size != 1 or parallel.tensor_parallel_size != 1:
        raise PersistentStateError("Persistent phase 1 requires PP1 and TP1")
    if getattr(parallel, "data_parallel_size", 1) != 1:
        raise PersistentStateError("Persistent phase 1 requires DP1")
    if getattr(config, "kv_transfer_config", None) is not None:
        raise PersistentStateError("Persistent phase 1 does not support a KV transfer connector")
    if getattr(config, "ec_transfer_config", None) is not None:
        raise PersistentStateError("Persistent phase 1 does not support an encoder transfer connector")
    if config.model_config.runner_type != "generate":
        raise PersistentStateError("Persistent phase 1 supports generation only")
    return {
        "model_runner": "v2" if config.use_v2_model_runner else "v1",
        "async_scheduling": bool(config.scheduler_config.async_scheduling),
        "max_model_len": int(config.model_config.max_model_len),
        "mtp": 0,
        "prefix_caching": False,
        "one_inflight_step_per_persistent_request": True,
    }


def validate_sampling(params: Any) -> None:
    if not marked_params(params):
        raise PersistentStateError("Persistent continuation is missing its explicit marker")
    if getattr(params, "n", None) != 1:
        raise PersistentStateError("Persistent phase 1 requires n=1")
    kind = getattr(params, "output_kind", None)
    if getattr(kind, "name", None) != "DELTA":
        raise PersistentStateError("Persistent phase 1 requires RequestOutputKind.DELTA")
    if getattr(params, "detokenize", True) is not False:
        raise PersistentStateError("Persistent phase 1 requires detokenize=False")
    if getattr(params, "stop", None):
        raise PersistentStateError("Persistent phase 1 does not support stop strings")
    if getattr(params, "structured_outputs", None) is not None:
        raise PersistentStateError("Persistent phase 1 does not support a grammar")
    if getattr(params, "max_tokens", None) is None or params.max_tokens <= 0:
        raise PersistentStateError("Persistent chunks require a positive explicit max_tokens")
    import one_lookahead
    one_lookahead.validate_sampling(params)


def assert_settled(request: Any) -> None:
    """A CPU-visible boundary must describe one exact committed model prefix."""
    for name in ("num_in_flight_tokens", "num_output_placeholders", "num_stale_output_tokens"):
        if getattr(request, name) != 0:
            raise PersistentStateError(f"Persistent boundary still has {name}={getattr(request, name)}")
    if request.spec_token_ids:
        raise PersistentStateError("Persistent MTP0 boundary contains speculative tokens")
    if not 0 <= request.num_computed_tokens <= len(request._all_token_ids):
        raise PersistentStateError("Processed prefix is outside the accepted token stream")


def append_session(scheduler: Any, session: Any, update: Any, *, statuses: Any, queued_event: Any) -> None:
    """Move all prior accepted output into input history without touching KV.

    Called from the stock lifecycle only after admission validation. All checks
    and transformed MM metadata are prepared before mutating the session.
    """
    validate_sampling(update.sampling_params)
    assert_settled(session)
    if session.num_preemptions:
        raise PersistentStateError("No-refill control was preempted; do not silently report persistence")
    old_ids = session._all_token_ids
    prompt_ids = session.prompt_token_ids
    if prompt_ids is None or session.prompt_embeds is not None:
        raise PersistentStateError("Persistent phase 1 requires explicit token IDs")
    if prompt_ids != old_ids[: session.num_prompt_tokens]:
        raise PersistentStateError("Stored prompt no longer matches the accepted token prefix")
    if session._output_token_ids != old_ids[session.num_prompt_tokens :]:
        raise PersistentStateError("Stored output IDs no longer match the accepted token suffix")
    delta = list(update.prompt_token_ids or ())
    base = len(old_ids)
    resident = base + len(delta)
    if resident >= scheduler.max_model_len:
        raise PersistentStateError("Persistent append exceeds the resident context budget; no eviction implemented")
    if update.max_tokens != update.sampling_params.max_tokens or update.max_tokens <= 0:
        raise PersistentStateError("Inconsistent chunk output budget")
    if update.max_tokens > scheduler.max_model_len - resident:
        raise PersistentStateError("Chunk output budget exceeds remaining resident capacity")
    # With one step in flight, normal AR stop leaves exactly its sampled last
    # token uncomputed. Empty continuation is legal only when that token exists.
    import one_lookahead
    one_lookahead.validate_append(session, update, delta)
    allowed_delta = (0, 1) if one_lookahead.marked(session) else (1,)
    if base - session.num_computed_tokens not in allowed_delta:
        raise PersistentStateError("Expected exactly one accepted, uncomputed final token at pause")
    if session.status != statuses.WAITING_FOR_STREAMING_REQ:
        raise PersistentStateError("Append requires an acknowledged paused session")
    if session.streaming_queue:
        raise PersistentStateError("A second input chunk was queued before the turn boundary")
    if scheduler.num_waiting_for_streaming_input <= 0:
        raise PersistentStateError("Paused-session admission counter would underflow")
    mm_features = []
    for feature in update.mm_features or ():
        position = feature.mm_position
        if position.offset < 0 or position.length < 0 or position.offset + position.length > len(delta):
            raise PersistentStateError("Multimodal feature lies outside the new token chunk")
        # Never shift caller-owned offsets in place: retry/reuse must not shift twice.
        mm_features.append(replace(feature, mm_position=replace(position, offset=position.offset + base)))

    import prefix_share_v1   # 2026-09-27 stage 15: system-prompt KV sharing (ARC3_PREFIX_SHARE); before any mutation
    prefix_share = prefix_share_v1.on_append(scheduler, session)
    kept_output = list(session._output_token_ids)
    prompt_ids.extend(kept_output)
    prompt_ids.extend(delta)
    old_ids.extend(delta)
    session._output_token_ids.clear()
    session.mm_features.extend(mm_features)
    session.update_block_hashes()
    session.num_prompt_tokens = len(prompt_ids)
    session.arrival_time = update.arrival_time
    session.sampling_params = update.sampling_params
    session.max_tokens = update.max_tokens
    session.stop_reason = None
    session.is_prefill_chunk = True
    session.next_decode_eligible_step = 0
    scheduler.num_waiting_for_streaming_input -= 1
    session.status = statuses.WAITING
    session._arc3_persistent_turn = getattr(session, "_arc3_persistent_turn", 0) + 1
    session._arc3_last_append = {
        "turn": session._arc3_persistent_turn,
        "old_resident_tokens": base,
        "appended_tokens": len(delta),
        "new_prompt_tokens": resident,
        "num_computed_tokens": session.num_computed_tokens,
        "pending_prefill_tokens": resident - session.num_computed_tokens,
        "retained_output_tokens": len(kept_output),
        "num_preemptions": session.num_preemptions,
        "prefix_share": prefix_share,
    }
    if scheduler.log_stats:
        session.record_event(queued_event)


def should_skip_inflight(request: Any) -> bool:
    import one_lookahead
    if one_lookahead.marked(request):
        return one_lookahead.skip(request)
    return bool(is_persistent(request) and request.num_in_flight_tokens > 0)


def _patched_schedule_ast(source: str) -> ast.FunctionDef:
    """Add one narrow guard to the exact known running-request iteration."""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    function = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule"))
    anchor = ast.parse("request = self.running[req_index]").body[0]
    found = 0
    for node in ast.walk(function):
        if not isinstance(node, ast.While):
            continue
        for i, statement in enumerate(node.body):
            if ast.dump(statement, include_attributes=False) == ast.dump(anchor, include_attributes=False):
                guard = ast.parse(
                    "if _arc3_persistent_skip_inflight(request):\n"
                    "    req_index += 1\n"
                    "    continue\n"
                ).body[0]
                node.body.insert(i + 1, guard)
                found += 1
                break
    if found != 1:
        raise PersistentStateError(f"Expected one schedule guard insertion site, found {found}")
    return ast.fix_missing_locations(function)


def verify_sources(runtime_root: Path) -> dict[str, str]:
    actual = {}
    for relative, expected in SOURCE_HASHES.items():
        path = runtime_root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise PersistentStateError(f"Unreviewed runtime source {relative}: {digest}, expected {expected}")
        actual[relative] = digest
    return actual


def _token_digest(tokens: Any) -> str:
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


_AUDIT_POOL = None


def _audit_pool():
    """Single background writer (2026-09-23): digests, JSON and the O_APPEND write leave the core thread; order kept."""
    global _AUDIT_POOL
    if _AUDIT_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _AUDIT_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arc3-engine-audit")
    return _AUDIT_POOL


def _audit_write(destination: str, item: dict, all_ids: list, computed: int) -> None:
    try:
        item["tokens_sha256"] = _token_digest(all_ids)
        item["computed_prefix_sha256"] = _token_digest(all_ids[:computed])
        encoded = (json.dumps(item, separators=(",", ":")) + "\n").encode()
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab", closefd=True) as stream:
            stream.write(encoded)
    except Exception as exc:  # noqa: BLE001 - evidence sink failures must not take the engine down asynchronously
        print(f"[arc3 engine audit] write failed: {type(exc).__name__}: {exc}", flush=True)


def _audit(scheduler: Any, event: str, request: Any, **extra: Any) -> None:
    destination = os.environ.get("ARC3_PERSISTENT_ENGINE_AUDIT")
    if not destination or not is_persistent(request):
        return
    block_ids = scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    item = {
        "event": event,
        "patch": PATCH_VERSION,
        "time": time.time(),
        "pid": os.getpid(),
        "request_id": request.request_id,
        "turn": getattr(request, "_arc3_persistent_turn", 0),
        "status": request.status.name,
        "resident_tokens": len(request._all_token_ids),
        "prompt_tokens": request.num_prompt_tokens,
        "output_tokens": len(request._output_token_ids),
        "computed_tokens": request.num_computed_tokens,
        "in_flight_tokens": request.num_in_flight_tokens,
        "output_placeholders": request.num_output_placeholders,
        "preemptions": request.num_preemptions,
        "block_ids": [list(group) for group in block_ids],
        "turn_work": dict(getattr(request, "_arc3_turn_work", {})),
        **extra,
    }
    # Snapshot the token ids now (they change at the next append/eviction); hash and write off-thread.
    _audit_pool().submit(_audit_write, destination, item, list(request._all_token_ids), int(request.num_computed_tokens))


def _new_work(request: Any) -> dict[str, int]:
    return {"scheduled_tokens": 0, "prefill_tokens": 0, "decode_tokens": 0,
            "steps": 0, "resume_computed_tokens": request.num_computed_tokens,
            "minimum_scheduled_position": request.num_computed_tokens}


def scheduler_snapshot(scheduler: Any) -> dict[str, Any]:
    """Small CPU-side diagnostic; no CUDA synchronization or token disclosure."""
    sessions = {}
    for request_id, req in scheduler.requests.items():
        if not is_persistent(req):
            continue
        blocks = scheduler.kv_cache_manager.get_blocks(request_id).get_block_ids()
        sessions[request_id] = {
            "status": req.status.name,
            "resident_tokens": len(req._all_token_ids),
            "computed_tokens": req.num_computed_tokens,
            "in_flight_tokens": req.num_in_flight_tokens,
            "output_placeholders": req.num_output_placeholders,
            "preemptions": req.num_preemptions,
            "block_ids": [list(group) for group in blocks],
            "last_append": getattr(req, "_arc3_last_append", None),
        }
    return {"patch": PATCH_VERSION, "sessions": sessions}


def _install_eviction_components(receipt: dict[str, Any]) -> dict[str, Any]:
    # General plugins run in the API, core, and worker processes. Installing
    # runner hooks here precedes Worker.load_model in a spawned worker, and the
    # hooks themselves reject unsupported execution modes when initialized.
    from engine_eviction import install as install_core_eviction
    from eviction_runner import install_runner_hooks
    from eviction_diagnostics import install_worker_diagnostics

    result = {**receipt, "eviction_core": install_core_eviction(),
              "eviction_worker": install_runner_hooks(),
              "eviction_diagnostics":install_worker_diagnostics()}
    if os.environ.get("ARC3_EVICTION_FUSED_CONTROL", "0") == "1":
        from control_transaction import install as install_control
        result["eviction_control"] = install_control()
    if os.environ.get("ARC3_EVICTION_MODE", "fused") == "async":
        from engine_eviction_async import install as install_async
        result["eviction_async"] = install_async()
    if os.environ.get("ARC3_RESUME_PROFILE", "0") == "1":
        from resume_profile import install as install_resume_profile
        result["resume_profile"] = install_resume_profile()
    if os.environ.get("ARC3_PERSISTENT_SINGLE_BATCH", "0") == "1":
        from persistent_batch_policy import install as install_batch_policy
        result["batch_policy"] = install_batch_policy()
    from one_lookahead import install as install_lookahead
    result["one_lookahead"] = install_lookahead()
    return result


def install() -> dict[str, Any]:
    """Install source-pinned hooks in this process; safe to call repeatedly."""
    global _INSTALL_RECEIPT
    if _INSTALL_RECEIPT is not None:
        return copy.deepcopy(_INSTALL_RECEIPT)
    import vllm
    from vllm.v1.core.sched import scheduler as sm
    from vllm.v1.engine.input_processor import InputProcessor

    runtime_root = Path(vllm.__file__).resolve().parent
    hashes = verify_sources(runtime_root)
    Scheduler = sm.Scheduler
    if getattr(Scheduler, "_arc3_persistent_patch", None) is not None:
        if Scheduler._arc3_persistent_patch != PATCH_VERSION:
            raise PersistentStateError("A different persistent-state patch is already installed")
        _INSTALL_RECEIPT = _install_eviction_components(
            {"patch": PATCH_VERSION, "runtime_root": str(runtime_root), "source_hashes": hashes})
        return copy.deepcopy(_INSTALL_RECEIPT)

    # Compile before mutating classes. Code executes in the original module's
    # namespace so stock globals and instrumentation remain authoritative.
    schedule_ast = _patched_schedule_ast((runtime_root / "v1/core/sched/scheduler.py").read_text())
    compiled = compile(ast.fix_missing_locations(ast.Module(body=[schedule_ast], type_ignores=[])), str(runtime_root / "v1/core/sched/scheduler.py") + ":arc3-persistent", "exec", flags=__import__("__future__").annotations.compiler_flag)
    namespace = dict(sm.__dict__)
    namespace["_arc3_persistent_skip_inflight"] = should_skip_inflight
    exec(compiled, namespace)
    patched_schedule = namespace["schedule"]
    original_append = Scheduler._update_request_as_session
    original_add = Scheduler.add_request
    original_stop = Scheduler._handle_stopped_request
    original_after_schedule = Scheduler._update_after_schedule
    original_free = Scheduler._free_request
    original_preempt = Scheduler._preempt_request
    original_input = InputProcessor.process_inputs
    original_validate_len = InputProcessor._validate_prompt_len
    input_signature = inspect.signature(original_input)

    @functools.wraps(original_append)
    def update_session(self, session, update):
        if not is_persistent(session):
            return original_append(self, session, update)
        append_session(self, session, update, statuses=sm.RequestStatus, queued_event=sm.EngineCoreEventType.QUEUED)
        session._arc3_turn_work = _new_work(session)
        _audit(self, "resume", session, append=session._arc3_last_append)
        return None

    @functools.wraps(original_add)
    def add_request(self, request):
        existing = self.requests.get(request.request_id)
        if is_persistent(request):
            validate_config(self.vllm_config)
            validate_sampling(request.sampling_params)
            import one_lookahead
            one_lookahead.validate_admission(self, request, existing)
            if request.prompt_embeds is not None or request.prompt_is_token_ids is not None:
                raise PersistentStateError("Mixed or embedded persistent prompts are not supported")
            if request.use_structured_output or request.pooling_params is not None:
                raise PersistentStateError("Persistent requests cannot use pooling or structured-output state")
            if existing is None:
                if not request.prompt_token_ids:
                    raise PersistentStateError("A new persistent session must have a nonempty initial prompt")
                if request.num_tokens + request.max_tokens > self.max_model_len:
                    raise PersistentStateError("Initial prompt and output budget exceed resident capacity")
            elif not is_persistent(existing):
                raise PersistentStateError("Cannot append persistent input to an unrelated request")
            elif existing.status != sm.RequestStatus.WAITING_FOR_STREAMING_REQ:
                raise PersistentStateError("Persistent append must wait for the prior turn finish")
        elif existing is not None and is_persistent(existing) and request.resumable:
            raise PersistentStateError("Persistent update must keep its explicit marker")
        result = original_add(self, request)
        if existing is None and is_persistent(request):
            request._arc3_turn_work = _new_work(request)
            _audit(self, "start", request, config=validate_config(self.vllm_config))
        return result

    @functools.wraps(original_stop)
    def handle_stopped(self, request):
        import one_lookahead
        if one_lookahead.marked(request):
            one_lookahead.prepare_stop(self, request)
            return original_stop(self, request)
        if is_persistent(request):
            assert_settled(request)
            if len(request._all_token_ids) - request.num_computed_tokens != 1:
                raise PersistentStateError("Persistent stop is not at the accepted AR frontier")
            if request.streaming_queue and any(item is not None for item in request.streaming_queue):
                raise PersistentStateError("Persistent input was queued before the acknowledged stop")
        finished = original_stop(self, request)
        if is_persistent(request) and not finished:
            _audit(self, "pause", request)
        return finished

    @functools.wraps(original_after_schedule)
    def after_schedule(self, scheduler_output):
        for request_id, count in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            if not is_persistent(request):
                continue
            import one_lookahead
            if one_lookahead.marked(request):
                one_lookahead.after_schedule_check(request, count)
            elif request.num_in_flight_tokens:
                raise PersistentStateError("A persistent request was scheduled with a step already in flight")
            work = getattr(request, "_arc3_turn_work", None)
            if work is None:
                work = request._arc3_turn_work = _new_work(request)
            start = request.num_computed_tokens
            prefill = max(0, min(count, request.num_prompt_tokens - start))
            work["scheduled_tokens"] += count
            work["prefill_tokens"] += prefill
            work["decode_tokens"] += count - prefill
            work["steps"] += 1
            work["minimum_scheduled_position"] = min(work["minimum_scheduled_position"], start)
        return original_after_schedule(self, scheduler_output)

    @functools.wraps(original_free)
    def free_request(self, request, *args, **kwargs):
        import one_lookahead
        if one_lookahead.marked(request) and not getattr(request, "_arc3_la_allow_free", False):
            return one_lookahead.defer_free(self, request, args, kwargs)
        if is_persistent(request):
            _audit(self, "close", request, release_phase="before_existing_fenced_free")
        result = original_free(self, request, *args, **kwargs)
        if is_persistent(request):
            _audit(self, "close_released", request,
                   request_still_registered=request.request_id in self.requests,
                   release_phase="after_existing_fenced_free")
        return result

    @functools.wraps(original_preempt)
    def preempt_request(self, request, *args, **kwargs):
        import one_lookahead
        if one_lookahead.marked(request):
            raise PersistentStateError("One-lookahead numerical candidate forbids preemption")
        if is_persistent(request):
            _audit(self, "preempt", request, control_valid=False)
        return original_preempt(self, request, *args, **kwargs)

    @functools.wraps(original_input)
    def process_inputs(self, *args, **kwargs):
        bound = input_signature.bind(self, *args, **kwargs)
        allow = bool(bound.arguments.get("resumable", False) and marked_params(bound.arguments.get("params")))
        if allow:
            validate_sampling(bound.arguments["params"])
        token = _ALLOW_EMPTY_DELTA.set(allow)
        try:
            return original_input(self, *args, **kwargs)
        finally:
            _ALLOW_EMPTY_DELTA.reset(token)

    @functools.wraps(original_validate_len)
    def validate_prompt_len(self, prompt_len, prompt_type):
        if prompt_len == 0 and prompt_type == "decoder" and _ALLOW_EMPTY_DELTA.get():
            # No dummy token. Scheduler admits only an already paused session
            # with the one accepted residual token that still needs forwarding.
            return None
        return original_validate_len(self, prompt_len, prompt_type)

    Scheduler.schedule = patched_schedule
    Scheduler._update_request_as_session = update_session
    Scheduler.add_request = add_request
    Scheduler._handle_stopped_request = handle_stopped
    Scheduler._update_after_schedule = after_schedule
    Scheduler._free_request = free_request
    Scheduler._preempt_request = preempt_request
    Scheduler.arc3_persistent_snapshot = scheduler_snapshot
    InputProcessor.process_inputs = process_inputs
    InputProcessor._validate_prompt_len = validate_prompt_len
    Scheduler._arc3_persistent_patch = PATCH_VERSION
    _INSTALL_RECEIPT = _install_eviction_components(
        {"patch": PATCH_VERSION, "runtime_root": str(runtime_root), "source_hashes": hashes})
    return copy.deepcopy(_INSTALL_RECEIPT)
