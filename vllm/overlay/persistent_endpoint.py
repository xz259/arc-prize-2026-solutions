"""Explicitly allowlisted endpoint for the isolated managed-suffix experiment."""
from __future__ import annotations

import asyncio
import os
import time
from copy import deepcopy
from contextlib import asynccontextmanager
from fastapi import Request

from session_service import SessionError, SessionManager, TurnInput, token_digest, session_token_digest


SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "max_tokens", "seed",
                 "logprobs", "ignore_eos", "stop_token_ids", "arc3_one_lookahead"}


def _eviction_verification(initial_payload, payload, *, initial):
    original = False if initial else initial_payload.get("arc3_verify_eviction", False)
    requested = payload.get("arc3_verify_eviction", original)
    if type(original) is not bool or type(requested) is not bool:
        raise SessionError("arc3_verify_eviction must be a boolean", 400)
    if not initial and requested != original:
        raise SessionError("Cannot change the eviction verification session contract", 400)
    return requested


class VllmBackend:
    def __init__(self, engine, serving_chat):
        self.engine = engine
        self.serving_chat = serving_chat
        self.tokenizer = engine.renderer.get_tokenizer()
        self.eviction_lock = asyncio.Lock()

    async def prepare(self, session, payload, *, initial):
        _eviction_verification(session.initial, payload, initial=initial)
        from protocol import render_initial, render_delta
        raw = payload.get("raw")
        records, retire_ids, observation_starts = [], [], []
        injected_ids = []
        if not initial and ((raw is not None) != (session.initial.get("raw") is not None)):
            raise SessionError("Cannot change between raw and chat session modes", 400)
        if raw is not None:
            if not isinstance(raw, dict) or set(raw) - {"token_ids", "suffix_records", "evict_observation_ids", "observation_starts"}:
                raise SessionError("Unsupported raw diagnostic fields", 400)
            ids = raw.get("token_ids")
            if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                raise SessionError("Invalid raw token IDs", 400)
            prompt = {"type": "token", "prompt_token_ids": list(ids)}
            context = None
            sampling = dict(payload.get("sampling", {}))
            records = deepcopy(raw.get("suffix_records", []))
            retire_ids = deepcopy(raw.get("evict_observation_ids", []))
            observation_starts = deepcopy(raw.get("observation_starts", []))
        elif initial:
            chat = payload.get("chat")
            if not isinstance(chat, dict):
                raise SessionError("start requires chat or raw input", 400)
            rendered = await render_initial(self.serving_chat, chat,
                                            managed_suffixes=payload.get("managed_suffixes"))
            prompt, context = rendered.engine_input, rendered.state
            records, retire_ids = rendered.suffix_records, rendered.evict_observation_ids
            observation_starts = rendered.observation_starts
            sampling = {k: chat[k] for k in SAMPLING_KEYS if k in chat}
        else:
            if session.parse_context is None:
                raise SessionError("Raw sessions require raw appends", 400)
            context = session.parse_context
            tail = payload.get("tail_eviction")
            if tail is not None:
                if (not isinstance(tail, dict) or set(tail) - {"assistant_offset", "reason"}
                        or type(tail.get("assistant_offset")) is not int or tail["assistant_offset"] < 0
                        or type(tail.get("reason", "")) is not str or not payload.get("continuation")):
                    raise SessionError("tail_eviction requires a nonnegative assistant_offset on a length continuation", 400)
            injection = payload.get("assistant_injection")
            if injection is not None:
                if (not isinstance(injection, dict) or set(injection) - {"text", "kind", "close"}
                        or type(injection.get("text")) is not str or not injection["text"]
                        or type(injection.get("kind", "")) is not str
                        or type(injection.get("close", False)) is not bool):
                    raise SessionError("assistant_injection requires text, optional kind and boolean close", 400)
                if not payload.get("continuation"):
                    raise SessionError("assistant_injection is valid only on a length continuation", 400)
            rendered = await render_delta(self.serving_chat, context,
                                        payload.get("external_messages", []),
                                        continuation=bool(payload.get("continuation")),
                                        managed_suffixes=payload.get("managed_suffixes"),
                                        injection_text=injection["text"] if injection is not None else None)
            prompt = rendered.engine_input
            injected_ids = list(rendered.injected_ids)
            records, retire_ids = rendered.suffix_records, rendered.evict_observation_ids
            observation_starts = rendered.observation_starts
            sampling = dict(payload.get("sampling", {}))
        if set(sampling) - SAMPLING_KEYS:
            raise SessionError("Unsupported sampling settings: " + str(sorted(set(sampling) - SAMPLING_KEYS)), 400)
        if not initial:
            original = session.initial.get("sampling", session.initial.get("chat", {}))
            initial_flag = original.get("arc3_one_lookahead", False)
            requested_flag = sampling.get("arc3_one_lookahead", initial_flag)
            if type(requested_flag) is not bool or requested_flag != initial_flag:
                raise SessionError("Cannot change the one-lookahead session contract", 400)
            if initial_flag:
                sampling["arc3_one_lookahead"] = True
        base = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
        base.update(sampling)
        injection = payload.get("assistant_injection") if not initial and raw is None else None
        tail = payload.get("tail_eviction") if not initial and raw is None else None
        if injection is not None and injection.get("close", False):
            # Hard close: append the text, then generate exactly one forced <|im_end|> so the turn ends on a real stop.
            from protocol import _ids
            base["max_tokens"] = 1
            base["allowed_token_ids"] = list(_ids(self.tokenizer, "<|im_end|>"))
        # Sampler schema validation is CPU-only and precedes session mutation.
        self.sampling_params({**base, "max_tokens": base.get("max_tokens") or 1})
        return TurnInput(prompt, base, context, records, retire_ids, observation_starts,
                         injected_ids=injected_ids, injection=dict(injection) if injection is not None else None,
                         tail_eviction=dict(tail) if tail is not None else None)

    def sampling_params(self, values):
        from vllm.sampling_params import SamplingParams, RequestOutputKind
        values = dict(values)
        if os.environ.get("ARC3_EVICTION_NLL_PROBE", "0") not in ("", "0"):
            values.setdefault("logprobs", 0)   # eviction-NLL probe: sampled-token logprobs on every turn (fixed per request)
        lookahead = values.pop("arc3_one_lookahead", False)
        if type(lookahead) is not bool:
            raise SessionError("arc3_one_lookahead must be a boolean", 400)
        extra = {"arc3_persistent_state": True, "arc3_managed_eviction": True}
        if lookahead:
            extra["arc3_persistent_one_lookahead"] = True
        params = SamplingParams(**values, n=1, detokenize=False,
                              output_kind=RequestOutputKind.DELTA,
                              extra_args=extra)
        import one_lookahead
        one_lookahead.validate_sampling(params)
        return params

    async def preflight(self, session, turn, *, initial):
        params = self.sampling_params(turn.sampling)
        if turn.sampling.get("arc3_one_lookahead") is True and not initial:
            # Explicit opt-in API contract: empty continuation remains available
            # after a length stop. EOS/stop boundaries require nonempty input,
            # conservatively including the EOS cases without a queued lookahead.
            if not turn.prompt.get("prompt_token_ids") and session.finish_reason != "length":
                raise SessionError("One-lookahead EOS/stop requires nonempty append", 400)
        self.engine.input_processor.process_inputs(
            request_id=session.engine_request_id, prompt=deepcopy(turn.prompt),
            params=params, supported_tasks=await self.engine.get_supported_tasks(),
            resumable=True)

    async def frequency_snapshot(self, session):
        """Read-only counter transfer for one paused owner; no scheduler pause.

        Any utility failure is reported as QSAFrequencyError so the planner records a
        chronological fallback ("no_frequency_snapshot") instead of failing the turn.
        """
        from qsa_frequency import QSAFrequencyError
        compact = os.environ.get("ARC3_QSA_SNAPSHOT", "compact") != "legacy"   # stage B (2026-09-24)
        try:
            if not compact:
                return await self.engine.engine_core.call_utility_async(
                    "arc3_qsa_frequency_snapshot", session.engine_request_id)
            snapshot = await self.engine.engine_core.call_utility_async(
                "arc3_qsa_frequency_snapshot_compact", session.engine_request_id)
            ledger = getattr(session, "residency", None); ctx = snapshot.get("context") or {}
            if ledger is not None and hasattr(ledger, "evicted") and hasattr(ledger, "logical_tokens"):
                if ([list(r) for r in ledger.evicted] != ctx.get("evicted")
                        or int(ledger.logical_tokens) != int(ctx.get("logical_tokens", -1))):
                    raise QSAFrequencyError("Session and engine resident ledgers differ at the snapshot")
            return snapshot
        except QSAFrequencyError:
            raise
        except Exception as exc:
            raise QSAFrequencyError(f"{type(exc).__name__}: {exc}") from exc

    async def evict(self, session, plan):
        """Drain model work, mutate one paused owner, then resume scheduling.

        The opt-in fused utility owns pause/drain/mutation/resume on the core.
        Caller cancellation still waits for that fenced transaction to finish.
        """
        payload = {"operation_id": session.current_operation,
                   "expected_logical_tokens": len(session.token_ids),
                   "logical_prefix_sha256": session_token_digest(session),
                   "logical_ranges": [list(r) for r in plan.ranges],
                   "expected_eviction_revision": session.eviction_revision,
                   "resident_before": plan.resident_before,
                   "resident_after": plan.resident_after,
                   "prior_evicted": [list(r) for r in plan.prior_evicted],
                   "plan_digest": plan.digest}
        if _eviction_verification(session.initial, {}, initial=False):
            payload["verify_eviction"] = True
        if getattr(session, "frequency_epoch_reset_pending", False):
            payload["frequency_epoch_reset"] = True

        async def transaction():
            profiling = os.environ.get("ARC3_EVICTION_PROFILE", "0") == "1"
            started = time.perf_counter()
            phases = {}
            async with self.eviction_lock:
                phases["lock_wait_seconds"] = time.perf_counter() - started
                core = self.engine.engine_core
                if os.environ.get("ARC3_EVICTION_MODE", "fused") == "async":
                    # stage C: begin (plan + launch copies on the eviction stream, no drain), then poll the commit
                    phase_start = time.perf_counter()
                    begun = await core.call_utility_async("arc3_evict_begin", session.engine_request_id, payload)
                    phases["begin_seconds"] = time.perf_counter() - phase_start
                    if not isinstance(begun, dict) or begun.get("begun") is not True:
                        raise RuntimeError("Invalid asynchronous eviction begin acknowledgement")
                    phase_start = time.perf_counter(); polls = 0
                    while True:
                        ack = await core.call_utility_async("arc3_evict_commit", session.engine_request_id)
                        if isinstance(ack, dict) and ack.get("pending"):
                            polls += 1
                            if time.perf_counter() - phase_start > 120:
                                raise RuntimeError("Asynchronous eviction did not complete")
                            await asyncio.sleep(0.02); continue
                        break
                    if not isinstance(ack, dict):
                        raise RuntimeError("Invalid asynchronous eviction acknowledgement")
                    phases["commit_seconds"] = time.perf_counter() - phase_start; phases["commit_polls"] = polls
                    phases["core_control"] = {"mode": "async", "begin": begun.get("core_profile"), "device": begun.get("device")}
                    phases["total_seconds"] = time.perf_counter() - started
                    session.eviction_endpoint_profile = phases if profiling else None
                    return ack
                if os.environ.get("ARC3_EVICTION_FUSED_CONTROL", "0") == "1":
                    phase_start = time.perf_counter()
                    envelope = await core.call_utility_async(
                        "arc3_evict_resident_transaction", session.engine_request_id, payload)
                    phases["fused_rpc_seconds"] = time.perf_counter() - phase_start
                    if (not isinstance(envelope, dict)
                            or not isinstance(envelope.get("ack"), dict)
                            or not isinstance(envelope.get("control_profile"), dict)):
                        raise RuntimeError("Invalid fused eviction transaction envelope")
                    phases["core_control"] = envelope["control_profile"]
                    phases["total_seconds"] = time.perf_counter() - started
                    session.eviction_endpoint_profile = phases if profiling else None
                    # No endpoint resume here: the future resolves only after
                    # the core-owned lease has been released and scheduling resumed.
                    return envelope["ack"]
                phase_start = time.perf_counter()
                if await core.is_scheduler_paused_async():
                    raise SessionError("Scheduler is already paused outside this eviction transaction")
                phases["status_seconds"] = time.perf_counter() - phase_start
                paused = False
                try:
                    # The actual EngineCoreProc keep pause drains the queued
                    # batches before its future resolves. No cache reset.
                    paused = True
                    phase_start = time.perf_counter()
                    await core.pause_scheduler_async(mode="keep", clear_cache=False)
                    phases["drain_seconds"] = time.perf_counter() - phase_start
                    phase_start = time.perf_counter()
                    result = await core.call_utility_async(
                        "arc3_evict_resident", session.engine_request_id, payload)
                    phases["core_rpc_seconds"] = time.perf_counter() - phase_start
                finally:
                    if paused:
                        phase_start = time.perf_counter()
                        await core.resume_scheduler_async()
                        phases["resume_seconds"] = time.perf_counter() - phase_start
                if profiling:
                    phases["total_seconds"] = time.perf_counter() - started
                    # Transport timings are not part of the engine's signed
                    # transaction acknowledgement. Preserve that object exactly.
                    session.eviction_endpoint_profile = phases
                else:
                    session.eviction_endpoint_profile = None
                return result

        task = asyncio.create_task(transaction())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Complete the fenced transaction/resume before the caller aborts
            # its now-uncertain session. Never retry this as another eviction.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    # A timeout plus an explicit close may cancel twice. Neither
                    # is allowed to cancel a transaction already mutating KV.
                    continue
                except BaseException:
                    break
            if task.done() and not task.cancelled():
                task.exception()  # consume any transaction failure before abort
            raise

    async def turn(self, session, turn, *, initial):
        from vllm.engine.protocol import StreamingInput
        params = self.sampling_params(turn.sampling)
        # Validate before queueing, so schema errors do not partially mutate the
        # engine. No image reprocessing occurs: turn.prompt is already rendered.
        self.engine.input_processor.process_inputs(
            request_id=session.engine_request_id, prompt=deepcopy(turn.prompt),
            params=params, supported_tasks=await self.engine.get_supported_tasks(),
            resumable=True)
        session.output_future = asyncio.get_running_loop().create_future()
        if initial:
            async def input_stream():
                while True:
                    item = await session.input_queue.get()
                    if item is None:
                        return
                    yield item

            async def collect():
                ids, logprobs = [], []; prompt_lp = None
                try:
                    async for output in self.engine.generate(input_stream(), params,
                                                              session.engine_request_id):
                        if getattr(output, "prompt_logprobs", None):
                            prompt_lp = (prompt_lp or []) + list(output.prompt_logprobs)
                        for completion in output.outputs:
                            ids.extend(completion.token_ids)
                            if completion.logprobs:
                                logprobs.extend([
                                    {str(k): {"logprob": v.logprob, "rank": v.rank}
                                     for k, v in row.items()} for row in completion.logprobs])
                            if completion.finish_reason is not None:
                                future = session.output_future
                                if future is None or future.done():
                                    raise RuntimeError("Unexpected or duplicate streaming turn boundary")
                                future.set_result({"token_ids": ids, "logprobs": logprobs, "prompt_logprobs": prompt_lp,
                                                   "finish_reason": completion.finish_reason,
                                                   "stop_reason": completion.stop_reason})
                                ids, logprobs = [], []; prompt_lp = None
                    if session.state not in ("closed", "error"):
                        raise RuntimeError("Engine closed a persistent session without explicit close")
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    if session.output_future is not None and not session.output_future.done():
                        session.output_future.set_exception(exc)
                    raise

            session.collector = asyncio.create_task(collect())
            session.collector.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        if session.collector is None or session.collector.done():
            raise RuntimeError("Persistent engine collector is not alive")
        session.input_queue.put_nowait(StreamingInput(prompt=turn.prompt, sampling_params=params))
        return await session.output_future

    async def restart_request(self, session):
        """oldest_recompute: drop the engine request (its KV is freed by the abort) and give the session a fresh request
        id, input queue and collector slot; the caller's next turn(initial=True) prefills the rebuilt prompt."""
        await self.close(session)
        session.restarts = getattr(session, "restarts", 0) + 1
        session.engine_request_id = session.engine_request_id.split("-rc")[0] + f"-rc{session.restarts}"
        session.input_queue = asyncio.Queue(); session.collector = None; session.output_future = None

    async def probe(self, prompt, sampling, *, tag):
        """Eviction-NLL probe (2026-09-23): one internal persistent request on a constructed engine input (fresh
        prefill, no session registry entry); returns the collector result (token_ids, logprobs, prompt_logprobs,
        finish_reason) and always closes/aborts the request."""
        import uuid
        from session_service import Session, TurnInput
        session = Session(session_id=f"probe-{tag}", engine_request_id=f"arc3-probe-{uuid.uuid4().hex}", initial={"probe": True})
        turn = TurnInput(prompt=prompt, sampling=dict(sampling))
        try:
            return await asyncio.wait_for(self.turn(session, turn, initial=True), timeout=900)
        finally:
            session.state = "closed"
            await self.close(session)

    async def response(self, session, result):
        from protocol import parse_output
        if session.parse_context is None:
            text = self.tokenizer.decode(result["token_ids"], skip_special_tokens=False)
            choice = {"index": 0, "message": {"role": "assistant", "content": text},
                      "token_ids": list(result["token_ids"]),
                      "finish_reason": result["finish_reason"], "logprobs": result["logprobs"]}
        else:
            choice = parse_output(self.serving_chat, session.parse_context, result["token_ids"],
                                  finish_reason=result["finish_reason"], stop_reason=result["stop_reason"])
        return {"id": session.current_operation, "object": "chat.completion",
                "created": int(time.time()), "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
                "choices": [choice]}

    async def close(self, session):
        if session.collector is not None:
            session.collector.cancel()
            await asyncio.gather(session.collector, return_exceptions=True)
        # Explicitly abort by external request ID as well. Abort is idempotent;
        # generate cancellation handles the randomized internal request ID.
        await self.engine.abort(session.engine_request_id)
        if session.output_future is not None and not session.output_future.done():
            session.output_future.cancel()


class PersistentEndpoint:
    name = "arc3_persistent"
    required_tasks = ("generate",)

    def attach_router(self, app):
        from fastapi import APIRouter
        from fastapi.responses import JSONResponse
        router = APIRouter(prefix="/arc3/persistent")

        async def invoke(request, method, *args):
            manager = request.app.state.arc3_persistent_manager
            try:
                return await getattr(manager, method)(*args)
            except SessionError as e:
                return JSONResponse({"error": {"message": str(e), "type": "persistent_session_error",
                                                "history_eviction": e.receipt}},
                                    status_code=e.status)
            except Exception as e:
                return JSONResponse({"error": {"message": f"{type(e).__name__}: {e}",
                                                "type": "persistent_engine_error"}}, status_code=500)

        @router.post("/start")
        async def start(request: Request):
            return await invoke(request, "start", await request.json())

        @router.post("/{session_id}/append")
        async def append(session_id: str, request: Request):
            return await invoke(request, "append", session_id, await request.json())

        @router.post("/{session_id}/close")
        async def close(session_id: str, request: Request):
            return await invoke(request, "close", session_id)

        @router.get("/{session_id}/status")
        async def status(session_id: str, request: Request):
            try:
                return request.app.state.arc3_persistent_manager.status(session_id)
            except SessionError as e:
                return JSONResponse({"error": str(e)}, status_code=e.status)

        @router.get("/{session_id}/lookahead-diagnostic")
        async def lookahead_diagnostic(session_id: str, request: Request):
            import one_lookahead
            if not one_lookahead.enabled():
                return JSONResponse({"error": "Numerical candidate is disabled"}, status_code=404)
            manager = request.app.state.arc3_persistent_manager
            session = manager.sessions.get(session_id)
            if session is None or session.state != "paused":
                return JSONResponse({"error": "Diagnostic requires a paused owned session"}, status_code=409)
            return await manager.backend.engine.engine_core.call_utility_async(
                "arc3_one_lookahead_diagnostic", session.engine_request_id)

        @router.get("/health")
        async def health(request: Request):
            manager = request.app.state.arc3_persistent_manager
            return {"state": "ready", "mode": "persistent-history-eviction-mtp0",
                    "history_eviction_policies": ["off", "oldest_interactions", "qsa_frequency"],
                    "max_context": manager.max_context, "max_sessions": manager.max_sessions,
                    "prefix_caching": False, "engine_patch": request.app.state.arc3_persistent_receipt,
                    "sessions": {sid: {"state": s.state, "revision": s.revision,
                                       "resident_tokens": s.residency.resident_tokens,
                                       "logical_tokens": len(s.token_ids),
                                       "eviction_revision": s.eviction_revision}
                                 for sid, s in manager.sessions.items()}}

        app.include_router(router)

        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def persistent_lifespan(application):
            async with original_lifespan(application) as lifespan_state:
                try:
                    yield lifespan_state
                finally:
                    if hasattr(application.state, "arc3_persistent_manager"):
                        await application.state.arc3_persistent_manager.close_all()
        app.router.lifespan_context = persistent_lifespan

    async def init_state(self, engine_client, state, args):
        from engine_patch import install, validate_config
        receipt = install()
        validation = validate_config(engine_client.vllm_config)
        state.arc3_persistent_receipt = {"installation": receipt, "config": validation}
        # The server is local only, matching the existing private experiment API.
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            raise RuntimeError("Persistent experiment requires a loopback-bound server")
        capacity = min(int(os.environ.get("ARC3_PERSISTENT_MAX_SESSIONS", "4")),
                       engine_client.vllm_config.scheduler_config.max_num_seqs)
        if capacity < 1:
            raise RuntimeError("Persistent session admission must be positive")
        state.arc3_persistent_manager = SessionManager(
            VllmBackend(engine_client, state.openai_serving_chat),
            max_context=engine_client.model_config.max_model_len, max_sessions=capacity,
            max_logical_tokens=engine_client.model_config.hf_text_config.max_position_embeddings,
            audit_path=os.environ.get("ARC3_PERSISTENT_SERVICE_AUDIT"),
            turn_timeout=float(os.environ.get("ARC3_PERSISTENT_TURN_TIMEOUT", "600")))


def create_plugin():
    return PersistentEndpoint()
