"""Private event-only timings via per-instance subclasses and local bindings.

No production module/class/global importer is patched. No telemetry is passed
to model messages, tool outputs, workspace logs or PNGs. Outer timings include
nested work; HTTP transport includes generation and must not be called friction.
"""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
import functools
import json
import os
from pathlib import Path
import threading
import time
from types import FunctionType

VERSION = 'arc3-gameplay-harness-profile-v1'
_CONTEXT = contextvars.ContextVar('arc3_harness_profile_context', default=None)
_RECORDER = None
_CLASSES = None


class Recorder:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.fd = None
        self.attempted = self.written = self.dropped = self.counter = 0
        self.first_error = None

    def emit(self, event, **fields):
        record = dict(version=VERSION, event=event, pid=os.getpid(),
                      thread_id=threading.get_ident(), **fields)
        with self.lock:
            self.attempted += 1
            try:
                # The existing runner owns creating its fresh output directory.
                if self.fd is None:
                    self.fd = os.open(self.path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
                data = (json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n').encode()
                while data:
                    n = os.write(self.fd, data)
                    if n <= 0:
                        raise OSError('Profile write made no progress')
                    data = data[n:]
                self.written += 1
            except Exception as exc:
                self.dropped += 1
                self.first_error = self.first_error or type(exc).__name__

    def next_id(self):
        with self.lock:
            self.counter += 1
            return self.counter

    def status(self):
        with self.lock:
            return dict(version=VERSION, journal=str(self.path), attempted=self.attempted,
                        written=self.written, dropped=self.dropped, first_error=self.first_error,
                        valid=self.written > 0 and self.dropped == 0)

    def close(self):
        with self.lock:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


def activate(path):
    global _RECORDER
    if _RECORDER is not None:
        raise RuntimeError('A harness profile is already active in this process')
    _RECORDER = Recorder(path)
    return _RECORDER


def deactivate():
    global _RECORDER
    if _RECORDER is not None:
        _RECORDER.close()
    _RECORDER = None


def emit(event, **fields):
    if _RECORDER is not None:
        context = {k:v for k,v in (_CONTEXT.get() or {}).items()
                   if k not in ('agent', 'receipt')}
        _RECORDER.emit(event, **dict(context, **fields))


@contextmanager
def context(**fields):
    value = dict(_CONTEXT.get() or {}, **fields)
    token = _CONTEXT.set(value)
    try:
        yield value
    finally:
        _CONTEXT.reset(token)


@contextmanager
def span(phase, **fields):
    if _RECORDER is None:
        yield {}
        return
    parent = (_CONTEXT.get() or {}).get('span_id')
    with context(span_id=_RECORDER.next_id(), **fields) as ctx:
        wall_start, mono_start = time.time(), time.monotonic()
        extra = {}; error = None
        try:
            yield extra
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            mono_end, wall_end = time.monotonic(), time.time()
            emit('span', phase=phase, parent_span_id=parent, wall_start=wall_start,
                 wall_end=wall_end, mono_start=mono_start, mono_end=mono_end,
                 seconds=mono_end-mono_start, error_type=error, **extra)


def timed(function, phase):
    @functools.wraps(function)
    def call(*args, **kwargs):
        with span(phase):
            return function(*args, **kwargs)
    return call


def rebind(function, **replacements):
    """Preserve original closure (__class__/super), defaults and descriptors."""
    unknown = set(replacements)-set(function.__globals__)
    if unknown:
        raise ValueError('Unexpected binding names: '+','.join(sorted(unknown)))
    bound = FunctionType(function.__code__, dict(function.__globals__, **replacements),
                         function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = function.__annotations__
    return functools.update_wrapper(bound, function)


def _agent_fields(agent):
    return dict(agent=agent, game_id=getattr(agent, 'game_id', None),
                session_id=getattr(agent, '_persistent_session_id', None),
                operation_id=getattr(agent, '_profile_last_operation_id', None))


def known_received(payload):
    if not isinstance(payload, dict):
        return None
    usage = payload.get('usage') or {}
    persistent = payload.get('persistent') or {}
    if not isinstance(usage, dict) or not isinstance(persistent, dict):
        return None
    ids = persistent.get('token_ids')
    n = usage.get('completion_tokens')
    if (type(n) is int and n >= 0 and isinstance(ids, list) and len(ids) == n
            and all(type(i) is int and i >= 0 for i in ids)):
        return n
    return None


def build_classes():
    """Called only after the runner verifies the reviewed harness source pins."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    import requests
    import append_only_history as ah
    import managed_suffix_history as mh
    import persistent_harness_adapter as ph
    import prolong_duck as pd
    import prolong_loop as pl
    import prolong_sandbox as ps
    import prolong_snapshots as snaps
    import prolong_images as images
    import prolong_memory as memory
    import prolong_output as output
    import prolong_program_index as programs
    import gameplay_duck as gd
    import scheduled_duck as scheduled

    class ProfiledResponse(requests.Response):
        def json(self, **kwargs):
            with span('http_response_parse'):
                result = super().json(**kwargs)
            state = _CONTEXT.get()
            if state is not None and 'receipt' in state:
                state['receipt']['received_completion_tokens'] = known_received(result)
            return result

    class ProfiledRequestsSession(requests.Session):
        prepare_request = timed(requests.Session.prepare_request, 'http_serialization')

        def send(self, request, **kwargs):
            with span('http_transport', scope='network_wait_including_entire_generation'):
                response = super().send(request, **kwargs)
            # Requests can re-enter send while following redirects. Its final
            # response may already be profiled; custom response subclasses keep
            # their behavior instead of becoming a profiler-induced exception.
            if type(response) is requests.Response:
                response.__class__ = ProfiledResponse
            return response

    class ProfiledTransport(ph.HTTPPersistentTransport):
        def __init__(self, *args, **kwargs):
            if kwargs.get('http') is None:
                kwargs['http'] = ProfiledRequestsSession()
            super().__init__(*args, **kwargs)

    png = timed(images.png_bytes, 'board_png')
    image_part = rebind(images.image_part, png_bytes=png)
    save_grid = rebind(images.save_grid, png_bytes=png)
    class ProfiledBoardOutput(images.BoardOutput):
        attach = rebind(images.BoardOutput.attach, image_part=image_part, save_grid=save_grid)

    class ProfiledProgramIndex(programs.ProgramIndex):
        text = timed(programs.ProgramIndex.text, 'program_index')
        observe = timed(programs.ProgramIndex.observe, 'program_index_update')

    class ProfiledWorkspace(memory.GameWorkspace):
        __init__ = rebind(memory.GameWorkspace.__init__, ProgramIndex=ProfiledProgramIndex)
        record = timed(memory.GameWorkspace.record, 'workspace_journal')
        frame = timed(memory.GameWorkspace.frame, 'native_frame_asset')
        archive_call = timed(memory.GameWorkspace.archive_call, 'python_preparation')

    snapshot_image = rebind(snaps.snapshot_image, image_part=image_part)
    snapshot_parts = rebind(snaps.snapshot_parts, snapshot_image=snapshot_image)
    snapshot_refresh = timed(rebind(snaps.refresh_snapshot_suffix, snapshot_parts=snapshot_parts), 'snapshot')
    decorate = timed(rebind(ah.decorate_observation,
        refresh_program_suffix=timed(ah.refresh_program_suffix, 'program_suffix'),
        refresh_snapshot_suffix=snapshot_refresh), 'managed_suffix')

    class ProfiledAppendState(ah.AppendOnlyMessageState):
        stage = timed(rebind(ah.AppendOnlyMessageState.stage, decorate_observation=decorate), 'external_history_stage')

    class ProfiledManagedState(mh.ManagedSuffixMessageState):
        __init__ = rebind(mh.ManagedSuffixMessageState.__init__, AppendOnlyMessageState=ProfiledAppendState)

    initialize = rebind(ph._initialize_persistent_state, ManagedSuffixMessageState=ProfiledManagedState,
                        HTTPPersistentTransport=ProfiledTransport)
    lazy_completion = rebind(ph.PersistentProlongAgent._chat_completion, _initialize_persistent_state=initialize)
    def sandbox_binding(function, **kw):
        result = ps.with_globals(function, **kw)
        return timed(result, 'python_subprocess') if function is ps.original.run_sandboxed_python else result
    run_python = rebind(ps.run_python, with_globals=sandbox_binding)
    run_tool = timed(rebind(ps.run_tool, run_python=timed(run_python, 'python_lifecycle')), 'python_dispatch')

    def normalized(*args, **kwargs):
        with span('response_normalization'):
            result = pl.normalize_response(*args, **kwargs)
        agent = (_CONTEXT.get() or {}).get('agent')
        if agent is not None:
            agent._profile_tool_calls = list(result.calls)
        return result
    analyze = rebind(pl.ContinuationToolLoop.analyze, normalize_response=normalized,
        _write_prompt_log_snapshot=timed(pl._write_prompt_log_snapshot, 'prompt_snapshot'),
        _append_request_snapshot=timed(pl._append_request_snapshot, 'request_snapshot'),
        _append_transcript_section=timed(pl._append_transcript_section, 'transcript_journal'))

    class ProfiledAgent(ph.PersistentProlongAgent):
        _chat_completion = lazy_completion
        _ensure_session = timed(rebind(pd.ProlongAgent._ensure_session, GameWorkspace=ProfiledWorkspace), 'session_prepare')
        _stage_persistent_messages = timed(ph.PersistentProlongAgent._stage_persistent_messages, 'persistent_stage')
        _persistent_result = timed(ph.PersistentProlongAgent._persistent_result, 'persistent_result_parse')
        _request_payload = timed(pd.ProlongAgent._request_payload, 'request_payload_display_view')
        _record_prepared_request = timed(gd.GameplayAgent._record_prepared_request, 'prepared_request_snapshot')
        _append_raw_evidence = timed(pd.ProlongAgent._append_raw_evidence, 'raw_journal')
        _build_user_prompt = timed(pd.ProlongAgent._build_user_prompt, 'build_user_prompt')
        _build_user_message = timed(rebind(pd.ProlongAgent._build_user_message,
            image_part=timed(image_part, 'current_board_png')), 'build_user_message')
        _run_python_tool = timed(rebind(pd.ProlongAgent._run_python_tool, run_tool=run_tool), 'python_tool')
        _render_tool_payload = timed(rebind(pd.ProlongAgent._render_tool_payload,
            BoardOutput=ProfiledBoardOutput, prepare_output=timed(output.prepare_output, 'output_filter')), 'output_render')

        def analyze(self, state_path, action_num, *args, **kwargs):
            with context(**_agent_fields(self), action_num=action_num):
                with span('observation'):
                    return analyze(self, state_path, action_num, *args, **kwargs)

        def _allocated_completion(self, messages, tools, admission):
            operation = f"{self._persistent_session_id}-r{admission['request']}"
            self._profile_last_operation_id = operation
            current = _CONTEXT.get()
            if current is not None:
                current.update(session_id=self._persistent_session_id, operation_id=operation)
            with context(**_agent_fields(self), receipt={}) as ctx:
                emit('operation_start', wall_start=time.time(), mono_start=time.monotonic())
                try:
                    with span('completion'):
                        result = super()._allocated_completion(messages, tools, admission)
                except BaseException as exc:
                    emit('completion_result', accepted=False,
                         completion_tokens=ctx['receipt'].get('received_completion_tokens'), error_type=type(exc).__name__,
                         deadline_observed=time.time() >= self.allocation.deadline,
                         wall_end=time.time(), mono_end=time.monotonic())
                    raise
                emit('completion_result', accepted=True, completion_tokens=result.usage['completion_tokens'],
                     prompt_tokens=result.usage['prompt_tokens'], finish_reason=result.finish_reason,
                     resident_tokens=result.persistent['resident_tokens'],
                     logical_tokens=result.persistent.get('logical_tokens'),
                     appended_prompt_tokens=result.persistent['appended_prompt_tokens'],
                     error_type=None, wall_end=time.time(), mono_end=time.monotonic())
                return result

        def _dispatch_tool(self, state_path, name, arguments):
            model_id = None
            queue = getattr(self, '_profile_tool_calls', [])
            for i, call in enumerate(queue):
                fn = call.get('function', {})
                parsed, error = pl.valid_arguments(fn.get('arguments'))
                if not error and fn.get('name') == name and parsed == arguments:
                    model_id = call.get('id'); del queue[:i+1]; break
            self._profile_tool_serial = getattr(self, '_profile_tool_serial', 0)+1
            fields = _agent_fields(self)
            fields.update(tool_id=model_id or f"{self._profile_last_operation_id}-tool-{self._profile_tool_serial}",
                          model_tool_call_id=model_id, tool_name=name)
            with context(**fields), span('tool_dispatch'):
                return super()._dispatch_tool(state_path, name, arguments)

    class ProfiledSession(pd.ProlongSession):
        write_runtime_state = timed(pd.ProlongSession.write_runtime_state, 'runtime_write')
        step_env = timed(pd.ProlongSession.step_env, 'action_batch')
        _execute_action = timed(pd.ProlongSession._execute_action, 'action')

    play = rebind(ph.PersistentProlongSolver._play_one, AllocatedAgent=ProfiledAgent,
                  AllocatedSession=ProfiledSession)
    class ProfiledSolver(ph.PersistentProlongSolver):
        def _play_one(self, game, index, pass_index, local_server=None):
            with context(game_id=game.game_run.game_id), span('game'):
                return play(self, game, index, pass_index, local_server)

    _CLASSES = dict(agent=ProfiledAgent, solver=ProfiledSolver, transport=ProfiledTransport,
                    workspace=ProfiledWorkspace, session=ProfiledSession, managed_state=ProfiledManagedState,
                    append_state=ProfiledAppendState, response=ProfiledResponse)
    return _CLASSES


def configure_solver(solver, **kwargs):
    import persistent_harness_adapter as ph
    ph.configure_persistent_solver(solver, **kwargs)
    solver.__class__ = build_classes()['solver']
    return solver
