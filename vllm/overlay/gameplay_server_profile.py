"""Opt-in API timing observer. No token payloads, GPU reads, or synchronization.

Intervals are nested and nonadditive. JSON writes occur after measured spans;
observer failures are counted and never replace serving results/exceptions.
"""
from __future__ import annotations
import ast
import asyncio
from contextlib import contextmanager, asynccontextmanager
import contextvars
import copy
from dataclasses import dataclass, field
import functools
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import threading
import time

VERSION='arc3-gameplay-server-profile-v1'
FLAG='ARC3_GAMEPLAY_PROFILE'
PATH='ARC3_GAMEPLAY_PROFILE_PATH'
SOURCE_HASHES={
 'persistent_endpoint.py':'ceead0163e99ffcdf3f376affee74a597a343ef06781a737e0b09bf9a56b0001',
 'session_service.py':'7ce0eaef072e45354cfae8476719313a401387ea379b3fe9c3ca9c1ab40193e8',
 'protocol.py':'7a11929c16dc682ef92a907ba8aaaeda0c4a9061f04b31eb799bebbf18660a23',
 'renderers/hf.py':'b0e83d95fc0aca6e248e28aa727d795bd74da681296634b15a0a0cf68a9feb48',
 'renderers/base.py':'a358921359b241df2e5c08b547897b874a0eb5c73bc9051293bab800511aeedb'}
_CURRENT=contextvars.ContextVar('arc3_gameplay_profile_current',default=None)
_PARENT=contextvars.ContextVar('arc3_gameplay_profile_parent',default=None)
_RENDER_KIND=contextvars.ContextVar('arc3_gameplay_render_kind',default='main')
_RENDER_FLOW=contextvars.ContextVar('arc3_gameplay_render_flow',default=None)
_IDS=itertools.count(1)
_LOCK=threading.Lock()
_STATS={'attempted':0,'written':0,'dropped':0,'metadata_errors':0,'unmatched_output_events':0}
_ACTIVE={}
_INSTALLED=False
_AVAILABILITY={}
_LOOP_TASKS=set()


def enabled():
    value=os.environ.get(FLAG,'0')
    if value not in ('0','1'):raise RuntimeError(FLAG+' must be 0 or 1')
    return value=='1'


def _id():return f'{os.getpid()}-{next(_IDS)}'


def _safe(fn,default=None):
    try:return fn()
    except Exception:
        _STATS['metadata_errors']+=1
        return default


def _write(record):
    with _LOCK:
        _STATS['attempted']+=1
        try:
            blob=(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode()
            fd=os.open(f'{os.environ[PATH]}.{os.getpid()}.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
            try:
                remaining=memoryview(blob)
                while remaining:
                    count=os.write(fd,remaining)
                    if count<=0:raise OSError('No diagnostic write progress')
                    remaining=remaining[count:]
            finally:os.close(fd)
            _STATS['written']+=1
        except Exception as exc:
            _STATS['dropped']+=1
            _STATS.setdefault('first_error_type',type(exc).__name__)


def status():
    with _LOCK:
        return dict(version=VERSION,installed=_INSTALLED,pid=os.getpid(),
            journal=f'{os.environ.get(PATH,"")}.{os.getpid()}.jsonl',
            active_output_turns=len(_ACTIVE),loop_lag_tasks=sum(not t.done() for t in _LOOP_TASKS),
            availability=dict(_AVAILABILITY),**_STATS)


@dataclass
class Trace:
    info: dict=field(default_factory=dict)
    trace_id: str=field(default_factory=_id)


@dataclass
class Span:
    event: str
    trace: Trace
    parent_span_id: str|None
    details: dict=field(default_factory=dict)
    span_id: str=field(default_factory=_id)
    wall_start: float=field(default_factory=time.time)
    mono_start: float=field(default_factory=time.monotonic)
    thread_id: int=field(default_factory=threading.get_ident)

    def finish(self,error=None,*,instant=False):
        end=self.mono_start if instant else time.monotonic()
        return dict(version=VERSION,event=self.event,pid=os.getpid(),thread_id=self.thread_id,
            trace_id=self.trace.trace_id,span_id=self.span_id,parent_span_id=self.parent_span_id,
            wall_start=self.wall_start,wall_end=self.wall_start if instant else time.time(),
            mono_start=self.mono_start,mono_end=end,seconds=end-self.mono_start,
            session_id=self.trace.info.get('session_id'),operation_id=self.trace.info.get('operation_id'),
            turn=self.trace.info.get('turn'),engine_request_id=self.trace.info.get('engine_request_id'),
            error_type=type(error).__name__ if error is not None else None,
            details=self.details)


@contextmanager
def span(event,details=None,*,trace=None):
    trace=trace or _CURRENT.get()
    if trace is None:
        yield None
        return
    item=Span(event,trace,_PARENT.get(),details or {})
    token=_PARENT.set(item.span_id);error=None
    try:yield item
    except BaseException as exc:
        error=exc;raise
    finally:
        _PARENT.reset(token)
        _safe(lambda:_write(item.finish(error)))


def instant(event,*,trace=None,parent=None,**details):
    trace=trace or _CURRENT.get()
    if trace is None:return
    item=Span(event,trace,parent if parent is not None else _PARENT.get(),details)
    _safe(lambda:_write(item.finish(instant=True)))


def _bind(session=None,payload=None,sid=None):
    trace=_CURRENT.get()
    if trace is None:return
    if isinstance(payload,dict):
        trace.info.update(session_id=sid or payload.get('session_id'),operation_id=payload.get('request_id'))
    if session is not None:
        trace.info.update(session_id=session.session_id,operation_id=session.current_operation,
            turn=session.revision,engine_request_id=session.engine_request_id)


def _async_span(method,name,details=None):
    @functools.wraps(method)
    async def wrapped(*args,**kwargs):
        if _CURRENT.get() is None:return await method(*args,**kwargs)
        info=_safe(lambda:details(*args,**kwargs),{}) if details else {}
        with span(name,info):return await method(*args,**kwargs)
    return wrapped


def _sync_span(method,name):
    @functools.wraps(method)
    def wrapped(*args,**kwargs):
        with span(name):return method(*args,**kwargs)
    return wrapped


class ProfileMiddleware:
    """Pure ASGI pass-through: does not consume, buffer, or rewrite request bodies."""
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        path=scope.get('path','')
        if scope['type']!='http' or scope.get('method')!='POST' or not (
                path=='/arc3/persistent/start' or path.startswith('/arc3/persistent/') and path.endswith('/append')):
            return await self.app(scope,receive,send)
        trace=Trace();token=_CURRENT.set(trace);parent=_PARENT.set(None)
        details={'route':path,'request_body_bytes':0,'response_body_bytes':0,'response_status':None}
        async def observed_receive():
            message=await receive()
            if message.get('type')=='http.request':details['request_body_bytes']+=len(message.get('body',b''))
            return message
        async def observed_send(message):
            if message.get('type')=='http.response.start':details['response_status']=message.get('status')
            elif message.get('type')=='http.response.body':details['response_body_bytes']+=len(message.get('body',b''))
            return await send(message)
        try:
            with span('server_request',details):return await self.app(scope,observed_receive,observed_send)
        finally:
            _PARENT.reset(parent);_CURRENT.reset(token)


async def loop_lag(*,interval=.1,aggregate=1.,clock=time.monotonic,wall=time.time,sleep=asyncio.sleep):
    """API event-loop wakeup lateness only; one aggregate, never per-tick I/O.

    Reset the next deadline after each wakeup rather than doing catch-up work.
    Queue/OS/GIL stalls all contribute; this does not measure GPU activity.
    The final partial (<aggregate) window is intentionally omitted at shutdown.
    """
    started=clock();wall_started=wall();expected=started+interval
    ticks=0;total=maximum=0.
    while True:
        await sleep(interval)
        now=clock();lag=max(0.,now-expected);expected=now+interval
        ticks+=1;total+=lag;maximum=max(maximum,lag)
        if now-started>=aggregate:
            end_wall=wall()
            details=dict(ticks=ticks,interval_seconds=interval,lag_sum_seconds=total,
                         lag_max_seconds=maximum,lag_mean_seconds=total/ticks)
            item=Span('api_loop_lag',Trace(),None,details,wall_start=wall_started,mono_start=started)
            row=item.finish(instant=True)
            row.update(wall_end=end_wall,mono_end=now,seconds=now-started)
            _safe(lambda:_write(row))
            started=now;wall_started=end_wall;ticks=0;total=maximum=0.


def install_loop_lag(app):
    original=app.router.lifespan_context
    @asynccontextmanager
    async def lifespan(application):
        async with original(application) as state:
            task=asyncio.create_task(loop_lag(),name='arc3-gameplay-api-loop-lag')
            _LOOP_TASKS.add(task);_AVAILABILITY['api_loop_lag']=True
            def done(t):
                _LOOP_TASKS.discard(t)
                if not t.cancelled() and t.exception() is not None:
                    _STATS['metadata_errors']+=1
            task.add_done_callback(done)
            try:yield state
            finally:
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
    app.router.lifespan_context=lifespan


def _collector_observe(backend,session,event,count=0):
    """Invoked only at first nonempty output/final boundary; no per-token JSON."""
    def record():
        active=_ACTIVE.get((id(backend),session.engine_request_id))
        if active is None:
            _STATS['unmatched_output_events']+=1;return
        trace,parent,seen=active
        if event in seen:
            _STATS['unmatched_output_events']+=1;return
        seen.add(event)
        instant(event,trace=trace,parent=parent,output_tokens=count)
    _safe(record)


def _queue_submit(backend,session,turn):
    _safe(lambda:instant('backend_submit',input_tokens=len(turn.token_ids),boundary='input_queue_put_nowait'))


def transform_turn(source):
    """Insert three observer calls; all collector/Future/cancellation code stays native."""
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_HASHES['persistent_endpoint.py']:
        raise RuntimeError('Endpoint source changed; review profiler collector hook')
    tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VllmBackend')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='turn'))
    counts={'first':0,'complete':0,'submit':0}
    class Observe(ast.NodeTransformer):
        def visit_Expr(self,node):
            call=node.value
            if not isinstance(call,ast.Call):return node
            target=ast.unparse(call.func)
            if target=='ids.extend' and ast.unparse(call.args[0])=='completion.token_ids':
                counts['first']+=1
                extra=ast.parse("if not ids and len(completion.token_ids):\n    _arc3_gameplay_observe(self,session,'backend_first_output',len(completion.token_ids))").body[0]
                return [extra,node]
            if target=='future.set_result':
                counts['complete']+=1
                return [ast.parse("_arc3_gameplay_observe(self,session,'backend_collection_complete',len(ids))").body[0],node]
            if target=='session.input_queue.put_nowait':
                counts['submit']+=1
                return [node,ast.parse('_arc3_gameplay_submit(self,session,turn)').body[0]]
            return node
    fn=Observe().visit(fn)
    if counts!={'first':1,'complete':1,'submit':1}:raise RuntimeError('Unexpected native collector observer anchors')
    return ast.fix_missing_locations(fn)


def build_turn(source,namespace):
    fn=transform_turn(source);scope=dict(namespace)
    scope.update(_arc3_gameplay_observe=_collector_observe,_arc3_gameplay_submit=_queue_submit)
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),'<arc3-gameplay-collector-profile>','exec'),scope)
    return scope['turn']


def _renderer_hooks(renderer):
    if getattr(renderer,'_arc3_gameplay_profile',False):return
    names={'_apply_chat_template_async':'renderer_template','_tokenize_prompt_async':'renderer_tokenize',
           '_process_multimodal_async':'renderer_multimodal','render_messages_async':'renderer_messages'}
    for name,event in names.items():
        method=getattr(renderer,name,None)
        _AVAILABILITY[event]=callable(method)
        if callable(method):setattr(renderer,name,_async_span(method,event))
    renderer._arc3_gameplay_profile=True


def install_targets(endpoint,service,protocol,*,native_hf=None):
    """Dependency-injected installation for CPU tests; production verifies file hashes first."""
    undo=[]
    def put(owner,name,value):
        undo.append((owner,name,getattr(owner,name)));setattr(owner,name,value)
    # Build the pinned nested collector before changing any methods.
    collector=build_turn(Path(endpoint.__file__).read_text(),endpoint.__dict__)
    backend=endpoint.VllmBackend;manager=service.SessionManager
    original_init=backend.__init__
    @functools.wraps(original_init)
    def backend_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        _safe(lambda:_renderer_hooks(self.serving_chat.renderer))
    put(backend,'__init__',backend_init)
    for method in ('prepare','preflight','response'):
        put(backend,method,_async_span(getattr(backend,method),{'response':'response_parse'}.get(method,method)))
    put(backend,'evict',_async_span(backend.evict,'backend_evict'))
    original_turn=backend.turn
    @functools.wraps(original_turn)
    async def backend_turn(self,session,turn,*,initial):
        trace=_CURRENT.get()
        if trace is None:return await collector(self,session,turn,initial=initial)
        with span('backend_turn',_safe(lambda:{'initial':initial,'input_tokens':len(turn.token_ids)},{})):
            key=(id(self),session.engine_request_id);active=(trace,_PARENT.get(),set())
            if key in _ACTIVE:_STATS['unmatched_output_events']+=1
            _ACTIVE[key]=active
            try:return await collector(self,session,turn,initial=initial)
            finally:
                if _ACTIVE.get(key) is active:_ACTIVE.pop(key,None)
    put(backend,'turn',backend_turn)
    for name in ('start','append'):
        original=getattr(manager,name)
        def make_entry(method,kind):
            @functools.wraps(method)
            async def entry(self,*args,**kwargs):
                if kind=='start':payload=args[0] if args else kwargs.get('payload');sid=None
                else:
                    sid=args[0] if args else kwargs.get('sid');payload=args[1] if len(args)>1 else kwargs.get('payload')
                _safe(lambda:_bind(payload=payload,sid=sid))
                return await method(self,*args,**kwargs)
            return entry
        put(manager,name,make_entry(original,name))
    original_run=manager._run_turn
    @functools.wraps(original_run)
    async def run_turn(self,session,payload,*,initial):
        old=_CURRENT.get();trace=Trace(dict(old.info) if old else {},old.trace_id if old else _id())
        token=_CURRENT.set(trace)
        _safe(lambda:_bind(session=session))
        try:
            with span('service_turn',_safe(lambda:{'initial':initial,'continuation':bool(payload.get('continuation'))},{})):
                return await original_run(self,session,payload,initial=initial)
        finally:_CURRENT.reset(token)
    put(manager,'_run_turn',run_turn)
    for name in ('render_initial','render_delta'):
        original=getattr(protocol,name)
        def make_render(method,kind):
            @functools.wraps(method)
            async def render(*args,**kwargs):
                token=_RENDER_FLOW.set({'initial':kind=='render_initial','main_calls':0})
                try:
                    with span(kind):return await method(*args,**kwargs)
                finally:_RENDER_FLOW.reset(token)
            return render
        put(protocol,name,make_render(original,name))
    for name,kind in (('_suffix_records','base'),('_observation_starts','preceding')):
        original=getattr(protocol,name)
        def make_kind(method,value):
            @functools.wraps(method)
            async def wrapped(*args,**kwargs):
                token=_RENDER_KIND.set(value)
                try:return await method(*args,**kwargs)
                finally:_RENDER_KIND.reset(token)
            return wrapped
        put(protocol,name,make_kind(original,kind))
    original_render=protocol._render
    @functools.wraps(original_render)
    async def render(serving_chat,payload,request_factory):
        kind=_RENDER_KIND.get();flow=_RENDER_FLOW.get()
        if kind=='main' and flow is not None:
            kind='anchor' if flow['initial'] and flow['main_calls'] else 'main';flow['main_calls']+=1
        def metadata():
            messages=payload.get('messages',[])
            return {'render_kind':kind,'messages':len(messages),'images':sum(
                part.get('type')=='image_url' for m in messages if isinstance(m.get('content'),list) for part in m['content'])}
        with span('render',_safe(metadata,{'render_kind':kind})):
            return await original_render(serving_chat,payload,request_factory)
    put(protocol,'_render',render)
    put(protocol,'parse_output',_sync_span(protocol.parse_output,'protocol_parse'))
    if native_hf is not None:
        put(native_hf,'parse_chat_messages_async',_async_span(native_hf.parse_chat_messages_async,'renderer_parse_messages'))
        _AVAILABILITY['renderer_parse_messages']=True
    original_attach=endpoint.PersistentEndpoint.attach_router
    @functools.wraps(original_attach)
    def attach(self,app):
        result=original_attach(self,app)
        app.add_middleware(ProfileMiddleware)
        install_loop_lag(app)
        @app.get('/arc3/persistent/profile-status')
        async def diagnostics():return status()
        return result
    put(endpoint.PersistentEndpoint,'attach_router',attach)
    return undo


def install():
    global _INSTALLED
    if not enabled():return {'version':VERSION,'installed':False}
    if _INSTALLED:return status()
    path=Path(os.environ.get(PATH,''))
    if not path.is_absolute() or not path.parent.is_dir():raise RuntimeError('Absolute existing-parent profile journal prefix required')
    import persistent_endpoint,session_service,protocol
    from vllm.renderers import hf,base
    modules={'persistent_endpoint.py':persistent_endpoint,'session_service.py':session_service,
             'protocol.py':protocol,'renderers/hf.py':hf,'renderers/base.py':base}
    for name,module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE_HASHES[name]:
            raise RuntimeError('Profiling source changed: '+name)
    install_targets(persistent_endpoint,session_service,protocol,native_hf=hf)
    _INSTALLED=True
    instant('profiler_installed',trace=Trace(),source_hashes=dict(SOURCE_HASHES))
    return status()
