"""CPU observer parity using the actual pinned collector, manager and protocol.

Fake model outputs exercise streaming/Future ownership only; no GPU timing claim.
"""
from __future__ import annotations
import ast
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
try:
    from fastapi import FastAPI
except ModuleNotFoundError:
    FastAPI=None

import gameplay_server_profile as p
if FastAPI is None:
    # CPU collector/protocol tests need only the route-annotation name. The
    # actual ASGI framework parity test below explicitly skips, never fakes it.
    with patch.dict(sys.modules,{'fastapi':NS(Request=object)}):
        import persistent_endpoint as endpoint
    sys.modules['persistent_endpoint']=endpoint
else:
    import persistent_endpoint as endpoint
import persistent_plugin
import protocol
import session_service as service

HERE=Path(__file__).resolve().parent
STACK=Path(os.environ.get('ARC3_STACK_PATH',next((x for x in HERE.parents if x.name=='inference_stack'),HERE)))


class FakeEngine:
    def __init__(self):
        tokenizer=NS(decode=lambda ids,**kw:','.join(map(str,ids)))
        self.renderer=NS(get_tokenizer=lambda:tokenizer)
        self.inputs=[];self.aborted=[];self.gate=None;self.error=None
        self.input_processor=NS(process_inputs=lambda **kw:self.inputs.append(kw))
    async def get_supported_tasks(self):return ('generate',)
    async def abort(self,rid):self.aborted.append(rid)
    async def generate(self,stream,params,rid):
        async for _item in stream:
            if self.gate:await self.gate.wait()
            if self.error:raise self.error
            for ids,finish in (([],None),([101,102],None),([103],'length')):
                yield NS(outputs=[NS(token_ids=ids,logprobs=None,finish_reason=finish,stop_reason=None)])


class ProfileTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ,{p.FLAG:'1',p.PATH:str(Path(self.temp.name)/'profile')})
        self.env.start()
        self.modules=patch.dict(sys.modules,{'vllm.engine.protocol':NS(StreamingInput=lambda **kw:NS(**kw))})
        self.modules.start()
        p._ACTIVE.clear();p._AVAILABILITY.clear();p._STATS.clear()
        p._STATS.update(attempted=0,written=0,dropped=0,metadata_errors=0,unmatched_output_events=0)
        self.undo=p.install_targets(endpoint,service,protocol)
        self.sampling=patch.object(endpoint.VllmBackend,'sampling_params',lambda _,values:NS(**values))
        self.sampling.start()
        self.managers=[]
    async def asyncTearDown(self):
        for manager in self.managers:await manager.close_all()
        self.sampling.stop()
        for owner,name,original in reversed(self.undo):setattr(owner,name,original)
        self.modules.stop();self.env.stop();self.temp.cleanup()
        self.assertIsNone(p._CURRENT.get());self.assertIsNone(p._PARENT.get())
        self.assertFalse(p._ACTIVE)
    def records(self,event=None):
        file=Path(f'{os.environ[p.PATH]}.{os.getpid()}.jsonl')
        values=[json.loads(x) for x in file.read_text().splitlines()] if file.exists() else []
        return values if event is None else [x for x in values if x['event']==event]
    def make_manager(self):
        engine=FakeEngine();backend=endpoint.VllmBackend(engine,NS(renderer=NS()))
        manager=service.SessionManager(backend,max_context=128,max_sessions=8,turn_timeout=1)
        self.managers.append(manager)
        return manager,engine
    def body(self,sid='game',op='initial',**kw):
        return dict(session_id=sid,request_id=op,raw={'token_ids':[1,2]},sampling={'max_tokens':3},**kw)
    async def traced(self,call):
        token=p._CURRENT.set(p.Trace())
        try:return await call
        finally:p._CURRENT.reset(token)
    def check_intervals(self,records):
        for row in records:
            self.assertEqual(row['pid'],os.getpid())
            self.assertGreaterEqual(row['seconds'],0)
            self.assertAlmostEqual(row['seconds'],row['mono_end']-row['mono_start'])
            self.assertNotIn('token_ids',row['details'])

    async def test_actual_collector_two_turns_exact_output_and_first_nonempty(self):
        manager,engine=self.make_manager()
        a=await self.traced(manager.start(payload=self.body()))
        b=await self.traced(manager.append(sid='game',payload={
            'request_id':'append','expected_revision':1,'raw':{'token_ids':[4]},'sampling':{'max_tokens':3}}))
        self.assertEqual(a['choices'][0]['token_ids'],[101,102,103])
        self.assertEqual(b['choices'][0]['token_ids'],[101,102,103])
        self.assertEqual(manager.get('game').token_ids,[1,2,101,102,103,4,101,102,103])
        first=self.records('backend_first_output');last=self.records('backend_collection_complete')
        self.assertEqual([(x['turn'],x['operation_id'],x['details']['output_tokens']) for x in first],
                         [(0,'initial',2),(1,'append',2)])
        self.assertEqual([x['details']['output_tokens'] for x in last],[3,3])
        self.assertEqual([x['details']['input_tokens'] for x in self.records('backend_submit')],[2,1])
        self.assertEqual(len(self.records('service_turn')),2)
        for item in first+last:
            self.assertEqual(item['engine_request_id'],'arc3-persistent-game')
            self.assertEqual(item['seconds'],0)
        self.assertEqual(p.status()['unmatched_output_events'],0)
        self.check_intervals(self.records())

    async def test_retry_cancellation_keeps_one_native_turn_and_one_first_output(self):
        manager,engine=self.make_manager();engine.gate=asyncio.Event()
        request=self.body()
        task=asyncio.create_task(self.traced(manager.start(request)))
        while not self.records('backend_submit'):await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        engine.gate.set()
        result=await self.traced(manager.start(request))
        self.assertEqual(result['persistent']['revision'],1)
        self.assertEqual(len(self.records('backend_submit')),1)
        self.assertEqual(len(self.records('backend_first_output')),1)
        self.assertEqual(len(self.records('service_turn')),1)

    async def test_concurrent_turns_contexts_do_not_cross(self):
        manager,_=self.make_manager()
        await asyncio.gather(*(self.traced(manager.start(self.body('game'+str(i),'op'+str(i)))) for i in range(3)))
        for event in ('service_turn','backend_first_output','backend_collection_complete','response_parse'):
            rows=self.records(event)
            self.assertEqual({(r['session_id'],r['operation_id']) for r in rows},
                             {('game'+str(i),'op'+str(i)) for i in range(3)})

    async def test_original_collector_exception_identity_and_cleanup(self):
        manager,engine=self.make_manager();failure=RuntimeError('native sentinel');engine.error=failure
        with self.assertRaises(RuntimeError) as caught:
            await self.traced(manager.start(self.body()))
        self.assertIs(caught.exception,failure)
        self.assertEqual(self.records('backend_first_output'),[])
        self.assertEqual(self.records('backend_turn')[0]['error_type'],'RuntimeError')
        self.assertEqual(manager.get('game').state,'error')

    async def test_explicit_close_cancels_original_inflight_future(self):
        manager,engine=self.make_manager();engine.gate=asyncio.Event()
        caller=asyncio.create_task(self.traced(manager.start(self.body())))
        while not self.records('backend_submit'):await asyncio.sleep(0)
        await manager.close('game')
        with self.assertRaises(asyncio.CancelledError):await caller
        self.assertEqual(self.records('backend_turn')[0]['error_type'],'CancelledError')
        self.assertEqual(self.records('backend_collection_complete'),[])

    async def test_source_pinned_ast_has_only_three_observer_insertions(self):
        source=(HERE/'persistent_endpoint.py').read_text()
        changed=p.transform_turn(source)
        tree=ast.parse(source)
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='VllmBackend')
        original=next(x for x in cls.body if isinstance(x,ast.AsyncFunctionDef) and x.name=='turn')
        counts={'first':0,'instant':0}
        class Strip(ast.NodeTransformer):
            def visit_If(self,node):
                if (len(node.body)==1 and isinstance(node.body[0],ast.Expr)
                    and ast.unparse(node.body[0].value.func)=='_arc3_gameplay_observe'):
                    counts['first']+=1;return None
                return self.generic_visit(node)
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in ('_arc3_gameplay_observe','_arc3_gameplay_submit'):
                    counts['instant']+=1;return None
                return node
        stripped=Strip().visit(changed)
        self.assertEqual(counts,{'first':1,'instant':2})
        self.assertEqual(ast.dump(stripped),ast.dump(original))
        with self.assertRaisesRegex(RuntimeError,'source changed'):p.transform_turn(source+'\n')

    @unittest.skipIf(FastAPI is None,'FastAPI absent locally; run this actual framework gate in pinned runtime')
    async def test_asgi_routes_preserve_response_status_bytes_and_invalid_json(self):
        manager,_=self.make_manager();app=FastAPI();app.state.arc3_persistent_manager=manager
        endpoint.create_plugin().attach_router(app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,raise_app_exceptions=False),base_url='http://test') as client:
            response=await client.post('/arc3/persistent/start',json=self.body())
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['choices'][0]['token_ids'],[101,102,103])
            bad=await client.post('/arc3/persistent/game/append',json={
                'request_id':'bad','expected_revision':0,'raw':{'token_ids':[4]}})
            self.assertEqual(bad.status_code,409)
            invalid=await client.post('/arc3/persistent/start',content=b'{broken')
            self.assertEqual(invalid.status_code,500)
            diag=await client.get('/arc3/persistent/profile-status')
            self.assertEqual(diag.status_code,200)
        rows=self.records('server_request')
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['details']['response_status'],200)
        self.assertEqual(rows[0]['details']['response_body_bytes'],len(response.content))
        self.assertEqual(rows[0]['operation_id'],'initial')
        self.assertEqual(rows[1]['details']['response_status'],409)
        self.assertEqual(rows[2]['error_type'],'JSONDecodeError')

    async def test_pure_asgi_cancellation_and_unobserved_routes(self):
        failure=asyncio.CancelledError('sentinel');calls=[]
        async def app(scope,receive,send):calls.append(scope);raise failure
        middleware=p.ProfileMiddleware(app)
        async def noop(*a):pass
        scope={'type':'http','method':'POST','path':'/arc3/persistent/start'}
        with self.assertRaises(asyncio.CancelledError) as caught:await middleware(scope,noop,noop)
        self.assertIs(caught.exception,failure)
        self.assertEqual(self.records('server_request')[0]['error_type'],'CancelledError')
        with self.assertRaises(asyncio.CancelledError):
            await middleware(dict(scope,path='/v1/chat/completions'),noop,noop)
        self.assertEqual(len(self.records('server_request')),1)

    async def test_observer_disk_error_does_not_change_results(self):
        manager,_=self.make_manager()
        with patch.object(p.os,'open',side_effect=OSError('observer only')):
            result=await self.traced(manager.start(self.body()))
        self.assertEqual(result['choices'][0]['token_ids'],[101,102,103])
        stats=p.status();self.assertGreater(stats['dropped'],0)
        self.assertEqual(stats['written'],0);self.assertEqual(stats['attempted'],stats['dropped'])

    async def test_renderer_async_observers_pass_keywords_and_errors(self):
        sentinel=object();seen=[]
        async def operation(*args,**kwargs):seen.append((args,kwargs));return sentinel
        renderer=NS(**{name:operation for name in ('_apply_chat_template_async','_tokenize_prompt_async',
                    '_process_multimodal_async','render_messages_async')})
        p._renderer_hooks(renderer)
        for name in ('_apply_chat_template_async','_tokenize_prompt_async','_process_multimodal_async','render_messages_async'):
            self.assertIs(await self.traced(getattr(renderer,name)(scheduler_output='x')),sentinel)
        self.assertEqual(seen,[((),{'scheduler_output':'x'})]*4)
        self.assertEqual(len(self.records()),4)
        self.assertTrue(all(p.status()['availability'].values()))
        failure=asyncio.CancelledError('renderer cancellation')
        async def broken():raise failure
        with self.assertRaises(asyncio.CancelledError) as caught:
            await self.traced(p._async_span(broken,'renderer_multimodal')())
        self.assertIs(caught.exception,failure)

    async def test_protocol_real_template_main_anchor_base_preceding_labels(self):
        import test_protocol as fixtures
        from tokenizers import Tokenizer
        from managed_spans import content_hash
        tokenizer=Path(os.environ.get('ARC3_TOKENIZER_JSON',STACK/'evidence/optimization-20260909/kernel-checkpoint-v1/model-metadata/tokenizer.json'))
        if not tokenizer.exists() or not (STACK/'prolong_chat_template.jinja').exists():
            self.skipTest('Set ARC3_STACK_PATH/ARC3_TOKENIZER_JSON for real local template fixture')
        with patch.object(fixtures,'STACK',STACK):
            serving=fixtures.FakeServing(Tokenizer.from_file(str(tokenizer)))
        text=lambda value:dict(type='text',text=value)
        parts=[text('Current'),fixtures.image('current'),text('Programs: find'),text('Snapshot'),fixtures.image('past')]
        payload=dict(model='pinned-test-model',messages=[dict(role='system',content='Solve the game'),dict(role='user',content=parts)],
            chat_template_kwargs=dict(enable_thinking=True,preserve_thinking=True))
        managed=dict(version=1,observations=[dict(observation_id='obs-1',message_index=1,
            content_sha256=content_hash(parts),base_parts_end=2,program_parts=[2,3],snapshot_parts=[3,5])],evict_observation_ids=[])
        result=await self.traced(protocol.render_initial(serving_chat=serving,payload=payload,
                                      managed_suffixes=managed,request_factory=fixtures.request))
        self.assertEqual(len(result.suffix_records),1)
        rows=self.records('render')
        self.assertEqual([r['details']['render_kind'] for r in rows],['main','anchor','base','preceding'])
        self.assertEqual([r['details']['images'] for r in rows],[2,0,1,0])
        before=len(rows)
        await self.traced(protocol.render_delta(serving,state=result.state,external_messages=[dict(role='user',content='New')],
            previous_output_ids=[serving.renderer.tokenizer.token_to_id('<|im_end|>')],request_factory=fixtures.request))
        self.assertEqual([r['details']['render_kind'] for r in self.records('render')[before:]],['main'])
        self.assertEqual(p.status()['metadata_errors'],0)

    async def test_bad_render_metadata_preserves_original_factory_exception(self):
        failure=ValueError('original factory')
        def factory(payload):raise failure
        with self.assertRaises(ValueError) as caught:
            await self.traced(protocol._render(serving_chat=None,payload={'messages':[None]},request_factory=factory))
        self.assertIs(caught.exception,failure)
        self.assertEqual(p.status()['metadata_errors'],1)

    def test_flag_off_factory_returns_original_plugin_without_install(self):
        marker=object()
        with patch.dict(os.environ,{p.FLAG:'0'}),patch.object(endpoint,'create_plugin',return_value=marker),patch.object(p,'install',side_effect=AssertionError('disabled')):
            self.assertIs(persistent_plugin.create_plugin(),marker)
        with patch.dict(os.environ,{p.FLAG:'0'}):self.assertEqual(p.install()['installed'],False)
        with patch.dict(os.environ,{p.FLAG:'unexpected'}):
            with self.assertRaises(RuntimeError):p.enabled()

    def test_profile_source_dependencies_and_entrypoint_are_pinned(self):
        for name in ('persistent_endpoint.py','session_service.py','protocol.py'):
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),p.SOURCE_HASHES[name])
        native=Path(os.environ.get('ARC3_RUNTIME_SOURCE',STACK/'experiments/persistent_state_20260912/source-snapshot/vllm'))
        for name in ('renderers/hf.py','renderers/base.py'):
            if (native/name).exists():
                self.assertEqual(hashlib.sha256((native/name).read_bytes()).hexdigest(),p.SOURCE_HASHES[name])
        entry=(HERE/'arc3_persistent_experiment-0.2.dist-info/entry_points.txt').read_text()
        self.assertIn('arc3_persistent = persistent_plugin:create_plugin',entry)
        self.assertIn('arc3_persistent_engine = engine_patch:install',entry)

    async def test_loop_lag_aggregates_without_per_tick_records(self):
        values=iter([.1,.2,1.3,1.4]);now=[0.]
        async def sleep(_):
            try:now[0]=next(values)
            except StopIteration:raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await p.loop_lag(clock=lambda:now[0],wall=lambda:1000+now[0],sleep=sleep)
        rows=self.records('api_loop_lag');self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['details']['ticks'],3)
        self.assertAlmostEqual(rows[0]['details']['lag_max_seconds'],1.)
        self.assertAlmostEqual(rows[0]['details']['lag_mean_seconds'],1./3)
        self.assertEqual(rows[0]['seconds'],1.3)
        self.assertIsNone(rows[0]['session_id'])

    async def test_loop_lag_lifespan_stops_own_task_and_preserves_state(self):
        events=[];entered=asyncio.Event();stopped=asyncio.Event();marker=object()
        @asynccontextmanager
        async def native(app):
            events.append('native-start')
            try:yield marker
            finally:events.append('native-stop')
        async def fake_lag():
            entered.set()
            try:await asyncio.Event().wait()
            finally:stopped.set()
        app=NS(router=NS(lifespan_context=native))
        with patch.object(p,'loop_lag',fake_lag):
            p.install_loop_lag(app)
            async with app.router.lifespan_context(app) as state:
                self.assertIs(state,marker)
                await entered.wait();self.assertEqual(p.status()['loop_lag_tasks'],1)
            self.assertTrue(stopped.is_set());self.assertEqual(p.status()['loop_lag_tasks'],0)
        self.assertEqual(events,['native-start','native-stop'])


if __name__=='__main__':unittest.main()
