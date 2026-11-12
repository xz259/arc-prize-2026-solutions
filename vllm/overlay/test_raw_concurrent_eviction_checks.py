"""Exercise the concurrent client against real local policy without network/GPU."""
import asyncio
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import raw_concurrent_eviction_checks as panel
from raw_gpu_checks import ContractFailure
from raw_history_eviction_checks import HistoryClient
from session_service import SessionManager
from test_raw_history_eviction_checks import Backend


class CountedBackend(Backend):
    async def turn(self,session,turn,*,initial):
        self.submissions.append((session.session_id,list(turn.token_ids)))
        await asyncio.sleep(0)
        return dict(token_ids=[501]*turn.sampling['max_tokens'],finish_reason='length',stop_reason=None)


class ThreadClient(HistoryClient):
    def __init__(self,*args,manager,loop,**kwargs):
        super().__init__(*args,**kwargs);self.manager=manager;self.loop=loop
    def call(self,method,route,payload=None):
        async def request():
            if route.endswith('/start'):return await self.manager.start(copy.deepcopy(payload))
            if route.endswith('/append'):return await self.manager.append(self.sid,copy.deepcopy(payload))
            if route.endswith('/close'):return await self.manager.close(self.sid)
            raise AssertionError(route)
        return asyncio.run_coroutine_threadsafe(request(),self.loop).result(timeout=10)


class ConcurrentPanelTests(unittest.TestCase):
    def test_dry_run_never_calls_execution(self):
        with patch.object(panel,'run',side_effect=AssertionError('execution')),redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(panel.main([]),0)
        self.assertEqual(json.loads(stream.getvalue())['maximum_emitted_tokens'],3840)

    def test_three_session_policy_slot_replacement_workload_is_bounded(self):
        loop=asyncio.new_event_loop();thread=threading.Thread(target=loop.run_forever,daemon=True);thread.start()
        backend=CountedBackend();manager=SessionManager(backend,max_context=40000,max_sessions=3)
        clients=[ThreadClient('http://127.0.0.1:1234',Path('unused'),1,'local-'+lane,
            deadline=time.monotonic()+30,budget=8192,output_tokens=128,manager=manager,loop=loop)
            for lane in 'abcd']
        rounds=[];reuse=[]
        try:
            panel.exercise(clients,lambda n,offset=0:list(range(offset,offset+n)),
                on_round=lambda phase,index:rounds.append((phase,index)),
                on_reuse=lambda phase,old,new:reuse.append(phase))
            self.assertEqual([len(c.turns) for c in clients],[10,5,10,5])
            self.assertEqual(sum(t['output_tokens'] for c in clients for t in c.turns),3840)
            self.assertEqual(len(rounds),10)
            self.assertEqual(reuse,['before-close','after-close','after-replacement'])
            self.assertIn(clients[1].sid,backend.closed)
            self.assertEqual(sum(t['history_eviction']['triggered'] for c in clients for t in c.turns),4)
            self.assertEqual(sum(t['history_eviction']['suffix_removed_tokens']>0 for c in clients for t in c.turns),16)
            self.assertLessEqual(max(t['resident_tokens'] for c in clients for t in c.turns),8192)
            for c in clients:
                first_history=next(t for t in c.turns if t['history_eviction']['triggered'])
                self.assertEqual(first_history['turn'],4)
                self.assertEqual(first_history['history_eviction']['selected_block_ids'],['obs-01'])
        finally:
            asyncio.run_coroutine_threadsafe(manager.close_all(),loop).result(timeout=10)
            loop.call_soon_threadsafe(loop.stop);thread.join(timeout=10);loop.close()

    def test_parallel_failures_wait_for_every_owned_thread(self):
        seen=[];lock=threading.Lock()
        class Job:
            def __init__(self,index):self.index=index
            def submit(self,*args,**kwargs):
                with lock:seen.append(self.index)
                if self.index==0:raise RuntimeError('intentional')
                return self.index
        with self.assertRaisesRegex(ContractFailure,'intentional'):
            panel.parallel_round([(Job(i),'test',[],None) for i in range(3)])
        self.assertEqual(sorted(seen),[0,1,2])

    def test_graph_batch_delta_requires_real_three_padded_four(self):
        before={'prepared_batches':[dict(mode='FULL',real_requests=3,padded_requests=4,count=10)]}
        after={'prepared_batches':[
            dict(mode='FULL',real_requests=3,padded_requests=4,count=23),
            dict(mode='FULL',real_requests=4,padded_requests=4,count=999),
            dict(mode='NONE',real_requests=3,padded_requests=3,count=99)]}
        self.assertEqual(panel.batch_delta(before,after,'graph')['count'],13)
        self.assertEqual(panel.batch_delta(before,after,'eager')['count'],99)

    def test_resident_slot_matches_exact_owner_not_prefix_collision(self):
        state={'request_slots':{'arc3-persistent-x-deadbeef':29,'arc3-persistent-xy-deadbeef':27}}
        self.assertEqual(panel.slot_for(state,'x'),29)
        with self.assertRaises(ContractFailure):panel.slot_for(state,'missing')

    def test_output_token_constructor_rejects_unbounded_or_invalid_counts(self):
        for count in (0,129,True,1.5):
            with self.subTest(count=count),self.assertRaises(ValueError):
                HistoryClient('http://127.0.0.1:1234',Path('unused'),1,'x',deadline=time.monotonic()+10,output_tokens=count)


if __name__=='__main__':unittest.main()
