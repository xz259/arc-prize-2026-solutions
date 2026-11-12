"""Execute the actual patched methods with controlled copy-completion boundaries."""
import ast
from contextlib import nullcontext
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).with_name('connector.py')


def methods(torch, source=SOURCE):
    tree=ast.parse(source.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PleOffloadConnector')
    functions=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in
               ('_launch','_copy_cuda_inputs','_process_request')]
    code=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
                         ast.ClassDef(name='Connector',bases=[],keywords=[],body=functions,decorator_list=[])],type_ignores=[])
    ns=dict(torch=torch,PleOffloadRequest=lambda **kw:SimpleNamespace(**kw),
            msgspec=SimpleNamespace(msgpack=SimpleNamespace(encode=lambda x:x)))
    exec(compile(ast.fix_missing_locations(code),str(source),'exec'),ns)
    return ns['Connector']


class HandoffTest(unittest.TestCase):
    def make(self, cuda=True, tp_rank=0):
        log=[];allow_submit=threading.Event();allow_complete=threading.Event()
        class Tensor:
            def __init__(self,name):self.name=name
            def __getitem__(self,key):return self
            def copy_(self,source,non_blocking):
                self_test.assertTrue(non_blocking)
                self_test.assertTrue(allow_submit.wait(2))
                log.append(('copy',self.name,threading.current_thread().name))
        class Event:
            def record(self,stream):log.append(('record',stream))
            def synchronize(self):
                log.append(('wait-completion',))
                self_test.assertTrue(allow_complete.wait(2))
        class Stream:
            def wait_event(self,event):log.append(('wait-ready',))
        self_test=self
        stream=Stream()
        torch=SimpleNamespace(accelerator=SimpleNamespace(device_index=lambda x:nullcontext()),
            cuda=SimpleNamespace(stream=lambda x:nullcontext(),current_stream=lambda x:'main',
                                 nvtx=SimpleNamespace(range=lambda x:nullcontext())))
        c=methods(torch)();c.device=SimpleNamespace(index=0);c.tp_rank=tp_rank;c.dp_rank=0
        c._uses_cuda_inputs=cuda;c._d2h_stream=stream;c._input_ready_event=Event();c._d2h_done_event=Event()
        c._input_copy_submitted=threading.Event();c._request_queue=queue.Queue(maxsize=1)
        for name in ('input_ids','query_start_loc','ngram_context'):
            setattr(c,'_'+name+'_buf',Tensor(name));setattr(c,'_'+name+'_source',Tensor('source_'+name))
        return c,log,allow_submit,allow_complete

    def test_submission_before_launch_return_but_completion_stays_async(self):
        c,log,submit,complete=self.make();returned=threading.Event();errors=[]
        socket=SimpleNamespace(send=lambda request:log.append(('send',vars(request))))
        def caller():
            try:c._launch(4,4);log.append(('launch-return',));returned.set()
            except BaseException as e:errors.append(e)
        def worker():
            try:c._process_request(c._request_queue.get(timeout=2),socket)
            except BaseException as e:errors.append(e)
        t=threading.Thread(target=caller,name='model');w=threading.Thread(target=worker,name='notifier')
        t.start();w.start()
        self.assertFalse(returned.wait(.03))
        submit.set();self.assertTrue(returned.wait(1))
        self.assertFalse(any(x[0]=='send' for x in log))
        self.assertEqual([x[1] for x in log if x[0]=='copy'],['input_ids','query_start_loc','ngram_context'])
        self.assertLess(max(i for i,x in enumerate(log) if x[0]=='copy'),next(i for i,x in enumerate(log) if x[0]=='launch-return'))
        complete.set();t.join(2);w.join(2)
        self.assertFalse(t.is_alive() or w.is_alive());self.assertFalse(errors,errors)
        self.assertEqual(log[-1],('send',dict(dp_rank=0,num_tokens=4,num_reqs=4)))

    def test_cpu_path_and_nonpublisher_unchanged(self):
        c,log,_,_=self.make(cuda=False);c._launch(3,5)
        self.assertEqual(vars(c._request_queue.get_nowait()),dict(dp_rank=0,num_tokens=5,num_reqs=3));self.assertEqual(log,[])
        c,log,_,_=self.make(tp_rank=1);c._launch(3,5);self.assertTrue(c._request_queue.empty());self.assertEqual(log,[])

    def test_missing_submission_fails_bounded(self):
        c,_,_,_=self.make()
        class Never:
            def clear(self):pass
            def wait(self,timeout):self.timeout=timeout;return False
        c._input_copy_submitted=Never()
        with self.assertRaisesRegex(RuntimeError,'submission'):c._launch(4,4)
        self.assertEqual(c._input_copy_submitted.timeout,30)

    def test_null_optional_ngram(self):
        c,log,submit,complete=self.make();c._ngram_context_buf=None;c._ngram_context_source=None
        submit.set();complete.set();c._copy_cuda_inputs(SimpleNamespace(num_tokens=4,num_reqs=4))
        self.assertTrue(c._input_copy_submitted.is_set())
        self.assertEqual([x[1] for x in log if x[0]=='copy'],['input_ids','query_start_loc'])

if __name__=='__main__':unittest.main()
