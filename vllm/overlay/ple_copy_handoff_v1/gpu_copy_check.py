"""Compare actual D2H bytes with original and patched PLE methods, no model."""
import json
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import torch
from test_handoff import methods

root=Path(__file__).resolve().parent
assert torch.cuda.get_device_capability()==(12,0)
records=[]
for name in ['connector_original.py','connector.py']:
    c=methods(torch,root/name)();c.device=torch.device('cuda:0');c.tp_rank=0;c.dp_rank=0
    c._uses_cuda_inputs=True;c._request_queue=queue.Queue(maxsize=1)
    c._input_copy_submitted=threading.Event();c._d2h_stream=torch.cuda.Stream()
    c._input_ready_event=torch.cuda.Event();c._d2h_done_event=torch.cuda.Event()
    for key,shape in [('input_ids',(8192,)),('query_start_loc',(31,)),('ngram_context',(30,2))]:
        setattr(c,'_'+key+'_source',torch.empty(shape,device='cuda',dtype=torch.int32))
        setattr(c,'_'+key+'_buf',torch.empty(shape,pin_memory=True,dtype=torch.int32))
    done=threading.Event();errors=[];expected={};count=[0]
    def send(request):
        for key,n in [('input_ids',request.num_tokens),('query_start_loc',request.num_reqs+1),('ngram_context',request.num_reqs)]:
            actual=getattr(c,'_'+key+'_buf')[:n]
            assert torch.equal(actual,expected[key][:n]),(name,key,count[0])
        count[0]+=1;done.set()
    socket=SimpleNamespace(send=send)
    def worker():
        try:
            while True:
                request=c._request_queue.get()
                if request is None:return
                c._process_request(request,socket)
        except BaseException as e:errors.append(repr(e));done.set()
    t=threading.Thread(target=worker,daemon=True);t.start();ack=[]
    for i in range(96):
        n=[30,25,8,5,4,3,2,1][i%8];tokens=8192 if i%12==0 else n
        for key in ['input_ids','query_start_loc','ngram_context']:
            source=getattr(c,'_'+key+'_source')
            value=(torch.arange(source.numel(),dtype=torch.int32).reshape(source.shape)+i*1024)
            expected[key]=value;source.copy_(value)
        done.clear();start=time.perf_counter();c._launch(n,tokens);ack.append(time.perf_counter()-start)
        assert done.wait(10),'copy completion timeout'
        assert not errors,errors
    c._request_queue.put(None);t.join(10);assert not t.is_alive()
    torch.cuda.synchronize();assert count[0]==96
    records.append(dict(source=name,cases=count[0],exact=True,mean_launch_us=sum(ack)/len(ack)*1e6))
report=dict(passed=True,scope='192 actual GPU D2H copy/optional-context cases; no model or graph-deadlock claim',records=records)
(root/'gpu-copy-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
