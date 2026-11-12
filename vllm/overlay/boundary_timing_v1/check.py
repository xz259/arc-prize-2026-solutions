"""CPU gates: wrappers record period/duration/composition and RPC durations; the journal thread writes JSON lines."""
import json, os, sys, tempfile, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tmp = tempfile.mkdtemp(); os.environ['ARC3_BOUNDARY_TIMING_RECEIPTS'] = tmp; os.environ['ARC3_BOUNDARY_TIMING'] = '1'
from boundary_timing_v1 import runtime as rt
class Runner:
    def execute_model(self, so, *a, **k): time.sleep(0.002); return 'out'
mr = types.SimpleNamespace(GPUModelRunner=Runner, __name__='mr'); rt.install_model_runner(mr)
SO = lambda n, tok, new, fin: types.SimpleNamespace(num_scheduled_tokens={f'r{i}': 1 for i in range(n)}, total_num_scheduled_tokens=tok, scheduled_new_reqs=[0] * new, finished_req_ids=set(range(fin)))
r = Runner(); assert r.execute_model(SO(39, 39, 0, 0)) == 'out'; time.sleep(0.01); r.execute_model(SO(39, 700, 1, 1))
ol = types.SimpleNamespace(__name__='one_lookahead', worker_settle=lambda worker, wires: 'settled', settle_core=lambda core, outputs, f, c: outputs, pending=lambda sched: {'a': 1})
rt.install_one_lookahead(ol); assert ol.worker_settle(None, [1, 2]) == 'settled'; assert ol.settle_core(types.SimpleNamespace(scheduler=None), {'x': 1}, None, None) == {'x': 1}
ee = types.SimpleNamespace(__name__='engine_eviction', worker_compact=lambda *a: 'c'); rt.install_engine_eviction(ee); assert ee.worker_compact(1) == 'c'
time.sleep(0.2); rt._Q.put(None); rt._THREAD.join(2)
lines = [json.loads(l) for f in os.listdir(tmp) if f.endswith('.jsonl') for l in open(os.path.join(tmp, f))]
kinds = [l['k'] for l in lines]; assert kinds == ['step', 'step', 'worker_settle', 'settle_core', 'worker_compact'], kinds
s0, s1 = lines[0], lines[1]; assert s0['p'] is None and s1['p'] > 0.01 and s1['n'] == 39 and s1['tok'] == 700 and s1['new'] == 1 and s1['fin'] == 1 and s0['d'] >= 0.002
assert lines[2]['n'] == 2 and lines[3]['n'] == 1 and all(l['d'] >= 0 for l in lines)
print('cpu gates ok', rt.STATS['installed'])
