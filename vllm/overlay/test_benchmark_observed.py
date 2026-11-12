"""No-network graph snapshot validation plus exact AST workload parity."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'frozen-v3-scalar-proof/benchmark.py'
sys.path.insert(0,str(BASE.parent))
spec=importlib.util.spec_from_file_location('benchmark_observed_under_test',HERE/'benchmark_observed.py')
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)


def state():
    return dict(graph_mapping=True,trace_enabled=True,enforce_eager=False,
                prepared_batches=[dict(mode='FULL',real_requests=23,padded_requests=24,count=1023)])


class SnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_read_only_rpc_and_counter_payload_are_retained(self):
        calls=[];value=state()
        async def request(*args):calls.append(args);return {'results':[value]}
        result=await b.capture_graph_state(request,'graph')
        self.assertIs(result,value)
        self.assertEqual(calls,[('POST','/collective_rpc',dict(method='arc3_eviction_graph_status',args=[],timeout=20))])
    async def test_empty_before_histogram_and_eager_state_are_valid(self):
        value=state();value['prepared_batches']=[];value['enforce_eager']=True
        async def request(*args):return {'results':[value]}
        self.assertEqual(await b.capture_graph_state(request,'eager'),value)
    async def test_missing_duplicate_and_noninteger_counts_are_rejected(self):
        broken=[]
        value=state();value.pop('prepared_batches');broken.append(value)
        value=state();value['prepared_batches']*=2;broken.append(value)
        for key,number in [('count',True),('count',-1),('real_requests',0),('padded_requests',22)]:
            value=state();value['prepared_batches'][0][key]=number;broken.append(value)
        for value in broken:
            async def request(*args):return {'results':[value]}
            with self.assertRaises(b.BenchmarkFailure):await b.capture_graph_state(request,'graph')
    async def test_wrong_worker_or_mode_and_transport_failure_propagate(self):
        for response in [{'results':[]},{'results':[state(),state()]},
                         {'results':[dict(state(),trace_enabled=False)]},
                         {'results':[dict(state(),enforce_eager=True)]}]:
            async def request(*args):return response
            with self.assertRaises(b.BenchmarkFailure):await b.capture_graph_state(request,'graph')
        async def failed(*args):raise TimeoutError('unavailable')
        with self.assertRaises(TimeoutError):await b.capture_graph_state(failed,'graph')


class SourceParityTests(unittest.TestCase):
    def test_all_code_except_observer_and_source_receipt_is_ast_identical(self):
        base=ast.parse(BASE.read_text());candidate=ast.parse((HERE/'benchmark_observed.py').read_text())
        candidate.body[0]=copy.deepcopy(base.body[0])  # explanatory module docstring
        class StripObserver(ast.NodeTransformer):
            assignments=0;helpers=0;source_receipts=0
            def visit_AsyncFunctionDef(self,node):
                if node.name=='capture_graph_state':self.helpers+=1;return None
                return self.generic_visit(node)
            def visit_Assign(self,node):
                for target in node.targets:
                    if any(isinstance(child,ast.Constant) and child.value in ('graph_states','graph_counter_scope')
                           for child in ast.walk(target)):
                        self.assignments+=1;return None
                return self.generic_visit(node)
            def visit_comprehension(self,node):
                if isinstance(node.iter,ast.Tuple) and len(node.iter.elts)==2 and isinstance(node.iter.elts[0],ast.Attribute):
                    self.assert_path_receipt(node.iter.elts[0]);self.source_receipts+=1
                    node.iter.elts[0]=ast.Constant(value='benchmark.py')
                return self.generic_visit(node)
            @staticmethod
            def assert_path_receipt(node):
                assert ast.dump(node)==ast.dump(ast.parse('Path(__file__).name',mode='eval').body)
        normalize=StripObserver();candidate=normalize.visit(candidate)
        self.assertEqual((normalize.assignments,normalize.helpers,normalize.source_receipts),(3,1,1))
        self.assertEqual(ast.dump(candidate),ast.dump(base))
    def test_snapshots_strictly_bracket_and_do_not_enter_measurement(self):
        source=(HERE/'benchmark_observed.py').read_text()
        before=source.index("result['graph_states']={'before_measurement'")
        metric=source.index('before=await metrics();samples.append(before)')
        start=source.index('start=time.monotonic()-origin\n')
        end=source.index('end=time.monotonic()-origin;after=await metrics()')
        receipt=source.index("result['measurement']=dict")
        after=source.index("result['graph_states']['after_measurement']")
        self.assertEqual(sorted([before,metric,start,end,receipt,after]),[before,metric,start,end,receipt,after])
        self.assertIn("zero-token worker calls return earlier and are not counted",source)


if __name__=='__main__':unittest.main()
