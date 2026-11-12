"""Source-bound actual-router inventory gates; no Torch/model construction."""
import hashlib
import inspect
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import arc3_private_moe_adapter as a


class FusedTopKRouter:
    top_k=10;global_num_experts=512;scoring_func='softmax';renormalize=True;eplb_state=None;capture_fn=None
class MoERunner:
    def __init__(self):
        self.router=FusedTopKRouter();self.routed_experts=NS(quant_method=NS(is_monolithic=False))
        self._quant_method=NS(topk_indices_dtype='i32');self._fse_fuse_gate=False


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.rows=[('layers.%d.experts'%i,MoERunner()) for i in range(48)]
        self.runner=NS(get_model=lambda:NS(named_modules=lambda:iter(self.rows)))
        self.settings=dict(runtime_root=str(self.root),router_sources={})
        p1='model_executor/layers/fused_moe/router/fused_topk_router.py';p2='model_executor/layers/fused_moe/runner/moe_runner.py'
        modules={}
        for name in ['vllm.model_executor.layers.fused_moe.router.fused_topk_router','vllm.model_executor.layers.fused_moe.runner.moe_runner','torch']:
            modules[name]=ModuleType(name)
        modules['vllm.model_executor.layers.fused_moe.router.fused_topk_router'].FusedTopKRouter=FusedTopKRouter
        modules['vllm.model_executor.layers.fused_moe.runner.moe_runner'].MoERunner=MoERunner
        modules['torch'].int32='i32'
        patch.dict(sys.modules,modules).start();patch.object(inspect,'getsourcefile',side_effect=lambda c:str(self.root/(p1 if c is FusedTopKRouter else p2))).start()
        self.addCleanup(patch.stopall)
    def test_all48_actual_router_paths_recorded(self):
        r=a.verify_routers(self.runner,self.settings);self.assertEqual(r['count'],48);self.assertEqual(len(r['routers']),48)
    def test_missing_router_count_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(RuntimeError,'all48'):a.verify_routers(self.runner,self.settings)
    def test_custom_router_and_eplb_rejected(self):
        self.rows[0][1].router.eplb_state=object()
        with self.assertRaisesRegex(RuntimeError,'EPLB'):a.verify_routers(self.runner,self.settings)
        self.rows[0][1].router=object()
        with self.assertRaisesRegex(RuntimeError,'Active router'):a.verify_routers(self.runner,self.settings)
    def test_monolithic_or_fused_gate_cannot_silently_bypass_producer(self):
        self.rows[0][1].routed_experts.quant_method.is_monolithic=True
        with self.assertRaisesRegex(RuntimeError,'bypass'):a.verify_routers(self.runner,self.settings)
    def test_default_indices_dtype_resolves_to_native_int32(self):
        self.rows[0][1]._quant_method.topk_indices_dtype=None
        r=a.verify_routers(self.runner,self.settings)
        self.assertEqual(r['routers'][0]['topk_indices_dtype'],'torch.int32')
        self.rows[0][1]._quant_method.topk_indices_dtype='i64'
        with self.assertRaisesRegex(RuntimeError,'bypass'):a.verify_routers(self.runner,self.settings)
    def test_pending_native_producer_receipt_cannot_admit(self):
        with self.assertRaisesRegex(RuntimeError,'still pending'):a.verify_route_eligibility({})

if __name__=='__main__':unittest.main()
