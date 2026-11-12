"""CPU source/lifecycle checks; real source getter with fake native constructor."""
import ast
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import typing
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arc3_private_moe_adapter as a


def original_getter():
    raw = (HERE / 'fixtures/core.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '9c5f0bb8b9d36586d0c16cd0f7ca837d5b75f06a5e74a1da75e2e27b3091c1c8'
    tree = ast.parse(raw)
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'get_cutlass_fused_moe_module')
    body = ast.Module(body=[node], type_ignores=[])
    core = ModuleType('flashinfer.fused_moe.core'); core.__package__ = 'flashinfer.fused_moe'
    namespace = dict(functools=functools, TunableRunner=object, TuningConfig=lambda **k: k,
        DynamicTensorSpec=lambda *x: x, get_hybrid_num_tokens_buckets=lambda n: [],
        make_hybrid_bucket_mapper=lambda n: None, torch=NS(dtype=str, Tensor=object,bfloat16='bf16', device=lambda x:x),
        ActivationType=NS(Swiglu='swiglu'), device_support_pdl=lambda _: True,
        SimpleNamespace=NS, register_custom_op=lambda *a, **k: lambda f:f,
        register_fake_op=lambda *a, **k: lambda f:f,
        Dict=typing.Dict, Tuple=typing.Tuple, List=typing.List, Optional=typing.Optional, Any=typing.Any,
        OptimizationProfile=object)
    core.__dict__.update(namespace)
    exec(compile(body, str(HERE / 'fixtures/core.txt'), 'exec'), core.__dict__)
    return core


class Native:
    def __init__(self):
        self.mode = 0; self.args = []; self.includes = []; self.workspaces = 0
        self.interleave_moe_weights_for_sm90_mixed_gemm = lambda x: x
    def arc3_private_get_modes_v1(self): return self.mode
    def arc3_private_configure_v1(self, mode, valid):
        old = self.mode; self.mode = mode; return old
    def set_deepgemm_jit_include_dirs(self, dirs): self.includes.append(dirs)
    def init(self, *args): self.args.append(args); return self
    def get_gemm1_tactic_count(self): return 20
    def get_gemm2_tactic_count(self): return 20
    def get_tactic_num(self): return 40
    def get_workspace_size(self, *args): self.workspaces += 1; return 1234


class FakeTuner:
    current = None
    @classmethod
    def get(cls): return cls.current
    def __init__(self):
        self._lock = threading.RLock(); self._file_configs = {}; self.profiling_cache = {}
        self.reject = False
    def load_configs(self, path):
        if self.reject: return False
        self._file_configs = json.loads(Path(path).read_text()); return True


class AdapterTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root = Path(tmp.name).resolve()
        self.native = Native(); self.core = original_getter(); self.original = lambda *args: (_ for _ in ()).throw(AssertionError('AOT generator must not run'))
        self.core.gen_cutlass_fused_moe_sm120_module = self.original
        self.lib = self.root/'private.so'; self.lib.write_bytes(b'fixture'); self.prepared = self.root/'PREPARE_RESULT.json'; self.prepared.write_text('{}')
        evidence = self.root/'actual-layer23'; evidence.mkdir()
        d = dict(complete=True,passed=True,controls_complete=True,host_mode_reset=True,modes_restored=True,
                 private_library_sha256=a.sha(self.lib),build_result_sha256='build',rows=[dict(mode=i,valid=True) for i in range(4)])
        (evidence/'private-layer23-n25.json').write_text(json.dumps(d))
        self.include = self.root/'data/csrc/nv_internal/tensorrt_llm';self.include.mkdir(parents=True)
        self.state = a.State.__new__(a.State); self.state.core = self.state.native = self.state.original_getter = None
        self.state.init_count = 0; self.state.mode_fixed = False; self.state.output = self.root/'output'; self.state.receipt = dict(startup_events=[])
        self.state.settings = dict(mode=3,flashinfer_root=str(self.root),prepared_sha256=a.sha(self.prepared),
            private=dict(prepared_root=str(self.root),private_library=str(self.lib),private_library_sha256=a.sha(self.lib),build_result_sha256='build'),
            operator_evidence_root=str(self.root),operator_evidence={'actual-layer23/private-layer23-n25.json':a.sha(evidence/'private-layer23-n25.json')})
        self.loaded = False
        def load(path): self.assertEqual(path,str(self.lib)); self.loaded=True; return self.native
        self.addCleanup(patch.stopall)
        patch.object(a,'verify_build',return_value=({},dict(module_name='private'),self.include)).start()
        patch.object(a,'verify_route_eligibility',return_value=dict(verified=True)).start()
        patch.object(a,'verify_routers',return_value=dict(count=48,actual_model_routers_verified=True)).start()
        patch.object(a,'mapped_paths',side_effect=lambda: {str(self.lib)} if self.loaded else set()).start()
        patch.dict('os.environ', {'ARC3_MOE_FINALIZE':'unfused'}).start()
        self.modules={}
        for name in ['flashinfer','flashinfer.jit','flashinfer.jit.env','tvm_ffi','torch','vllm','vllm.compilation','vllm.compilation.caching','flashinfer.autotuner']:
            self.modules[name]=ModuleType(name); self.modules[name].__path__=[]
        self.modules['tvm_ffi'].load_module=load
        self.modules['flashinfer.jit.env'].FLASHINFER_CSRC_DIR=self.root/'data/csrc'
        self.modules['torch'].__dict__.update(uint8='u8',int64='i64',bfloat16='bf16')
        patch.dict(sys.modules,self.modules).start()

    def setup_core(self):
        self.state.core_ready(self.core)
        w=self.core.get_cutlass_fused_moe_module('120')
        result=w.cutlass_fused_moe_workspace_size(25,2560,640,512,10,x_dtype='u8',weight_dtype='i64',use_fused_finalize=False)
        self.assertEqual(result,1234)
        return w

    def test_exact_source_getter_and_one_native_init_no_forward_proxy(self):
        w=self.setup_core()
        w.cutlass_fused_moe_workspace_size(30,2560,640,512,10,x_dtype='u8',weight_dtype='i64',use_fused_finalize=False)
        self.assertEqual(self.native.args,[('u8','i64','bf16',False,False,False,False,False)])
        self.assertEqual(self.native.workspaces,2)
        self.assertIs(self.core.gen_cutlass_fused_moe_sm120_module,self.original)
        self.assertEqual(self.core.get_cutlass_fused_moe_module.cache_info().currsize,1)
        self.assertEqual(self.state.receipt['native_init_count'],1)
        self.assertEqual(self.native.mode,0)

    def test_nonempty_getter_rejected_without_clear(self):
        self.core.gen_cutlass_fused_moe_sm120_module=lambda *args:NS(build_and_load=lambda:self.native)
        self.core.get_cutlass_fused_moe_module('120')
        with self.assertRaisesRegex(RuntimeError,'Existing MoE'):self.state.core_ready(self.core)
        self.assertEqual(self.core.get_cutlass_fused_moe_module.cache_info().currsize,1)
        self.assertFalse(self.loaded)

    def test_generator_restored_if_getter_registration_fails(self):
        self.core.register_custom_op=lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('registration failure'))
        with self.assertRaisesRegex(ValueError,'registration'):self.state.core_ready(self.core)
        self.assertIs(self.core.gen_cutlass_fused_moe_sm120_module,self.original)
        self.assertIs(self.state.native,self.native) # native handle retained, no unsafe unload

    def test_wrong_constructor_or_second_key_rejected(self):
        self.setup_core(); w=self.core.get_cutlass_fused_moe_module('120')
        with self.assertRaisesRegex(RuntimeError,'repeated native'):
            w.cutlass_fused_moe_workspace_size(25,2560,640,512,10,x_dtype='u8',weight_dtype='i64',use_fused_finalize=True)

    def warmup(self, mutate=False):
        self.setup_core()
        cache=self.root/'namespace/autotune_configs.json';cache.parent.mkdir()
        plans={f"('{op}', 'MoERunner', (({n}, 1280),), (False,))":['MoERunner',t] for op,t in zip(a.OPS,[17,37]) for n in range(21)}
        cache.write_text(json.dumps(plans)); tuner=FakeTuner();FakeTuner.current=tuner
        config=NS(speculative_config=None,parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,enable_expert_parallel=False),
            model_config=NS(enforce_eager=False,max_model_len=40000),scheduler_config=NS(max_num_seqs=30),
            cache_config=NS(enable_prefix_caching=False,gpu_memory_utilization=.95),
            compilation_config=NS(compile_sizes=[],cudagraph_capture_sizes=[1,2,4,8,16,24,25,30,32]),compute_hash=lambda:'config')
        factors=['env','config','tail'];self.modules['vllm.compilation.caching'].aot_compile_hash_factors=lambda _:factors
        self.modules['flashinfer.autotuner'].AutoTuner=FakeTuner
        self.state.settings.update(baseline_cache=str(cache),baseline_namespace='namespace',baseline_cache_sha256=a.sha(cache),
            baseline_config_hash='config',baseline_env_hash='env',baseline_tail_sha256=hashlib.sha256(str(factors[2:]).encode()).hexdigest(),autotuner_source_sha256=a.sha(__file__))
        module=ModuleType('warmup')
        module.resolve_flashinfer_autotune_file=lambda runner:cache
        def original(runner):
            tuner.load_configs(str(cache))
            if mutate:tuner._file_configs[next(iter(plans))]=['MoERunner',999]
            return 'original-result'
        module.flashinfer_autotune=original
        self.state.wrap_warmup(module)
        return module,original,tuner,NS(vllm_config=config)

    def test_cache42_load_and_warmup_restored(self):
        m,original,tuner,runner=self.warmup()
        self.assertEqual(m.flashinfer_autotune(runner),'original-result')
        self.assertIs(m.flashinfer_autotune,original)
        self.assertNotIn('load_configs',tuner.__dict__)
        r=self.state.receipt
        self.assertEqual(r['state'],'warmup-verified')
        self.assertTrue(r['loaded_main_plans_verified'] and r['after_warmup_main_plans_verified'])
        self.assertEqual(r['main_plan_count'],42)
        self.assertEqual(self.native.mode,3)
        self.assertTrue(r['mode_set_after_actual_router_admission'])

    def test_changed_plans_fail_and_restore_hooks(self):
        m,original,tuner,runner=self.warmup(mutate=True)
        with self.assertRaisesRegex(RuntimeError,'altered42'):m.flashinfer_autotune(runner)
        self.assertIs(m.flashinfer_autotune,original);self.assertNotIn('load_configs',tuner.__dict__)
        self.assertTrue(self.state.receipt['process_must_exit'])

    def test_cache_load_rejection_fail_closed(self):
        m,original,tuner,runner=self.warmup();tuner.reject=True
        with self.assertRaisesRegex(RuntimeError,'load identical42'):m.flashinfer_autotune(runner)
        self.assertIs(m.flashinfer_autotune,original);self.assertNotIn('load_configs',tuner.__dict__)

    def test_nonbaseline_compile_and_namespace_fail_before_warmup(self):
        m,original,tuner,runner=self.warmup();runner.vllm_config.compilation_config.compile_sizes=[25,30]
        with self.assertRaisesRegex(RuntimeError,'Exact-compile'):m.flashinfer_autotune(runner)
        self.assertEqual(tuner._file_configs,{})
        self.assertEqual(self.native.mode,0)


if __name__=='__main__': unittest.main()
