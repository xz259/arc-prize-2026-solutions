"""Fixed private module at startup; unchanged FlashInfer wrapper/native calls."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import time

from arc3_private_build_contracts.contracts import require, sha, verify_build, mapped_paths

CORE = 'flashinfer.fused_moe.core'
WARMUP = 'vllm.model_executor.warmup.kernel_warmup'
OPS = ('trtllm::fused_moe::gemm1', 'trtllm::fused_moe::gemm2')
STATE = None


def main_plans(document):
    return {k: list(v) for k, v in document.items() if isinstance(k, str) and
            any(k.startswith("('" + op + "', 'MoERunner',") for op in OPS)}


def loaded_plans(tuner):
    with tuner._lock:
        result = main_plans(tuner._file_configs)
        for key, value in tuner.profiling_cache.items():
            if main_plans({key.file_key: [key.runner_class_name, value[0]]}):
                require(type(value[0]) is int, 'Unexpected MoE tactic type')
                result[key.file_key] = [key.runner_class_name, value[0]]
    return result


def config_receipt(config):
    require(config.speculative_config is None, 'MTP must remain disabled')
    p = config.parallel_config
    require((p.tensor_parallel_size, p.pipeline_parallel_size, p.data_parallel_size) == (1, 1, 1),
            'Require TP1/PP1/DP1')
    require(not p.enable_expert_parallel, 'Expert parallelism unsupported')
    require(config.model_config.enforce_eager is False, 'Normal compilation is required')
    require(config.model_config.max_model_len == 40000 and config.scheduler_config.max_num_seqs == 30,
            'Serving capacity differs')
    require(config.cache_config.enable_prefix_caching is False and
            config.cache_config.gpu_memory_utilization == .95, 'Cache policy differs')
    require(list(config.compilation_config.compile_sizes) == [], 'Exact-compile experiment is not baseline')
    sizes = list(config.compilation_config.cudagraph_capture_sizes)
    require(sizes == [1, 2, 4, 8, 16, 24, 25, 30, 32], 'Capture sizes differ')
    return dict(hash=config.compute_hash(), compile_sizes=[], capture_sizes=sizes,
                enforce_eager=False, max_num_seqs=30, max_model_len=40000,
                gpu_memory_utilization=.95, prefix_caching=False, mtp_tokens=0,
                parallelism=[1, 1, 1], expert_parallel=False)


def verify_route_eligibility(settings):
    gate = settings.get('route_eligibility')
    require(isinstance(gate, dict) and gate.get('path') and gate.get('sha256'),
            'Native top-k route eligibility receipt is still pending')
    path = Path(gate['path'])
    require(path.is_absolute() and sha(path) == gate['sha256'], 'Native top-k evidence changed')
    proof = json.loads(path.read_text())
    require(proof['schema'] == 'arc3.moe.topk-admission.v1' and proof['complete'] is True and
            proof['eligible'] is True and proof['native_bytes_unchanged'] is True,
            'Native top-k did not pass distinct-valid-ID admission')
    require(len(proof['checks']) == 210 and len(proof['graph_checks']) == 50 and
            all(x['passed'] is True for x in proof['checks'] + proof['graph_checks']),
            'Native top-k coverage incomplete')
    runtime = proof['runtime']
    require(runtime['native_sha256'] == 'b34e1e3ce1b26088e717d1483a3801f1737cc68fd99263d91349e47b779d032a' and
            runtime['ops_sha256'] == '9f11980f43c44857acd88e6f5c96cbd2d767b657b05207e26f064566200dae11',
            'Native top-k authority differs')
    source = Path(settings['runtime_root'])
    require(Path(runtime['native_path']) == source / '_moe_C_stable_libtorch.abi3.so' and
            sha(runtime['native_path']) == runtime['native_sha256'] and
            sha(source / '_custom_ops.py') == runtime['ops_sha256'], 'Current top-k producer changed')
    return dict(path=str(path), sha256=gate['sha256'], native_sha256=runtime['native_sha256'],
                active_and_masked_graph_eligibility_verified=True)


def verify_routers(runner, settings):
    """Read the actual main model once before native warmup; no tensor operations."""
    root = Path(settings['runtime_root'])
    for relative, expected in settings['router_sources'].items():
        require(sha(root / relative) == expected, 'Router source changed: ' + relative)
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    import torch
    require(Path(inspect.getsourcefile(FusedTopKRouter)).resolve() == root /
            'model_executor/layers/fused_moe/router/fused_topk_router.py', 'Router class shadows pinned source')
    require(Path(inspect.getsourcefile(MoERunner)).resolve() == root /
            'model_executor/layers/fused_moe/runner/moe_runner.py', 'MoE runner shadows pinned source')
    rows = []
    for name, holder in runner.get_model().named_modules():
        if not hasattr(holder, 'router'):
            continue
        router = holder.router
        require(type(holder) is MoERunner and type(router) is FusedTopKRouter,
                'Active router/runner is not the reviewed native producer path')
        require(router.top_k == 10 and router.global_num_experts == 512 and
                router.scoring_func == 'softmax' and router.renormalize is True and
                router.eplb_state is None and router.capture_fn is None,
                'Active routing config/EPLB/callback differs')
        require(holder.routed_experts.quant_method.is_monolithic is False and
                holder._quant_method.topk_indices_dtype in (None, torch.int32) and not holder._fse_fuse_gate,
                'Routing may bypass reviewed topk producer')
        rows.append(dict(module_path=name, router=type(router).__module__ + '.' + type(router).__qualname__,
                         top_k=10, global_num_experts=512, renormalize=True, scoring_func='softmax',
                         eplb=False, capture_callback=False, modular=True,
                         requested_indices_dtype=str(holder._quant_method.topk_indices_dtype),
                         topk_indices_dtype='torch.int32'))
    require(len(rows) == 48, 'Expected all48 main MoE routers')
    return dict(count=48, actual_model_routers_verified=True, routers=rows,
                source_sha256=settings['router_sources'])


class State:
    def __init__(self, settings, settings_path):
        require(os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') == 'spawn',
                'Pinned runtime must spawn fresh worker interpreters')
        self.settings = settings
        self.settings_path = Path(settings_path)
        self.output = Path(os.environ['ARC3_PRIVATE_MOE_RECEIPTS'])
        self.original_getter = self.core = self.native = None
        self.init_count = 0
        self.mode_fixed = False
        self.receipt = dict(schema='arc3.private-moe-serving.startup.v1', state='installed',
            pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            hostname=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            settings_sha256=sha(self.settings_path), fixed_mode=settings['mode'],
            private_library=settings['private']['private_library'],
            private_library_sha256=settings['private']['private_library_sha256'],
            build_result_sha256=settings['private']['build_result_sha256'],
            generator_restored=False, original_warmup_restored=False,
            per_decode_hooks=False, native_init_count=0, startup_events=[])
        require(self.receipt['hostname'] == settings['expected_hostname'] and
                self.receipt['boot_id'] == settings['expected_boot_id'], 'Host/boot differs')
        self.record('installed')

    def record(self, state, **fields):
        self.receipt.update(state=state, **fields)
        self.receipt['startup_events'].append(dict(state=state, epoch=time.time()))
        self.output.mkdir(parents=True, exist_ok=True)
        target = self.output / (str(os.getpid()) + '.json')
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(self.receipt, indent=2, sort_keys=True) + '\n')
        temp.replace(target)

    def core_ready(self, core):
        require(self.core is None and self.native is None, 'Core setup must occur once')
        getter = core.get_cutlass_fused_moe_module
        require(getter.cache_info().currsize == 0, 'Existing MoE wrapper must never be cleared/replaced')
        require(list(inspect.signature(getter).parameters) == ['backend', 'use_fast_build'],
                'Getter API differs')
        settings = self.settings
        require(os.environ.get('ARC3_MOE_FINALIZE') == 'unfused', 'Preserve deterministic unfused path')
        require(sha(Path(settings['private']['prepared_root']) / 'PREPARE_RESULT.json') ==
                settings['prepared_sha256'], 'Preparation identity changed')
        built, plan, private_include = verify_build(settings['private'])
        self.receipt['route_eligibility'] = verify_route_eligibility(settings)
        evidence = Path(settings['operator_evidence_root'])
        for relative, expected in settings['operator_evidence'].items():
            require(sha(evidence / relative) == expected, 'Admitted actual operator evidence changed')
        document = json.loads((evidence / 'actual-layer23/private-layer23-n25.json').read_text())
        require(document['complete'] and document['passed'] and document['controls_complete'] and
                document['host_mode_reset'] and document['modes_restored'], 'Actual operator gate incomplete')
        require(document['private_library_sha256'] == settings['private']['private_library_sha256'] and
                document['build_result_sha256'] == settings['private']['build_result_sha256'],
                'Operator/build linkage differs')
        require([row['mode'] for row in document['rows']] == [0, 1, 2, 3] and
                all(row['valid'] for row in document['rows']), 'Actual mode gates incomplete')
        library = Path(settings['private']['private_library'])
        before_maps = mapped_paths()
        require(str(library) not in before_maps and not any('arc3_private_moe_sm120_' in p for p in before_maps),
                'Private module already mapped before startup admission')
        import tvm_ffi
        native = tvm_ffi.load_module(str(library))
        self.native = native  # Keep resident on all success/failure paths; never unload/retry.
        require(str(library) in mapped_paths() and sha(library) == settings['private']['private_library_sha256'],
                'Loaded private module differs')
        require(int(native.arc3_private_get_modes_v1()) == 0, 'Fresh private module must start in mode0')
        self.core, self.original_getter = core, getter
        original_generator = core.gen_cutlass_fused_moe_sm120_module
        proxy = ModuleProxy(self, native, Path(settings['flashinfer_root']) / 'data/csrc/nv_internal/tensorrt_llm')
        calls = []
        class Spec:
            def build_and_load(inner):
                require(calls == ['generator'], 'Unexpected repeated module load')
                calls.append('load')
                return proxy
        def generator(use_fast_build=False):
            require(use_fast_build is False and not calls, 'Unexpected SM120 generator invocation')
            calls.append('generator')
            return Spec()
        core.gen_cutlass_fused_moe_sm120_module = generator
        try:
            # Ordinary callers use exactly this functools-cache key, not ("120", False).
            wrapper = getter('120')
            require(calls == ['generator', 'load'] and getter.cache_info().currsize == 1,
                    'Getter was not initialized exactly once')
            require(getter('120') is wrapper and self.init_count == 0, 'Unexpected native initialization at import')
        finally:
            core.gen_cutlass_fused_moe_sm120_module = original_generator
            self.receipt['generator_restored'] = True
        self.record('core-verified', module_name=plan['module_name'],
                    original_getter_preserved=core.get_cutlass_fused_moe_module is getter,
                    getter_cache_key=['120'], native_mode_readback=0,
                    include_root=str(proxy.include_root), module_mapped=True)

    def wrap_warmup(self, module):
        require(not hasattr(module, '_arc3_private_startup'), 'Warmup adapter already installed')
        original = module.flashinfer_autotune
        require(list(inspect.signature(original).parameters) == ['runner'], 'Warmup API differs')
        @functools.wraps(original)
        def warmup(runner):
            module.flashinfer_autotune = original
            self.receipt['original_warmup_restored'] = True
            tuner = original_load = None
            instance_load = False
            observed = []
            try:
                require(self.native is not None, 'Private module was not installed before warmup')
                actual = config_receipt(runner.vllm_config)
                self.receipt['router_inventory'] = verify_routers(runner, self.settings)
                require(not self.mode_fixed and int(self.native.arc3_private_get_modes_v1()) == 0,
                        'Private mode changed before actual-router admission')
                require(actual['hash'] == self.settings['baseline_config_hash'], 'Effective baseline config differs')
                from vllm.compilation.caching import aot_compile_hash_factors
                factors = aot_compile_hash_factors(runner.vllm_config)
                require(factors[0] == self.settings['baseline_env_hash'] and
                        hashlib.sha256(str(factors[2:]).encode()).hexdigest() == self.settings['baseline_tail_sha256'],
                        'Effective environment/Inductor hash differs')
                target = Path(module.resolve_flashinfer_autotune_file(runner))
                require(str(target) == self.settings['baseline_cache'] and
                        target.parent.name == self.settings['baseline_namespace'], 'FlashInfer namespace differs')
                require(sha(target) == self.settings['baseline_cache_sha256'], 'Original42-plan cache changed before warmup')
                expected = main_plans(json.loads(target.read_text()))
                require(len(expected) == 42, 'Expected42 main MoE plans')
                from flashinfer.autotuner import AutoTuner
                require(sha(inspect.getsourcefile(AutoTuner)) == self.settings['autotuner_source_sha256'],
                        'AutoTuner implementation differs')
                # Early memory profiling used the original mode0. Switch once,
                # only after actual model routing and baseline cache admission.
                require(int(self.native.arc3_private_configure_v1(self.settings['mode'], True)) == 0 and
                        int(self.native.arc3_private_get_modes_v1()) == self.settings['mode'],
                        'Fixed native mode did not apply before warmup')
                self.mode_fixed = True
                self.record('mode-fixed', native_mode_readback=self.settings['mode'],
                            mode_set_after_actual_router_admission=True)
                tuner = AutoTuner.get()
                original_load = tuner.load_configs
                instance_load = 'load_configs' in tuner.__dict__
                def restore_load():
                    if instance_load:
                        tuner.load_configs = original_load
                    else:
                        tuner.__dict__.pop('load_configs', None)
                def load(path):
                    require(not observed and Path(path) == target, 'Unexpected/repeated official cache load')
                    observed.append(str(path)); restore_load()
                    require(sha(target) == self.settings['baseline_cache_sha256'], 'Cache changed during warmup')
                    accepted = original_load(path)
                    require(accepted is True and loaded_plans(tuner) == expected,
                            'Official AutoTuner did not load identical42 plans')
                    self.record('plans-loaded', loaded_main_plans_verified=True,
                                main_plan_count=42, main_plans=expected, actual_config=actual,
                                namespace=target.parent.name, cache_path=str(target))
                    return accepted
                tuner.load_configs = load
                result = original(runner)
                require(self.init_count == 1, 'Private native runner was not initialized exactly once')
                require(len(observed) == 1 and loaded_plans(tuner) == expected and
                        main_plans(json.loads(target.read_text())) == expected, 'Warmup altered42 effective plans')
                require(self.core.get_cutlass_fused_moe_module is self.original_getter and
                        int(self.native.arc3_private_get_modes_v1()) == self.settings['mode'], 'Startup module/mode changed')
                self.record('warmup-verified', after_warmup_main_plans_verified=True,
                            post_warmup_cache_sha256=sha(target), native_mode_readback=self.settings['mode'])
                return result
            except BaseException as error:
                self.record('failed', error=type(error).__name__ + ': ' + str(error), process_must_exit=True)
                raise
            finally:
                if tuner is not None and original_load is not None:
                    if instance_load:
                        tuner.load_configs = original_load
                    else:
                        tuner.__dict__.pop('load_configs', None)
        module.flashinfer_autotune = warmup
        module._arc3_private_startup = self


class ModuleProxy:
    """Only setup-time init/include calls are checked; returns the real runner."""
    def __init__(self, state, native, include_root):
        self.state, self.native, self.include_root = state, native, include_root.resolve()
        self.include_seen = False
        self.interleave_moe_weights_for_sm90_mixed_gemm = native.interleave_moe_weights_for_sm90_mixed_gemm

    def set_deepgemm_jit_include_dirs(self, dirs):
        require(not self.include_seen and dirs == [str(self.include_root)] and self.include_root.is_dir(),
                'Original DeepGEMM include root differs')
        self.include_seen = True
        return self.native.set_deepgemm_jit_include_dirs(dirs)

    def init(self, *args):
        import torch
        require(self.include_seen and self.state.init_count == 0, 'Unexpected/repeated native initialization')
        require(args == (torch.uint8, torch.int64, torch.bfloat16, False, False, False, False, False),
                'Native constructor differs from actual-bank validation')
        expected_mode = self.state.settings['mode'] if self.state.mode_fixed else 0
        require(int(self.native.arc3_private_get_modes_v1()) == expected_mode, 'Native mode changed')
        raw = self.native.init(*args)
        require((int(raw.get_gemm1_tactic_count()), int(raw.get_gemm2_tactic_count()),
                 int(raw.get_tactic_num())) == (20, 20, 40), 'Native enumeration differs')
        self.state.init_count += 1
        self.state.record('native-initialized', native_init_count=1,
                          native_init=[str(a) for a in args[:3]] + list(args[3:]),
                          native_mode_readback=expected_mode)
        return raw


def initialized(settings, path):
    global STATE
    require(STATE is None, 'Startup adapter already initialized')
    STATE = State(settings, path)
    return STATE
