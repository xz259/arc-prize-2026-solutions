"""Optional CUDA inventory and bounded headroom tests for private Runpod profiling.

Loaded through vLLM's worker-extension API. Optional determinism controls require
separately installed diagnostic hooks and an idle, eager server.
"""
class WorkerDiagnostics:
    def inspect_autotune_namespaces(self):
        """Read config hashes and predict explicitly bounded environment changes."""
        from arc3_autotune_namespace_diagnostics import inspect_autotune_namespaces
        return inspect_autotune_namespaces(self.model_runner)

    def inspect_sampler_topk_guard(self):
        """Read the loaded sampler binding, source identity and adapter calls."""
        from arc3_sampler_guard_diagnostics import loaded_sampler_guard_status
        report = loaded_sampler_guard_status()
        runner = self.model_runner
        def class_name(value):
            return None if value is None else dict(
                module=type(value).__module__, name=type(value).__qualname__)
        report['loaded_runner'] = class_name(runner)
        report['loaded_sampler'] = class_name(getattr(runner, 'sampler', None))
        report['loaded_rejection_sampler'] = class_name(getattr(runner, 'rejection_sampler', None))
        return report

    def inspect_moe_backend_layout(self):
        """Read loaded MoE backend and registered weight/scale metadata."""
        from arc3_moe_backend_inventory import inspect_moe_backend_layout
        return inspect_moe_backend_layout(self)

    def arc3_qsa_reader(self):
        from arc3_qsa_cache import reader_status
        return reader_status()

    def inspect_shared_experts(self):
        """Read loaded overlap decisions; no tensor allocation or forward call.

        Shape-only decisions describe the current Python selector, not the
        contents of an existing CUDA graph. Change overlap settings at reload.
        """
        import hashlib
        import inspect
        import os
        from enum import Enum
        from pathlib import Path
        from types import SimpleNamespace
        import vllm.envs as envs

        rows = (1, 8, 32, 256, 257, 8192)
        source_files = {}

        def plain(value):
            if isinstance(value, Enum):
                return value.name
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            if isinstance(value, dict):
                return {str(key): plain(item) for key, item in value.items()}
            return str(value)

        def identity(value):
            target = value if (inspect.ismodule(value) or inspect.isroutine(value)
                               or inspect.isclass(value)) else type(value)
            name = getattr(target, '__qualname__', getattr(target, '__name__', None))
            result = dict(name=name, module=getattr(target, '__module__', None))
            try:
                source = inspect.getsourcefile(target)
            except TypeError:
                # Built-in classes/functions do not have a Python source file.
                source = None
            result['source_file'] = source
            if source is not None and source not in source_files:
                path = Path(source)
                if path.is_file():
                    data = path.read_bytes()
                    source_files[source] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                else:
                    source_files[source] = dict(unavailable='source file is absent')
            return result

        runner = self.model_runner
        draft_getter = getattr(runner, 'get_draft_model', None)
        if draft_getter is not None and not callable(draft_getter):
            raise TypeError('model_runner.get_draft_model is not callable')
        draft = draft_getter() if draft_getter is not None else None
        models = []
        for role, model in (('main', runner.model), ('draft', draft)):
            model_info = dict(role=role, present=model is not None, moe_layers=[])
            if model is None:
                models.append(model_info)
                continue
            model_info['class'] = identity(model)
            # named_modules walks wrappers as well as the actual model. A
            # registered child lives in _modules, not necessarily __dict__.
            for path, moe in model.named_modules():
                if ('_shared_experts' not in vars(moe)
                        and '_shared_experts' not in vars(moe).get('_modules', {})):
                    continue
                shared = moe._shared_experts
                entry = dict(module_path=path, object_id=id(moe), class_=identity(moe),
                             layer_name=moe.layer_name, shared_present=shared is not None,
                             enable_dbo=moe.enable_dbo, fse_fuse_gate=moe._fse_fuse_gate)
                if moe.gate is not None:
                    entry['gate'] = dict(class_=identity(moe.gate), dtype=str(moe.gate.weight.dtype))
                if shared is not None:
                    method = moe._quant_method
                    kernel = method.moe_kernel
                    pc = shared._moe_config.moe_parallel_config
                    # Read known selector properties directly. Their failures
                    # must propagate instead of masquerading as missing data.
                    entry['shared'] = dict(
                        class_=identity(shared), layer_class=identity(shared._layer),
                        selector=identity(shared._determine_shared_experts_order),
                        has_aux_stream=shared._stream is not None,
                        overlap_disabled=shared._disable_shared_experts_overlap,
                        mk_can_overlap=shared._mk_can_overlap_shared_experts(),
                        shard_sequence_parallel=getattr(shared._layer, 'shard_sequence_parallel', False),
                        parallel=dict(enable_eplb=pc.enable_eplb, all2all_backend=pc.all2all_backend,
                                      use_fi_nvl_two_sided_kernels=pc.use_fi_nvl_two_sided_kernels),
                        order_by_rows={str(n): shared._determine_shared_experts_order(
                            SimpleNamespace(shape=(n,))).name for n in rows})
                    entry['quant_method'] = identity(method)
                    entry['moe_kernel'] = (None if kernel is None else dict(
                        class_=identity(kernel), can_overlap_shared_experts=kernel.can_overlap_shared_experts))
                model_info['moe_layers'].append(entry)
            models.append(model_info)

        compilation = self.vllm_config.compilation_config
        env_names = ('VLLM_DISABLE_SHARED_EXPERTS_STREAM',
                     'VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD')
        return dict(
            schema_version=1, rows=list(rows), models=models,
            draft_accessor_present=draft_getter is not None,
            resolved_env={name: plain(getattr(envs, name)) for name in env_names},
            environment_overrides={name: os.environ.get(name) for name in env_names},
            environment_source=identity(envs), diagnostic_source=identity(WorkerDiagnostics.inspect_shared_experts),
            compilation=dict(enforce_eager=self.vllm_config.model_config.enforce_eager,
                             **{name: plain(getattr(compilation, name, None)) for name in
                                ('mode', 'cudagraph_mode', 'cudagraph_capture_sizes',
                                 'max_cudagraph_capture_size')}),
            source_files=source_files,
            note='Shape-only loaded Python selector decisions; source hashes identify on-disk files. '
                 'This does not inspect already-captured CUDA graphs or execute model tensors.')

    def arc3_prefill_profile(self, action='status', profile='stock', confirm_idle=False):
        """A bounded eager-only tuning control; no graph recapture or mutation."""
        import torch
        import arc3_determinism as diag
        if action == 'status':
            return dict(profile=diag.PREFILL_PROFILE,counters=dict(diag.PREFILL_COUNTERS))
        # Keep the RPC as conservative as the selector switch. A fresh graph
        # server uses its launch-time profile, never a mutated captured graph.
        if action != 'set' or profile not in {'stock','tile32'} or confirm_idle is not True:
            return {'error':'Invalid sparse-prefill tuning request'}
        if not self.vllm_config.model_config.enforce_eager:
            return {'error':'Sparse-prefill tuning requires an idle eager server'}
        if diag.PREFILL_MIN_ROWS != 1024:
            return {'error':'Unexpected sparse-prefill threshold'}
        torch.cuda.synchronize()
        diag.PREFILL_PROFILE=profile
        for key in diag.PREFILL_COUNTERS:diag.PREFILL_COUNTERS[key]=0
        return dict(profile=diag.PREFILL_PROFILE,counters=dict(diag.PREFILL_COUNTERS))

    def arc3_determinism(self, action='status', qsa='native', moe='fused', confirm_idle=False):
        # Return validation errors instead of raising through a worker RPC.
        import torch
        import arc3_determinism as diag
        if action == 'status':
            return diag.status()
        if confirm_idle is not True or not self.vllm_config.model_config.enforce_eager:
            return {'error': 'Mode changes/capture require an idle eager server'}
        if action not in {'set', 'capture', 'check_capture'} or qsa not in {'native', 'exact', 'radix'} or moe not in {'fused', 'unfused'}:
            return {'error': 'Invalid diagnostic arguments'}
        torch.cuda.synchronize()
        if action == 'set':
            diag.QSA_MODE = qsa
            diag.MOE_FUSED_FINALIZE = moe == 'fused'
            for key in diag.COUNTERS:
                diag.COUNTERS[key] = 0
        elif action == 'capture':
            diag.CAPTURE_PENDING = True
            diag.CAPTURE = None
        elif action == 'check_capture':
            return diag.check_captured_topk()
        return diag.status()

    def arc3_memory_snapshot(self):
        import torch
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        model = self.model_runner.model
        buffers = []
        for name, tensor in model.named_buffers():
            if tensor.is_cuda and tensor.numel()*tensor.element_size() >= 1024**2:
                buffers.append(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                                    bytes=tensor.numel()*tensor.element_size()))
        # Resolved worker specs are authoritative for state dtypes/page layout;
        # the frontend config can still contain unresolved 'auto' values.
        def plain(value):
            if value is None or isinstance(value,(str,int,float,bool)): return value
            if isinstance(value,(list,tuple)): return [plain(v) for v in value]
            if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
            return str(value)
        try:
            config = self.model_runner.kv_cache_config
            cache_inventory = dict(num_blocks=config.num_blocks, groups=[])
            for group in config.kv_cache_groups:
                spec=group.kv_cache_spec
                cache_inventory['groups'].append(dict(layer_names=list(group.layer_names),
                    spec_type=type(spec).__name__, spec=plain(vars(spec)), page_size_bytes=spec.page_size_bytes))
            cache_inventory['raw_tensors'] = [dict(shape=list(t.shape),dtype=str(t.dtype),
                bytes=t.numel()*t.element_size()) for t in self.model_runner.kv_caches]
            storages={}
            for view,tensor in zip(cache_inventory['raw_tensors'],self.model_runner.kv_caches):
                storage=tensor.untyped_storage(); key=(storage.data_ptr(),storage.nbytes())
                if key not in storages: storages[key]=len(storages)
                view['storage_id']=storages[key]
            cache_inventory['unique_storage_bytes']=sum(size for _,size in storages)
            cache_inventory['note']='Tensor views can alias; do not sum view bytes as physical allocation.'
        except Exception as exc:
            cache_inventory = {'error':f'{type(exc).__name__}: {exc}'}
        runtime_memory = {key:getattr(self,key,None) for key in
                          ('requested_memory','available_kv_cache_memory_bytes',
                           'peak_activation_memory','cudagraph_memory_estimate','total_consumed')}
        runtime_memory.update(model_loading_bytes=self.model_runner.model_memory_usage,
                              max_model_len=self.model_runner.model_config.max_model_len,
                              mm_ipc_gpu_memory_gb=getattr(self.model_runner.model_config.multimodal_config,
                                                          'mm_ipc_gpu_memory_gb',0))
        return dict(free_device_bytes=free, total_device_bytes=total,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    inactive_split_bytes=stats.get("inactive_split_bytes.all.current"),
                    allocator_retries=stats.get('num_alloc_retries'),
                    allocator_ooms=stats.get('num_ooms'),
                    buffers=sorted(buffers, key=lambda r:r["bytes"], reverse=True),
                    cache_inventory=cache_inventory, runtime_memory=plain(runtime_memory))

    def arc3_reset_memory_peaks(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return {"reset":True}

    def arc3_reserve_test_memory(self, gib):
        """Hold a bounded diagnostic allocation; it is not usable KV capacity."""
        import torch
        size = float(gib)
        if not 0 <= size <= 4:
            raise ValueError("Diagnostic reservation must be between zero and 4 GiB")
        if getattr(self, '_arc3_test_memory', None) is not None:
            raise RuntimeError("Release the existing diagnostic reservation first")
        try:
            self._arc3_test_memory = torch.empty(int(size*1024**3), dtype=torch.uint8, device='cuda')
            torch.cuda.synchronize()
            return {'reserved_bytes':self._arc3_test_memory.numel()}
        except torch.OutOfMemoryError as exc:
            self._arc3_test_memory = None
            return {'error':str(exc), 'reserved_bytes':0}

    def arc3_release_test_memory(self):
        import torch
        torch.cuda.synchronize()
        self._arc3_test_memory = None
        torch.cuda.empty_cache()
        return {'released':True}

    def arc3_zero_cache_for_diagnostic(self, confirm_idle_and_cache_reset=False):
        """Only for isolated tests after the engine's cache reset succeeds.

        No live requests may exist. This changes free cache contents, not weights.
        The caller must invalidate prefix hashes first, including on cache-on runs.
        """
        if confirm_idle_and_cache_reset is not True:
            raise ValueError('Caller must verify idle engine and successful prefix-cache reset')
        import torch
        torch.cuda.synchronize()
        seen=set();total=0
        for tensor in self.model_runner.kv_caches:
            storage=tensor.untyped_storage();key=(storage.data_ptr(),storage.nbytes())
            if key in seen:continue
            seen.add(key)
            # Covers padding too, without allocating a second copy of the pool.
            raw=torch.empty(0,dtype=torch.uint8,device=tensor.device).set_(storage)
            raw.zero_();total+=storage.nbytes()
        torch.cuda.synchronize()
        return {'zeroed_storage_bytes':total,'storages':len(seen)}
