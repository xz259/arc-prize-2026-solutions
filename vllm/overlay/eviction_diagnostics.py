"""Read-only metadata for the actual bound worker caches; no tensor values."""
from __future__ import annotations


VERSION="arc3-eviction-layout-v1"


def describe_cache(cache):
    if isinstance(cache,(tuple,list)):
        return {"type":type(cache).__name__,"items":[describe_cache(value) for value in cache]}
    result={"type":type(cache).__name__}
    if hasattr(cache,"shape") and hasattr(cache,"stride"):
        result.update(shape=list(cache.shape),stride=list(cache.stride()),
                      dtype=str(cache.dtype),device=str(cache.device),
                      storage_offset=int(cache.storage_offset()),data_ptr=int(cache.data_ptr()))
    return result


def spec_description(spec):
    result={"type":type(spec).__name__}
    for field in ("block_size","storage_block_size","compress_ratio","num_kv_heads","head_size"):
        value=getattr(spec,field,None)
        if value is not None:result[field]=int(value)
    if getattr(spec,"dtype",None) is not None:result["dtype"]=str(spec.dtype)
    return result


def arc3_eviction_layout(worker):
    """Expose metadata by existing collective_rpc without touching GPU state."""
    runner=worker.model_runner
    config_forward=runner.vllm_config.compilation_config.static_forward_context
    bound_forward=runner.compilation_config.static_forward_context
    allocated=list(getattr(runner,"kv_caches",()))
    allocated_descriptions=[describe_cache(cache) for cache in allocated]
    owners=[]
    for name,layer in sorted(config_forward.items()):
        if not hasattr(layer,"kv_cache"):continue
        cache=layer.kv_cache
        description=describe_cache(cache)
        owners.append({"layer_name":name,"layer_type":type(layer).__name__,"cache":description,
            "same_runtime_bound_layer":bound_forward.get(name) is layer,
            "runtime_bound_cache":describe_cache(getattr(bound_forward.get(name),"kv_cache",None)),
            "runner_cache_identity_indices":[i for i,other in enumerate(allocated) if other is cache],
            "runner_cache_same_view_indices":[i for i,other in enumerate(allocated_descriptions)
                if "data_ptr" in description and all(description.get(key)==other.get(key)
                    for key in ("data_ptr","shape","stride","dtype"))]})
    groups=[]
    for group_id,group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec=group.kv_cache_spec
        specs=getattr(spec,"kv_cache_specs",None) or {name:spec for name in group.layer_names}
        groups.append({"group_id":group_id,"spec":spec_description(spec),
                       "owners":[{"layer_name":name,"spec":spec_description(owner)}
                                 for name,owner in specs.items()]})
    return {"diagnostic":VERSION,"read_only":True,"tensor_values_read":False,
            "runner_type":type(runner).__name__,
            "same_compilation_forward_context":config_forward is bound_forward,
            "kernel_block_sizes":list(runner.kernel_block_sizes),"groups":groups,
            "bound_owners":owners,"allocated_caches":allocated_descriptions}


def install_worker_diagnostics():
    from vllm.v1.worker.worker_base import WorkerBase
    previous=getattr(WorkerBase,"_arc3_eviction_layout_version",None)
    if previous is not None:
        if previous!=VERSION:raise RuntimeError("Different eviction-layout diagnostic already installed")
        return {"installed":True,"version":VERSION,"already_installed":True}
    WorkerBase.arc3_eviction_layout=arc3_eviction_layout
    WorkerBase._arc3_eviction_layout_version=VERSION
    return {"installed":True,"version":VERSION}
