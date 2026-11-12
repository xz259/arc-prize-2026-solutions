"""Pinned-runtime import and hook installation checks, without loading a model.

Run this separately from unit tests which replace vLLM modules with CPU fakes.
No inference, socket connection or active-server mutation occurs.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys


def run(output):
    # runtime_python.sh deliberately replaces PYTHONPATH; add the already
    # verified custom kernel modules after selecting that runtime.
    sys.path.insert(1, os.environ['ARC3_ROOT'])
    import engine_patch
    import engine_eviction
    import eviction_runner
    import vllm
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.worker.gpu.mm.rope import RopeState
    from vllm.v1.worker.worker_base import WorkerBase

    assert importlib.metadata.version('vllm') == '0.1.dev20073+g8e685d198'
    first = engine_patch.install()
    hooks = [(Scheduler, 'add_request'), (Scheduler, '_update_request_as_session'),
             (EngineCore, 'arc3_evict_resident'), (GPUModelRunner, 'add_requests'),
             (GPUModelRunner, 'load_model'), (RopeState, 'init_prefill_positions'),
             (WorkerBase, 'arc3_eviction_layout')]
    installed = [getattr(cls, name) for cls, name in hooks]
    second = engine_patch.install()
    assert all(getattr(cls, name) is value for (cls, name), value in zip(hooks, installed))
    assert Scheduler._arc3_eviction_patch == engine_eviction.VERSION
    assert eviction_runner._INSTALLED is True
    signatures = {}
    for name in ('pause_scheduler_async', 'resume_scheduler_async',
                 'is_scheduler_paused_async', 'call_utility_async'):
        method = getattr(AsyncMPClient, name)
        assert inspect.iscoroutinefunction(method), name
        signatures[name] = str(inspect.signature(method))
    signatures['collective_rpc'] = str(inspect.signature(EngineCore.collective_rpc))
    root = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
               for name in ('engine_patch.py', 'engine_eviction.py', 'eviction_runner.py',
                            'eviction_qsa.py', 'eviction_storage.py', 'persistent_endpoint.py',
                            'eviction_diagnostics.py')}
    receipt = dict(passed=True, scope='Pinned-runtime imports, source pins, hook installation and idempotence only',
                   runtime_root=str(Path(vllm.__file__).parent), first=first, second=second,
                   signatures=signatures, source_sha256=sources)
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('passed', 'scope', 'signatures')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
