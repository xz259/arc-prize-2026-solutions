"""L4 port bootstrap v5 (2026-10-30): bootstrap_ple4/sitecustomize.py run unchanged by path (ple2's chain + vptopk, the PP drafter
hook, the PP warmup cap), then ARC3_L4PORT_KVDUMP=1 installs l4port_kvdump.py (prints vLLM's KV cache groups and per-worker
layout; read-only). Put this directory first on PYTHONPATH (w3_try's L4_EXTRA_PREFIX)."""
import importlib.util
import os
import runpy
import traceback

try:
    runpy.run_path("/mnt/shared/arc2/l4port/bootstrap_ple4/sitecustomize.py", run_name="l4port_bootstrap_ple4")
    if os.environ.get("ARC3_L4PORT_KVDUMP") == "1":
        _spec = importlib.util.spec_from_file_location("l4port_kvdump", "/mnt/shared/arc2/l4port/bootstrap_ple5/l4port_kvdump.py")
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.install()
except BaseException:
    traceback.print_exc()
    os._exit(93)
