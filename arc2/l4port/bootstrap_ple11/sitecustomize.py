"""L4 port bootstrap v11 (2026-10-31): bootstrap_ple10/sitecustomize.py run unchanged by path (ple2's chain + vptopk, the PP drafter hook,
the PP warmup cap, the KV dump, the PP broadcast pad, the PP draft broadcast, the PP host embeddings), then ARC3_L4PORT_PP_SPLIT=a,b
installs l4port_ppsplit.py (an uneven layer split for the two main PP stages only). Put this directory first on PYTHONPATH."""
import importlib.util
import os
import runpy
import traceback

try:
    runpy.run_path("/mnt/shared/arc2/l4port/bootstrap_ple10/sitecustomize.py", run_name="l4port_bootstrap_ple10")
    if os.environ.get("ARC3_L4PORT_PP_SPLIT"):
        _spec = importlib.util.spec_from_file_location("l4port_ppsplit", "/mnt/shared/arc2/l4port/bootstrap_ple11/l4port_ppsplit.py")
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.install()
except BaseException:
    traceback.print_exc()
    os._exit(93)
