"""L4 port bootstrap v10 (2026-10-30): bootstrap_ple9/sitecustomize.py run unchanged by path (ple2's chain + vptopk, the PP drafter hook,
the PP warmup cap, the KV dump, the PP broadcast pad, the PP draft broadcast), then ARC3_L4PORT_PP_EMBHOST=1 installs
l4port_ppembhost.py (the target's embedding on stage 0 and the drafter's own copy on the last stage moved to pinned host memory).
Put this directory first on PYTHONPATH."""
import importlib.util
import os
import runpy
import traceback

try:
    runpy.run_path("/mnt/shared/arc2/l4port/bootstrap_ple9/sitecustomize.py", run_name="l4port_bootstrap_ple9")
    if os.environ.get("ARC3_L4PORT_PP_EMBHOST") == "1":
        _spec = importlib.util.spec_from_file_location("l4port_ppembhost", "/mnt/shared/arc2/l4port/bootstrap_ple10/l4port_ppembhost.py")
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.install()
except BaseException:
    traceback.print_exc()
    os._exit(93)
