"""L4 port bootstrap v14 (2026-10-31): bootstrap_ple13/sitecustomize.py run unchanged by path (ple11's chain + the drafter head v2),
then ARC3_L4PORT_EVPP=1 installs l4prod/l4port_evpp.py (registered by name). In the stripped core cells only its EngineCore patch
acts: ARC3_L4PORT_PP_QUEUE=N caps the batch queue at N (the production path caps it at PP under persistent sessions; this
measures that cap's cost). Put this directory first on PYTHONPATH."""
import importlib.util
import os
import runpy
import sys
import traceback

try:
    runpy.run_path("/mnt/shared/arc2/l4port/bootstrap_ple13/sitecustomize.py", run_name="l4port_bootstrap_ple13")
    if os.environ.get("ARC3_L4PORT_EVPP") == "1":
        _spec = importlib.util.spec_from_file_location("l4port_evpp", "/mnt/shared/arc2/l4port/l4prod/l4port_evpp.py")
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        sys.modules["l4port_evpp"] = _m
        _m.install()
except BaseException:
    traceback.print_exc()
    os._exit(93)
