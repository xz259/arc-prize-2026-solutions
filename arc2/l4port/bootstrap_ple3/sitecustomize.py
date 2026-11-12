"""L4 port bootstrap v3 (2026-10-30): bootstrap_ple2/sitecustomize.py run unchanged by path (bootstrap_ple's chain, then vptopk when
ARC3_L4PORT_VPTOPK > 0), then ARC3_L4PORT_PP_MTP=1 installs l4port_ppmtp.py (the MTP drafter as a single-stage module on the last
PP rank). Put this directory first on PYTHONPATH (w3_try's L4_EXTRA_PREFIX) so its sitecustomize is the one Python imports."""
import importlib.util
import os
import runpy
import traceback

try:
    runpy.run_path("/mnt/shared/arc2/l4port/bootstrap_ple2/sitecustomize.py", run_name="l4port_bootstrap_ple2")
    if os.environ.get("ARC3_L4PORT_PP_MTP") == "1":
        _spec = importlib.util.spec_from_file_location("l4port_ppmtp", "/mnt/shared/arc2/l4port/bootstrap_ple3/l4port_ppmtp.py")
        _pm = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_pm)
        _pm.install()
except BaseException:
    traceback.print_exc()
    os._exit(93)
