"""L4 port bootstrap v4 (2026-10-30): bootstrap_ple2/sitecustomize.py run unchanged by path (bootstrap_ple's chain, then vptopk when
ARC3_L4PORT_VPTOPK > 0), then ARC3_L4PORT_PP_MTP=1 installs bootstrap_ple3/l4port_ppmtp.py (the MTP drafter as a single-stage module
on the last PP rank) and ARC3_L4PORT_PP_WARMCAP=N installs l4port_ppwarm.py (the same warmup request count on every PP stage).
Put this directory first on PYTHONPATH (w3_try's L4_EXTRA_PREFIX) so its sitecustomize is the one Python imports."""
import importlib.util
import os
import runpy
import traceback


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install()


try:
    runpy.run_path("/mnt/shared/arc2/l4port/bootstrap_ple2/sitecustomize.py", run_name="l4port_bootstrap_ple2")
    if os.environ.get("ARC3_L4PORT_PP_MTP") == "1":
        _load("l4port_ppmtp", "/mnt/shared/arc2/l4port/bootstrap_ple3/l4port_ppmtp.py")
    if int(os.environ.get("ARC3_L4PORT_PP_WARMCAP", "0") or 0) >= 2:
        _load("l4port_ppwarm", "/mnt/shared/arc2/l4port/bootstrap_ple4/l4port_ppwarm.py")
except BaseException:
    traceback.print_exc()
    os._exit(93)
