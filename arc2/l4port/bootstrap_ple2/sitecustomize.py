"""L4 port bootstrap v2 (2026-10-30; throughput work): bootstrap_ple/sitecustomize.py run unchanged by path (the PLE copy handoff,
the stage-2 bootstrap, the PP hooks), then optional throughput hooks: ARC3_L4PORT_VPTOPK=K installs l4port_vptopk.py (the
vocab-parallel top-K logits gather). Put this directory ahead of bootstrap_ple on PYTHONPATH (w3_try10's L4_EXTRA_PREFIX) so
its sitecustomize is the one Python imports; trials without it keep bootstrap_ple's."""
import importlib.util
import os
import runpy
import traceback

try:
    runpy.run_path("/mnt/shared/arc2/l4port/bootstrap_ple/sitecustomize.py", run_name="l4port_bootstrap_ple")
    if int(os.environ.get("ARC3_L4PORT_VPTOPK", "0") or 0) > 0:
        _spec = importlib.util.spec_from_file_location("l4port_vptopk", "/mnt/shared/arc2/l4port/bootstrap_ple2/l4port_vptopk.py")
        _vp = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_vp)
        _vp.install()
except BaseException:
    traceback.print_exc()
    os._exit(93)
