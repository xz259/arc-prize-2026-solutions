"""L4 port bootstrap v13 (2026-10-31): bootstrap_ple11/sitecustomize.py run unchanged by path (ple2's chain + vptopk, the PP drafter
hook, warmup cap, KV dump, broadcast pad, draft broadcast, host embeddings, layer split), then ARC3_L4PORT_DHEAD_TP=1 installs this
directory's l4port_dheadtp.py (v2: the drafter's head over production's 32K draft vocabulary split evenly over the TP ranks;
ARC3_L4PORT_DHEAD_EVEN=0 restores v1's split by vocab shard). Put this directory first on PYTHONPATH."""
import importlib.util
import os
import runpy
import traceback

try:
    runpy.run_path("/mnt/shared/arc2/l4port/bootstrap_ple11/sitecustomize.py", run_name="l4port_bootstrap_ple11")
    if os.environ.get("ARC3_L4PORT_DHEAD_TP") == "1":
        _spec = importlib.util.spec_from_file_location("l4port_dheadtp", "/mnt/shared/arc2/l4port/bootstrap_ple13/l4port_dheadtp.py")
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.install()
except BaseException:
    traceback.print_exc()
    os._exit(93)
