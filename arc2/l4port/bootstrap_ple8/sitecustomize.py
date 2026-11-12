"""L4 port bootstrap v8 (2026-10-30): bootstrap_ple6/sitecustomize.py run unchanged by path (ple2's chain + vptopk, the PP drafter hook,
the PP warmup cap, the KV dump, the PP broadcast pad), then ARC3_L4PORT_PP_DRAFTBC=1 installs l4port_ppdraftbc.py (the MTP drafts
broadcast from the last PP stage to the others). Supersedes bootstrap_ple7's scheduler copy (placeholders in the PP path).
Put this directory first on PYTHONPATH."""
import importlib.util
import os
import runpy
import traceback

try:
    runpy.run_path("/mnt/shared/arc2/l4port/bootstrap_ple6/sitecustomize.py", run_name="l4port_bootstrap_ple6")
    if os.environ.get("ARC3_L4PORT_PP_DRAFTBC") == "1":
        _spec = importlib.util.spec_from_file_location("l4port_ppdraftbc", "/mnt/shared/arc2/l4port/bootstrap_ple8/l4port_ppdraftbc.py")
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.install()
except BaseException:
    traceback.print_exc()
    os._exit(93)
