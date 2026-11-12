"""L4 port bootstrap (2026-10-30): production's PLE copy handoff for TP=4 trials, then the next sitecustomize (the stage-2
bootstrap) when L4_NEXT_SITECUSTOMIZE names one. The copy handoff (runs/dense_fp8_v1/ple_copy_handoff_v1, source-pinned
import hook, original b911d00c -> patched 5c9e6822) makes the PLE connector acknowledge that TP0's input copies are enqueued
before the consumer graph launches; with the stock connector the graph trials hung right after capture. The hook module is
loaded by file path (the overlay directory is not put on sys.path). Receipts: ARC3_PLE_HANDOFF_RECEIPTS/worker-<pid>.json."""
import importlib.util
import os
import runpy
import traceback

try:
    _spec = importlib.util.spec_from_file_location(
        "l4port_ple_copy_hook", "/workspace/arc3-inference/runs/dense_fp8_v1/ple_copy_handoff_v1/import_hook.py")
    _hook = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_hook)
    _hook.install()
    _next = os.environ.get("L4_NEXT_SITECUSTOMIZE", "")
    if _next:
        runpy.run_path(_next, run_name="l4port_next_sitecustomize")
    if os.environ.get("ARC3_L4PORT_PP_SKIPMIXER") == "1":      # PP weight loading (l4port_pp.py): installed last so its finder
        _spec2 = importlib.util.spec_from_file_location(        # is consulted first and wraps whatever loader the others return
            "l4port_pp", "/mnt/shared/arc2/l4port/bootstrap_ple/l4port_pp.py")
        _pp = importlib.util.module_from_spec(_spec2)
        _spec2.loader.exec_module(_pp)
        _pp.install()
except BaseException:
    traceback.print_exc()
    os._exit(91)
