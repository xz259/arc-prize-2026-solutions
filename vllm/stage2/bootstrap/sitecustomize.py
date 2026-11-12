"""stage2_v1 bootstrap: run the existing chain first, then install the selected stage-2 parts (ARC3_STAGE2)."""
import os
import runpy

try:
    if os.environ.get('ARC3_S2_PYCACHE'):     # bytecode cache outside the runtime tree (milestone 2 A10; exact)
        import sys
        sys.pycache_prefix = os.environ['ARC3_S2_PYCACHE']   # runtime_python.sh sets PYTHONDONTWRITEBYTECODE=1
        sys.dont_write_bytecode = False
    if (os.environ.get('ARC3_S2_MEMDIAG', '0') not in ('', '0')   # CUDA-init tracer and allocator history, before the chain
            or os.environ.get('ARC3_S2_MEMDIAG_RETRY', '0') == '1'):   # v7.5 C3: the retry counter alone (no history without MEMDIAG)
        from stage2_v1 import memdiag
        memdiag.install_early()
    previous = os.environ.get('ARC3_S2_CHAIN_BOOTSTRAP')
    if previous:
        runpy.run_path(previous, run_name='arc3_chained_sitecustomize')
    if os.environ.get('ARC3_STAGE2', '0') != '0':
        from stage2_v1.install import install
        install()
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(94)
