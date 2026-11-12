"""Outermost bootstrap: run the existing combined bootstrap chain, then install the dense-fp8 hook on top."""
import os
import runpy

try:
    previous = os.environ.get('ARC3_DENSE_FP8_CHAIN_BOOTSTRAP')
    if previous:
        runpy.run_path(previous, run_name='arc3_chained_sitecustomize')
    if os.environ.get('ARC3_DENSE_FP8_TIER', 'off') != 'off':
        from dense_fp8_adapter_v1.import_hook import install
        install()
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(92)
