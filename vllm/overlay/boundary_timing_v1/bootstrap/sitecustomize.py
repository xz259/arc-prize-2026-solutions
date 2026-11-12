import os
import runpy

try:
    previous = os.environ.get('ARC3_BT_CHAIN_BOOTSTRAP')
    if previous:
        runpy.run_path(previous, run_name='arc3_chained_sitecustomize')
    if os.environ.get('ARC3_BOUNDARY_TIMING', '0') == '1':
        from boundary_timing_v1.import_hook import install
        install()
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(95)
