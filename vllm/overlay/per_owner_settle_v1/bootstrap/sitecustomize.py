import os
import runpy

try:
    previous = os.environ.get('ARC3_POS_CHAIN_BOOTSTRAP')
    if previous:
        runpy.run_path(previous, run_name='arc3_chained_sitecustomize')
    if os.environ.get('ARC3_PER_OWNER_SETTLE', '0') == '1':
        from per_owner_settle_v1.import_hook import install
        install()
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(96)
