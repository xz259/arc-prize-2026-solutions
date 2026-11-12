"""mtp0x_v1 bootstrap: run the existing chain first, then install the selected test parts (ARC3_MTP0X)."""
import os
import runpy

try:
    previous = os.environ.get('ARC3_MTP0X_CHAIN_BOOTSTRAP')
    if previous:
        runpy.run_path(previous, run_name='arc3_chained_sitecustomize')
    if os.environ.get('ARC3_MTP0X', '0') != '0':
        from mtp0x_v1.install import install
        install()
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(93)
