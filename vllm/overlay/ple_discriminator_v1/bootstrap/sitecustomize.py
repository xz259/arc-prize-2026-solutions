"""Outermost bootstrap: run the chained bootstrap (dense-fp8 -> combined), then install the PLE discriminator hook."""
import os
import runpy

try:
    previous = os.environ.get('ARC3_PLE_DISC_CHAIN_BOOTSTRAP')
    if previous:
        runpy.run_path(previous, run_name='arc3_chained_sitecustomize')
    if os.environ.get('ARC3_PLE_DISCRIMINATOR', 'off') != 'off':
        from ple_discriminator_v1.import_hook import install
        install()
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(93)
