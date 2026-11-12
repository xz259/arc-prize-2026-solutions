"""Outermost bootstrap: run the chained bootstrap (ple discriminator or dense-fp8 -> combined), then install the fold hook."""
import os
import runpy

try:
    previous = os.environ.get('ARC3_FOLD_CHAIN_BOOTSTRAP')
    if previous:
        runpy.run_path(previous, run_name='arc3_chained_sitecustomize')
    if os.environ.get('ARC3_FIXED_STATE_FOLD', 'off') != 'off':
        from fixed_state_fold_v1.import_hook import install
        install()
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(94)
