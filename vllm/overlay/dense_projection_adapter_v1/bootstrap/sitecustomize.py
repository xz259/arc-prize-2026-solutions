"""Compose the selected MoE bootstrap with the common dense projection hook."""
import os
import runpy
import sys

if os.environ.get('ARC3_DENSE_EXPERIMENT') == '1':
    try:
        from dense_projection_adapter_v1.import_hook import install
        install()
        previous = os.environ.get('ARC3_DENSE_CHAIN_BOOTSTRAP')
        if previous:
            runpy.run_path(previous, run_name='arc3_chained_sitecustomize')
    except BaseException:
        import traceback
        traceback.print_exc()
        # Python otherwise swallows sitecustomize exceptions and serves an
        # unpatched model. Refuse startup when an experiment hook fails.
        os._exit(91)
