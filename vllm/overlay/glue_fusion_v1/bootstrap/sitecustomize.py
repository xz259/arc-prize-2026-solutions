import os
import runpy

try:
    previous = os.environ.get('ARC3_GF_CHAIN_BOOTSTRAP')
    if previous:
        runpy.run_path(previous, run_name='arc3_chained_sitecustomize')
    if os.environ.get('ARC3_GLUE_FUSION', '0') != '0':
        from glue_fusion_v1.import_hook import install
        install()
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(97)
