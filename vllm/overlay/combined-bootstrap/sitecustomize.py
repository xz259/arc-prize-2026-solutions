import os
import runpy
from pathlib import Path
try:
    _impl = os.environ.get('ARC3_PLE_HANDOFF_IMPL', 'copy')
    if _impl == 'hostpoll':
        from ple_hostpoll_v1.import_hook import install   # stage 5: direct staging + host-polled results
    elif _impl == 'direct':
        from ple_direct_handoff_v1.import_hook import install   # stage 5: direct staging, no request thread
    else:
        from ple_copy_handoff_v1.import_hook import install
    install()
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'dense_projection_adapter_v1'/'bootstrap'/'sitecustomize.py'), run_name='arc3_dense_ple_chain')
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(91)
