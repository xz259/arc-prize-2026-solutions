"""probe_instance.py (2026-10-11, two-instance generation): run a production probe script against one instance's server.
minimal_inference reads ARC3_PORT at import (set by the caller) and its IDENTITY is pointed at the instance's identity
file before the probe and its helpers (benchmark_serving_config, run_decode_profile_candidate) import it, so they talk
to that port and verify that server's processes. The probe runs as __main__ with the probe's directory first on sys.path,
as a direct invocation would. minimal_inference.py and the probe are not modified.
usage: ARC3_PORT=<port> python probe_instance.py IDENTITY_JSON PROBE.py [probe args ...]"""
import os
from pathlib import Path
import runpy
import sys

identity, probe = Path(sys.argv[1]).resolve(), os.path.abspath(sys.argv[2])
assert os.environ.get('ARC3_PORT', '').isdigit(), 'set ARC3_PORT'
assert identity.is_file(), identity
sys.path[0] = os.path.dirname(probe)   # as `python PROBE.py` would have it (not this wrapper's directory)
import minimal_inference  # noqa: E402
assert minimal_inference.PORT == int(os.environ['ARC3_PORT']), minimal_inference.PORT
minimal_inference.IDENTITY = identity
sys.argv = [probe] + sys.argv[3:]
runpy.run_path(probe, run_name='__main__')
