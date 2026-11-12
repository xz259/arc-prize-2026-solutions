"""ple_hostpoll_tp installer (L4 port, 2026-10-31, ARC2 - main; the user: "Let's do it", the PLE handoff without the offload
process's CUDA contexts). Called by bootstrap_l4prod2 (v7) with ARC3_L4PORT_PLE_HPTP=1, after the production chain head ran,
in every process (API server, workers, the PLE offload process). Needs ARC3_PLE_HANDOFF_IMPL=hostpoll, so the combined
bootstrap installed production's ple_hostpoll_v1 hook and the launcher tagged the caches '-plehostpoll'. Production's files
are read, never written; this redirects three things before vLLM imports the PLE modules:
  - ple_hostpoll_v1.import_hook.SOURCE -> this directory's connector.py (derive_tp_connector.py from production's);
  - sys.modules['ple_hostpoll_v1.worker_patch'] -> worker_patch_tp.py (production's patch + a CUDA-free accept);
  - ARC3_PLE_HOSTPOLL_BUILD -> ARC3_L4PORT_PLE_HPTP_BUILD (default this directory's build_sm89: production's polling
    kernel source built for sm_89; the launcher points it at production's sm_120 build).
Production's receipt (ple-hostpoll-startup/worker-PID.json) gains 'l4port_tp'."""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = {"connector.py": "2e84b011ad02f9ac39bf4cfa5caaecad", "worker_patch_tp.py": "932407de25a08556f436e7dc700a5616"}


def install():
    assert os.environ.get("ARC3_PLE_HANDOFF_IMPL") == "hostpoll", "ple_hostpoll_tp needs ARC3_PLE_HANDOFF_IMPL=hostpoll"
    ih = sys.modules.get("ple_hostpoll_v1.import_hook")
    assert ih is not None and any(isinstance(f, ih.Finder) for f in sys.meta_path), "production's hostpoll hook is not installed"
    assert not {ih.CONNECTOR, ih.WORKER, ih.LAYER}.intersection(sys.modules), "ple_hostpoll_tp must precede vLLM's PLE imports"
    for name, md5 in PINS.items():
        got = hashlib.md5((HERE / name).read_bytes()).hexdigest()
        assert md5 is None or got == md5, (name, got, md5)
    build = Path(os.environ.get("ARC3_L4PORT_PLE_HPTP_BUILD", str(HERE / "build_sm89")))
    assert (build / "arc3_ple_hostpoll_v1.so").is_file(), ("no sm_89 polling-kernel build", str(build))
    os.environ["ARC3_PLE_HOSTPOLL_BUILD"] = str(build)
    ih.SOURCE = HERE / "connector.py"
    spec = importlib.util.spec_from_file_location("ple_hostpoll_v1.worker_patch", HERE / "worker_patch_tp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ple_hostpoll_v1.worker_patch"] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules["ple_hostpoll_v1"], "worker_patch", mod)
    ih.REPORT["l4port_tp"] = dict(connector=str(ih.SOURCE), worker_patch=str(HERE / "worker_patch_tp.py"), build=str(build),
                                  verify=os.environ.get("ARC3_L4PORT_PLE_HP_VERIFY", "0"),
                                  md5={n: hashlib.md5((HERE / n).read_bytes()).hexdigest() for n in PINS})
    ih._receipt()
