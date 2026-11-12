"""Import and tiny CUDA smoke check; never loads model weights."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

root = Path(os.environ.get("ARC3_ROOT", Path(__file__).resolve().parent))
site = (root / "runtime/base/usr/local/lib/python3.12/dist-packages").resolve()
report = {"python": sys.version, "executable": sys.executable,
          "host": platform.node(), "timestamp": time.time(),
          "model_loaded": False, "checks": {}, "failures": []}
assert sys.version_info[:2] == (3, 12), "Runtime requires Python 3.12"
for name in ("torch", "vllm", "flashinfer", "cutlass", "transformers",
             "vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch",
             "vllm.v1.ple_offload.worker", "vllm.v1.ple_offload.connector",
             "vllm.models.qwen3_8_flash_next.nvidia.model",
             "vllm.models.qwen3_8_flash_next.nvidia.mtp"):
    try:
        print(f"Checking import: {name}", flush=True)
        module = importlib.import_module(name)
        source = Path(module.__file__).resolve()
        assert source.is_relative_to(site), f"Wrong import source: {source}"
        report["checks"][name] = {"file": str(source),
            "version": getattr(module, "__version__", None)}
    except Exception:
        report["failures"].append({"check": name, "traceback": traceback.format_exc()})
try:
    import torch
    assert importlib.metadata.version("vllm") == "0.1.dev20073+g8e685d198"
    assert torch.cuda.is_available()
    x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    result = (x @ x.T).cpu()
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torch.testing.assert_close(result, expected @ expected.T)
    report["checks"]["cuda"] = {"gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch_cuda": torch.version.cuda, "tiny_matmul": "passed"}
except Exception:
    report["failures"].append({"check": "cuda", "traceback": traceback.format_exc()})
for name, command in {"nvcc": ["nvcc", "--version"],
                      "vllm_cli": [sys.executable, "-m", "vllm.entrypoints.cli.main", "--help"]}.items():
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=300)
        report["checks"][name] = {"returncode": result.returncode, "output": result.stdout[-12000:]}
        if result.returncode:
            report["failures"].append({"check": name, "returncode": result.returncode})
    except Exception:
        report["failures"].append({"check": name, "traceback": traceback.format_exc()})
report["status"] = "passed" if not report["failures"] else "failed"
(root / "evidence").mkdir(exist_ok=True)
(root / "evidence/runtime-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
raise SystemExit(bool(report["failures"]))
