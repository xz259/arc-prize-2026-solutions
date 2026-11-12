#!/usr/bin/env bash
# One-time online Runpod preparation of the PROVIDED runtime. No model launch.
set -euo pipefail
ARC3_ROOT="${ARC3_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export ARC3_ROOT
mkdir -p "$ARC3_ROOT/bootstrap" "$ARC3_ROOT/evidence" "$ARC3_ROOT/logs"
export UV_PYTHON_INSTALL_DIR="$ARC3_ROOT/python"
export UV_CACHE_DIR="$ARC3_ROOT/cache/uv"
if [[ ! -x "$ARC3_ROOT/bootstrap/uv/bin/uv" ]]; then
    python3 -m venv "$ARC3_ROOT/bootstrap/uv"
    "$ARC3_ROOT/bootstrap/uv/bin/python" -m pip install 'uv==0.12.11'
fi
arc3_uv="$ARC3_ROOT/bootstrap/uv/bin/uv"
"$arc3_uv" --version > "$ARC3_ROOT/evidence/bootstrap-uv-version.txt"
"$arc3_uv" python install 3.12.12
if [[ ! -x "$ARC3_ROOT/venv-runtime/bin/python" ]]; then
    "$arc3_uv" venv --python 3.12.12 "$ARC3_ROOT/venv-runtime"
fi
python3 "$ARC3_ROOT/prepare_runtime.py"
bash "$ARC3_ROOT/runtime_python.sh" "$ARC3_ROOT/check_runtime.py"
