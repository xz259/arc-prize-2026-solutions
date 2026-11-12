#!/usr/bin/env bash
# Launch one Python process against the extracted runtime. No extraction or pip.
set -euo pipefail
ARC3_ROOT="${ARC3_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export ARC3_ROOT
arc3_runtime="$ARC3_ROOT/runtime/base"
arc3_site="$arc3_runtime/usr/local/lib/python3.12/dist-packages"
arc3_cuda="$arc3_runtime/usr/local/cuda-13.0"
arc3_python="$ARC3_ROOT/venv-runtime/bin/python"
test -f "$ARC3_ROOT/runtime/extraction.json"
test -x "$arc3_python"
"$arc3_python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "complete", "Runtime extraction is incomplete"' "$ARC3_ROOT/runtime/extraction.json"
export PYTHONPATH="$arc3_site/nvidia_cutlass_dsl/dsl_packages:$arc3_site"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME="$arc3_cuda" CUDACXX="$arc3_cuda/bin/nvcc"
export PATH="$arc3_cuda/bin:$arc3_runtime/usr/local/bin:$PATH"
arc3_libs="$arc3_cuda/targets/x86_64-linux/lib:$arc3_cuda/lib64:$arc3_site/torch/lib"
for arc3_lib in "$arc3_site"/nvidia/*/lib; do
    if [[ -d "$arc3_lib" ]]; then arc3_libs="$arc3_libs:$arc3_lib"; fi
done
export LD_LIBRARY_PATH="$arc3_libs:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME="$ARC3_ROOT/cache/runtime"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_CUDA_COMPATIBILITY=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
unset PYTORCH_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# Offload, RadixArk checkpoint gates, MTP, and cache choices belong to the future
# model launcher. This wrapper only selects the unchanged runtime environment.
exec "$arc3_python" "$@"
