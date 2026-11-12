# The base layer

The first layer of the engine: how the runtime is taken from the public image, the 12 files of it that we patch in place, and the helper modules those patches import. On the pods this directory is `/workspace/arc3-inference` (`$ARC3_ROOT`).

## Preparing the runtime

| File | Role |
|---|---|
| `reference/runtime-manifest.json` | Pins the image `vllm/vllm-openai:qwen38-flash-next` by digest and lists the six layers we use (7.9 GB compressed). |
| `reference/download_runtime.py` | Fetches those layers from the registry by digest and checks their hashes. |
| `prepare_runtime.sh`, `prepare_runtime.py` | Extract the layers into `runtime/base` and create the Python 3.12 environment. Nothing is patched at this step. |
| `check_runtime.py` | An import check and a small CUDA check; it loads no model. |
| `runtime_python.sh` | Runs one Python process against the extracted runtime. The pod launchers start through it; [`../launch/k_runtime_python.sh`](../launch/k_runtime_python.sh) is its Kaggle copy. |

## Files patched in place

Each script checks the hash of the original file, writes the patched one and records both hashes in `patches/*.json`. The patches are short hooks into the helper modules below, between 2 and about 40 added lines per file. `patches/diffs/` holds one diff per file, and [`../vllm/`](../vllm/) is the patched source.

- **`apply_nvidia_patches.py`.** Loader fixes for the NVIDIA checkpoint: the FP8 PLE embedding (following the upstream fix), and the MTP head's dispatch and layer remapping (adapted from vLLM PR 55513). It changes:
  - `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`
  - `vllm/models/qwen3_8_flash_next/nvidia/mtp.py`
  - `vllm/model_executor/layers/quantization/modelopt.py`
- **`apply_qsa_quantization.py`.** Quantized storage for the keys and values of the QSA layers (FP8 or the 4-bit NVFP4 layout), chosen by `ARC3_QSA_KV_FORMAT`, with every scale byte counted in vLLM's KV pages. The shipped profiles use the NVFP4 layout, which the stage-2 part `kvrot` fills with the codebook format. It changes:
  - `vllm/models/qwen3_8_flash_next/nvidia/qsa.py`
  - `vllm/models/qwen3_8_flash_next/common/qsa_cache.py`
- **`apply_determinism_diagnostics.py`.** Hooks that let a launch choose the QSA selection kernel (`ARC3_QSA_MODE`: vLLM's own, an exact reference, or the radix selector) and the MoE finalizer mode. The shipped profiles run the radix selector and the unfused finalizer. It changes:
  - `vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py`
  - `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`
  - `flashinfer/fused_moe/core.py` (in FlashInfer)
- **`apply_mtp_min_p.py`.** `min_p` sampling under speculative decoding, behind `ARC3_MTP_MIN_P=1`. The launchers set the switch, but the harness samples with `min_p` 0, so it does not affect the submitted runs. It changes:
  - `vllm/sampling_params.py`
  - `vllm/v1/sample/logits_processor/__init__.py`
  - `vllm/v1/sample/rejection_sampler.py`
- **`apply_sampler_topk_guard.py`.** With `ARC3_SKIP_REDUNDANT_TOPK=1`, the sampler imports a copy of vLLM's Triton top-k / top-p kernel that skips the top-k search for rows with at most k finite logits (`arc3_sampler_topk_guard*.py`). The patch stays applied, but in the shipped profiles the stage-2 part `topkfix` replaces the masking function, so neither Triton kernel runs. It changes:
  - `vllm/v1/sample/ops/topk_topp_sampler.py`

The runtime bundle on Kaggle carried these patches already applied, so the scripts are here to show how the files were changed; `apply_sampler_topk_guard.py` reads a staging bundle that is not in the repository.

## Helper modules

| File | Role |
|---|---|
| `arc3_qsa_cache.py`, `arc3_qsa_quant.py` | The storage dispatch of the quantized QSA cache and its FP8 format. |
| `arc3_qsa_nvfp4.py`, `arc3_qsa_nvfp4_attention.py` | The 4-bit layout (148 bytes per key or value vector) and the sparse attention kernel that reads it. |
| `arc3_qsa_profiles.py` | The tile and warp settings of the quantized readers. The settings inherited from the BF16 reader spilled registers once the kernel also had to dequantize; these keep it spill-free (a large prefill went from 221 to 5 ms). |
| `arc3_determinism.py` | The selection and finalizer switches named above. |
| `arc3_topk_radix.py`, `kernels/` | The exact radix top-k for the QSA selection: `exact_topk_radix.cuh` is the selector of vLLM PR 55122, kept verbatim, and `exact_topk_extension.cu` our launcher for it (`build_exact_topk.py` compiles it; `exact_topk_provenance.json` records the source). |
| `arc3_sampler_topk_guard.py`, `arc3_sampler_topk_guard_impl.py` | The sampler kernel of the last patch. |
| `arc3_qsa_fp8_scale_placement.py` | An FP8 layout experiment that was never used for serving. |

## Tools of the first weeks

`minimal_inference.py` is the first launcher; the later launchers and the NLL probe still import its helpers, and the probe also imports `gameplay_config.py` and `benchmark_serving_config.py`. `game_allocation.py` holds the token allowances of the first harness. `run_quantized_serving_candidate.py`, `run_decode_profile_candidate.py`, `profile_monitor.py` and `worker_diagnostics.py` are the benchmark and profiling controllers of September.
