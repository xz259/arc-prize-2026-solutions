# What we changed in vLLM

Three layers, in the order they are applied. The paths are relative to this directory.

## 1. Files patched in place (`base/`)

Each script verifies the hash of the original file, writes the patched one and records both hashes in `base/patches/`. The patches are small hooks into our helper modules, 2 to 44 added lines per file; `base/patches/diffs/` has one diff per file.

| Script | Patch | Files |
|---|---|---|
| `apply_nvidia_patches.py` | `nvidia-ple-mtp-v1`: loader fixes for the NVIDIA checkpoint, the FP8 PLE embedding (following the upstream fix) and the MTP head's dispatch and layer remapping (adapted from vLLM PR #55513) | `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`, `vllm/models/qwen3_8_flash_next/nvidia/mtp.py`, `vllm/model_executor/layers/quantization/modelopt.py` |
| `apply_qsa_quantization.py` | `qsa-quantized-main-v2`: opt-in quantized storage for the main QSA keys and values (FP8, NVFP4), with every scale byte counted in the KV pages | `vllm/models/qwen3_8_flash_next/nvidia/qsa.py`, `vllm/models/qwen3_8_flash_next/common/qsa_cache.py` |
| `apply_determinism_diagnostics.py` | `qsa-moe-diagnostics-v2`: hooks for an exact reference and a radix implementation of the QSA selection, and for the MoE finalizer modes; the shipped profiles select the radix one | `vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py`, `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`, `flashinfer/fused_moe/core.py` |
| `apply_mtp_min_p.py` | `mtp-min-p-v1`: `min_p` under speculative decoding, opt-in (the shipped profiles sample with `min_p` 0) | `vllm/sampling_params.py`, `vllm/v1/sample/logits_processor/__init__.py`, `vllm/v1/sample/rejection_sampler.py` |
| `apply_sampler_topk_guard.py` | a guarded top-k / top-p kernel for the sampler, selected by a startup flag (`arc3_sampler_topk_guard*.py`) | `vllm/v1/sample/ops/topk_topp_sampler.py` |

## 2. The overlay (`overlay/`)

| Component | Role |
|---|---|
| `persistent_endpoint.py`, `protocol.py` | the endpoint (`/arc3/persistent/start`, `/{session}/append`, `/{session}/close`) and the chat rendering around a raw-token session |
| `session_service.py`, `residency.py`, `managed_spans.py`, `append_only_history.py` | one session per game: its token history, what is resident, and its budget |
| `engine_patch.py`, `persistent_batch_policy.py`, `control_transaction.py` | the engine side of a persistent request |
| `qsa_frequency.py`, `history_eviction.py` | selection counts from the QSA indexers and the eviction plan built from them |
| `eviction_runner.py`, `eviction_qsa.py`, `engine_eviction.py`, `engine_eviction_async.py`, `eviction_storage.py` | carrying an eviction out: original positions kept, 4-token groups handled, KV rows compacted while the other lanes keep stepping |
| `fixed_state_fold_v1/`, `per_owner_settle_v1/` | the KV block layout, and stopping one lane without draining the others |
| `ple_hostpoll_v1/` (on `ple_direct_handoff_v1/`, `ple_copy_handoff_v1/`) | the PLE handoff through a pinned buffer and a polled flag |
| `glue_fusion_v1/`, `dense_fp8_adapter_v1/`, `mtp0x_v1/`, `dense_projection_adapter_v1/`, `private_moe/` | kernels for the dense layers and the decode sampler; FP8 weights for dense layers where a profile asks for them |
| `launch_scaling.py`, `launch_candidate.py`, `launch_optimized.py` | the launch chain the stage-2 launchers build on; the overlay reaches the engine and worker processes as a `PYTHONPATH` entry with its entry-point metadata |
| `test_*.py`, `*_checks.py`, `run_*.py`, `benchmark*.py` | tests, GPU checks and the controllers of early experiments |

## 3. Stage-2 parts (`stage2/`)

The profiles of both ARC-AGI-3 packages install these 43 parts (`ARC3_STAGE2` in `../arc3/kaggle/kaggle_defs/KDEF.json`); `82kx1632a30p16` adds `eprune`. The header of `stage2/install.py` describes each one.

| Group | Parts |
|---|---|
| Speculative decoding | `guards`, `rollback`, `wy`, `vsampler`, `persist`, `draftqsa`, `drafthead`, `draftwin`, `topkfix` |
| KV cache and recurrent state | `kvrot`, `kvring`, `draftring`; the exact set is switched on by `ARC3_S2_KVRING_EXACT` (`stage2/kvring_exact.py`) and the recurrent-state format by `ARC3_S2_GDN_STATE` (`stage2/s16.py`) |
| Eviction, rebase and positions | `agestat`, `agecap`, `hollowgroup`, `sumpin`, `rebaseobs`, `rebase74`, `promptmove`, `rope2m`, `padbudget` |
| Scheduling and overcommit | `schedfast`, `overlap`, `prestage`, `resumefast`, `poolpress`, `stallguard`, `deadlinefix` |
| Host-side work and memory | `ctxfree`, `topkshare`, `embedhost`, `pleconv`, `ctxtail`, `mmstrip`, `mmonly`, `sessmem`, `fqfast`, `agefix`, `splitfix` |
| Checkpoint formats and startup | `b1inc`, `marlindet`, `loadskip`, `ckptfetch` |

The directory also holds modules outside that list: diagnostics switched on by their own environment variables (the step timer, the canary, memory and host profilers), levers that were measured and set aside (for example `kvq4`, the KV format with scales shared by 4-token groups) and the export tools for the drafter's training data (`replayfx`, `dflashtaps`).
