# The inference engine

This directory holds the engine that served both solvers: the vLLM build we started from and everything we changed in it. The design is described in section 2 of the [ARC-AGI-3 write-up](../arc3/docs/writeup.md); this page says where each piece lives.

## The base build

- vLLM `0.1.dev20073+g8e685d198`, a development build shipped in the public image `vllm/vllm-openai:qwen38-flash-next`. [`base/reference/runtime-manifest.json`](base/reference/runtime-manifest.json) pins the image by digest and lists the layers we extract.
- With it: Python 3.12.12, PyTorch 2.13.0 (CUDA 13.0), FlashInfer 0.6.17, Triton 3.7.1, Transformers 5.15.1.

[`vllm/`](vllm/) is the Python source of that build's `vllm` package (2,473 modules and its small data files), with our in-place patches applied. The commit that adds it holds the unmodified source, checked file by file against the hashes recorded when the runtime was extracted; the next commit applies the patches, so `git log -p -- vllm/vllm` shows exactly what they change. The same changes are in `base/patches/diffs/`. The compiled extensions and the bundled CUDA headers are not included. [UPSTREAM_METADATA.txt](UPSTREAM_METADATA.txt) is the installed package's metadata.

## The layers

Our changes come in three layers, applied in this order.

| Directory | Layer | How it is applied |
|---|---|---|
| [`base/`](base/) | The patches to 12 files of the build (11 in vLLM, one in FlashInfer), the helper modules they import (`arc3_*.py`), the exact top-k CUDA extension (`kernels/`), and the scripts that fetch and prepare the runtime. | Once, when the runtime is prepared: `prepare_runtime.py`, then the `apply_*.py` scripts. Each script checks the hash of the file it patches; `patches/*.json` record the hashes before and after. |
| [`overlay/`](overlay/) | Persistent sessions and their endpoint, the session service, history eviction, the KV layout, the PLE handoff, and the dense-layer kernels. | At server start, through vLLM's plugin entry points and import hooks. No file of the runtime is modified. `overlay-manifest.json` pins every overlay file by hash. |
| [`stage2/`](stage2/) | Speculative decoding with single-slot rollback, the 4-bit KV cache and its exact window, the scaled FP16 recurrent state, the scheduler and memory changes, and the policies that eviction and rebase need on the server. | At launch, as in-memory patches chosen by the `ARC3_STAGE2` list. Each part is pinned to the hash of the source it patches. The header of [`stage2/install.py`](stage2/install.py) describes every part. |

[`launch/`](launch/) holds the server launchers: `launch_scaling_s2i.py` for the pods and `k_launch_scaling_s2i.py` for Kaggle, which is the same launcher with every check that would stop the launch turned into a log line. The profiles they start are defined in [`../arc3/kaggle/kaggle_defs/KDEF.json`](../arc3/kaggle/kaggle_defs/KDEF.json).

[CHANGES.md](CHANGES.md) lists the patches, the overlay's components and the parts that ran in production.

## Where the write-up's pieces are

| Write-up | Files |
|---|---|
| Persistent sessions (§1, item 1) | `overlay/persistent_endpoint.py`, `overlay/session_service.py`, `overlay/engine_patch.py`; the client is `../arc3/harness/v62/transport.py` |
| Attention-guided eviction | `overlay/qsa_frequency.py` (selection counts), `overlay/history_eviction.py` (planning), `overlay/eviction_runner.py`, `overlay/engine_eviction*.py`, `overlay/eviction_storage.py`; `stage2/agecap.py` (the 120K age cap), `stage2/hollowgroup.py` (blocks), `stage2/sumpin.py` (the pinned notes and summary) |
| Rebase and the moving system prompt (§1, item 3) | `stage2/rebase74.py`, `stage2/rebaseobs.py`, `stage2/promptmove.py`, `stage2/rope2m.py` |
| Memory for more lanes (§2.1) | `stage2/draftring.py` (the drafter's KV outside the shared pool), `stage2/embedhost.py` (the embedding table in host memory), `overlay/fixed_state_fold_v1/`, `stage2/poolpress.py` and `stage2/stallguard.py` (the two extra lanes) |
| PLE handoff | `overlay/ple_hostpoll_v1/` |
| Speculative decoding and rollback | `stage2/rollback.py`, `stage2/rb_kernels.py`, `stage2/wy_kernels.py`, `stage2/persist.py`, `stage2/draftqsa.py`, `stage2/drafthead.py` |
| Profiling items | `base/kernels/exact_topk_extension.cu` (radix top-k), `stage2/kvlut_kernels.py` (the 4-bit reader), `stage2/schedfast.py` (KV accounting), `stage2/overlap.py` and `stage2/prestage.py` (work overlapped with the running step), `overlay/glue_fusion_v1/` (dense-layer kernels) |
| 4-bit codebook KV (§2.2) | `stage2/kvrot.py`, `stage2/kvlut_kernels.py`, and the NVFP4 layout they reuse in `base/arc3_qsa_nvfp4*.py` |
| Exact window and exact set | `stage2/kvring.py`, `stage2/kvring_kernels.py`, `stage2/kvring_fast.py`, `stage2/kvring_exact.py` |
| GDN recurrent state | `stage2/s16.py` (the scaled FP16 format and its stochastic rounding) |
| Exact speculative sampling | `stage2/vsampler.py`, `stage2/cand_verify.py`, `stage2/topkfix.py` |
| Determinism and the NLL fingerprint | `stage2/marlindet.py`, `base/arc3_determinism.py`, `launch/nll_panel_i.sh`, `launch/stage0-reference/` |
| Startup | `stage2/ckptfetch.py`, `stage2/loadskip.py` |
| Expert pruning and FP8 input projections (§3, in `arc3-v16`) | `stage2/eprune.py`, `stage2/eprune_k16.json`, `overlay/dense_fp8_adapter_v1/`, `overlay/mtp0x_v1/` |

The ARC-AGI-2 entry runs the same engine on four L4s. Its port and the engine files it changes are in [`../arc2/`](../arc2/).

## What is left out

Compiled objects (`kernels/build/`, the PLE poll kernel's build directory, jemalloc), compile and AOT caches, startup receipts, the fixed scoring set behind the NLL fingerprint (`launch/nll-panels/*/panel.json`; the `DERIVED.json` beside it records its hashes), and backup copies of files that sat beside the live ones in the package.

## Licence

Apache-2.0 ([LICENSE](LICENSE)), as vLLM. Several overlay modules and stage-2 parts are modified copies of vLLM functions. The GDN rollback kernels (`stage2/rb_kernels.py`) are derived from the flash-linear-attention kernels bundled with vLLM, which are MIT-licensed (`vllm/third_party/flash_linear_attention/LICENSE`).
