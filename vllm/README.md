# The inference engine

This directory holds the engine that served both solvers: the vLLM build we started from and everything we changed in it. Section 2 of the [ARC-AGI-3 write-up](../arc3/docs/writeup.md) describes the design; this page says where each piece lives.

## The base build

- vLLM `0.1.dev20073+g8e685d198`, a development build shipped in the public image `vllm/vllm-openai:qwen38-flash-next`. [`base/reference/runtime-manifest.json`](base/reference/runtime-manifest.json) pins the image by digest and lists the layers we extract.
- With it: Python 3.12.12, PyTorch 2.13.0 (CUDA 13.0), FlashInfer 0.6.17, Triton 3.7.1, Transformers 5.15.1.

[`vllm/`](vllm/) is the Python source of that build's `vllm` package (2,473 modules and its small data files) with our in-place patches applied. The commit that adds it holds the unmodified source, checked file by file against the hashes recorded when the runtime was extracted. The next commit applies the patches, so the history of `vllm/vllm` shows exactly what they change; [`base/patches/diffs/`](base/patches/diffs/) holds the same changes as diff files.

The compiled extensions and the bundled CUDA headers are not included. [UPSTREAM_METADATA.txt](UPSTREAM_METADATA.txt) is the installed package's metadata.

## The layers

Our changes come in three layers, applied in this order. Each directory has its own README with the details.

| Directory | What it holds | How it is applied |
|---|---|---|
| [`base/`](base/) | Short patches to 12 files of the build (11 in vLLM, one in FlashInfer), the helper modules they import, the radix top-k extension, and the scripts that fetch and prepare the runtime. | Once, when the runtime is prepared. Each patch script checks the hash of the file it changes. |
| [`overlay/`](overlay/) | Persistent sessions and their endpoint, the session service, eviction, the KV layout, the PLE handoff, and the dense-layer kernels. | At server start, through vLLM's plugin entry points and through import hooks. No file of the runtime is modified. |
| [`stage2/`](stage2/) | Speculative decoding with single-slot rollback, the 4-bit KV cache with its exact window, the scaled FP16 recurrent state, the scheduler and memory changes, and the server side of the harness's eviction, rebase and parking policies. | At launch, as in-memory patches chosen by the `ARC3_STAGE2` list. Each part is pinned to the hash of the source it patches. |

[`launch/`](launch/) holds the launchers that assemble the three layers into one server.

Every server process starts through a chain of `sitecustomize.py` files. The one in `stage2/bootstrap/` is the entry: it runs the overlay's chain and then installs the stage-2 parts. Each link registers import hooks, and a hook patches its module right after vLLM imports it.

## Where the write-up's pieces are

| Write-up | Files |
|---|---|
| Persistent sessions (§1, item 1) | `overlay/persistent_endpoint.py`, `overlay/session_service.py`, `overlay/engine_patch.py`. The client is `../arc3/harness/v62/transport.py`. |
| Attention-guided eviction (§1, item 1) | `overlay/qsa_frequency.py` (the selection counts), `overlay/history_eviction.py` (the plan), `overlay/eviction_runner.py`, `overlay/engine_eviction*.py` and `overlay/eviction_storage.py` (carrying it out). `stage2/agecap.py` is the 120K age cap, `stage2/hollowgroup.py` the block rule, `stage2/sumpin.py` the pinned notes and summary. |
| Rebase and the moving system prompt (§1, item 3) | `stage2/rebase74.py`, `stage2/rebaseobs.py`, `stage2/promptmove.py`, `stage2/rope2m.py` |
| Parking (§1, item 4) | The suspend and resume of a session in `stage2/rebase74.py`. The queue is in the harness. |
| Memory for more lanes (§2.1) | `stage2/draftring.py` (the drafter's KV outside the shared pool), `stage2/embedhost.py` (the embedding table in host memory), `overlay/fixed_state_fold_v1/` (the block layout), `stage2/poolpress.py` and `stage2/stallguard.py` (the two extra lanes) |
| PLE handoff (§2.1) | `overlay/ple_hostpoll_v1/` |
| Speculative decoding and rollback (§2.1) | `stage2/rollback.py`, `stage2/rb_kernels.py`, `stage2/wy_kernels.py`, `stage2/persist.py`, `stage2/draftqsa.py`, `stage2/drafthead.py` |
| Radix top-k (§2.1) | `base/kernels/`, `base/arc3_topk_radix.py` |
| Register-spill fix of the 4-bit reader (§2.1) | `base/arc3_qsa_profiles.py`, used by the readers `base/arc3_qsa_nvfp4_attention.py` and `stage2/kvlut_kernels.py` |
| Constant-time KV accounting (§2.1) | `stage2/schedfast.py` |
| Work overlapped with the running step (§2.1) | `stage2/overlap.py`, `stage2/prestage.py` |
| Split-K kernels for the dense layers (§2.1) | `overlay/glue_fusion_v1/` |
| 4-bit codebook KV (§2.2) | `stage2/kvrot.py`, `stage2/kvlut_kernels.py`, on the byte layout of `base/arc3_qsa_nvfp4.py` |
| Exact window and exact set (§2.2) | `stage2/kvring.py`, `stage2/kvring_kernels.py`, `stage2/kvring_fast.py`, `stage2/kvring_exact.py` |
| GDN recurrent state (§2.2) | `stage2/s16.py` (the scaled FP16 format and its stochastic rounding) |
| Exact speculative sampling (§2.2) | `stage2/vsampler.py`, `stage2/cand_verify.py`, `stage2/topkfix.py` |
| Determinism and the NLL fingerprint (§2.2) | `stage2/marlindet.py`, `base/arc3_determinism.py`, `launch/nll_panel_i.sh`, `launch/stage0-reference/` |
| Startup | `stage2/ckptfetch.py`, `stage2/loadskip.py` |
| Expert pruning and FP8 input projections (§3, in `arc3-v16`) | `stage2/eprune.py`, `stage2/eprune_k16.json`, `overlay/dense_fp8_adapter_v1/`, `overlay/mtp0x_v1/` |

The ARC-AGI-2 entry runs the same engine on four L4s. Its port and the engine files it changes are in [`../arc2/`](../arc2/).

## Upstream work in these layers

Some pieces follow or port changes that were open or merged upstream at the time:

- the MTP head's dispatch and layer remapping, adapted from vLLM PR 55513, and the FP8 PLE fix (`base/apply_nvidia_patches.py`);
- the radix selector of vLLM PR 55122, kept verbatim in `base/kernels/exact_topk_radix.cuh`;
- vLLM PR 58430, ported in `stage2/splitfix.py`, and vLLM PR 58784, backported in `stage2/vsampler.py`;
- the bounded shard prefetch of `stage2/ckptfetch.py` follows the design of a patch in @dfranzen's milestone-2 solution, and `stage2/lowm_kernels.py` (not used by the shipped profiles) builds on his low-M BF16 GEMM patch, itself derived from a patch by Gabriel Olympie.

## What is left out

Compiled objects (`base/kernels/build/`, the PLE poll kernel's build directory, jemalloc), compile and AOT caches, startup receipts, the scoring set behind the NLL fingerprint, the drafter's vocabulary list (see [`launch/README.md`](launch/README.md)), and backup copies of files that sat beside the live ones in the package.

## Licence

Apache-2.0 ([LICENSE](LICENSE)), as vLLM: several overlay modules and stage-2 parts are modified copies of vLLM functions. The GDN rollback kernels (`stage2/rb_kernels.py` and the kernels built on it) derive from the flash-linear-attention kernels bundled with vLLM, which are MIT-licensed (`vllm/third_party/flash_linear_attention/LICENSE`).
