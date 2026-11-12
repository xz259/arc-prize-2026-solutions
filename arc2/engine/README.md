# Engine files in the ARC-AGI-2 package that differ from `vllm/`

The ARC-AGI-2 package carries the ARC-AGI-3 engine. Its engine tree branched from ARC-AGI-3 at package v12, while [`../../vllm/`](../../vllm/) on `main` is the later v16. Of the engine files in the package, 555 are byte-identical to `main` and live only there. The 14 files below differ, and each sits here at the path it has under `vllm/` (or under `arc3/kaggle/`).

## Changed for the port

| File here | Counterpart | Change |
|---|---|---|
| `launch/k_launch_scaling_l4.py` | none (added) | The launcher for two pipeline stages of two tensor-parallel ranks on four GPUs. |
| `stage2/eprune.py` | `vllm/stage2/eprune.py` | Expert pruning under expert parallelism and pipeline stages; an optional placement order of the kept experts. |
| `stage2/kvrot.py` | `vllm/stage2/kvrot.py` | The centering means sliced to the KV heads that a tensor-parallel rank holds. |
| `stage2/kvfq.py`, `stage2/memopt.py`, `stage2/seloverlap.py` | the same names under `vllm/stage2/` | Only the pinned hash of `qsa.py`, which the package replaces in the runtime ([`../kaggle/runtime_patch/`](../kaggle/runtime_patch/)). |
| `stage2/splitfix.py` | `vllm/stage2/splitfix.py` | Only the pinned hash of `gpu_worker.py`, for the same reason. |

## Unchanged since v12

These seven are the v12 versions of files that ARC-AGI-3 changed afterwards, for v13 to v16. They are identical to the files at the tag `arc3-v12`.

| File here | Counterpart |
|---|---|
| `launch/k_launch_scaling_s2i.py` | `vllm/launch/k_launch_scaling_s2i.py` |
| `overlay/dense_fp8_adapter_v1/runtime.py` | `vllm/overlay/dense_fp8_adapter_v1/runtime.py` |
| `overlay/mtp0x_v1/install.py` | `vllm/overlay/mtp0x_v1/install.py` |
| `overlay/overlay-manifest.json` | `vllm/overlay/overlay-manifest.json` |
| `overlay/gameplay-canary-pins.json` | `vllm/overlay/gameplay-canary-pins.json` |
| `stage2/eprune_k16.json` | `vllm/stage2/eprune_k16.json` |
| `../kaggle/kaggle_defs/KDEF.json` | `arc3/kaggle/kaggle_defs/KDEF.json` |
