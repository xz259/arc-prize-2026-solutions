# ARC Prize 2026: ARC-AGI-3 and ARC-AGI-2

Code behind my ARC Prize 2026 entries on Kaggle.

| Track | Leaderboard | Hardware and time limit | Description |
|---|---|---|---|
| ARC-AGI-3 | 58.59 (4th on the public leaderboard) | one RTX PRO 6000, 9 hours | [write-up](arc3/docs/writeup.md), also [on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/writeups/public-lb-4th) |
| ARC-AGI-2 | 76.81 | four L4s, 12 hours | [arc2/README.md](arc2/README.md) |

Both entries run Intel's W4A16 AutoRound quantization of Qwen3.8-Flash-Next as an agent that works through a Python tool. The model itself is not trained; only the small MTP drafter used for speculative decoding is fine-tuned. Most of the work is in the inference engine and in how the harness shares it between tasks.

## Where to start

- **The method:** the ARC-AGI-3 write-up, [here](arc3/docs/writeup.md) or [on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/writeups/public-lb-4th), and the first section of [arc2/README.md](arc2/README.md) for ARC-AGI-2.
- **The engine:** [vllm/README.md](vllm/README.md) maps each technique of the write-up to its files. The 4-bit KV cache, the rollback of rejected drafts and the scaled recurrent state are in [`vllm/stage2/`](vllm/stage2/); persistent sessions and eviction are in [`vllm/overlay/`](vllm/overlay/).
- **The harnesses:** [`arc3/harness/`](arc3/harness/) plays the games, and [`arc2/harness/`](arc2/harness/) solves the puzzles; the second has tests and a dry run that work without a server.

Every directory named here has a README that says what its files do.

## Layout

| Directory | Contents |
|---|---|
| [`arc3/`](arc3/) | The ARC-AGI-3 solver: the game-playing harness, the Kaggle notebook and driver, the write-up and the validation run index. |
| [`arc2/`](arc2/) | The ARC-AGI-2 solver: its harness, the port of the engine to four L4s, and the Kaggle notebook and driver. |
| [`vllm/`](vllm/) | The inference engine both solvers use: the vLLM build we started from and our changes to it. |
| [`training/`](training/) | Fine-tuning of the MTP drafter. |

## The inference engine

The engine is the vLLM development build shipped in the public image `vllm/vllm-openai:qwen38-flash-next`, with three layers of changes, applied in this order:

1. [`vllm/base/`](vllm/base/): 12 files of the build patched in place, the helper modules they import, and the scripts that prepare the runtime from the image.
2. [`vllm/overlay/`](vllm/overlay/): modules that hook into vLLM through its plugin entry points and through import hooks. They add persistent sessions, the session service, eviction, and the changes to the dense layers and to the per-layer embedding (PLE) handoff.
3. [`vllm/stage2/`](vllm/stage2/): parts installed at launch as in-memory patches, each pinned to the hash of the source it replaces. They add speculative decoding with single-slot rollback, the 4-bit KV cache with its exact window, the scaled FP16 recurrent state, and the scheduler and memory changes.

[`vllm/vllm/`](vllm/vllm/) is the Python source of the build with the in-place patches applied; its first commit is the unmodified source.

## What is not here

- **The compiled runtime.** The engine needs the image above (PyTorch 2.13, CUDA 13.0, and the compiled extensions of vLLM and FlashInfer). `vllm/base/reference/download_runtime.py` fetches the image layers by digest and `vllm/base/prepare_runtime.py` extracts them.
- **Model weights.** The target is Intel's AutoRound W4A16 checkpoint of Qwen3.8-Flash-Next, with its PLE table resharded into 19 files for the startup read (`arc3/kaggle/build/reshard_ple.py`). The fine-tuned drafter and the KV centering means are not in the repository.
- **Games and puzzles.** The 25 public ARC-AGI-3 games and the ARC-AGI-2 tasks come from the competitions' data. The 30 custom games used for validation were made by @sonphamorg and @markbarney ([gallery](https://arc.markbarney.net/arc3/gallery)) and are not redistributed.
- **Caches and records.** The Kaggle packages also carried compile caches, kernel binaries and startup receipts. Those, the run logs and the game transcripts are left out.

So the servers do not run from a fresh clone. The repository holds the code as it was packaged for Kaggle, for reading and for reuse of its parts.

## Versions

`main` holds the last package of each track, ARC-AGI-3 v16 and ARC-AGI-2 v2k, with the documentation added on top. Each tag holds the code files of one Kaggle package unchanged, except that a path on my machine is replaced by `<project>` in four of them (a metadata field and a README).

| Tag | Package | Leaderboard |
|---|---|---|
| `arc3-v12` | arc3-code v12, profile `82kx1280` | the notebook built for it scored 58.59 and 58.28 |
| `arc3-v16` | arc3-code v16, profile `82kx1632a30p16` (FP8 attention input projections, 16 experts pruned per layer, 30+2 lanes) | 57.97 |
| `arc2-v2j`, `arc2-v2k` | arc2-code v2j and v2k (v2k adds a guard against repeated invalid tool calls) | 75.97 and 76.81 |

The best ARC-AGI-3 run is therefore the code at `arc3-v12`. v16 still defines `82kx1280` with the same settings and differs from v12 in nine files: the faster startup reads, the FP8 tier, the expert set and the new profiles.

```bash
git diff --stat arc3-v12 arc3-v16
```

## How the code was laid out when it ran

The code refers to absolute paths on our development pods. The Kaggle notebooks extract the packages at the same paths before they start the server.

| Repository path | Path on the pods and on Kaggle |
|---|---|
| `vllm/base/` | `/workspace/arc3-inference/` (`$ARC3_ROOT`) |
| `vllm/overlay/` | `/workspace/arc3-inference/runs/dense_fp8_v1/` |
| `vllm/stage2/` | `/workspace/kvring_dev/fin74/s2/stage2_v1/` |
| `vllm/launch/` | `/workspace/kvring_dev/fin74/s2/` |
| `arc3/harness/` | `/workspace/kvring_dev/fin74/s2/harness_v6/` |
| `arc3/kaggle/` | `/workspace/kvring_dev/kaggle/` |
| `arc2/harness/` | `/mnt/shared/arc2/harness_v2q/` |
| `arc2/l4port/` | `/mnt/shared/arc2/l4port/` |
| `arc2/kaggle/` | `/mnt/shared/arc2/kaggle/` |

## Names in the code

The code and its comments use the project's working names. The most common ones:

| Name | Meaning |
|---|---|
| lane | one concurrent session on the server |
| GDN, QSA | the model's Gated DeltaNet linear-attention layers and its query-sparse attention layers |
| PLE | the per-layer n-gram embedding table, kept in host memory |
| MTP, drafter | the model's multi-token-prediction head, used to draft tokens for speculative decoding |
| stage 2, `s2` | the parts installed at launch (`vllm/stage2/`) and the production tree that holds them |
| `fin74` | the name of the production tree |
| `lutc`, `kvrot`, `kvlut` | the 4-bit KV format: centered, Hadamard-rotated, with a 16-level codebook |
| `kvring`, exact set | the BF16 window over the newest tokens; the BF16 copies of the first prompt tokens and of the most-selected older tokens |
| `scaled16`, `scaled16sr` | the recurrent state as FP16 with FP32 scales, without and with stochastic rounding |
| `rollback`, `wy` | single-slot rollback of rejected drafts, and the chunked (WY) form of the delta rule |
| `hostpoll` | the PLE handoff through a pinned buffer and a flag that a GPU kernel polls |
| `promptmove` | re-rotating the system prompt's cached keys so that it stays just behind the oldest cached token |
| refresh, rebase, park | starting a level's conversation again from notes; rebuilding a conversation from position 0; suspending a game so that a waiting one gets its lane |
| profile | a named serving configuration, for example `82kx1280`: an 82,528-token context with the exact set (`x`) and a 1,280-token window |
| KDEF | the frozen definition from which the Kaggle driver starts the server and the harness |
| pass (ARC-AGI-2) | one independent attempt at a puzzle, in its own session and Python kernel |

The comments also carry working notes: dates, run names, and quotes of my instructions to the coding agents ("the user: ..."). The solution was built with Codex and Claude Code, and the notes are left as they were.

## Third-party code and licences

| Path | Licence | Origin |
|---|---|---|
| `vllm/` | Apache-2.0 ([vllm/LICENSE](vllm/LICENSE)) | vLLM, and our layers on it, which replace vLLM functions with modified copies. Details and the upstream pull requests that some pieces follow are in [vllm/README.md](vllm/README.md). |
| `arc2/engine/`, `arc2/l4port/`, `arc2/kaggle/runtime_patch/` | Apache-2.0 | The engine code of the ARC-AGI-2 port, likewise derived from vLLM. |
| `arc3/harness/` | Apache-2.0 (`arc3/harness/src/LICENSE.dfranzen`) | `src/` is @dfranzen's milestone-2 solution ([da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution) at `10882e3`), unmodified: his agent, built on Tufa Labs' Duck harness, and Tufa Labs' framework `taaf`. Our layers patch it and in places carry modified copies of its code and prompts. |
| `training/mtp_drafter/kit/qwen38_mtp.py` | Apache-2.0 | A port of vLLM's model code for the MTP head. |
| `vllm/launch/draft_vocab_32k.json` | none stated at the source | The drafter's vocabulary subset, `configs/draft_vocab_32k.json` from the public Kaggle dataset `lordhansolo/taaf-kaggle-source` (a milestone-2 solution), unchanged. |
| everything else | MIT-0 ([LICENSE](LICENSE)) | Ours: the ARC-AGI-2 harness, the Kaggle kits, the training and analysis scripts, the documents. |

Some GDN kernels in `vllm/stage2/` derive from the flash-linear-attention kernels that vLLM bundles (MIT; its licence is in `vllm/vllm/third_party/flash_linear_attention/`).
