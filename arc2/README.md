# ARC-AGI-2 solver

The same model and engine as the ARC-AGI-3 entry, moved to Kaggle's four L4s, with a harness for static puzzles. The model is not trained on ARC tasks. The two final submissions scored 75.97 and 76.81 on the leaderboard.

## How it works

**Harness.** Each puzzle is attempted in independent *passes*. A pass is one persistent session on the server and one Python kernel that holds the puzzle. The model reads the grids as text, writes Python, tests a transformation against the demonstration pairs with `check(fn)` and records up to two answers with `submit(...)`. A pass ends when the model stops with a valid answer saved, when it stalls, or at 200,000 generated tokens. Every puzzle gets passes until two of them have ended with a valid answer, at most four. Time left after that goes to further passes and to review sessions, in the order that paid most per token on the public evaluation tasks. For each test input, the distinct answers are ranked by the number of passes that verified them on the demonstrations, and the first two become the two attempts.

**Serving.** One server spans the four GPUs as two pipeline stages of two tensor-parallel ranks each: layers 0 to 25 on GPUs 0 and 1, layers 26 to 47 with the output head and the drafter on GPUs 2 and 3. It runs 35 lanes at a 60,240-token context, at about 800 tokens per second. Compared with the ARC-AGI-3 configuration, it stores the input projections of the GDN and attention layers in FP8 and prunes 128 of the 512 experts in each layer, ranked by routing frequency on public tasks.

## Layout

| Path | Contents |
|---|---|
| `harness/` | Harness v2q. `arc2/runner.py` schedules the passes over the lanes, `agent.py` runs one pass, `prelude.py` defines what the model sees in its kernel (`train`, `test`, `check`, `submit`, ...), `kernel.py` and `kernel_child.py` are the persistent Python process, `slots.py` picks the two attempts, `review.py` is the review session, `texts.json` holds every model-facing text. `shared.py` imports the ARC-AGI-3 client (`../arc3/harness/v62/transport.py`) by path. The `README.md` inside dates from the first version. |
| `l4port/` | The port of the engine to four L4s. `l4prod/` holds the hooks of the submitted configuration: persistent sessions and eviction under tensor and pipeline parallelism (`l4port_evpp.py`), the QSA indexer's keys shared between the two ranks (`l4port_idxshard_v3.py`, with `idx_shard/`), balanced pipeline microbatches (`l4port_ppbalance.py`), the fix for a decode stall seen only on Kaggle's driver (`l4port_ppdefer.py`), the drafter's KV under pipeline parallelism (`l4port_draftring_pp.py`) and the determinism hooks (`l4port_marlindet_ep.py`, `l4port_qsasort.py`). `ple_hostpoll_tp/` is the PLE handoff for two ranks, `fp8tp_v1/` and `fp8tier/` the FP8 input projections, `bootstrap_*/` the chain that installs the hooks, `sidepod/` the expert sets and the profiles of the single-GPU development runs. |
| `engine/` | The engine files that differ from [`../vllm/`](../vllm/) in this package, at the paths they have there; [`engine/DELTA.md`](engine/DELTA.md) lists them. `engine/launch/k_launch_scaling_l4.py` is the launcher. |
| `kaggle/` | `solution_notebook.py` (the submission notebook as a script), `kaggle_run_arc2_v2f.py` (the driver it runs), `kaggle_defs/KDEF.json` (the frozen serving definition) and `runtime_patch/` (three vLLM files that the package replaces in the runtime). |

## Notes on the port

- `l4port_idxshard_v3.py` puts the sharded indexer call behind the engine's eviction wrapper. Versions 1 and 2 skipped the wrapper, so a session lost its index remapping and its selection counts after its first eviction; the packages before v2j carried that defect.
- On Kaggle the first packages decoded at about 515 tokens per second against about 800 on our pods. Stage 0's CUDA graph was stalling behind the receive of stage 1's sampled tokens, which the stock handler posts one step early. `l4port_ppdefer.py` posts those receives at the step that consumes them.
- A context that is an exact multiple of the attention page size broke evictions; the profiles keep 256 tokens of slack below the last page.

The ARC-AGI-2 tasks come from the competition's data and are not in this repository.

Licence: `engine/`, `l4port/` and `kaggle/runtime_patch/` contain modified vLLM code and are under Apache-2.0, as [`../vllm/`](../vllm/LICENSE); the harness and the rest of `kaggle/` are under MIT-0.
