# ARC-AGI-2 solver

The same model and engine as the ARC-AGI-3 entry, moved to Kaggle's four L4s, with a harness for static puzzles. The model is not trained on ARC tasks. The two final submissions scored 75.97 and 76.81 on the leaderboard.

## How it works

**Harness.** Each puzzle is attempted in independent *passes*. A pass is one persistent session on the server and one Python kernel that holds the puzzle. The model reads the grids as text, writes Python, tests a transformation against the demonstration pairs with `check(fn)` and records up to two answers with `submit(...)`. A pass ends when the model stops with a valid answer saved, when it stalls, or at 200,000 generated tokens. Every puzzle gets passes until two of them have ended with a valid answer, at most four. Time left after that goes to further passes and to review sessions, in the order that paid most per token on the public evaluation tasks. For each test input, the distinct answers are ranked by the number of passes that verified them on the demonstrations, and the first two become the two attempts.

**Serving.** One server spans the four GPUs as two pipeline stages of two tensor-parallel ranks each: layers 0 to 25 on GPUs 0 and 1, layers 26 to 47 with the output head and the drafter on GPUs 2 and 3. It runs 35 lanes at a 60,240-token context, at about 800 tokens per second. Compared with the ARC-AGI-3 configuration, it stores the input projections of the GDN and attention layers in FP8 and prunes 128 of the 512 experts in each layer, ranked by routing frequency on public tasks.

## Layout

| Path | Contents |
|---|---|
| [`harness/`](harness/) | Harness v2q: the runner that schedules the passes, the agent of one pass, the kernel, the prompts, and the rule that picks the two attempts. Its tests and a dry run work without a server. |
| [`l4port/`](l4port/) | The port of the engine to four L4s: the hooks that make persistent sessions, eviction, the drafter and the PLE handoff work under tensor and pipeline parallelism, and the serving definition with the submitted profile. |
| [`engine/`](engine/) | The 14 engine files of this package that differ from [`../vllm/`](../vllm/), at the paths they have there. Six of them change for the port and one is the launcher; the other seven are the v12 versions of files that ARC-AGI-3 changed later. |
| [`kaggle/`](kaggle/) | The submission notebook, the driver it runs, and three files of vLLM that the package replaces in the runtime. |

`main` holds package v2k. The tag `arc2-v2j` is the package before it, which differs only in the harness and in the notebook's choice of it: v2k adds the guard against repeated invalid tool calls. The packages also carried the earlier harness versions v2m to v2p (the notebook falls back to v2n if v2q cannot be configured); they are not included here.

## Notes on the port

- `l4port/l4prod/l4port_idxshard_v3.py` puts the sharded indexer call behind the engine's eviction wrapper. Versions 1 and 2 skipped the wrapper, so a session lost its index remapping and its selection counts after its first eviction; the packages before v2j carried that defect.
- On Kaggle the first packages decoded at about 515 tokens per second against about 800 on our pods. Stage 0's CUDA graph was stalling behind the receive of stage 1's sampled tokens, which the stock handler posts one step early. `l4port/l4prod/l4port_ppdefer.py` posts those receives at the step that consumes them.
- A context that is an exact multiple of the attention page size broke evictions; the profiles keep 256 tokens of slack below the last page.

The ARC-AGI-2 tasks come from the competition's data and are not in this repository.

Licence: `engine/`, `l4port/` and `kaggle/runtime_patch/` contain modified vLLM code and are under Apache-2.0, as [`../vllm/`](../vllm/LICENSE); the harness and the rest of `kaggle/` are under MIT-0.
