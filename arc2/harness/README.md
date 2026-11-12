# The ARC-AGI-2 harness (v2q)

One puzzle attempt, a *pass*, is a persistent session on the server and a Python kernel that holds the puzzle. The runner schedules passes over the server's lanes and builds the submission from their answers. The method is described in [`../README.md`](../README.md); this page maps it to the files. On the pods and on Kaggle this directory is `/mnt/shared/arc2/harness_v2q`.

| File | Role |
|---|---|
| `arc2/runner.py` | Schedules the passes over the lanes, decides which puzzle gets the next pass, and rewrites `submission.json` after every pass that ends. |
| `arc2/agent.py` | One pass: the turn loop on the session, the Python calls, the token budget, and the stops. |
| `arc2/prelude.py` | What the model finds in its kernel: `train`, `test`, `check`, `submit`, `note`, `history`, `source`, `show`. Expected outputs are checked against private copies, so the model's code cannot change what is verified. |
| `arc2/kernel.py`, `arc2/kernel_child.py` | The persistent Python process of a pass, with a time limit per call. After a timeout the process restarts and the model's definitions are replayed. |
| `arc2/prompts.py`, `arc2/texts.json` | Every model-facing text. The puzzle's grids are pinned on the server: eviction leaves them alone until the model has stopped attending to them. |
| `arc2/slots.py` | Which two answers are submitted for a test input, and how far along a puzzle is. |
| `arc2/review.py` | The review session: a fresh session that sees the puzzle and the answers of two passes, and saves its own two attempts. |
| `arc2/program.py` | Rebuilds the program behind an answer from the calls of a pass and checks it in a fresh kernel. |
| `arc2/tasks.py` | Loading the tasks, the text form of a grid, the submission format, and the local scorer. |
| `arc2/shared.py`, `shared_modules.json` | Imports the session client of the ARC-AGI-3 harness (`../../arc3/harness/v62/transport.py`) by path and refuses a copy whose hash differs from the pin. |
| `arc2/images.py`, `arc2/judge.py`, `arc2/represent.py` | Options that the submitted runs leave off: grid images beside the text, a judge session that ranks candidate answers, and transformed copies of the puzzles as a control for memorization. |
| `arc2/fake.py` | A scripted stand-in for the server, for dry runs and tests. |
| `tests/test_arc2.py` | 82 tests; they need no server. |

The options of the submitted runs are the `HARNESS_ARGS` list in [`../kaggle/kaggle_run_arc2_v2f.py`](../kaggle/kaggle_run_arc2_v2f.py), with the order of the later passes set in the notebook.

## Trying it without a server

The tests and a dry run need the competition's task files (`arc-agi_training_challenges.json`, `arc-agi_evaluation_challenges.json` and their solutions) in `arc2/official/data/` of this repository, and Python with numpy.

```bash
export ARC2_ARC3_HARNESS_ROOT=../../arc3/harness
```

```bash
python3 -m unittest tests.test_arc2
```

```bash
python3 -m arc2.runner --out /tmp/arc2_dry --fake --limit 3
```

Run them from this directory. The dry run answers every request with a scripted reply, so it exercises the kernel, the turn loop, the logs and the scorer, and scores zero.
