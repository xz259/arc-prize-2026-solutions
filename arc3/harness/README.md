# The ARC-AGI-3 harness

The harness plays the games: it shows the model each board, runs the Python the model writes, executes the model's actions, and decides which game gets a lane. On the pods this directory is `/workspace/kvring_dev/fin74/s2/harness_v6`; a run starts with `python -m v62.run_v6` from here.

## The base and our layer

`src/` is @dfranzen's milestone-2 solution, unmodified ([da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution) at commit `10882e3`, Apache-2.0, `src/LICENSE.dfranzen`): his agent in `src/inference/`, which builds on Tufa Labs' Duck harness, and Tufa Labs' orchestration framework in `src/taaf/`.

`v62/` is our layer on top of it. It does not edit the base. It subclasses the base's agent and solver, and it patches the base's modules when they are imported; every patch is an exact text replacement with an asserted anchor, so a changed base fails loudly instead of running half-patched.

| File in `v62/` | Role |
|---|---|
| `run_v6.py` | The runner: one run of a list of games against the running server. It sets up the base's benchmark and solver with our agent and writes the logs and the score. |
| `agent.py` | The agent of one game: the turn loop on a persistent session, and the policies of the write-up (notes and refresh, rebase, parking and the order of the waiting queue, level-up summaries, the warnings and the loop detection). |
| `transport.py` | The client of the server's persistent sessions (`/arc3/persistent/start`, `append`, `close`): it sends only new messages, checks the session's revision and token counts, and applies the caps on one reply. |
| `solver.py` | The base's solver with our action logs and the session's lifecycle. |
| `patches.py`, `session_ext.py` | The changes to the base's game session: the verdict of each action, the stops of a batch of actions, the cap on actions per call. `session_ext.py` holds modified copies of two of the base's methods. |
| `kernel_ext.py` | The stateful Python kernel: one worker process per conversation, so that variables persist between calls. |
| `sandbox_ext.py` | What the model's code can call besides the base's tools: `programs()`, `read_code()`, `save_summary()`, `summaries()`, the frames of an animation, and the helpers in `seeds/`. |
| `lazy_history.py` | The history of boards fetched on demand, so that the cost of a call does not grow with the length of the game. |
| `retention.py` | Constants written in capitals persist between calls, as functions do. |
| `tool_output.py` | Turns boards that the model prints into images. |
| `variants.py`, `texts.json`, `extract_texts.py` | Every model-facing text and its revisions, each selected by an environment variable. |
| `fake_session.py` | A scripted stand-in for the server, for dry runs and tests. |

The lanes are shared through the base's priority gate (`_PriorityGate` in `src/inference/agent/tool_agent.py`). `agent.py` sets the order in which the gate admits waiting games: games that have not started first, then parked games by how often they have parked on their current level, and within such a band by restart value.

## The other directories

| Path | Contents |
|---|---|
| `v70/` | `swap70.py`, the index and swap rule of an expectation-based allocator, and its fitted constants. `agent.py` imports the module; the submitted runs rank parked games by restart value instead. |
| `v6/`, `v61/` | Versions 6.0 and 6.1 of the layer. They shipped in the package, and the submitted runs do not import them. |
| `tests/` | Tests of the layer against scripted sessions. They run as `PYTHONPATH=src:. python3 tests/<name>.py` in an environment that has the base's dependencies (`arcengine` among them); some expect recorded logs that are not in the repository. |
| `BUILD_PLAN.md` | The build plan of version 6.0, of October 3. Later changes are documented in the docstrings and comments of the modules they touch. |

## Configuration

The harness is configured through environment variables (`ARC3_V6*`, `ARC3_V7*`); the names carry the version in which a switch appeared. The values of the submitted runs are in the `client_run_v6` environment of [`../kaggle/kaggle_defs/KDEF.json`](../kaggle/kaggle_defs/KDEF.json), and the driver adds the restart-value ranking and the stateful kernel ([`../kaggle/kaggle_run.py`](../kaggle/kaggle_run.py)).

## Licence

Apache-2.0, as the base: `v62/session_ext.py` carries modified copies of its code, and the prompts in `texts.json` grew out of its prompts.
