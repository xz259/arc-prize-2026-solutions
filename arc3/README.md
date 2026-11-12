# ARC-AGI-3 solver

The model plays each game through a Python tool. One RTX PRO 6000 serves 28 games at a time for nine hours; the harness keeps every game in a persistent session on the server and shares the 28 lanes between the 110 games.

- The write-up: [docs/writeup.md](docs/writeup.md), posted [on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/writeups/public-lb-4th)
- The engine: [../vllm/](../vllm/)

## Layout

| Path | Contents |
|---|---|
| [`harness/`](harness/) | The game-playing harness. `harness/v62/` is our layer as submitted; `harness/src/` is the base it builds on, @dfranzen's milestone-2 agent with Tufa Labs' framework, unmodified. |
| [`kaggle/`](kaggle/) | The submission notebook, the driver it runs, and the frozen definition of the serving profiles. |
| [`analysis/`](analysis/) | The index of validation runs and the script behind Figure 1 of the write-up. |
| [`docs/`](docs/) | The write-up and its figure. |

## Where the write-up's harness items are

| Write-up, section 1 | Code |
|---|---|
| 1. Persistent sessions with attention-guided eviction | The client is `harness/v62/transport.py`. The server side is in [`../vllm/`](../vllm/README.md). |
| 2. In-level notes and refresh at 200K tokens | `harness/v62/agent.py` (`ARC3_V71_LEVEL_TOKENS`) |
| 3. Rebase at 250K positions | `harness/v62/agent.py` (`ARC3_V74_REBASE_AT`; `ARC3_V76_REFRESH_AT` for the level that gets a refresh instead), with `rebase74` and `promptmove` in `../vllm/stage2/` |
| 4. Priority queue with parking | `harness/v62/agent.py` (`ARC3_V72_PARK_BANDS`, `ARC3_V72_PARK_RANK`, `restart_value`), on the base's priority gate |
| 5. Level-up summaries | `harness/v62/sandbox_ext.py` (`save_summary`, `summaries`) and `harness/v62/agent.py` |
| The stateful Python kernel | `harness/v62/kernel_ext.py` |

The harness is configured through environment variables; the values of the submitted runs are in `kaggle/kaggle_defs/KDEF.json`.

## How a run fits together

1. `kaggle/solution_notebook.py` mounts the model, the runtime bundle and the code package, extracts them at the paths the code expects, and calls the driver.
2. `kaggle/kaggle_run.py` renders the environment of the chosen profile from `KDEF.json` and starts the server through `../vllm/launch/k_launch_scaling_s2i.py`.
3. When the server answers, or ten minutes after the notebook started, the driver starts the harness (`python -m v62.run_v6` in `harness/`), which plays the games through the server's persistent sessions.
4. The driver ends the games 150 seconds before the nine hours are over and stops the server.

## Games

The public games come from the competition's data and the custom validation games from their authors ([gallery](https://arc.markbarney.net/arc3/gallery)); neither is in this repository.
