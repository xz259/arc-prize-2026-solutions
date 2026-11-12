# ARC-AGI-3 solver

The model plays each game through a Python tool. One RTX PRO 6000 serves 28 games at a time for nine hours; the harness keeps every game in a persistent session on the server and shares the 28 lanes between the 110 games.

- The write-up: [docs/writeup.md](docs/writeup.md)
- The engine: [../vllm/](../vllm/)

## Layout

| Path | Contents |
|---|---|
| `harness/v62/` | The harness as submitted. `run_v6.py` is the runner and the lane allocator, `agent.py` the turn loop and the policies (refresh, rebase, parking, summaries), `transport.py` the persistent-session client, `kernel_ext.py` the stateful Python kernel, `variants.py` with `texts.json` every model-facing text. |
| `harness/v70/` | The allocator's policy constants. |
| `harness/v6/`, `harness/v61/` | Earlier layers of the harness that shipped in the package. The submitted run uses `v62` only. |
| `harness/src/` | The base: @dfranzen's ARC3-Inference agent and Tufa Labs' `taaf` framework, unmodified (Apache-2.0, `LICENSE.dfranzen`). |
| `harness/tests/` | CPU tests of the harness. |
| `kaggle/` | The submission notebook, the driver it runs and the frozen definition of the serving profiles: [kaggle/README.md](kaggle/README.md). |
| `analysis/` | The index of validation runs and the script behind Figure 1 of the write-up. |
| `docs/` | The write-up and its figure. |

## Where the write-up's harness items are

| Write-up, section 1 | Code |
|---|---|
| 1. Persistent sessions with attention-guided eviction | The client is `harness/v62/transport.py`. The server side is in `../vllm/` (see its README). |
| 2. In-level notes and refresh at 200K tokens | `harness/v62/agent.py` (`ARC3_V71_LEVEL_TOKENS`) |
| 3. Rebase at 250K positions | `harness/v62/agent.py` (`ARC3_V74_REBASE_AT`, `ARC3_V76_REFRESH_AT`), with `rebase74` and `promptmove` in `../vllm/stage2/` |
| 4. Priority queue with parking | `harness/v62/run_v6.py` and `harness/v62/agent.py` (`ARC3_V72_PARK_BANDS`, `ARC3_V72_PARK_RANK`, `restart_value`) |
| 5. Level-up summaries | `harness/v62/sandbox_ext.py` (`save_summary`, `summaries`) and `harness/v62/agent.py` |
| The stateful Python kernel | `harness/v62/kernel_ext.py` |

The harness is configured through environment variables; the values of the submitted runs are in `kaggle/kaggle_defs/KDEF.json`.

## Games

The public games come from the competition's data and the custom validation games from their authors ([gallery](https://arc.markbarney.net/arc3/gallery)); neither is in this repository.
