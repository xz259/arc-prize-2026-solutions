# ARC-2 harness v0

A minimal client for the ARC-3 serving stack: one persistent session per puzzle attempt, a python tool with a persistent
kernel, exact verification on the demonstrations, and a token budget per attempt. The plan and its decisions:
[../PLAN_V0_2026-10-29.md](../PLAN_V0_2026-10-29.md).

## Layout

| File | Role |
|---|---|
| `arc2/tasks.py` | loading, grid text (ARC-3 colour letters by default, digits optional), Kaggle submission format, local scorer |
| `arc2/prelude.py` | runs inside each attempt's kernel: `train`, `test`, `check`, `submit`, `history`, `programs`, `source`, `show` |
| `arc2/kernel_child.py`, `arc2/kernel.py` | the persistent python process per attempt: 60-s limit per call, restart with definition replay, ARC-3's tool-message JSON |
| `arc2/texts.json`, `arc2/prompts.py` | every model-facing text; system prompt, opener (pinned on the server), cue texts for the transport |
| `arc2/agent.py` | one attempt: the turn loop, length pauses, closes, nudges, budget notice, fallback |
| `arc2/runner.py` | lanes over the task queue (FIFO, a finished lane takes the next task), logs, `submission.json`, `score.json`, `summary.md` |
| `arc2/shared.py`, `shared_modules.json` | imports ARC-3's `v62/transport.py` by path, refused unless its sha256 matches the pin |
| `arc2/fake.py` | scripted stand-in for the server session (`--fake`) |
| `arc2/represent.py` | the re-presented panel (memorization control): one symmetry and a 1-9 colour permutation per task, scored in its own frame; seed 1 is in `../official/represented/` |
| `tests/test_arc2.py` | 23 local tests (no server) |
| `scripts/run_eval_v0.sh` | the pod command |

## Run

```bash
cd arc2_stack/harness
python3 -m unittest tests.test_arc2                       # local, no server
python3 -m arc2.runner --out /tmp/arc2_dry --fake          # dry run over the 120 evaluation tasks
```

On the pod (server up, profile 82kr1632): `bash scripts/run_eval_v0.sh NAME [--limit 3 --lanes 2 --budget 20000]`. Sampling defaults to T 0.6 (ARC-3's shipped value). The re-presented panel: add `--challenges ../official/represented/represented_s1_challenges.json --solutions ../official/represented/represented_s1_solutions.json`.

Outputs per run: `settings.json`, `tasks/<id>/log.jsonl` (every server request and response, every tool call),
`tasks/<id>/attempt.json`, `results.jsonl`, `submission.json` (rewritten after each task), `score.json`, `summary.md`,
`metrics_before.txt` / `metrics_after.txt` (the server's `/metrics`, for drafter acceptance).
