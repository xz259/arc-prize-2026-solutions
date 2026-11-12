# The Kaggle kit (ARC-AGI-3)

On the pods and on Kaggle this directory is `/workspace/kvring_dev/kaggle`.

| File | Role |
|---|---|
| `solution_notebook.py` | The submission notebook as a script; the `# %%` lines split it into cells. It mounts three Kaggle Models (the target checkpoint with its PLE table; a bundle with the runtime, the drafter and the calibration files; the code package), extracts them at the paths the code expects and runs the driver. |
| `kaggle_run.py` | The driver. It renders the environment of the chosen profile from `kaggle_defs/KDEF.json`, starts the server through `../../vllm/launch/k_launch_scaling_s2i.py`, starts the harness, prints a status line every ten minutes and stops everything before the time limit. No check in it stops a run: each one logs and the run continues, because a run that stops scores zero. |
| `kaggle_defs/KDEF.json` | The frozen definition: the command lines and environments of the server and the harness, captured from a development launch, and the named profiles that override parts of them. |
| `build/` | How the code package and the definition are made: `mk_code_bundle.py` (the archive, built from traces of the files a run opens), `mk_kdef.py` (the definition), `mk_klaunch.py` (the Kaggle launcher, generated from the pod launcher), `reshard_ple.py` (the PLE table split into 19 files, so that startup can read them in parallel), `build_kaggle_stage.py`, `stage_core55.py`, and the file tracers in `filetrace*/`. |

## Profiles

A profile is a named serving configuration in `KDEF.json`.

| Profile | Lanes | Notes |
|---|---|---|
| `82kx1280` | 26 + 2 | An 82,528-token context, a BF16 window of 1,280 tokens and the 352-slot exact set. The profile of the 58.59 and 58.28 runs. |
| `82kx1632a30p16` | 30 + 2 | The same with a window of 1,632 tokens, FP8 attention input projections and 16 experts pruned per layer. The profile of the 57.97 run. |
| `82kr1632`, `80kr512`, `82kx1280a28` | 26 + 2, 27 + 2, 28 + 2 | Earlier and intermediate profiles. |

"26 + 2" means that the KV pool is sized for 26 full sessions and 28 run (the stage-2 parts `poolpress` and `stallguard` make that safe).

## Which notebook is which

`main` holds the notebook of the last package, `arc3-code` v16: it selects `82kx1632a30p16`, with `82kx1280` as the fallback.

The 58.59 and 58.28 runs used the notebook of v12. At the tag `arc3-v12`, `solution_notebook.py` carries that package's defaults, and the submitted copy changed four settings in it: `PROFILE = "82kx1280"`, `FALLBACK_PROFILE = "82kr1632"`, `SHUFFLE_GAMES = True` and `CKPTFETCH_FILES = 4`.
