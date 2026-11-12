# The Kaggle kit (ARC-AGI-3)

| File | Role |
|---|---|
| `solution_notebook.py` | The submission notebook as a script; the `# %%` lines split it into cells. It mounts three Kaggle Models (the target checkpoint with its PLE table, the bundle with the runtime, the drafter and the calibration files, and the code package), extracts them at the paths the code expects and runs the driver. |
| `kaggle_run.py` | The driver. It renders the environment of the chosen profile from `kaggle_defs/KDEF.json`, starts the server through `../../vllm/launch/k_launch_scaling_s2i.py`, waits for it, starts the harness, prints a status line every ten minutes and stops everything before the time limit. No check in it stops a run: each one logs and the run continues. |
| `kaggle_defs/KDEF.json` | The frozen definition: the server's and the harness's command lines and environments, captured from the development launch, and the named profiles. |
| `build/` | How the code package and the definition are made: `mk_code_bundle.py` (the archive, from traces of the files a run opens), `mk_kdef.py`, `mk_klaunch.py` (the Kaggle launcher, generated from the pod launcher), `reshard_ple.py` (the PLE table in 19 files), `build_kaggle_stage.py`, `stage_core55.py`, and the file tracers in `filetrace*/`. |

The profiles:

| Profile | Lanes | Notes |
|---|---|---|
| `82kx1280` | 26 + 2 | 82,528-token context, a 1,280-token BF16 window plus the 352-slot exact set. The profile of the 58.59 and 58.28 runs. |
| `82kx1632a30p16` | 30 + 2 | The same with a 1,632-token window, FP8 attention input projections and 16 experts pruned per layer. The profile of the 57.97 run. |
| `82kr1632`, `80kr512`, `82kx1280a28` | 26 + 2, 27 + 2, 28 + 2 | Earlier and intermediate profiles. |

At the tag `arc3-v12` the notebook file carries its development defaults; it was submitted with `PROFILE = "82kx1280"`, `FALLBACK_PROFILE = "82kr1632"`, `SHUFFLE_GAMES = True` and `CKPTFETCH_FILES = 4`.
