# The Kaggle kit (ARC-AGI-2)

On the pods and on Kaggle this directory is `/mnt/shared/arc2/kaggle`.

| File | Role |
|---|---|
| `solution_notebook.py` | The submission notebook as a script; the `# %%` lines split it into cells. It mounts three Kaggle Models (the target checkpoint, the bundle with the runtime and the drafter, and the code package), extracts them, applies the runtime patch, and runs the driver. In a submission it solves the test puzzles for 12 hours; otherwise it runs the evaluation puzzles for two hours and scores them. |
| `kaggle_run_arc2_v2f.py` | The driver. It renders the profile `g60f8inp128` from [`../l4port/sidepod/KDEF_sp20.json`](../l4port/sidepod/KDEF_sp20.json), adds the settings for four L4s, starts one server on all four GPUs, and runs the harness until the deadline. `submission.json` holds an entry for every puzzle from the first minutes on, and the harness's answers are merged into it every minute. No check stops a run. `HARNESS_ARGS` in this file are the harness's options. |
| `kaggle_run_arc2.py` | The driver's previous version; `kaggle_run_arc2_v2f.py` adds one log line to it. |
| `runtime_patch/` | Three files of vLLM that the notebook copies over the runtime's, with their hashes in `RUNTIME_PATCH.json`: the hyper-connection projection split over the tensor-parallel ranks, the quantized QSA cache allowed under tensor parallelism, and the PLE connection under pipeline parallelism. |
| `kaggle_defs/KDEF.json` | The ARC-AGI-3 definition as this package carried it (its v12 state). The ARC-AGI-2 driver does not read its profiles. |
| `build/filetrace_l4/` | The tracer that records which files a run opens, from which the package was built. |

At the tag `arc2-v2j` the notebook selects harness v2p and the code package `arc2-code-v2j`; on `main` it selects v2q and `arc2-code-v2k`.
