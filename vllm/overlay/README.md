# The overlay

The second layer of the engine: persistent sessions, eviction, the KV layout, the PLE handoff and the dense-layer kernels. On the pods this directory is `/workspace/arc3-inference/runs/dense_fp8_v1`.

No file of the runtime is modified. The launcher puts this directory on `PYTHONPATH`, and the code reaches vLLM in two ways:

- **Plugin entry points** (`arc3_persistent_experiment-0.2.dist-info/entry_points.txt`). vLLM's engine plugin calls `lookahead_bootstrap.py`, which checks the pinned files and installs `engine_patch.py`; its endpoint plugin calls `persistent_plugin.py`, which registers the session endpoint.
- **Import hooks.** Each `*/bootstrap/sitecustomize.py` runs at interpreter start in every server process, chains to the next one, and registers a hook that patches its target module right after vLLM imports it.

`overlay-manifest.json` pins every file here by hash. The pod launcher refuses to start if one differs; the Kaggle launcher logs it.

## Components

| Files | Role |
|---|---|
| `persistent_endpoint.py`, `protocol.py` | The endpoint (`/arc3/persistent/start`, `/{session}/append`, `/{session}/close`) and the chat rendering around a session that is kept as raw tokens. |
| `session_service.py`, `residency.py`, `managed_spans.py`, `append_only_history.py` | One session per game: its token history, what is still cached, and its budget. |
| `engine_patch.py`, `persistent_batch_policy.py`, `control_transaction.py` | The engine side of a persistent request. |
| `qsa_frequency.py`, `history_eviction.py` | The selection counts gathered from the QSA indexers, and the eviction plan built from them. |
| `eviction_runner.py`, `eviction_qsa.py`, `engine_eviction.py`, `engine_eviction_async.py`, `eviction_storage.py` | Carrying an eviction out: tokens keep their original positions, the 4-token groups of the indexer stay consistent, and the KV rows are compacted while the other lanes keep decoding. |
| `fixed_state_fold_v1/` | The KV block layout: per-session state of fixed size shares blocks instead of taking one each. |
| `per_owner_settle_v1/` | Stopping one lane without draining the others. |
| `ple_hostpoll_v1/` | The PLE handoff through a pinned buffer and a flag that a GPU kernel polls. It builds on the two earlier handoffs, `ple_direct_handoff_v1/` and `ple_copy_handoff_v1/`. |
| `glue_fusion_v1/` | Triton kernels for the small matrix multiplications of a decode step, split-K among them. |
| `dense_fp8_adapter_v1/`, `mtp0x_v1/` | FP8 weights for the dense projections that a profile selects (none in `82kx1280`; the input projections in the `a28` and `a30p16` profiles), their deterministic kernel, and the decode sampler. |
| `dense_projection_adapter_v1/` | Generated kernels for the BF16 projections at batches of 25 and 30 rows; other sizes keep vLLM's path. |
| `launch_scaling.py`, `launch_candidate.py`, `launch_optimized.py` | The launch chain that the stage-2 launchers in [`../launch/`](../launch/) extend. |

## What else is here

The directory is the experiment tree as the packages shipped it, so it also holds material that the submitted runs do not use:

- **Earlier candidates that are switched off:** `prefix_share_v1/` (a shared system-prompt cache), `private_moe/` (a startup adapter for a private build of the MoE kernels), `ple_discriminator_v1/` and `boundary_timing_v1/` (diagnostics), `one_lookahead.py` with `lookahead_snapshot_v2/` and `lookahead_image_oracle/` (one-step lookahead, which the stage-2 part `persist` no longer uses, and its state checkers).
- **Tests, GPU checks and controllers of the September experiments:** `test_*.py`, `*_checks.py`, `run_*.py`, `benchmark*.py`, the `*-job.json` files and `fixtures/`.
- **Profiling observers:** `gameplay_server_profile.py`, `gameplay_harness_profile.py`, `resume_profile.py`, `host_profile.py`, `profile_observer.py`.
- **Design notes of that time:** `HARNESS_HISTORY_DESIGN.md`, `HARNESS_SUFFIX_DESIGN.md`, `HARNESS_PROFILE_SCHEMA.md`, `PRIVATE_SERVING.md` and the READMEs in three subdirectories. Several describe mechanisms that were replaced later, for example eviction of the oldest interactions, which gave way to eviction by selection counts.
- **Pins and lineage records:** `gameplay-canary-pins.json`, `runtime-pins.json`, `BASE.json`, `BASELINE_*`, `PRIVATE_LINEAGE.json`, `STAGE_VALIDATION.json`.

`private_moe/fixtures/core.txt` is FlashInfer's `fused_moe/core.py` (Apache-2.0), kept as a test fixture.
