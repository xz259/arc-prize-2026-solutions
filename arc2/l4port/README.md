# The port to four L4s

The ARC-AGI-3 engine assumes one GPU and one worker process. For ARC-AGI-2 it runs as one server over four L4s: two pipeline stages (layers 0 to 25, then layers 26 to 47 with the output head and the drafter), each stage split over two GPUs by tensor parallelism. This directory holds what that took. On the pods and on Kaggle it is `/mnt/shared/arc2/l4port`.

Nothing in the engine is edited for the port, apart from the files listed in [`../engine/`](../engine/) and three runtime files in [`../kaggle/runtime_patch/`](../kaggle/runtime_patch/). Everything else is a hook: a module that patches the engine in memory after it is imported, switched on by its own environment variable.

## How the hooks are loaded

`bootstrap_l4prod9/sitecustomize.py` is the bootstrap of the two submitted packages. It first runs the engine's own bootstrap chain and then loads the hooks below in a fixed order; its docstring gives the history of that order. The launcher is `l4prod/k_launch_scaling_l4.py`. The environment that switches the hooks on comes from `L4ENV` in `l4prod/arc2_server_l4.py` and from `L4_SIZING` in the [notebook](../kaggle/solution_notebook.py).

## The hooks of the submitted configuration

| What had to work | Hook |
|---|---|
| Persistent sessions and eviction with several workers | `l4prod/l4port_evpp.py` |
| The drafter's private KV ring per pipeline stage | `l4prod/l4port_draftring_pp.py` |
| Loading and running a stage that lacks the model's final mixer | `bootstrap_ple/l4port_pp.py` |
| The drafter on the last stage: its inputs, the warmup, the broadcast of sampled tokens, and its drafts carried back to the first stage | `bootstrap_ple3/l4port_ppmtp.py`, `bootstrap_ple4/l4port_ppwarm.py`, `bootstrap_ple6/l4port_ppbcast.py`, `bootstrap_ple9/l4port_ppdraftbc.py` |
| The embedding table in host memory on both stages | `bootstrap_ple10/l4port_ppembhost.py` |
| An uneven layer split between the stages (26 and 22) | `bootstrap_ple11/l4port_ppsplit.py` |
| The PLE handoff for two ranks, without CUDA contexts in the PLE process | `ple_hostpoll_tp/` |
| Only the top 20 logits of each rank gathered, instead of the whole vocabulary | `bootstrap_ple2/l4port_vptopk.py` |
| The drafter's 32,000-token head split over the two ranks | `bootstrap_ple13/l4port_dheadtp.py` |
| The QSA indexer's keys stored once per stage, split between its two ranks, instead of on both (about 10% more KV tokens) | `l4prod/l4port_idxshard_v3.py`, with the merge in `idx_shard/merge_v2.py` |
| Even pipeline microbatches when sessions pause and resume | `l4prod/l4port_ppbalance.py` |
| The decode stall seen on Kaggle's driver | `l4prod/l4port_ppdefer.py` |
| A smaller workspace for the QSA selection (16 MiB) | `l4prod/l4port_qsaws.py` |
| Outputs that repeat bit for bit: the MoE's token order under expert parallelism, and a fixed order for the QSA selection | `l4prod/l4port_marlindet_ep.py` with `l4prod/marlindet_ep_kernels.py`, `l4prod/l4port_qsasort.py` |
| FP8 input projections under tensor parallelism | `fp8tier/`, `fp8tp_v1/` |

`sidepod/KDEF_sp20.json` is the serving definition that the Kaggle driver reads; the submitted profile is `g60f8inp128` (a 60,240-token context, K = 4, 128 experts pruned per layer, FP8 input projections). `sidepod/prune_sets_hope/eprune_freq_k128.json` is the submitted expert set.

## What else is here

The packages carried the port's whole working tree, so the directory also holds files that the submitted runs do not load:

- **Earlier versions.** `bootstrap_l4prod` to `bootstrap_l4prod8` and the `sitecustomize.py` in each `bootstrap_ple*` directory are earlier bootstraps. `l4port_idxshard.py` and `l4port_idxshard_v2.py`, `l4port_ppbalance_v1_4eb02756.py`, `bootstrap_ple7/`, `bootstrap_ple8/`, `bootstrap_ple12/l4port_dheadtp.py` and the files in `ple_hostpoll_tp/` whose names end in eight hex digits are earlier versions of hooks. Versions 1 and 2 of the indexer hook skipped the engine's eviction wrapper; see [`../README.md`](../README.md).
- **Diagnostics and checks:** `l4port_tpident.py` with `compare_tpident.py`, `l4port_idxfake.py`, `bootstrap_ple5/l4port_kvdump.py`, `check_l4prod.py`, `dryrun_l4.py`, and the tests in `idx_shard/`.
- **Run scripts of the development runs:** `l4prod/chain*.sh`, `try_*.sh`, `twin_pp.sh`, `tpident_pp.sh`. They refer to pod paths and run names.
- `sidepod/KDEF_sp20g.json` adds the profile of package v2g, whose context was an exact multiple of the attention page and broke evictions. `sidepod/prune_sets/eprune_arc2_k128.json` is the expert set that the profile names; the notebook overrides it with the one above.
