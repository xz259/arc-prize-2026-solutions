# Training

The only training in either solution is the fine-tuning of the MTP drafter, the model's multi-token-prediction head, which proposes the draft tokens for speculative decoding. The target model stays frozen and unmodified. Section 2.1 of the [ARC-AGI-3 write-up](../arc3/docs/writeup.md) gives the result: on held-out games the tokens emitted per decoding step rose from 3.03 to 3.35.

The training data are recorded gameplay sessions, replayed through the server with the stage-2 part `dflashtaps`, which writes the target's hidden states for every token (`../vllm/stage2/dflashtaps.py`, `replayfx.py`). The data and the checkpoints are not in this repository.

## `mtp_drafter/kit/`

The first version, written and tested on the CPU.

| File | Role |
|---|---|
| `qwen38_mtp.py` | The drafter as a standalone, trainable module: a port of the serving code's MTP head that loads the checkpoint's tensors under their own names. |
| `train_mtp.py` | The first trainer. |
| `export_mtp_drafter.py` | Turns a trained head back into a drafter folder that the server loads. |
| `apply_dflashtaps_mtp.py` | Extends the export of hidden states with the input that the drafter consumes. |
| `test_qwen38_mtp_cpu.py`, `test_train_mtp_cpu.sh` | CPU tests. |

## `mtp_drafter/scripts/`

What trained the shipped drafter, as it ran on our pods.

| Files | Role |
|---|---|
| `train_mtp_sp.py` | The trainer. To the kit's it adds the losses on the distributions that serving compares (after temperature, top-k and top-p), importance weights on the accepted path, frozen routed experts, and training states that a run can resume from. |
| `mtp_data_prep.sh`, `extract_streams.py`, `mtp_full_sessions.py`, `select_val_*.py` | Data preparation: the token streams of the recorded sessions, the list of sessions to replay, and the held-out games. |
| `mtp_eval_val50.py`, `live_tau.py`, `paired_live_tau.py` | Evaluation by the expected number of tokens emitted per step in live decoding, and the paired comparison of two heads. |
| `state_to_head.py`, `stitch_mtp_head.py` | From a training state to a head, and from a head to a drafter folder with the served experts unchanged. |
| `sp_round*.sh`, `sp_fullrun*.sh`, `mtp_launch_r*.sh`, `fr1c_launch*.sh` | The launch scripts: the ablation rounds, the full run, the refit rounds, and the continuation `fr1c` of the full run, whose final head is the shipped drafter. |
| `dev_stepemb_v2/`, `step_emb.py` | A variant with a learned vector per draft step. It did not beat the shipped head. |
| `dflash_*.py`, `compare_mtp_dflash.py` | Comparisons with a DFlash drafter, which we trained first and did not adopt. |
| `test_*` | CPU and GPU tests of the trainer. |

The scripts are as they ran: they refer to pod paths and to the run names of the project.

`kit/qwen38_mtp.py` follows vLLM's model code closely and is under Apache-2.0, as [`../vllm/`](../vllm/LICENSE); the rest is under MIT-0.
