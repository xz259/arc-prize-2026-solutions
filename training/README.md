# Training

The only training in either solution is the fine-tuning of the MTP drafter, the model's multi-token-prediction head, which proposes the draft tokens for speculative decoding. The target model stays frozen and unmodified. Section 2.1 of the [ARC-AGI-3 write-up](../arc3/docs/writeup.md) gives the result: on held-out games the tokens emitted per decoding step rose from 3.03 to 3.35.

## `mtp_drafter/`

| Path | Contents |
|---|---|
| `kit/` | The first version: `qwen38_mtp.py` (the drafter as a standalone module), `train_mtp.py` (the trainer), `export_mtp_drafter.py` (a checkpoint back into the serving format), `apply_dflashtaps_mtp.py` (the export of the target's hidden states that the drafter trains on), and CPU tests. |
| `scripts/` | What trained the shipped drafter. `train_mtp_sp.py` is the trainer: it adds the served-pair losses, importance weights on the accepted path, frozen routed experts and resumable training states to the kit's. Around it are the data preparation (`mtp_data_prep.sh`, `extract_streams.py`, `mtp_full_sessions.py`), the evaluation by expected accepted length (`mtp_eval_val50.py`, `live_tau.py`, `paired_live_tau.py`), the head's assembly for serving (`stitch_mtp_head.py`, `state_to_head.py`) and the launch scripts of each round. |

The training data are recorded gameplay sessions replayed through the server with the stage-2 part `dflashtaps`, which writes the target's hidden states per token (`../vllm/stage2/dflashtaps.py`, `replayfx.py`). The data and the checkpoints are not in this repository.

The scripts are as they ran on our pods: they refer to pod paths and to run names of the project.
