#!/usr/bin/env bash
# run_t_compact_gpu.sh (2026-10-31, ARC2 - main): idx_shard/t_compact_gpu.py on GPUs 0-1 of the 4090 pod; ends 't_compact_gpu ended'.
D=/mnt/shared/arc2/l4port; L=$D/logs/t_compact_gpu_4090.out
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || { echo "STOP: $n GPU processes" >> $L; echo "t_compact_gpu ended" >> $L; exit 1; }
echo "[$(date -u +'%F %T')] start (t_compact_gpu $(md5sum < $D/idx_shard/t_compact_gpu.py | cut -c1-8), idxshard v2 $(md5sum < $D/l4prod/l4port_idxshard_v2.py | cut -c1-8))" >> $L
CUDA_VISIBLE_DEVICES=0,1 timeout 900 bash /workspace/arc3-inference/runtime_python.sh $D/idx_shard/t_compact_gpu.py /workspace/arc3-inference/runs/dense_fp8_v1 >> $L 2>&1
echo "[$(date -u +'%F %T')] rc $?" >> $L
echo "t_compact_gpu ended" >> $L
