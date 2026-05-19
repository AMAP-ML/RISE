#!/bin/bash
# evaluate_safe.sh - launch evaluation in two 4-GPU batches.
# Usage: bash question_evaluate/evaluate_safe.sh <model_name> <save_name>

model_name=$1
save_name=$2
timeout_duration=3600

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] evaluate_safe: start ==="
echo "Model: $model_name"
echo "Save name: $save_name"
echo "Strategy: two batches, 4 GPUs per batch"

# --- Batch 1: GPU 0-3 ---
echo ""
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Batch 1: start GPU 0-3 ==="
pids_batch1=()
for i in {0..3}; do
  CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate.py --model $model_name --suffix $i --save_name $save_name &
  pids_batch1+=($!)
  echo "  GPU $i -> PID ${pids_batch1[-1]}"
done

# Wait for batch 1.
for pid in "${pids_batch1[@]}"; do
  wait $pid 2>/dev/null
done
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Batch 1: finished GPU 0-3 ==="

# --- Batch 2: GPU 4-7 ---
echo ""
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Batch 2: start GPU 4-7 ==="
pids_batch2=()
for i in {4..7}; do
  CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate.py --model $model_name --suffix $i --save_name $save_name &
  pids_batch2+=($!)
  echo "  GPU $i -> PID ${pids_batch2[-1]}"
done

# Timeout guard for batch 2.
(
  sleep $timeout_duration
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Timeout after ${timeout_duration}s; killing remaining batch-2 processes..."
  for pid in "${pids_batch2[@]}"; do
    if kill -0 $pid 2>/dev/null; then
      kill -9 $pid 2>/dev/null
      echo "  killed PID $pid"
    fi
  done
) &
timeout_pid=$!

# Wait for batch 2.
for pid in "${pids_batch2[@]}"; do
  wait $pid 2>/dev/null
done

# Stop timeout guard.
kill $timeout_pid 2>/dev/null
wait $timeout_pid 2>/dev/null

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Batch 2: finished GPU 4-7 ==="
echo ""
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] evaluate_safe: finished ==="
