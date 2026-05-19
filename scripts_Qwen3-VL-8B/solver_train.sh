#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"

solver_model_path=$1
questioner_model_path=$2
experiment_name=$3
train_steps=${4:-${GLOBAL_STEP:-5}}
load_checkpoint_path=${5:-}

# Data and storage roots inherited from main.sh. They can also be overridden when this
# stage script is launched directly.
data_dir="${DATA_DIR:-../datasets}"
micro_steps="${MICRO_STEPS:-5}"

# Number of images/questions generated for this solver stage before filtering.
# Larger values provide more candidate pseudo labels but increase generation cost.
num_samples_per_micro="${QUESTION_NUM_SAMPLES_PER_MICRO:-2000}"

export PROJECT_NAME="${PROJECT_NAME:-RISE_Qwen3-VL-8B}"
export STORAGE_PATH="${STORAGE_PATH:-../storage_${PROJECT_NAME}}"
model_save_root="${MODEL_SAVE_ROOT:-${STORAGE_PATH}/models}"
parquet_dir="${STORAGE_PATH}/local_parquet"

source scripts_Qwen3-VL-8B/runtime_cleanup.sh
trap runtime_cleanup_all EXIT

mkdir -p "${model_save_root}" "${parquet_dir}"

micro_idx=$((train_steps / micro_steps))
if [ "${micro_idx}" -le 0 ]; then
  micro_idx=1
fi
slice_start=$(((micro_idx - 1) * num_samples_per_micro))
data_save_name="${experiment_name}_step${train_steps}"
solver_train_parquet="${parquet_dir}/${data_save_name}_train.parquet"

echo "Train solver: ${experiment_name}"
echo "Solver model: ${solver_model_path}"
echo "Questioner model: ${questioner_model_path}"
echo "Target steps: ${train_steps}"
echo "Training data: ${solver_train_parquet}"

# Disable vLLM compile cache to avoid stale kernel/cache issues across alternating stages.
export VLLM_DISABLE_COMPILE_CACHE=1

# Explicit role marker used by the dataset pipeline. Solver training must keep the
# pseudo-labeled problem/answer fields because the answer is the solver reward target.
export RISE_TRAINING_ROLE=solver

if [ ! -s "${solver_train_parquet}" ]; then
  echo "Generating questions with the current questioner..."
  bash question_generate/question_generate.bash \
    "${questioner_model_path}" \
    "${num_samples_per_micro}" \
    "${data_save_name}" \
    "${slice_start}"

  runtime_cleanup_all

  echo "Evaluating generated questions with the current solver..."
  bash question_evaluate/evaluate.sh "${solver_model_path}" "${data_save_name}"

  runtime_cleanup_all

  echo "Building solver training parquet..."
  python -u question_evaluate/upload.py \
    --max_score 0.8 \
    --min_score 0.3 \
    --save_name "${data_save_name}" \
    --strict_row_count
else
  echo "Reuse existing solver training parquet: ${solver_train_parquet}"
fi

trainer_args=(
  config=train_examples/cot_config.yaml

  # Solver responses are trained to end with a boxed final answer.
  data.max_response_length=2048
  "data.train_files=${solver_train_parquet}"
  "data.val_files=${data_dir}/MMStar"
  data.format_prompt=./train_examples/format_prompt/solver.jinja
  "worker.actor.model.model_path=${solver_model_path}"

  # Keep per-device micro batches small for VLM memory stability.
  worker.actor.micro_batch_size_per_device_for_update=1
  worker.actor.micro_batch_size_per_device_for_experience=1
  worker.actor.offload.offload_params=false
  worker.actor.offload.offload_optimizer=false

  # Maximum rollout tokens batched by vLLM during experience generation.
  worker.rollout.max_num_batched_tokens=20000

  # Keep the reference model on GPU to avoid CPU offload overhead.
  worker.ref.fsdp.enable_cpu_offload=false

  # Solver reward compares extracted boxed answers against pseudo labels.
  worker.reward.reward_function=./train_examples/reward_function/cot_val_solver.py:compute_score
  worker.val_reward.reward_function=./train_examples/reward_function/cot_val_solver.py:compute_score

  "trainer.project_name=${PROJECT_NAME:-RISE}"
  trainer.total_epochs=1
  "trainer.max_steps=${train_steps}"
  "trainer.save_freq=${train_steps}"
  "trainer.experiment_name=${experiment_name}"
  "trainer.save_checkpoint_path=${model_save_root}/${experiment_name}/"
  trainer.val_before_train=false
  trainer.load_dataloader_state=false
)

if [ -n "${load_checkpoint_path}" ]; then
  trainer_args+=("trainer.load_checkpoint_path=${load_checkpoint_path}")
fi

echo "Training solver..."
python3 -m verl.trainer.main "${trainer_args[@]}"

step_dir="${model_save_root}/${experiment_name}/global_step_${train_steps}"
python scripts_Qwen3-VL-8B/model_merger.py --local_dir "${step_dir}/actor"

if [ ! -f "${step_dir}/actor/huggingface/config.json" ]; then
  echo "ERROR: merged solver checkpoint is incomplete: ${step_dir}/actor/huggingface" >&2
  exit 1
fi

echo "Solver training finished: ${step_dir}/actor/huggingface"
