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
export PROJECT_NAME="${PROJECT_NAME:-RISE_Qwen3-VL-8B}"
export STORAGE_PATH="${STORAGE_PATH:-../storage_${PROJECT_NAME}}"
model_save_root="${MODEL_SAVE_ROOT:-${STORAGE_PATH}/models}"

source scripts_Qwen3-VL-8B/runtime_cleanup.sh
trap runtime_cleanup_all EXIT

mkdir -p "${model_save_root}"

# Distributed training timeout for long VLM rollout/reward steps.
export TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC:-3600}"

# Keep checkpoint loading/saving on GPU unless users need CPU offload for memory.
export VERL_CKPT_CPU_OFFLOAD="${VERL_CKPT_CPU_OFFLOAD:-0}"

# Explicit role marker used by the dataset pipeline. In questioner training, source
# dataset question/answer fields are masked so rewards depend only on the image,
# generated question, solver responses, and supervisor judgments.
export RISE_TRAINING_ROLE=questioner
export QUESTIONER_MASK_SOURCE_QA=1

# Unique ID for the vLLM reward servers used in this questioner stage.
export RUN_ID="${RUN_ID:-$(date +%s%N)}"

echo "Train questioner: ${experiment_name}"
echo "Solver model for reward: ${solver_model_path}"
echo "Questioner model: ${questioner_model_path}"
echo "Target steps: ${train_steps}"

bash vllm_service_init/start.sh "${solver_model_path}" "${RUN_ID}"

trainer_args=(
  config=train_examples/cot_config.yaml

  # Unlabeled image pool used to train the questioner.
  data.train_files=../datasets/parquet/LMMs-Lab-Turtle__Vision-SR1-47K
  "data.val_files=${data_dir}/MMStar" # not used dataset, just for compatibility
  data.prompt_key=problem
  data.answer_key=answer
  data.image_key=images
  "worker.actor.model.model_path=${questioner_model_path}"

  # Maximum context length for questioner rollouts.
  worker.rollout.max_model_len=12288

  # Number of rollouts per image for GRPO.
  worker.rollout.n=8

  "trainer.project_name=${PROJECT_NAME:-RISE}"
  "trainer.max_steps=${train_steps}"
  "trainer.save_freq=${train_steps}"
  "trainer.experiment_name=${experiment_name}"
  "trainer.save_checkpoint_path=${model_save_root}/${experiment_name}"
  trainer.total_epochs=1

  # The questioner is trained on GPUs 0-3 while solver reward servers use GPUs 4-7.
  trainer.n_gpus_per_node=4
  trainer.val_before_train=false
  trainer.val_only=false
)

if [ -n "${load_checkpoint_path}" ]; then
  trainer_args+=("trainer.load_checkpoint_path=${load_checkpoint_path}")
fi

echo "Training questioner..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main "${trainer_args[@]}"

step_dir="${model_save_root}/${experiment_name}/global_step_${train_steps}"
python scripts_Qwen3-VL-8B/model_merger.py --local_dir "${step_dir}/actor"

if [ ! -f "${step_dir}/actor/huggingface/config.json" ]; then
  echo "ERROR: merged questioner checkpoint is incomplete: ${step_dir}/actor/huggingface" >&2
  exit 1
fi

echo "Questioner training finished: ${step_dir}/actor/huggingface"
