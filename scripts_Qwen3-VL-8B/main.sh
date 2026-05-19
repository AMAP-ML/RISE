#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Dataset, model, and output roots. Override these when running on a new machine.
export DATA_DIR="${DATA_DIR:-../datasets}"
export MODEL_DIR="${MODEL_DIR:-../pretrained_LM}"
export PROJECT_NAME="${PROJECT_NAME:-RISE_Qwen3-VL-8B}"
export STORAGE_TAG="${STORAGE_TAG:-${PROJECT_NAME}}"
export STORAGE_PATH="${STORAGE_PATH:-../storage_${STORAGE_TAG}}"
export MODEL_SAVE_ROOT="${MODEL_SAVE_ROOT:-${STORAGE_PATH}/models}"
export HUGGINGFACENAME=""

# Skill-balancing hyperparameters used by the questioner reward and solver data upload.
# SKILL_BALANCE_WEIGHT controls the reward bonus for under-represented skills.
# SKILL_BALANCED_UPLOAD_TARGET is the target number of filtered samples for each solver stage.
# SKILL_BALANCED_UPLOAD_SEED makes skill-balanced sampling reproducible.
export SKILL_BALANCE_WEIGHT="${SKILL_BALANCE_WEIGHT:-0.2}"
export SKILL_BALANCED_UPLOAD_TARGET="${SKILL_BALANCED_UPLOAD_TARGET:-1500}"
export SKILL_BALANCED_UPLOAD_SEED="${SKILL_BALANCED_UPLOAD_SEED:-42}"

mkdir -p \
  "${STORAGE_PATH}/evaluation" \
  "${STORAGE_PATH}/generated_question" \
  "${STORAGE_PATH}/local_parquet" \
  "${STORAGE_PATH}/temp_results" \
  "${STORAGE_PATH}/wandb" \
  "${MODEL_SAVE_ROOT}"

BASE_MODEL="${MODEL_DIR}/Qwen3-VL-8B-Instruct"
MODEL_ABBR="Qwen3-VL-8B-Instruct"

# Fine-grained alternation schedule.
# BIG_ROUNDS repeats the whole questioner/solver alternation block.
# MICRO_ITERS is the number of short alternation cycles in each big round.
# MICRO_STEPS is the number of GRPO update steps added in each short cycle.
export BIG_ROUNDS="${BIG_ROUNDS:-1}"
export MICRO_ITERS="${MICRO_ITERS:-12}"
export MICRO_STEPS="${MICRO_STEPS:-5}"
export GLOBAL_STEP="${MICRO_STEPS}"

source scripts_Qwen3-VL-8B/runtime_cleanup.sh
trap runtime_cleanup_all EXIT

echo "Project: ${PROJECT_NAME}"
echo "Base model: ${BASE_MODEL}"
echo "Storage path: ${STORAGE_PATH}"
echo "Model save root: ${MODEL_SAVE_ROOT}"
echo "Schedule: BIG_ROUNDS=${BIG_ROUNDS}, MICRO_ITERS=${MICRO_ITERS}, MICRO_STEPS=${MICRO_STEPS}"

is_complete_step_dir() {
  local step_dir="$1"
  [ -f "${step_dir}/actor/huggingface/config.json" ]
}

find_hf_ckpt_by_step() {
  local exp_name="$1"
  local step="$2"
  local step_dir="${MODEL_SAVE_ROOT}/${exp_name}/global_step_${step}"

  if is_complete_step_dir "${step_dir}"; then
    echo "${step_dir}/actor/huggingface"
  fi
}

find_resume_ckpt_by_step() {
  local exp_name="$1"
  local step="$2"
  local step_dir="${MODEL_SAVE_ROOT}/${exp_name}/global_step_${step}"

  if is_complete_step_dir "${step_dir}"; then
    echo "${step_dir}"
  fi
}

find_latest_hf_ckpt() {
  local exp_name="$1"
  local exp_root="${MODEL_SAVE_ROOT}/${exp_name}"

  if [ ! -d "${exp_root}" ]; then
    return
  fi

  local step_dir=""
  while IFS= read -r step_dir; do
    if is_complete_step_dir "${step_dir}"; then
      echo "${step_dir}/actor/huggingface"
      return
    fi
  done < <(find "${exp_root}" -maxdepth 1 -type d -name 'global_step_*' | sort -V -r)
}

prev_questioner_ckpt="${BASE_MODEL}"
prev_solver_ckpt="${BASE_MODEL}"

for big in $(seq 1 "${BIG_ROUNDS}"); do
  echo "=== Big round ${big}/${BIG_ROUNDS} ==="

  q_exp="${MODEL_ABBR}_q_b${big}"
  s_exp="${MODEL_ABBR}_s_b${big}"

  latest_q_ckpt="$(find_latest_hf_ckpt "${q_exp}" || true)"
  latest_s_ckpt="$(find_latest_hf_ckpt "${s_exp}" || true)"
  if [ -n "${latest_q_ckpt}" ]; then
    prev_questioner_ckpt="${latest_q_ckpt}"
  fi
  if [ -n "${latest_s_ckpt}" ]; then
    prev_solver_ckpt="${latest_s_ckpt}"
  fi

  for micro in $(seq 1 "${MICRO_ITERS}"); do
    target_steps=$((micro * MICRO_STEPS))
    prev_steps=$(((micro - 1) * MICRO_STEPS))
    echo "--- Micro iteration ${micro}/${MICRO_ITERS}, target step ${target_steps} ---"

    runtime_cleanup_all

    q_ckpt="$(find_hf_ckpt_by_step "${q_exp}" "${target_steps}" || true)"
    if [ -z "${q_ckpt}" ]; then
      q_load_ckpt=""
      if [ "${prev_steps}" -gt 0 ]; then
        q_load_ckpt="$(find_resume_ckpt_by_step "${q_exp}" "${prev_steps}" || true)"
      fi

      bash scripts_Qwen3-VL-8B/questioner_train.sh \
        "${prev_solver_ckpt}" \
        "${prev_questioner_ckpt}" \
        "${q_exp}" \
        "${target_steps}" \
        "${q_load_ckpt}"

      q_ckpt="$(find_hf_ckpt_by_step "${q_exp}" "${target_steps}" || true)"
      if [ -z "${q_ckpt}" ]; then
        echo "ERROR: missing questioner checkpoint for ${q_exp} step ${target_steps}" >&2
        exit 1
      fi
    else
      echo "Questioner checkpoint already exists: ${q_ckpt}"
    fi

    runtime_cleanup_all

    s_ckpt="$(find_hf_ckpt_by_step "${s_exp}" "${target_steps}" || true)"
    if [ -z "${s_ckpt}" ]; then
      s_load_ckpt=""
      if [ "${prev_steps}" -gt 0 ]; then
        s_load_ckpt="$(find_resume_ckpt_by_step "${s_exp}" "${prev_steps}" || true)"
      fi

      bash scripts_Qwen3-VL-8B/solver_train.sh \
        "${prev_solver_ckpt}" \
        "${q_ckpt}" \
        "${s_exp}" \
        "${target_steps}" \
        "${s_load_ckpt}"

      s_ckpt="$(find_hf_ckpt_by_step "${s_exp}" "${target_steps}" || true)"
      if [ -z "${s_ckpt}" ]; then
        echo "ERROR: missing solver checkpoint for ${s_exp} step ${target_steps}" >&2
        exit 1
      fi
    else
      echo "Solver checkpoint already exists: ${s_ckpt}"
    fi

    prev_questioner_ckpt="${q_ckpt}"
    prev_solver_ckpt="${s_ckpt}"
    echo "Finished big=${big}, micro=${micro}, step=${target_steps}"
  done
done

echo "RISE training finished."
