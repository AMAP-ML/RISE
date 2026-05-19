# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
start_index=${4:-0}
export VLLM_DISABLE_COMPILE_CACHE=1

# If HF_CACHE_ROOT is explicitly provided, use isolated writable caches per process.
# Otherwise keep original HuggingFace default cache behavior.
use_custom_hf_cache=0
if [ -n "${HF_CACHE_ROOT:-}" ]; then
  use_custom_hf_cache=1
  mkdir -p "${HF_CACHE_ROOT}"
fi

for i in 0 1 2 3 4 5 6 7; do
  if [ "${use_custom_hf_cache}" = "1" ]; then
    proc_cache_root="${HF_CACHE_ROOT}/qg_${save_name}_${i}"
    mkdir -p "${proc_cache_root}"/datasets "${proc_cache_root}"/hub "${proc_cache_root}"/transformers
    env -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
      -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u RAY_LOCAL_RANK -u RAY_LOCAL_WORLD_SIZE \
      HF_HOME="${proc_cache_root}" \
      HF_DATASETS_CACHE="${proc_cache_root}/datasets" \
      HUGGINGFACE_HUB_CACHE="${proc_cache_root}/hub" \
      TRANSFORMERS_CACHE="${proc_cache_root}/transformers" \
      CUDA_VISIBLE_DEVICES="${i}" \
      python question_generate/question_generate.py \
      --model "$model_name" \
      --suffix "${i}" \
      --num_samples "$num_samples" \
      --save_name "$save_name" \
      --start_index "$start_index" &
  else
    env -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
      -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u RAY_LOCAL_RANK -u RAY_LOCAL_WORLD_SIZE \
      CUDA_VISIBLE_DEVICES="${i}" \
      python question_generate/question_generate.py \
      --model "$model_name" \
      --suffix "${i}" \
      --num_samples "$num_samples" \
      --save_name "$save_name" \
      --start_index "$start_index" &
  fi
  sleep 1
done

wait
