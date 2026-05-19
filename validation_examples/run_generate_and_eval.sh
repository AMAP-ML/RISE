#!/usr/bin/env bash
set -euo pipefail

if [[ "${RISE_EVAL_DEBUG:-0}" == "1" ]]; then
  set -x
fi

# Make all relative paths stable no matter where this script is launched from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISPLAY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VISPLAY_ROOT}/.." && pwd)"

# Avoid ~/.cache quota issues for datasets/filelock in multi-process eval.
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/tmp/${USER:-rise}/rise_hf_cache}"
export HF_HOME="${HF_CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" || true
echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"


# Dataset registry. It defines val_file, gt_file, prompt keys, join keys, and eval options.
REGISTRY="${REGISTRY:-Evaluation/dataset_registry.yaml}"
# Model or checkpoint path used for generation. Pass this by CLI or environment.
MODEL_PATH="${MODEL_PATH:-}"
# Model tag used in output file names, e.g. <dataset>_<model>_project.jsonl.
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B-Instruct}"
# Output subdirectory under OUTPUT_ROOT.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rise_eval}"
# Root directory for predictions and evaluation reports.
OUTPUT_ROOT="${OUTPUT_ROOT:-../storage_RISE_Qwen3-VL-8B/evaluation_metrics}"
# Optional single dataset name. Empty means all enabled datasets in the registry.
DATASET="${DATASET:-}"
# Evaluation mode: rule / llm / both. The open-source default avoids external API calls.
MODE="${MODE:-both}"
# LLM judge worker count.
MAX_WORKERS="${MAX_WORKERS:-8}"
# Max retries for one LLM judge sample. -1 means retry until success.
MAX_RETRIES="${MAX_RETRIES:--1}"
# Seconds to wait before each retry.
RETRY_INTERVAL_SEC="${RETRY_INTERVAL_SEC:-2}"
# Optional override for each dataset join_key: auto/id/orig_row_index/dataset_index.
JOIN_KEY_OVERRIDE="${JOIN_KEY_OVERRIDE:-}"
# Generation mode: per_dataset or merged. Merged reuses combined parquet groups.
GEN_MODE="${GEN_MODE:-merged}"
# Temporary directory for merged mode. Empty defaults to <OUT_DIR>/.tmp_merge.
MERGE_TMP_DIR="${MERGE_TMP_DIR:-}"

# GPU count used for generation.
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
# vLLM rollout tensor parallel size. It must divide N_GPUS_PER_NODE.
TP_SIZE="${TP_SIZE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# LLM judge provider settings. The key is read by eval_boxed_accuracy.py from env,
# so it is not passed through the command line.
JUDGE_API_URL="${JUDGE_API_URL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen-max}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --max_workers) MAX_WORKERS="$2"; shift 2 ;;
    --max_retries) MAX_RETRIES="$2"; shift 2 ;;
    --retry_interval_sec) RETRY_INTERVAL_SEC="$2"; shift 2 ;;
    --join_key) JOIN_KEY_OVERRIDE="$2"; shift 2 ;;
    --gen_mode) GEN_MODE="$2"; shift 2 ;;
    --merge_tmp_dir) MERGE_TMP_DIR="$2"; shift 2 ;;
    --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2 ;;
    --tensor_parallel_size) TP_SIZE="$2"; shift 2 ;;
    --judge_api_url) JUDGE_API_URL="$2"; shift 2 ;;
    --judge_model) JUDGE_MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" || -z "$MODEL_NAME" ]]; then
  echo "Usage: $0 --model_path <path> --model_name <name> [--dataset <name>] [other options]"
  exit 1
fi

if (( N_GPUS_PER_NODE <= 0 )); then
  echo "N_GPUS_PER_NODE must be > 0, got: ${N_GPUS_PER_NODE}"
  exit 1
fi

if (( TP_SIZE <= 0 )); then
  echo "TP_SIZE must be > 0, got: ${TP_SIZE}"
  exit 1
fi

if (( N_GPUS_PER_NODE % TP_SIZE != 0 )); then
  echo "N_GPUS_PER_NODE (${N_GPUS_PER_NODE}) must be divisible by TP_SIZE (${TP_SIZE})"
  exit 1
fi

# Normalize key paths to absolute based on VISPLAY_ROOT,
# so callers can always pass repo-relative paths.
if [[ "$REGISTRY" != /* ]]; then
  REGISTRY="${VISPLAY_ROOT}/${REGISTRY}"
fi
if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="${VISPLAY_ROOT}/${MODEL_PATH}"
fi
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="${VISPLAY_ROOT}/${OUTPUT_ROOT}"
fi
if [[ -n "$MERGE_TMP_DIR" && "$MERGE_TMP_DIR" != /* ]]; then
  MERGE_TMP_DIR="${VISPLAY_ROOT}/${MERGE_TMP_DIR}"
fi

OUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "$OUT_DIR" || true
if [[ -z "$MERGE_TMP_DIR" ]]; then
  MERGE_TMP_DIR="${OUT_DIR}/.tmp_merge"
fi

echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
echo "TP_SIZE=${TP_SIZE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cleanup_generate_processes() {
  pkill -9 -f "run_eval_batch.py" || true
  pkill -9 -f "verl.trainer.main|ray::Runner.run|start_vllm_server|vllm" || true
  pkill -9 -f "raylet|gcs_server|ray/dashboard/agent.py|ray::" || true
  if command -v ray >/dev/null 2>&1; then
    ray stop --force || true
  fi
}

# =========================
# Part A: Generate prediction jsonl files.
# - per_dataset: generate one dataset at a time.
# - merged: merge compatible datasets first, generate once per group, then split predictions back.
# =========================
echo "[1/2] Generate predictions"
mapfile -t GEN_DATASETS < <(
  python - "$REGISTRY" "$DATASET" <<'PY'
import sys, yaml
registry_path = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else ""
with open(registry_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
for ds in cfg.get("datasets", []):
    if not ds.get("enabled", True):
        continue
    name = ds.get("name", "")
    if not name:
        continue
    if target and name != target:
        continue
    print(name)
PY
)

if [[ ${#GEN_DATASETS[@]} -eq 0 ]]; then
  echo "No dataset selected from registry for generation."
  exit 1
fi

if [[ "$GEN_MODE" == "per_dataset" ]]; then
  for DS_NAME in "${GEN_DATASETS[@]}"; do
    echo "Generating for dataset: ${DS_NAME}"
    python "${VISPLAY_ROOT}/validation_examples/run_eval_batch.py" \
      --registry "$REGISTRY" \
      --model_path "$MODEL_PATH" \
      --model_name "$MODEL_NAME" \
      --experiment_name "$EXPERIMENT_NAME" \
      --output_root "$OUTPUT_ROOT" \
      --pred_out_mode auto \
      --n_gpus_per_node "$N_GPUS_PER_NODE" \
      --tensor_parallel_size "$TP_SIZE" \
      --only_dataset "$DS_NAME"

    echo "Cleaning processes after dataset: ${DS_NAME}"
    cleanup_generate_processes
  done
elif [[ "$GEN_MODE" == "merged" ]]; then
  mkdir -p "$MERGE_TMP_DIR" || true
  MERGE_META="${MERGE_TMP_DIR}/merge_metadata.json"

  MERGE_CACHE_STATUS="$(
    python - "$MERGE_META" "$MERGE_TMP_DIR" <<'PY'
import json
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
merge_tmp_dir = Path(sys.argv[2])

if not meta_path.exists():
    print("MISS metadata_missing")
    raise SystemExit(0)

try:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"MISS metadata_unreadable:{exc}")
    raise SystemExit(0)

if metadata.get("cache_format_version") != 1:
    print("MISS cache_format_version")
    raise SystemExit(0)

groups = metadata.get("groups", [])
if not groups:
    print("MISS groups_missing")
    raise SystemExit(0)

rewrote_metadata = False
for group in groups:
    merged_file = str(group.get("merged_file", "")).strip()
    group_id = str(group.get("group_id", "")).strip()
    candidates = []
    if merged_file:
        candidates.append(Path(merged_file))
        candidates.append(meta_path.parent / Path(merged_file).name)
    if group_id:
        candidates.append(merge_tmp_dir / f"{group_id}.parquet")

    resolved = None
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate
            break

    if resolved is None:
        print(f"MISS merged_file_missing:{merged_file or group_id}")
        raise SystemExit(0)
    resolved_str = str(resolved)
    if group.get("merged_file") != resolved_str:
        group["merged_file"] = resolved_str
        rewrote_metadata = True

if rewrote_metadata:
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"HIT groups={len(groups)}")
PY
  )"

  if [[ "$MERGE_CACHE_STATUS" == HIT* ]]; then
    echo "Reusing merged datasets from cache: ${MERGE_TMP_DIR} (${MERGE_CACHE_STATUS})"
  else
    echo "Building merged datasets into: ${MERGE_TMP_DIR} (${MERGE_CACHE_STATUS})"
    python -u "${VISPLAY_ROOT}/validation_examples/build_merged_eval_dataset.py" \
      --registry "$REGISTRY" \
      --dataset "$DATASET" \
      --output_dir "$MERGE_TMP_DIR" \
      --metadata_out "$MERGE_META"
  fi

  if [[ -n "$JOIN_KEY_OVERRIDE" && "$JOIN_KEY_OVERRIDE" != "orig_row_index" ]]; then
    echo "[warn] merged mode forces join_key=orig_row_index (override '${JOIN_KEY_OVERRIDE}' ignored)"
  fi
  JOIN_KEY_OVERRIDE="orig_row_index"
  echo "Merged mode: force join_key=${JOIN_KEY_OVERRIDE}"

  for DS_NAME in "${GEN_DATASETS[@]}"; do
    rm -f "${OUT_DIR}/${DS_NAME}_${MODEL_NAME}_project.jsonl" || true
  done

  mapfile -t GROUP_ROWS < <(
    python - "$MERGE_META" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for g in meta.get("groups", []):
    sig = g.get("signature", {})
    option_keys = sig.get("val_option_keys", ["options", "choices", "option"])
    option_csv = ",".join(str(x).strip() for x in option_keys if str(x).strip()) or "options,choices,option"
    print("\t".join([
        str(g.get("group_id", "")),
        str(g.get("merged_file", "")),
        str(sig.get("prompt_key", "question")),
        str(sig.get("answer_key", "answer")),
        str(sig.get("image_key", "image")),
        "true" if bool(sig.get("val_include_options_in_prompt", False)) else "false",
        "true" if bool(sig.get("val_force_choice_letter_output", False)) else "false",
        "true" if bool(sig.get("val_force_yes_no_output", False)) else "false",
        option_csv,
        "true" if bool(sig.get("eval_normalize_image_placeholders", True)) else "false",
        str(sig.get("format_prompt", "./train_examples/format_prompt/solver.jinja")),
        ",".join(g.get("datasets", [])),
    ]))
PY
  )

  if [[ ${#GROUP_ROWS[@]} -eq 0 ]]; then
    echo "No merged groups generated. metadata=${MERGE_META}"
    exit 1
  fi

  FAILED_GROUPS=()
  for row in "${GROUP_ROWS[@]}"; do
    IFS=$'\t' read -r GROUP_ID MERGED_FILE PROMPT_KEY ANSWER_KEY IMAGE_KEY INC_OPTIONS FORCE_CHOICE FORCE_YES_NO OPTION_KEYS_CSV NORM_PLACEHOLDERS FORMAT_PROMPT GROUP_DATASETS <<< "$row"

    if [[ "$MERGED_FILE" != /* ]]; then
      MERGED_FILE="${VISPLAY_ROOT}/${MERGED_FILE}"
    fi
    if [[ "$FORMAT_PROMPT" != /* ]]; then
      FORMAT_PROMPT="${VISPLAY_ROOT}/${FORMAT_PROMPT}"
    fi

    GROUP_PRED_FILE="${MERGE_TMP_DIR}/${GROUP_ID}_${MODEL_NAME}_combined_project.jsonl"
    rm -f "$GROUP_PRED_FILE"

    echo "Generating merged group: ${GROUP_ID} datasets=[${GROUP_DATASETS}]"
    GEN_CMD=(
      python -m verl.trainer.main
      "config=validation_examples/eval_config.yaml"
      "data.train_files=../datasets/MMStar"
      "data.val_files=${MERGED_FILE}"
      "data.prompt_key=${PROMPT_KEY}"
      "data.answer_key=${ANSWER_KEY}"
      "data.image_key=${IMAGE_KEY}"
      "data.val_include_options_in_prompt=${INC_OPTIONS}"
      "data.val_force_choice_letter_output=${FORCE_CHOICE}"
      "data.val_force_yes_no_output=${FORCE_YES_NO}"
      "data.val_option_keys=[${OPTION_KEYS_CSV}]"
      "data.eval_normalize_image_placeholders=${NORM_PLACEHOLDERS}"
      "worker.actor.model.model_path=${MODEL_PATH}"
      "worker.rollout.max_model_len=12800"
      "worker.rollout.n=8"
      "worker.rollout.tensor_parallel_size=${TP_SIZE}"
      "trainer.total_epochs=1"
      "trainer.experiment_name=${EXPERIMENT_NAME}"
      "trainer.save_checkpoint_path=./Evaluation/eval_outputs"
      "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
      "worker.actor.micro_batch_size_per_device_for_experience=1"
      "worker.actor.global_batch_size=8"
      "data.format_prompt=${FORMAT_PROMPT}"
      "trainer.val_only=true"
      "trainer.logger=[console]"
      "trainer.response_path=${GROUP_PRED_FILE}"
    )
    if ! "${GEN_CMD[@]}"; then
      echo "[warn] merged group generate failed: ${GROUP_ID}"
      FAILED_GROUPS+=("${GROUP_ID}:generate")
      cleanup_generate_processes
      continue
    fi

    if ! python "${VISPLAY_ROOT}/validation_examples/split_merged_predictions.py" \
      --combined_pred_file "$GROUP_PRED_FILE" \
      --output_dir "$OUT_DIR" \
      --model_name "$MODEL_NAME" \
      --allowed_datasets "$GROUP_DATASETS"; then
      echo "[warn] merged group split failed: ${GROUP_ID}"
      FAILED_GROUPS+=("${GROUP_ID}:split")
    fi

    cleanup_generate_processes
  done

  if [[ ${#FAILED_GROUPS[@]} -ge ${#GROUP_ROWS[@]} ]]; then
    echo "All merged groups failed:"
    printf '  %s\n' "${FAILED_GROUPS[@]}"
    echo "Merged temp kept: ${MERGE_TMP_DIR}"
    exit 1
  fi

  if [[ ${#FAILED_GROUPS[@]} -gt 0 ]]; then
    echo "[warn] Partial merged-group failures:"
    printf '  %s\n' "${FAILED_GROUPS[@]}"
    echo "Merged temp kept for debug: ${MERGE_TMP_DIR}"
  else
    echo "Merged temp kept for reuse: ${MERGE_TMP_DIR}"
  fi
else
  echo "Unknown --gen_mode: ${GEN_MODE}. Expected: per_dataset or merged"
  exit 1
fi

# =========================
# Part B: Evaluate accuracy, join stats, and judge failures.
# The registry provides gt_file, join_key, answer_format, and optional filters.
# =========================
echo "[2/2] Evaluate"
mapfile -t DATASET_ROWS < <(
  python - "$REGISTRY" "$DATASET" <<'PY'
import sys, yaml
registry_path = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else ""
with open(registry_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
for ds in cfg.get("datasets", []):
    if not ds.get("enabled", True):
        continue
    name = ds.get("name", "")
    if not name:
        continue
    if target and name != target:
        continue
    gt_file = ds.get("gt_file", "")
    join_key = ds.get("join_key", "auto")
    answer_format = ds.get("answer_format", "auto")
    file_name_filter = ds.get("file_name_filter")
    if file_name_filter is None:
        file_name_filter = ""
    print("\t".join([name, gt_file, join_key, str(file_name_filter), str(answer_format)]))
PY
)

if [[ ${#DATASET_ROWS[@]} -eq 0 ]]; then
  echo "No dataset selected from registry."
  exit 1
fi

# Run eval_boxed_accuracy.py for each dataset. Each report is written to:
#   <OUT_DIR>/<MODEL_NAME>_<DATASET>_results.txt
for row in "${DATASET_ROWS[@]}"; do
  IFS=$'\t' read -r DS_NAME GT_FILE DS_JOIN_KEY FILE_FILTER DS_ANSWER_FORMAT <<< "$row"

  # Backward compatibility: older rows may have only 4 fields where the 4th is answer_format.
  if [[ -z "${DS_ANSWER_FORMAT:-}" && "${FILE_FILTER:-}" =~ ^(auto|text|yes_no)$ ]]; then
    DS_ANSWER_FORMAT="$FILE_FILTER"
    FILE_FILTER=""
  fi
  if [[ -z "${DS_ANSWER_FORMAT:-}" ]]; then
    DS_ANSWER_FORMAT="auto"
  fi

  PRED_FILE="${OUT_DIR}/${DS_NAME}_${MODEL_NAME}_project.jsonl"

  if [[ "$GT_FILE" != /* ]]; then
    GT_FILE="${VISPLAY_ROOT}/${GT_FILE}"
  fi

  if [[ ! -f "$PRED_FILE" ]]; then
    echo "[warn] Missing prediction file, skip: $PRED_FILE"
    continue
  fi

  JOIN_KEY="${DS_JOIN_KEY}"
  if [[ -n "$JOIN_KEY_OVERRIDE" ]]; then
    JOIN_KEY="$JOIN_KEY_OVERRIDE"
  fi

  EVAL_CMD=(
    python "${VISPLAY_ROOT}/Evaluation/eval_boxed_accuracy.py"
    --model "$MODEL_NAME"
    --pred_file "$PRED_FILE"
    --gt_file "$GT_FILE"
    --join_key "$JOIN_KEY"
    --answer_format "$DS_ANSWER_FORMAT"
    --mode "$MODE"
    --max_workers "$MAX_WORKERS"
    --max_retries "$MAX_RETRIES"
    --retry_interval_sec "$RETRY_INTERVAL_SEC"
    --output_dir "$OUT_DIR"
    --report_tag "$DS_NAME"
    --judge_model "$JUDGE_MODEL"
  )

  if [[ -n "$JUDGE_API_URL" ]]; then
    EVAL_CMD+=(--api_url "$JUDGE_API_URL")
  fi

  if [[ -n "$FILE_FILTER" && "$FILE_FILTER" != "None" && "$FILE_FILTER" != "null" ]]; then
    EVAL_CMD+=(--file_name_filter "$FILE_FILTER")
  fi

  echo "Running eval for: $DS_NAME"
  "${EVAL_CMD[@]}"
done

# =========================
# Part C: Merge per-dataset result files into:
#   <OUT_DIR>/<MODEL_NAME>_ALL_results.txt
# =========================
COMBINED_REPORT="${OUT_DIR}/${MODEL_NAME}_ALL_results.txt"
: > "$COMBINED_REPORT"
for row in "${DATASET_ROWS[@]}"; do
  IFS=$'\t' read -r DS_NAME _ <<< "$row"
  ONE_REPORT="${OUT_DIR}/${MODEL_NAME}_${DS_NAME}_results.txt"
  if [[ -f "$ONE_REPORT" ]]; then
    {
      echo "===== ${DS_NAME} ====="
      cat "$ONE_REPORT"
      echo
    } >> "$COMBINED_REPORT"
  fi
done

echo "Done."
echo "Per-dataset reports: ${OUT_DIR}/${MODEL_NAME}_<DATASET>_results.txt"
echo "Combined report: $COMBINED_REPORT"

# =========================
# Part D: Compute simple and weighted averages across datasets.
# =========================
echo ""
echo "===== Summary with Averages ====="
python - "$COMBINED_REPORT" <<'PY'
import sys, re

report_path = sys.argv[1]
with open(report_path, "r", encoding="utf-8") as f:
    content = f.read()

# Collect all output lines, then print to stdout and append to report file.
output_lines = []

def emit(line=""):
    output_lines.append(line)
    print(line)

# Parse lines like: "MMMU llm: 439/857 = 51.23%" or "MMMU rule: 343/857 = 40.02%"
pattern = re.compile(r"^(\S+)\s+(rule|llm):\s+(\d+)/(\d+)\s+=\s+([\d.]+)%", re.MULTILINE)

rule_scores = {}
llm_scores = {}
rule_counts = {}
llm_counts = {}
for match in pattern.finditer(content):
    dataset_name = match.group(1)
    mode = match.group(2)
    correct = int(match.group(3))
    total = int(match.group(4))
    score = float(match.group(5))
    if mode == "rule":
        rule_scores[dataset_name] = score
        rule_counts[dataset_name] = total
    else:
        llm_scores[dataset_name] = score
        llm_counts[dataset_name] = total

emit("===== Summary with Averages =====")

for label, scores, counts in [("rule", rule_scores, rule_counts), ("llm", llm_scores, llm_counts)]:
    if not scores:
        continue
    emit(f"\n--- {label} scores ---")
    for name, score in scores.items():
        n = counts.get(name, 0)
        emit(f"  {name:20s} {score:6.2f}%  (n={n})")
    simple_avg = sum(scores.values()) / len(scores)
    total_n = sum(counts.get(name, 0) for name in scores)
    if total_n > 0:
        weighted_avg = sum(scores[name] * counts.get(name, 0) for name in scores) / total_n
    else:
        weighted_avg = simple_avg
    emit(f"  {'SIMPLE AVG':20s} {simple_avg:6.2f}%  ({len(scores)} datasets)")
    emit(f"  {'WEIGHTED AVG':20s} {weighted_avg:6.2f}%  (total_n={total_n})")

# Use the better rule/llm score for each dataset, then compute simple and weighted averages.
all_datasets = set(rule_scores) | set(llm_scores)
if all_datasets:
    emit("\n--- best(rule, llm) scores ---")
    best_scores = {}
    best_counts = {}
    for name in sorted(all_datasets):
        rule_s = rule_scores.get(name)
        llm_s = llm_scores.get(name)
        if rule_s is not None and llm_s is not None:
            if llm_s >= rule_s:
                best_scores[name] = llm_s
                best_label = "llm"
            else:
                best_scores[name] = rule_s
                best_label = "rule"
            best_counts[name] = llm_counts.get(name) or rule_counts.get(name, 0)
        elif rule_s is not None:
            best_scores[name] = rule_s
            best_label = "rule"
            best_counts[name] = rule_counts.get(name, 0)
        else:
            best_scores[name] = llm_s
            best_label = "llm"
            best_counts[name] = llm_counts.get(name, 0)
        n = best_counts[name]
        emit(f"  {name:20s} {best_scores[name]:6.2f}%  (n={n}, from={best_label})")
    simple_avg = sum(best_scores.values()) / len(best_scores)
    total_n = sum(best_counts.values())
    if total_n > 0:
        weighted_avg = sum(best_scores[name] * best_counts[name] for name in best_scores) / total_n
    else:
        weighted_avg = simple_avg
    emit(f"  {'SIMPLE AVG':20s} {simple_avg:6.2f}%  ({len(best_scores)} datasets)")
    emit(f"  {'WEIGHTED AVG':20s} {weighted_avg:6.2f}%  (total_n={total_n})")

emit()

# Append summary to the combined report file
with open(report_path, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("\n".join(output_lines))
    f.write("\n")
PY

# nohup bash validation_examples/run_generate_and_eval.sh > ./logs/run_generate_and_eval.log 2>&1 &
