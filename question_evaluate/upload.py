import json
import pandas as pd
import argparse
import os
import time
import pyarrow.parquet as pq
import re
import random
from collections import Counter

SUPERVISOR_VALIDITY_ENABLED = os.getenv("SUPERVISOR_VALIDITY_ENABLED", "1") == "1"
SUPERVISOR_ANSWER_ENABLED = os.getenv("SUPERVISOR_ANSWER_ENABLED", "1") == "1"
SKILL_BALANCED_UPLOAD_ENABLED = os.getenv("SKILL_BALANCED_UPLOAD_ENABLED", "1") == "1"
SKILL_BALANCED_UPLOAD_TARGET = int(os.getenv("SKILL_BALANCED_UPLOAD_TARGET", "1500"))
SKILL_BALANCED_UPLOAD_SEED = int(os.getenv("SKILL_BALANCED_UPLOAD_SEED", "42"))

ALLOWED_SKILLS = [
    "coarse perception",
    "fine-grained perception",
    "instance reasoning",
    "logical reasoning",
    "math & counting",
    "science & technology",
]
SKILL_ALIASES = {
    "coarse perception": "coarse perception",
    "fine grained perception": "fine-grained perception",
    "fine-grained perception": "fine-grained perception",
    "instance reasoning": "instance reasoning",
    "logical reasoning": "logical reasoning",
    "math": "math & counting",
    "math & counting": "math & counting",
    "math and counting": "math & counting",
    "science & technology": "science & technology",
    "science and technology": "science & technology",
}


def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[upload {ts}] {msg}", flush=True)


def normalize_skill_label(skill):
    if skill is None:
        return None
    normalized = str(skill).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return SKILL_ALIASES.get(normalized)


def count_skills(rows, key):
    counts = Counter()
    for row in rows:
        normalized = normalize_skill_label(row.get(key))
        if normalized:
            counts[normalized] += 1
        else:
            counts["unknown"] += 1
    return counts


def apply_skill_balanced_sampling(rows, target_size, rng):
    if not rows or target_size <= 0:
        return rows, {
            "requested_target": target_size,
            "per_skill_quota": 0,
            "selected_before_backfill": 0,
            "backfill_count": 0,
        }

    per_skill_quota = max(target_size // len(ALLOWED_SKILLS), 0)
    selected = []
    selected_ids = set()

    skill_to_rows = {skill: [] for skill in ALLOWED_SKILLS}
    unknown_rows = []
    for idx, row in enumerate(rows):
        normalized = normalize_skill_label(row.get("declared_skill"))
        if normalized in ALLOWED_SKILLS:
            skill_to_rows[normalized].append((idx, row))
        else:
            unknown_rows.append((idx, row))

    for skill in ALLOWED_SKILLS:
        bucket = list(skill_to_rows[skill])
        rng.shuffle(bucket)
        chosen = bucket[:per_skill_quota]
        selected.extend(row for _, row in chosen)
        selected_ids.update(idx for idx, _ in chosen)

    selected_before_backfill = len(selected)
    if len(selected) < min(target_size, len(rows)):
        remaining = [(idx, row) for idx, row in enumerate(rows) if idx not in selected_ids]
        rng.shuffle(remaining)
        need = min(target_size, len(rows)) - len(selected)
        chosen_backfill = remaining[:need]
        selected.extend(row for _, row in chosen_backfill)
    else:
        chosen_backfill = []

    return selected, {
        "requested_target": target_size,
        "per_skill_quota": per_skill_quota,
        "selected_before_backfill": selected_before_backfill,
        "backfill_count": len(chosen_backfill),
    }


STORAGE_PATH = os.getenv("STORAGE_PATH")
_log(f"STORAGE_PATH={STORAGE_PATH}")
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="", help="Output directory for parquet files")
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--save_name", type=str, default="vqa_generated", help="Base name for input and output files")
parser.add_argument("--verify_timeout_sec", type=int, default=600, help="Maximum seconds to wait for a readable parquet file")
parser.add_argument("--verify_interval_sec", type=int, default=5, help="Seconds between parquet readability checks")
parser.add_argument("--strict_row_count", action="store_true", help="Require parquet row count to match the filtered sample count")
args = parser.parse_args()

datas= []
# Find all matching result files
import glob
_log("Scanning result json files")
result_files = glob.glob(f'{STORAGE_PATH}/generated_question/{args.save_name}_*_results.json')
_log(f"Found {len(result_files)} result files: {result_files}")

for file_path in result_files:
    try:
        _log(f"Loading: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            datas.extend(data)
        _log(f"Loaded {len(data)} samples from {file_path}")
    except Exception as e:
        _log(f"Error loading {file_path}: {e}")
        continue

# Filter and save data as parquet files
invalid_answers = {"", "none", "n/a", "na", "null"}

def _is_valid_answer(answer):
    if answer is None:
        return False
    ans = str(answer).strip().lower()
    return ans not in invalid_answers


def _has_uppercase_abcd(question):
    if not question:
        return False
    return all(re.search(pattern, question) for pattern in (r"A", r"B", r"C", r"D"))


def _log_multiple_choice_abcd_stats(rows, stage, question_key, type_key):
    mc_rows = [row for row in rows if str(row.get(type_key, "")).strip().lower() == "multiple choice"]
    mc_total = len(mc_rows)
    mc_with_abcd = sum(1 for row in mc_rows if _has_uppercase_abcd(str(row.get(question_key, ""))))
    ratio = (mc_with_abcd / mc_total) if mc_total else 0.0
    _log(
        f"{stage}: multiple_choice={mc_total}, question_has_uppercase_ABCD={mc_with_abcd}, "
        f"ratio={ratio:.4f}"
    )


_log_multiple_choice_abcd_stats(datas, "Before upload filtering", question_key="question", type_key="question_type")

filtered_datas = [
    {
        'problem': data['question'],
        'answer': data['answer'],
        'score': data['score'],
        'images': data.get('image', ''),
        'problem_type': data.get('question_type', 'unknown'),
        'declared_skill': normalize_skill_label(data.get('declared_skill')) or str(data.get('declared_skill', 'unknown')),
        'skill_match': int(data.get('skill_match', 1)),
        'valid': int(data.get('valid', 1)),
    }
    for data in datas
    if data['score'] >= args.min_score
    and data['score'] <= args.max_score
    and _is_valid_answer(data.get('answer', None))
    and ((not SUPERVISOR_VALIDITY_ENABLED) or int(data.get('valid', 1)) == 1)
    and ((not SUPERVISOR_ANSWER_ENABLED) or int(data.get('supervisor_correct', 0)) == 1)
]
_log(f"Filtered {len(filtered_datas)} samples with score between {args.min_score} and {args.max_score}")
_log_multiple_choice_abcd_stats(filtered_datas, "After upload filtering", question_key="problem", type_key="problem_type")
declared_skill_counts_after_filter = count_skills(filtered_datas, "declared_skill")
_log(f"Declared skill counts after filter: {dict(declared_skill_counts_after_filter)}")
supervisor_reject_count = 0
supervisor_eligible_count = sum(
    1
    for data in datas
    if data.get('score', -1) >= args.min_score
    and data.get('score', -1) <= args.max_score
    and _is_valid_answer(data.get('answer', None))
)
if SUPERVISOR_ANSWER_ENABLED:
    supervisor_reject_count = sum(
        1
        for data in datas
        if data.get('score', -1) >= args.min_score
        and data.get('score', -1) <= args.max_score
        and _is_valid_answer(data.get('answer', None))
        and int(data.get('supervisor_correct', 0)) != 1
    )
    _log(f"Supervisor answer filter rejected {supervisor_reject_count} samples")

balanced_sampling_info = {
    "requested_target": 0,
    "per_skill_quota": 0,
    "selected_before_backfill": len(filtered_datas),
    "backfill_count": 0,
}
if filtered_datas and SKILL_BALANCED_UPLOAD_ENABLED and SKILL_BALANCED_UPLOAD_TARGET > 0:
    rng = random.Random(SKILL_BALANCED_UPLOAD_SEED)
    filtered_datas, balanced_sampling_info = apply_skill_balanced_sampling(
        filtered_datas,
        target_size=SKILL_BALANCED_UPLOAD_TARGET,
        rng=rng,
    )
    _log(f"Applied skill-balanced sampling: {balanced_sampling_info}")
declared_skill_counts_after_balance = count_skills(filtered_datas, "declared_skill")
_log(f"Declared skill counts after balance: {dict(declared_skill_counts_after_balance)}")


def _parquet_is_readable(parquet_path, expected_rows=None):
    if not os.path.exists(parquet_path):
        return False, "file does not exist"
    size = os.path.getsize(parquet_path)
    if size <= 0:
        return False, "file size is 0"
    try:
        meta = pq.ParquetFile(parquet_path).metadata
        if meta is None or meta.num_rows <= 0:
            return False, "missing parquet metadata or row count <= 0"
        if expected_rows is not None and meta.num_rows != expected_rows:
            return False, f"row count mismatch: expected={expected_rows}, actual={meta.num_rows}"
        return True, f"size={size} bytes, rows={meta.num_rows}"
    except Exception as e:
        return False, f"read failed: {e}"

if filtered_datas:
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{args.save_name}_train.parquet")
    else:
        # Default to STORAGE_PATH/generated_question/
        output_dir = f"{STORAGE_PATH}/local_parquet"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.save_name}_train.parquet")
    
    # Convert to DataFrame and save as parquet
    _log(f"Building DataFrame with {len(filtered_datas)} samples")
    df = pd.DataFrame(filtered_datas)
    tmp_output_path = f"{output_path}.tmp"
    if os.path.exists(tmp_output_path):
        os.remove(tmp_output_path)
    _log(f"Writing temporary parquet file: {tmp_output_path}")
    df.to_parquet(tmp_output_path, index=False)
    if hasattr(os, "sync"):
        os.sync()

    expected_rows = len(filtered_datas) if args.strict_row_count else None

    ok, detail = _parquet_is_readable(tmp_output_path, expected_rows=expected_rows)
    if not ok:
        raise RuntimeError(f"Temporary parquet validation failed: {tmp_output_path} ({detail})")

    os.replace(tmp_output_path, output_path)
    if hasattr(os, "sync"):
        os.sync()

    deadline = time.time() + args.verify_timeout_sec
    ok = False
    last_detail = "unknown"
    while time.time() < deadline:
        ok, last_detail = _parquet_is_readable(output_path, expected_rows=expected_rows)
        if ok:
            break
        _log(f"Waiting for readable parquet: {output_path} ({last_detail})")
        time.sleep(args.verify_interval_sec)

    if not ok:
        raise RuntimeError(
            f"Parquet was not readable before timeout ({args.verify_timeout_sec}s): "
            f"{output_path} ({last_detail})"
        )

    _log(f"Saved {len(filtered_datas)} samples to {output_path} ({last_detail})")
    
    # Also save a summary file
    summary_path = output_path.replace('.parquet', '_summary.json')
    summary = {
        "total_samples": len(filtered_datas),
        "score_range": [args.min_score, args.max_score],
        "experiment_name": args.save_name,
        "output_file": output_path,
        "supervisor_answer_enabled": SUPERVISOR_ANSWER_ENABLED,
        "supervisor_validity_enabled": SUPERVISOR_VALIDITY_ENABLED,
        "skill_balanced_upload_enabled": SKILL_BALANCED_UPLOAD_ENABLED,
        "skill_balanced_upload_target": SKILL_BALANCED_UPLOAD_TARGET,
        "skill_balanced_upload_seed": SKILL_BALANCED_UPLOAD_SEED,
        "balanced_sampling_info": balanced_sampling_info,
        "supervisor_reject_count": supervisor_reject_count,
        "supervisor_reject_rate": (supervisor_reject_count / supervisor_eligible_count) if supervisor_eligible_count else 0.0,
        "declared_skill_counts_after_filter": dict(declared_skill_counts_after_filter),
        "declared_skill_counts_after_balance": dict(declared_skill_counts_after_balance),
        "unknown_skill_after_filter": int(declared_skill_counts_after_filter.get("unknown", 0)),
        "unknown_skill_after_balance": int(declared_skill_counts_after_balance.get("unknown", 0)),
        "multiple_choice_total_before_filter": sum(
            1 for data in datas if str(data.get("question_type", "")).strip().lower() == "multiple choice"
        ),
        "multiple_choice_with_uppercase_abcd_before_filter": sum(
            1
            for data in datas
            if str(data.get("question_type", "")).strip().lower() == "multiple choice"
            and _has_uppercase_abcd(str(data.get("question", "")))
        ),
        "multiple_choice_total_after_filter": sum(
            1 for data in filtered_datas if str(data.get("problem_type", "")).strip().lower() == "multiple choice"
        ),
        "multiple_choice_with_uppercase_abcd_after_filter": sum(
            1
            for data in filtered_datas
            if str(data.get("problem_type", "")).strip().lower() == "multiple choice"
            and _has_uppercase_abcd(str(data.get("problem", "")))
        ),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    _log(f"Saved summary to {summary_path}")

    # Clean up result files only after parquet is written successfully.
    for file_path in result_files:
        try:
            os.remove(file_path)
            _log(f"Removed {file_path}")
        except Exception as e:
            _log(f"Error removing {file_path}: {e}")
            continue
else:
    _log("No data to save after filtering")
