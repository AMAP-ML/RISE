#!/usr/bin/env python3
"""
Unified evaluation script:
- join prediction jsonl with GT by key (id/orig_row_index/dataset_index)
- supports rule-based / LLM-judge / both
- supports single dataset args or registry yaml batch mode
"""

import argparse
import ast
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen3-max")
DEFAULT_JUDGE_API_URL = os.getenv("JUDGE_API_URL", os.getenv("QWEN_CHAT_API_URL", ""))


@dataclass
class EvalStats:
    total_pred_rows: int = 0
    matched_rows: int = 0
    unmatched_rows: int = 0
    match_rate: float = 0.0
    judge_failures: int = 0


@dataclass
class EvalResult:
    correct: int
    total: int
    accuracy: float
    errors: List[Dict[str, Any]]
    stats: EvalStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen3-VL-8B-Instruct", help="Model name used in output file names")
    parser.add_argument("--registry", type=str, default="Evaluation/dataset_registry.yaml", help="Registry yaml path")
    parser.add_argument("--dataset", type=str, default=None, help="Run only one dataset from registry")
    parser.add_argument(
        "--mode",
        type=str,
        default="rule",
        choices=["rule", "llm", "both"],
        help="Evaluation mode (default: rule)",
    )
    parser.add_argument("--max_workers", type=int, default=8, help="Max threads for LLM judge")
    parser.add_argument("--retry_interval_sec", type=float, default=2.0, help="Sleep seconds between LLM-judge retries")
    parser.add_argument(
        "--join_key",
        type=str,
        default="auto",
        choices=["auto", "id", "orig_row_index", "dataset_index"],
        help="Default key used to join prediction rows with ground truth",
    )
    parser.add_argument(
        "--answer_format",
        type=str,
        default="auto",
        choices=["auto", "text", "yes_no"],
        help="Answer format for rule-based matching",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="LLM judge API key. Prefer env JUDGE_API_KEY, DASHSCOPE_API_KEY, or QWEN_API_KEY.",
    )
    parser.add_argument("--api_url", type=str, default=DEFAULT_JUDGE_API_URL, help="OpenAI-compatible chat completions URL for the LLM judge")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL, help="LLM judge model name")
    parser.add_argument("--pred_file", type=str, default=None, help="Single-run prediction file (without registry)")
    parser.add_argument("--gt_file", type=str, default=None, help="Single-run GT file (without registry)")
    parser.add_argument("--file_name_filter", type=str, default=None, help="Optional GT file_name filter")
    parser.add_argument("--report_tag", type=str, default=None, help="Optional tag used in result file name")
    parser.add_argument(
        "--max_retries",
        type=int,
        default=-1,
        help="Max retries for one LLM-judge request; -1 means retry until success",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../storage_RISE_Qwen3-VL-8B/evaluation_metrics/multi_eval",
        help="Output directory for results/errors/summary",
    )
    return parser.parse_args()


def extract_boxed_content(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return text.strip()[:100]


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    ans = str(answer).strip().upper()
    return re.sub(r"[,\.\s]+", "", ans)


def normalize_yes_no(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    if text in {"1", "true", "yes", "y", "t"}:
        return "yes"
    if text in {"0", "false", "no", "n", "f"}:
        return "no"
    return ""


def normalize_options_text(options_value: Any) -> str:
    if options_value is None:
        return ""
    if isinstance(options_value, list):
        values = [str(x).strip() for x in options_value if str(x).strip()]
        return " | ".join(values)
    try:
        if pd.isna(options_value):
            return ""
    except Exception:
        pass
    if isinstance(options_value, str):
        text = options_value.strip()
        if not text:
            return ""
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                values = [str(x).strip() for x in parsed if str(x).strip()]
                return " | ".join(values)
        except Exception:
            pass
        return text
    return str(options_value).strip()


def load_table(path: str):
    cache_root = os.getenv("HF_HOME", f"/tmp/{os.getenv('USER', 'rise')}/rise_hf_cache")
    cache_dir = os.getenv("HF_DATASETS_CACHE", str(Path(cache_root) / "datasets"))
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    if path.endswith(".parquet"):
        return load_dataset("parquet", data_files=path, split="train", cache_dir=cache_dir)
    if path.endswith(".arrow"):
        return load_dataset("arrow", data_files=path, split="train", cache_dir=cache_dir)
    raise ValueError(f"Unsupported file format: {path}")


def load_ground_truth(gt_file: str, answer_key: str = "answer", file_name_filter: Optional[str] = None) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, int]]:
    ds = load_table(gt_file)

    ground_truth = {"dataset_index": {}, "id": {}, "orig_row_index": {}}
    skipped_no_answer = 0
    total_rows = 0

    for idx, row in enumerate(ds):
        total_rows += 1
        if file_name_filter and row.get("file_name") != file_name_filter:
            continue

        answer = row.get(answer_key)
        if answer is None or str(answer).strip() == "":
            skipped_no_answer += 1
            continue

        row_data = {
            "answer": str(answer).strip(),
            "options_text": normalize_options_text(row.get("options", row.get("choices", row.get("option", "")))),
        }

        key_idx = str(idx)
        ground_truth["dataset_index"][key_idx] = row_data
        ground_truth["orig_row_index"][key_idx] = row_data

        sample_id = row.get("id")
        if sample_id is not None and str(sample_id).strip() != "":
            ground_truth["id"][str(sample_id)] = row_data

        orig_row_index = row.get("orig_row_index")
        if orig_row_index is not None and str(orig_row_index).strip() != "":
            ground_truth["orig_row_index"][str(orig_row_index)] = row_data

    gt_stats = {
        "gt_total_rows": total_rows,
        "gt_scored_rows": len(ground_truth["dataset_index"]),
        "gt_skipped_no_answer": skipped_no_answer,
    }
    return ground_truth, gt_stats


def build_samples(predictions_file: str, ground_truth: Dict[str, Dict[str, Dict[str, str]]], join_key: str = "auto") -> Tuple[List[Dict[str, Any]], EvalStats]:
    if join_key == "auto":
        join_priority = ["id", "orig_row_index", "dataset_index"]
    else:
        join_priority = [join_key]

    samples: List[Dict[str, Any]] = []
    stats = EvalStats()

    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            stats.total_pred_rows += 1
            data = json.loads(line)

            matched_key_name = None
            matched_key_value = None
            true_data = None

            for key_name in join_priority:
                pred_value = data.get(key_name)
                if pred_value is None:
                    continue
                pred_key = str(pred_value)
                if pred_key in ground_truth[key_name]:
                    matched_key_name = key_name
                    matched_key_value = pred_key
                    true_data = ground_truth[key_name][pred_key]
                    break

            if true_data is None:
                stats.unmatched_rows += 1
                continue

            stats.matched_rows += 1
            response = data.get("response", "")
            samples.append(
                {
                    "join_key_name": matched_key_name,
                    "join_key_value": matched_key_value,
                    "predicted": extract_boxed_content(response),
                    "true_answer": true_data["answer"],
                    "options_text": true_data.get("options_text", ""),
                    "question": data.get("question", ""),
                    "response": response,
                }
            )

    stats.match_rate = (stats.matched_rows / stats.total_pred_rows * 100) if stats.total_pred_rows else 0.0
    return samples, stats


def _extract_qwen_text_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: List[str] = []
        for item in raw_content:
            if isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    parts.append(str(txt))
            elif item:
                parts.append(str(item))
        return "".join(parts)
    return str(raw_content or "")


def judge_answer_with_llm(
    predicted_answer: str,
    ground_truth_answer: str,
    options_text: str,
    question: str,
    api_key: str,
    api_url: str,
    judge_model: str,
) -> bool:
    if not predicted_answer:
        return False

    user_content = (
        "Please judge whether the predeict answer is right based on the question and the correct answer. "
        f"Question: {question}. "
        f"Correct answer: {ground_truth_answer}. "
        f"Predicted answer: {predicted_answer}. "
        "You don't need to reason, if correct return 1, if incorrect return 0. Don't say anything else, only return 1 or 0."
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": judge_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an answer evaluation assistant. Your task is to decide whether the predicted answer "
                    "is correct with respect to the correct answer. Treat answers as correct when they express the "
                    "same core meaning, even if wording or formatting is different. Do not require exact string match."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "enable_thinking": False,
    }
    response = requests.post(
        api_url,
        json=payload,
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    response_json = response.json()

    raw_content = response_json["choices"][0]["message"]["content"]
    result = _extract_qwen_text_content(raw_content).strip().lower()
    if result in {"1", "correct", "yes", "true"}:
        return True
    if result in {"0", "incorrect", "no", "false"}:
        return False
    if "correct" in result and "incorrect" not in result:
        return True
    if "incorrect" in result:
        return False
    raise RuntimeError(f"Unparseable judge response: {result!r}")


def infer_answer_format(samples: List[Dict[str, Any]], configured_format: str) -> str:
    if configured_format != "auto":
        return configured_format
    if not samples:
        return "text"
    true_parsed = sum(1 for s in samples if normalize_yes_no(s.get("true_answer", "")) in {"yes", "no"})
    if (true_parsed / len(samples)) >= 0.95:
        return "yes_no"
    return "text"


def evaluate_rule(samples: List[Dict[str, Any]], stats: EvalStats, answer_format: str = "text") -> EvalResult:
    correct = 0
    errors = []
    for sample in tqdm(samples, desc="Evaluating (rule-based)", leave=True):
        if answer_format == "yes_no":
            pred_cmp = normalize_yes_no(sample["predicted"])
            true_cmp = normalize_yes_no(sample["true_answer"])
        else:
            pred_cmp = normalize_answer(sample["predicted"])
            true_cmp = normalize_answer(sample["true_answer"])

        if pred_cmp == true_cmp and true_cmp != "":
            correct += 1
        else:
            errors.append(
                {
                    "index": sample["join_key_value"],
                    "index_key": sample["join_key_name"],
                    "predicted": sample["predicted"],
                    "true_answer": sample["true_answer"],
                    "question": sample["question"],
                    "response": sample["response"],
                }
            )

    total = len(samples)
    acc = correct / total * 100 if total else 0.0
    return EvalResult(correct=correct, total=total, accuracy=acc, errors=errors, stats=stats)


def evaluate_llm(
    samples: List[Dict[str, Any]],
    stats: EvalStats,
    api_key: str,
    api_url: str,
    judge_model: str,
    max_workers: int,
    max_retries: int,
    retry_interval_sec: float,
) -> EvalResult:
    correct = 0
    errors = []

    def judge_sample(sample: Dict[str, Any]):
        retries = 0
        while True:
            try:
                ok = judge_answer_with_llm(
                    sample["predicted"],
                    sample["true_answer"],
                    sample.get("options_text", ""),
                    sample.get("question", ""),
                    api_key,
                    api_url,
                    judge_model,
                )
                return sample, ok
            except Exception:
                retries += 1
                if max_retries >= 0 and retries > max_retries:
                    raise
                time.sleep(retry_interval_sec)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(judge_sample, sample): sample for sample in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating (LLM judge)", leave=True):
            sample = futures[future]
            try:
                sample, is_correct = future.result()
            except Exception as e:
                stats.judge_failures += 1
                errors.append(
                    {
                        "index": sample["join_key_value"],
                        "index_key": sample["join_key_name"],
                        "predicted": sample["predicted"],
                        "true_answer": sample["true_answer"],
                        "question": sample["question"],
                        "response": sample["response"],
                        "error": repr(e),
                    }
                )
                continue

            if is_correct:
                correct += 1
            else:
                errors.append(
                    {
                        "index": sample["join_key_value"],
                        "index_key": sample["join_key_name"],
                        "predicted": sample["predicted"],
                        "true_answer": sample["true_answer"],
                        "question": sample["question"],
                        "response": sample["response"],
                    }
                )

    total = len(samples)
    acc = correct / total * 100 if total else 0.0
    return EvalResult(correct=correct, total=total, accuracy=acc, errors=errors, stats=stats)


def load_registry(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    datasets = data.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("registry yaml must contain `datasets` list")
    return datasets


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())


def resolve_report_tag(args: argparse.Namespace) -> str:
    if args.report_tag:
        return sanitize_name(args.report_tag)
    if args.dataset:
        return sanitize_name(args.dataset)
    if args.pred_file and args.gt_file:
        return sanitize_name(Path(args.pred_file).stem.replace("_project", ""))
    return "all"


def run_one_dataset(
    model_name: str,
    ds_name: str,
    pred_file: str,
    gt_file: str,
    answer_key: str,
    file_name_filter: Optional[str],
    join_key: str,
    mode: str,
    output_dir: Path,
    api_key: Optional[str],
    api_url: Optional[str],
    judge_model: str,
    max_workers: int,
    max_retries: int,
    retry_interval_sec: float,
    answer_format: str,
) -> Dict[str, Any]:
    if not Path(pred_file).exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    if not Path(gt_file).exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    gt_map, gt_stats = load_ground_truth(gt_file, answer_key=answer_key, file_name_filter=file_name_filter)
    samples, stats = build_samples(pred_file, gt_map, join_key=join_key)
    resolved_answer_format = infer_answer_format(samples, answer_format)

    results: Dict[str, Any] = {
        "dataset": ds_name,
        "pred_file": pred_file,
        "gt_file": gt_file,
        "join_key": join_key,
        "answer_format": resolved_answer_format,
        "mode": mode,
        "match_stats": vars(stats),
        "gt_stats": gt_stats,
    }

    if mode in {"rule", "both"}:
        rule_result = evaluate_rule(samples, EvalStats(**vars(stats)), answer_format=resolved_answer_format)
        results["rule"] = {
            "correct": rule_result.correct,
            "total": rule_result.total,
            "accuracy": rule_result.accuracy,
            "stats": vars(rule_result.stats),
        }
        safe_model = sanitize_name(model_name)
        safe_ds = sanitize_name(ds_name)
        write_jsonl(output_dir / f"{safe_model}_{safe_ds}_rule_errors.jsonl", rule_result.errors)

    if mode in {"llm", "both"}:
        if not api_key:
            raise RuntimeError("Missing API key. Set --api_key or env JUDGE_API_KEY, DASHSCOPE_API_KEY, or QWEN_API_KEY.")
        if not api_url:
            raise RuntimeError("Missing API URL. Set --api_url or env JUDGE_API_URL.")

        llm_result = evaluate_llm(
            samples,
            EvalStats(**vars(stats)),
            api_key=api_key,
            api_url=api_url,
            judge_model=judge_model,
            max_workers=max_workers,
            max_retries=max_retries,
            retry_interval_sec=retry_interval_sec,
        )
        results["llm"] = {
            "correct": llm_result.correct,
            "total": llm_result.total,
            "accuracy": llm_result.accuracy,
            "stats": vars(llm_result.stats),
        }
        safe_model = sanitize_name(model_name)
        safe_ds = sanitize_name(ds_name)
        write_jsonl(output_dir / f"{safe_model}_{safe_ds}_llm_errors.jsonl", llm_result.errors)

    safe_model = sanitize_name(model_name)
    safe_ds = sanitize_name(ds_name)
    with (output_dir / f"{safe_model}_{safe_ds}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_tag = resolve_report_tag(args)

    datasets: List[Dict[str, Any]] = []
    if args.pred_file and args.gt_file:
        datasets.append(
            {
                "name": report_tag,
                "pred_out": args.pred_file,
                "gt_file": args.gt_file,
                "answer_key": "answer",
                "answer_format": args.answer_format,
                "file_name_filter": args.file_name_filter,
                "join_key": args.join_key,
                "eval_mode": args.mode or "both",
                "enabled": True,
            }
        )
    else:
        datasets = load_registry(args.registry)

    api_key = args.api_key or os.getenv("JUDGE_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    api_url = args.api_url or os.getenv("JUDGE_API_URL") or os.getenv("QWEN_CHAT_API_URL")
    report_lines: List[str] = []

    print("=" * 80)
    print("Starting evaluation")
    print(f"Registry: {args.registry}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    for ds in datasets:
        name = ds.get("name", "unknown")
        if args.dataset and name != args.dataset:
            continue
        if not ds.get("enabled", True):
            print(f"[skip] {name}: disabled")
            continue

        pred_file = ds.get("pred_out")
        gt_file = ds.get("gt_file")
        answer_key = ds.get("answer_key", "answer")
        answer_format = ds.get("answer_format", args.answer_format)
        file_name_filter = ds.get("file_name_filter")
        join_key = ds.get("join_key", args.join_key)
        mode = args.mode or ds.get("eval_mode", "both")

        print(f"\nEvaluating: {name}")
        print(f"  Prediction file: {pred_file}")
        print(f"  Ground truth: {gt_file}")
        print(f"  Mode: {mode}")
        print(f"  Answer format: {answer_format}")

        try:
            result = run_one_dataset(
                model_name=args.model,
                ds_name=name,
                pred_file=pred_file,
                gt_file=gt_file,
                answer_key=answer_key,
                file_name_filter=file_name_filter,
                join_key=join_key,
                mode=mode,
                output_dir=output_dir,
                api_key=api_key,
                api_url=api_url,
                judge_model=args.judge_model,
                max_workers=args.max_workers,
                max_retries=args.max_retries,
                retry_interval_sec=args.retry_interval_sec,
                answer_format=answer_format,
            )
        except Exception as e:
            print(f"  ✗ Evaluation error: {e}")
            report_lines.append(f"{name}: ERROR {e}")
            continue

        match_stats = result["match_stats"]
        print(
            "  ✓ Join stats: "
            f"total_pred_rows={match_stats['total_pred_rows']}, "
            f"matched_rows={match_stats['matched_rows']}, "
            f"unmatched_rows={match_stats['unmatched_rows']}, "
            f"match_rate={match_stats['match_rate']:.2f}%"
        )

        if "rule" in result:
            rule = result["rule"]
            print(f"  ✓ Rule: {rule['correct']}/{rule['total']} = {rule['accuracy']:.2f}%")
            report_lines.append(f"{name} rule: {rule['correct']}/{rule['total']} = {rule['accuracy']:.2f}%")

        if "llm" in result:
            llm = result["llm"]
            print(
                f"  ✓ LLM: {llm['correct']}/{llm['total']} = {llm['accuracy']:.2f}% "
                f"(judge_failures={llm['stats']['judge_failures']})"
            )
            report_lines.append(
                f"{name} llm: {llm['correct']}/{llm['total']} = {llm['accuracy']:.2f}% "
                f"judge_failures={llm['stats']['judge_failures']}"
            )

    report_path = output_dir / f"{args.model}_{report_tag}_results.txt"
    with report_path.open("w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")

    print("\n" + "=" * 80)
    print(f"Results saved to: {report_path}")
    print("Evaluation completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
