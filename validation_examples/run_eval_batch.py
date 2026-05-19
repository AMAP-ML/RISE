#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def find_rise_root(start: Path) -> Path:
    """Find project root from script location without assuming folder name."""
    for candidate in [start, *start.parents]:
        has_eval = (candidate / "Evaluation").is_dir()
        has_val = (candidate / "validation_examples").is_dir()
        has_train = (candidate / "train_examples").is_dir()
        if has_eval and has_val and has_train:
            return candidate
    raise RuntimeError(
        f"Cannot locate RISE project root from {start}. "
        "Expected directories: Evaluation/, validation_examples/, train_examples/."
    )


def load_registry(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    datasets = cfg.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("`datasets` must be a list in registry yaml")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch val-only generation from dataset registry")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("Evaluation/dataset_registry.yaml"),
        help="Path to dataset registry yaml",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="validation_examples/eval_config.yaml",
        help="Hydra config path relative to the RISE project root",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Model or checkpoint path")
    parser.add_argument("--model_name", type=str, required=True, help="Model short name for logs")
    parser.add_argument("--experiment_name", type=str, default="multi_eval", help="Experiment output folder name")
    parser.add_argument(
        "--output_root",
        type=str,
        default="../storage_RISE_Qwen3-VL-8B/evaluation_metrics",
        help="Root directory for generated prediction jsonl outputs (used in auto mode)",
    )
    parser.add_argument(
        "--pred_out_mode",
        type=str,
        default="auto",
        choices=["auto", "registry"],
        help="How to build prediction output path: auto uses output_root/experiment_name/model_name, registry uses registry pred_out",
    )
    parser.add_argument("--n_gpus_per_node", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=12800)
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--only_dataset", type=str, default=None, help="Only run one dataset name")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    visplay_root = find_rise_root(Path(__file__).resolve().parent)
    registry_path = args.registry if args.registry.is_absolute() else (visplay_root / args.registry)
    datasets = load_registry(registry_path)

    failed: List[str] = []
    succeeded: List[str] = []

    for item in datasets:
        name = item.get("name")
        if not name:
            continue
        if args.only_dataset and name != args.only_dataset:
            continue
        if not item.get("enabled", True):
            print(f"[skip] {name}: disabled in registry")
            continue

        val_file = str(item.get("val_file", "")).strip()
        pred_out_cfg = str(item.get("pred_out", "")).strip()
        prompt_key = str(item.get("prompt_key", "question")).strip()
        answer_key = str(item.get("answer_key", "answer")).strip()
        image_key = str(item.get("image_key", "image")).strip()
        answer_format = str(item.get("answer_format", "")).strip().lower()
        val_include_options_in_prompt = bool(item.get("val_include_options_in_prompt", False))
        val_force_choice_letter_output = bool(item.get("val_force_choice_letter_output", False))
        val_force_yes_no_output = bool(item.get("val_force_yes_no_output", answer_format == "yes_no"))
        eval_normalize_image_placeholders = bool(item.get("eval_normalize_image_placeholders", True))

        option_keys = item.get("val_option_keys", ["options", "choices", "option"])
        if not isinstance(option_keys, list) or not option_keys:
            option_keys = ["options", "choices", "option"]
        option_keys_str = ",".join(str(x).strip() for x in option_keys if str(x).strip())
        if not option_keys_str:
            option_keys_str = "options,choices,option"

        if not val_file:
            print(f"[skip] {name}: missing val_file")
            failed.append(name)
            continue
        val_path = Path(val_file)
        if not val_path.is_absolute():
            val_path = visplay_root / val_path
        if not val_path.exists():
            print(f"[skip] {name}: val_file does not exist -> {val_path}")
            failed.append(name)
            continue
        val_file = str(val_path)

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        if args.pred_out_mode == "registry":
            if not pred_out_cfg:
                print(f"[skip] {name}: pred_out_mode=registry but pred_out is empty")
                failed.append(name)
                continue
            pred_out = pred_out_cfg.format(
                model_name=args.model_name,
                experiment_name=args.experiment_name,
                dataset=name,
                dataset_safe=safe_name,
            )
            pred_out_path = Path(pred_out)
            if not pred_out_path.is_absolute():
                pred_out_path = visplay_root / pred_out_path
            pred_out = str(pred_out_path)
        else:
            pred_out = str(
                Path(args.output_root)
                / args.experiment_name
                / f"{safe_name}_{args.model_name}_project.jsonl"
            )

        Path(pred_out).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "verl.trainer.main",
            f"config={args.config}",
            "data.train_files=../datasets/MMStar",
            f"data.val_files={val_file}",
            f"data.prompt_key={prompt_key}",
            f"data.answer_key={answer_key}",
            f"data.image_key={image_key}",
            f"data.val_include_options_in_prompt={'true' if val_include_options_in_prompt else 'false'}",
            f"data.val_force_choice_letter_output={'true' if val_force_choice_letter_output else 'false'}",
            f"data.val_force_yes_no_output={'true' if val_force_yes_no_output else 'false'}",
            f"data.val_option_keys=[{option_keys_str}]",
            f"data.eval_normalize_image_placeholders={'true' if eval_normalize_image_placeholders else 'false'}",
            f"worker.actor.model.model_path={args.model_path}",
            f"worker.rollout.max_model_len={args.max_model_len}",
            f"worker.rollout.n={args.rollout_n}",
            f"worker.rollout.tensor_parallel_size={args.tensor_parallel_size}",
            "trainer.total_epochs=1",
            f"trainer.experiment_name={args.experiment_name}",
            "trainer.save_checkpoint_path=./Evaluation/eval_outputs",
            f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
            "worker.actor.micro_batch_size_per_device_for_experience=1",
            "worker.actor.global_batch_size=8",
            "data.format_prompt=./train_examples/format_prompt/solver.jinja",
            "trainer.val_only=true",
            "trainer.logger=[console]",
            f"trainer.response_path={pred_out}",
        ]

        print("=" * 80)
        print(f"[run] dataset={name}")
        print(" ".join(cmd))

        if args.dry_run:
            succeeded.append(name)
            continue

        try:
            subprocess.run(cmd, cwd=visplay_root, check=True, env=os.environ.copy())
            succeeded.append(name)
        except subprocess.CalledProcessError as exc:
            print(f"[fail] {name}: returncode={exc.returncode}")
            failed.append(name)

    print("=" * 80)
    print(f"done. success={len(succeeded)} failed={len(failed)}")
    if succeeded:
        print("success:", ", ".join(succeeded))
    if failed:
        print("failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
