#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, TextIO


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split merged prediction jsonl into per-dataset jsonl files")
    parser.add_argument("--combined_pred_file", type=str, required=True, help="Combined prediction jsonl path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for per-dataset files")
    parser.add_argument("--model_name", type=str, required=True, help="Model name used in output filename")
    parser.add_argument(
        "--allowed_datasets",
        type=str,
        default="",
        help="Comma-separated dataset names; when set, unknown dataset prefixes are skipped",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined_path = Path(args.combined_pred_file)
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined prediction file not found: {combined_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed = {x.strip() for x in args.allowed_datasets.split(",") if x.strip()}
    writers: Dict[str, TextIO] = {}
    counts: Dict[str, int] = defaultdict(int)
    malformed = 0
    skipped_unknown = 0

    try:
        with combined_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    malformed += 1
                    continue

                sample_id = row.get("id")
                if sample_id is None:
                    malformed += 1
                    continue
                sample_id = str(sample_id)
                if "@@" not in sample_id:
                    malformed += 1
                    continue

                ds_name = sample_id.split("@@", 1)[0].strip()
                if not ds_name:
                    malformed += 1
                    continue
                if allowed and ds_name not in allowed:
                    skipped_unknown += 1
                    continue

                safe_ds = sanitize_name(ds_name)
                out_path = output_dir / f"{safe_ds}_{args.model_name}_project.jsonl"
                writer = writers.get(safe_ds)
                if writer is None:
                    writer = out_path.open("a", encoding="utf-8")
                    writers[safe_ds] = writer

                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[safe_ds] += 1
    finally:
        for writer in writers.values():
            writer.close()

    print(
        "[merged-split] "
        f"datasets={len(counts)} rows={sum(counts.values())} "
        f"malformed={malformed} skipped_unknown={skipped_unknown}"
    )
    for ds_name, n in sorted(counts.items()):
        print(f"[merged-split] {ds_name}: {n}")


if __name__ == "__main__":
    main()
