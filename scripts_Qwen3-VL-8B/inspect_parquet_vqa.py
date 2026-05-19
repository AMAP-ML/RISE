#!/usr/bin/env python3
import argparse
import os
from typing import Any

import pandas as pd


def _to_text(value: Any, max_len: int) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _format_images(value: Any, max_len: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        text = ", ".join(str(v) for v in value)
    else:
        text = str(value)
    return _to_text(text, max_len)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect VQA parquet data and print readable samples."
    )
    parser.add_argument("parquet_path", type=str, help="Path to parquet file")
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=5,
        help="How many samples to print (default: 5)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=240,
        help="Max length for printed text fields (default: 240)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample rows instead of taking the first n",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed when using --random"
    )
    args = parser.parse_args()

    parquet_path = args.parquet_path
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    total = len(df)
    print(f"File: {parquet_path}")
    print(f"Total rows: {total}")
    print(f"Columns: {list(df.columns)}")
    if total == 0:
        print("Empty parquet file.")
        return

    n = min(max(args.num_samples, 1), total)
    sampled = df.sample(n=n, random_state=args.seed) if args.random else df.head(n)

    print("")
    print(f"Showing {n} sample(s):")
    for idx, (_, row) in enumerate(sampled.iterrows(), start=1):
        problem = _to_text(row.get("problem", ""), args.max_len)
        answer = _to_text(row.get("answer", ""), args.max_len)
        images = _format_images(row.get("images", ""), args.max_len)
        score = row.get("score", "N/A")
        problem_type = row.get("problem_type", "N/A")

        print("-" * 80)
        print(f"[Sample {idx}]")
        print(f"Type : {problem_type}")
        print(f"Score: {score}")
        print(f"Image: {images}")
        print(f"Q    : {problem}")
        print(f"A    : {answer}")


if __name__ == "__main__":
    main()
