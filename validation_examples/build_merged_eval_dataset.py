#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import yaml
from datasets import Dataset, Features, Image, Sequence as HFSequence, Value, concatenate_datasets, load_dataset


def find_rise_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "Evaluation").is_dir() and (candidate / "validation_examples").is_dir():
            return candidate
    raise RuntimeError(f"Cannot locate RISE project root from {start}")


def _normalize_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _normalize_option_keys(value: Any) -> List[str]:
    if isinstance(value, list):
        keys = [str(x).strip() for x in value if str(x).strip()]
        if keys:
            return keys
    return ["options", "choices", "option"]


def _signature_from_registry(ds: Dict[str, Any]) -> Dict[str, Any]:
    answer_format = str(ds.get("answer_format", "")).strip().lower()
    return {
        "prompt_key": str(ds.get("prompt_key", "question")).strip(),
        "answer_key": str(ds.get("answer_key", "answer")).strip(),
        "image_key": str(ds.get("image_key", "image")).strip(),
        "format_prompt": str(ds.get("format_prompt", "./train_examples/format_prompt/solver.jinja")).strip(),
        "val_include_options_in_prompt": _normalize_bool(ds.get("val_include_options_in_prompt"), False),
        "val_force_choice_letter_output": _normalize_bool(ds.get("val_force_choice_letter_output"), False),
        "val_force_yes_no_output": _normalize_bool(ds.get("val_force_yes_no_output"), answer_format == "yes_no"),
        "val_option_keys": _normalize_option_keys(ds.get("val_option_keys")),
        "eval_normalize_image_placeholders": _normalize_bool(ds.get("eval_normalize_image_placeholders"), True),
    }


def _signature_key(sig: Dict[str, Any]) -> str:
    return json.dumps(sig, ensure_ascii=False, sort_keys=True)


def _resolve_data_path(visplay_root: Path, path_str: str) -> Path:
    data_path = Path(path_str)
    if not data_path.is_absolute():
        data_path = visplay_root / data_path
    return data_path


def _file_stamp(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_local_table(path: Path):
    if path.is_dir():
        arrow_files = sorted(glob(str(path / "*.arrow")))
        parquet_files = sorted(glob(str(path / "*.parquet")))
        if arrow_files:
            return load_dataset("arrow", data_files=arrow_files, split="train")
        if parquet_files:
            return load_dataset("parquet", data_files=parquet_files, split="train")
        return load_dataset("parquet", data_dir=str(path), split="train")

    if path.is_file():
        if path.suffix == ".arrow":
            return load_dataset("arrow", data_files=str(path), split="train")
        if path.suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        raise ValueError(f"Unsupported file suffix: {path}")

    raise FileNotFoundError(f"Data path not found: {path}")


def _normalize_orig_row_index(value: Any, fallback_idx: int) -> Any:
    if value is None:
        return fallback_idx
    text = str(value).strip()
    if text == "":
        return fallback_idx
    return value


def _id_fallback(value: Any, fallback_orig_row_index: Any) -> str:
    if value is None:
        return str(fallback_orig_row_index)
    text = str(value).strip()
    if text == "":
        return str(fallback_orig_row_index)
    return text


def _normalize_image_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item is not None]
    return [value]


def _build_canonical_dataset(table: Dataset, ds_name: str, signature: Dict[str, Any]) -> Dataset:
    prompt_key = signature["prompt_key"]
    answer_key = signature["answer_key"]
    image_key = signature["image_key"]

    if prompt_key not in table.column_names:
        raise ValueError(f"{ds_name}: missing prompt_key column `{prompt_key}`")
    if answer_key not in table.column_names:
        raise ValueError(f"{ds_name}: missing answer_key column `{answer_key}`")

    row_count = len(table)
    prompt_values = ["" if v is None else str(v) for v in table[prompt_key]]
    answer_values = ["" if v is None else str(v) for v in table[answer_key]]
    raw_images = table[image_key] if image_key in table.column_names else [None] * row_count
    image_values = [_normalize_image_list(v) for v in raw_images]

    has_orig = "orig_row_index" in table.column_names
    has_id = "id" in table.column_names
    orig_col = table["orig_row_index"] if has_orig else [None] * row_count
    id_col = table["id"] if has_id else [None] * row_count

    orig_row_values: List[str] = []
    id_values: List[str] = []
    ds_name_values: List[str] = [ds_name] * row_count
    for idx in range(row_count):
        orig_row_index = _normalize_orig_row_index(orig_col[idx], idx)
        orig_row_text = str(orig_row_index)
        orig_row_values.append(orig_row_text)
        id_base = _id_fallback(id_col[idx], orig_row_text)
        id_values.append(f"{ds_name}@@{id_base}")

    features = Features(
        {
            prompt_key: Value("string"),
            answer_key: Value("string"),
            image_key: HFSequence(Image(decode=False)),
            "orig_row_index": Value("string"),
            "id": Value("string"),
            "__dataset_name": Value("string"),
        }
    )
    return Dataset.from_dict(
        {
            prompt_key: prompt_values,
            answer_key: answer_values,
            image_key: image_values,
            "orig_row_index": orig_row_values,
            "id": id_values,
            "__dataset_name": ds_name_values,
        },
        features=features,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged eval datasets grouped by generation config signature")
    parser.add_argument("--registry", type=str, required=True, help="Path to registry yaml")
    parser.add_argument("--dataset", type=str, default="", help="Only include one dataset name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to place merged parquet files")
    parser.add_argument("--metadata_out", type=str, required=True, help="Output metadata json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    visplay_root = find_rise_root(script_dir)

    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = visplay_root / registry_path
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    with registry_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    datasets_cfg = cfg.get("datasets", [])
    if not isinstance(datasets_cfg, list):
        raise ValueError("registry yaml must contain `datasets` list")

    selected: List[Dict[str, Any]] = []
    cache_inputs: List[Dict[str, Any]] = []
    for ds in datasets_cfg:
        if not ds.get("enabled", True):
            continue
        name = str(ds.get("name", "")).strip()
        if not name:
            continue
        if args.dataset and name != args.dataset:
            continue
        val_file = str(ds.get("val_file", "")).strip()
        if not val_file:
            continue
        selected.append(ds)
        resolved_val_file = _resolve_data_path(visplay_root, val_file)
        cache_inputs.append(
            {
                "name": name,
                "val_file": str(resolved_val_file),
                "val_file_stat": _file_stamp(resolved_val_file),
                "signature": _signature_from_registry(ds),
            }
        )

    if not selected:
        raise RuntimeError("No datasets selected from registry")
    print(f"[merged-build] selected_datasets={len(selected)}", flush=True)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    signatures: Dict[str, Dict[str, Any]] = {}
    for ds in selected:
        sig = _signature_from_registry(ds)
        key = _signature_key(sig)
        signatures[key] = sig
        grouped[key].append(ds)

    group_metas: List[Dict[str, Any]] = []
    group_serial = 0
    for sig_key, ds_list in grouped.items():
        signature = signatures[sig_key]
        print(
            "[merged-build] start signature-group "
            f"datasets={','.join(str(x.get('name', '')).strip() for x in ds_list)}",
            flush=True,
        )
        datasets_to_concat: List[Dataset] = []
        per_dataset_rows: Dict[str, int] = {}
        total_rows = 0

        for ds in ds_list:
            name = str(ds["name"]).strip()
            data_path = _resolve_data_path(visplay_root, str(ds["val_file"]).strip())
            print(f"[merged-build] loading {name} from {data_path}", flush=True)
            table = _load_local_table(data_path)
            print(f"[merged-build] loaded {name} rows={len(table)}", flush=True)
            patched = _build_canonical_dataset(table, name, signature)
            print(f"[merged-build] canonicalized {name} rows={len(patched)}", flush=True)
            datasets_to_concat.append(patched)
            per_dataset_rows[name] = len(patched)
            total_rows += len(patched)

        if not datasets_to_concat:
            continue

        group_serial += 1
        group_id = f"group_{group_serial:03d}"
        merged_ds = datasets_to_concat[0] if len(datasets_to_concat) == 1 else concatenate_datasets(datasets_to_concat)
        merged_file = output_dir / f"{group_id}.parquet"
        print(
            f"[merged-build] writing {group_id} rows={total_rows} to {merged_file}",
            flush=True,
        )
        merged_ds.to_parquet(str(merged_file))

        group_metas.append(
            {
                "group_id": group_id,
                "signature": signature,
                "datasets": [str(x.get("name", "")).strip() for x in ds_list],
                "merged_file": str(merged_file),
                "rows": total_rows,
                "per_dataset_rows": per_dataset_rows,
            }
        )

    if not group_metas:
        raise RuntimeError("No merged groups generated")

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "cache_format_version": 1,
        "registry": str(registry_path),
        "dataset_filter": args.dataset,
        "selected_datasets": [str(x.get("name", "")).strip() for x in selected],
        "cache_inputs": cache_inputs,
        "groups": group_metas,
    }
    with metadata_out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[merged-build] groups={len(group_metas)}")
    print(f"[merged-build] metadata={metadata_out}")
    for group in group_metas:
        print(
            "[merged-build] "
            f"{group['group_id']} rows={group['rows']} "
            f"datasets={','.join(group['datasets'])} file={group['merged_file']}"
        )


if __name__ == "__main__":
    main()
