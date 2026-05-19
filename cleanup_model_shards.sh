#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash cleanup_model_shards.sh <target_dir> [--apply] [--yes]

Notes:
  - Dry-run mode is used by default. It only prints files that would be deleted.
  - Pass --apply to delete files.
  - Pass --yes to skip confirmation.

Deleted filename patterns:
  extra_state_world_size_*_rank_*.pt
  model_world_size_*_rank_*.pt
  optim_world_size_*_rank_*.pt

Examples:
  bash cleanup_model_shards.sh "/path/to/global_step_5/actor"
  bash cleanup_model_shards.sh "/path/to/global_step_5/actor" --apply
  bash cleanup_model_shards.sh "/path/to/global_step_5/actor" --apply --yes
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

target_dir="$1"
shift || true

apply=false
assume_yes=false

for arg in "$@"; do
  case "$arg" in
    --apply) apply=true ;;
    --yes) assume_yes=true ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$target_dir" ]]; then
  echo "Directory not found: $target_dir"
  exit 1
fi

# Recursively scan the target directory.
mapfile -t files < <(
  find "$target_dir" -type f \( \
    -name "extra_state_world_size_*_rank_*.pt" -o \
    -name "model_world_size_*_rank_*.pt" -o \
    -name "optim_world_size_*_rank_*.pt" \
  \) | sort -u
)

count="${#files[@]}"
echo "Target directory: $target_dir"
echo "Matched shard files: $count"

if [[ "$count" -eq 0 ]]; then
  echo "No shard files to delete."
  exit 0
fi

printf '%s\n' "${files[@]}"

if [[ "$apply" != true ]]; then
  echo
  echo "[Dry run] No files were deleted."
  echo "Pass --apply to delete files."
  exit 0
fi

if [[ "$assume_yes" != true ]]; then
  echo
  read -r -p "Delete the $count files listed above? [y/N] " ans
  if [[ ! "$ans" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
  fi
fi

rm -f -- "${files[@]}"
echo "Deleted $count files."
