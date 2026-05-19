import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

from huggingface_hub import snapshot_download


def download_parquet_dataset(repo_id: str) -> None:
    dataset_root = Path(os.getenv("RISE_DATASET_ROOT", "../datasets/parquet"))
    output_dir = dataset_root / repo_id.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=["*.parquet"],
        max_workers=1, 
    )
    print(f"completed parquet download of {repo_id} -> {output_dir}")


# Login first: huggingface-cli login
dataset_repos = [
    "zli12321/MMMU",
    "zli12321/mm-vet",
    "zli12321/realWorldQA",
    "zli12321/mathVision",
    "zli12321/mathVerse",
    "LMMs-Lab-Turtle/Vision-SR1-47K",
    "zli12321/mmstar",
    "zli12321/mathvista",
    "zli12321/ChartQA",
]

for repo in dataset_repos:
    download_parquet_dataset(repo)
