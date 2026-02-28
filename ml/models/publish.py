# ml/models/publish.py
"""
Upload a trained LoRA adapter to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish adapter to HF Hub")
    ap.add_argument(
        "--repo",
        required=True,
        help="HF repo id (e.g., myorg/ministral14b-lora-fi-cc0)",
    )
    ap.add_argument(
        "--adapter", type=Path, required=True, help="Path to adapter directory"
    )
    a = ap.parse_args()

    api = HfApi()
    api.create_repo(repo_id=a.repo, exist_ok=True)
    api.upload_folder(
        repo_id=a.repo,
        folder_path=str(a.adapter),
        commit_message="Upload adapter",
    )
    print(f"Published to https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
