#!/usr/bin/env python3
"""Upload trained StopWise LoRA checkpoints to a Hugging Face Hub repository."""

from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Push StopWise LoRA checkpoints to Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target Hugging Face repo ID, e.g. 'your-username/stopwise-qwen2.5-3b-lora'",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints/full",
        help="Path to the checkpoints directory (default: checkpoints/full)",
    )
    parser.add_argument(
        "--include-intermediate-steps",
        action="store_true",
        help="Also upload intermediate step_25/step_50/step_75 subfolders (default: False, uploads final best adapters only).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face repository as private.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoints_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_path}")

    api = HfApi()
    url = api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Repository ready: {url}")

    ignore_patterns = [] if args.include_intermediate_steps else ["*/step_*/*", "*/step_*"]
    print(f"Uploading {ckpt_path} to {args.repo_id} (ignoring: {ignore_patterns})...")
    api.upload_folder(
        folder_path=str(ckpt_path),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
        commit_message="Upload StopWise trained LoRA adapters (Methods A, B, C, E6 transfer, E8 ablations)",
    )
    print(f"Done! View your weights at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
