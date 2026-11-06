"""Automated check ensuring zero test-set task leakage into any training split."""

from __future__ import annotations

from pathlib import Path

from src.utils.io import read_jsonl


def test_no_train_test_leakage() -> None:
    for subdir in ["splits_smoke", "splits"]:
        splits_dir = Path("data") / subdir
        if not splits_dir.exists():
            continue
        for domain in ["search_qa", "math", "code"]:
            train_path = splits_dir / f"{domain}_train.jsonl"
            val_path = splits_dir / f"{domain}_val.jsonl"
            test_path = splits_dir / f"{domain}_test.jsonl"
            if not (train_path.exists() and test_path.exists()):
                continue
            train_rows = read_jsonl(train_path)
            val_rows = read_jsonl(val_path) if val_path.exists() else []
            test_rows = read_jsonl(test_path)

            train_ids = {r["task_id"] for r in train_rows}
            val_ids = {r["task_id"] for r in val_rows}
            test_ids = {r["task_id"] for r in test_rows}

            assert len(train_ids & test_ids) == 0, f"Task ID leakage in {domain} ({subdir})"
            assert len(val_ids & test_ids) == 0, f"Val/Test ID leakage in {domain} ({subdir})"

            train_questions = {r["question"].strip() for r in train_rows}
            test_questions = {r["question"].strip() for r in test_rows}
            assert len(train_questions & test_questions) == 0, f"Question text leakage in {domain} ({subdir})"

