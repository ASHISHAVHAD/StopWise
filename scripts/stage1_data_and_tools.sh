#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

echo "=== Stage 1 (Smoke Test) ==="
.venv/bin/python -m src.tasks.prepare_data --config configs/default.yaml --smoke

echo "=== Stage 1 (Full Dataset Splits & Hybrid Index) ==="
.venv/bin/python -m src.tasks.prepare_data --config configs/default.yaml

echo "=== Stage 1 (Unit & Leakage Tests) ==="
.venv/bin/pytest tests/ -v
