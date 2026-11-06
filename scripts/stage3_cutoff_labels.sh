#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

.venv/bin/python -m src.labels.run_stage3 --config configs/default.yaml "$@"

