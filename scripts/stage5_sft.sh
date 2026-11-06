#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

.venv/bin/python -m src.train.run_stage5 --config configs/default.yaml "$@"

