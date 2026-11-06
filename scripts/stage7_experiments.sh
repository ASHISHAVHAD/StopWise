#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

.venv/bin/python -m src.eval.run_stage7 --config configs/default.yaml "$@"

