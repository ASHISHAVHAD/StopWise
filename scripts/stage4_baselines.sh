#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

.venv/bin/python -m src.baselines.run_stage4 --config configs/default.yaml "$@"

