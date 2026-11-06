#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

.venv/bin/python -m src.eval.make_figures --results-dir results --prefix ""
echo "Regenerated all report figures in results/ from CSVs."

