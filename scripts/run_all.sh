#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

bash scripts/stage1_data_and_tools.sh
bash scripts/stage2_collect_runs.sh "$@"
bash scripts/stage3_cutoff_labels.sh "$@"
bash scripts/stage4_baselines.sh "$@"
bash scripts/stage5_sft.sh "$@"
bash scripts/stage6_grpo.sh "$@"
bash scripts/stage7_experiments.sh "$@"
bash scripts/make_figures.sh
.venv/bin/pytest tests/ -v

