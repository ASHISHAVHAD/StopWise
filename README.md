# StopWise: Teaching AI Agents When to Stop

**StopWise** is an end-to-end framework for training and evaluating adaptive stopping policies in multi-step tool-using LLM agents across three domains: **Multi-Hop Search QA with RAG**, **Mathematical Reasoning with Exact Tools**, and **Python Code Generation with Sandbox Unit Tests**.

At every step $t$, the agent outputs a structured `<think>...</think>` trace, one action `<action>CONTINUE|GATHER|VERIFY|ANSWER</action>` with `<args>...</args>`, and a calibrated readiness score `<ready>0.xx</ready>`. We obtain step-level supervision automatically via the **cut-off test**—forcing answers at intermediate steps $t$ across saved trajectories and checking correctness—and compare three training paradigms:
- **Method A (SFT on Cut-Off Labels)**: Supervised LoRA fine-tuning on balanced ready/not-ready trajectory prefixes.
- **Method B (GRPO with Cost-Aware Reward)**: Group Relative Policy Optimization with `<result>` span masking and reward $R = \text{correct} - \lambda \cdot \text{effort} - 0.1 \cdot \text{invalid\_actions}$.
- **Method C (SFT $\rightarrow$ GRPO)**: Initializing from Method A and refining with cost-aware GRPO.

---

## Repository Structure

```text
stopwise/
  configs/default.yaml       # Single source of truth for models, seeds, budgets, and rewards
  data/                      # Deterministic train/val/test splits and hybrid BM25+FAISS index
  src/
    tasks/                   # Dataset loaders & checkers: search_qa.py, math.py, code.py, prepare_data.py
    tools/                   # Hybrid retriever.py (BM25 + FAISS + CrossEncoder), sandbox.py, calculator.py
    agent/                   # prompt.py, action_parser.py, loop.py, readiness_head.py
    runs/                    # collect_runs.py (multi-sample + few-shot long-budget trajectory logger)
    labels/                  # cutoff_labels.py, hindsight.py, run_stage3.py
    train/                   # reward.py, sft.py, grpo.py, run_stage5.py, run_stage6.py
    baselines/               # fixed_limit.py, own_decision.py, confidence_rule.py, hindsight_best.py, run_stage4.py
    eval/                    # evaluate.py, metrics.py, stats.py, calibration.py, failure_taxonomy.py, make_figures.py
    utils/                   # config.py, io.py
  tests/                     # Unit & leakage test suite (checkers, sandbox, cut-off labels, reward, masking)
  scripts/                   # Reproducible stage scripts + make_figures.sh + run_all.sh
  results/                   # CSV tables, PNG figures, JSON summaries, ENVIRONMENT.md, execution logs
  REPORT.md                  # Full technical report with tables, figures, and hypothesis evaluation
  PROGRESS.md                # Stage-by-stage progress and runtime log
  DECISIONS.md               # Architectural and experimental design decisions
```

---

## Environment Setup

```bash
# Using the preconfigured virtual environment (.venv) on NVIDIA RTX 6000 Ada (48 GB VRAM)
.venv/bin/pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

---

## Reproducing Each Pipeline Stage (Resumable & Idempotent)

Every script caches its intermediate and final outputs (`data/`, `runs/`, `checkpoints/`, `results/`). Re-running any stage script skips completed work automatically. Pass `--smoke` to run the fast 20-task smoke test first.

| Stage | Command | Description & Acceptance Check | Expected Runtime (1x RTX 6000 Ada) |
| :--- | :--- | :--- | :--- |
| **Unit Tests** | `.venv/bin/pytest tests/ -v` | Runs 12 unit & train/test leakage checks on checkers, sandbox, cut-off labels, and GRPO reward/masking. | ~6 sec |
| **Stage 1: Data & Tools** | `bash scripts/stage1_data_and_tools.sh` | Downloads datasets, creates deterministic splits, builds BM25+FAISS+CrossEncoder index, checks Recall@3 (0.98) & sandbox timeout. | ~3 min |
| **Stage 2: Collect Runs** | `bash scripts/stage2_collect_runs.sh` | Collects multi-sample base + few-shot trajectories across Search QA, Maths, and Code; verifies $\ge 25\%$ success rate. | ~12–18 min |
| **Stage 3: Cut-Off Labels** | `bash scripts/stage3_cutoff_labels.sh` | Forces 5 answers per step $t$, computes `ready_fraction_t` and hindsight stop steps, plots readiness curve. | ~15–20 min |
| **Stage 4: Baselines (E1, E2)** | `bash scripts/stage4_baselines.sh` | Evaluates problem size (E1) and Fixed Limits (1..10), Own Decision, Confidence Rule, and Hindsight Best (E2). | ~15 sec |
| **Stage 5: Method A (SFT, E3)** | `bash scripts/stage5_sft.sh` | Trains balanced LoRA SFT model (`r=16, alpha=32`) on cut-off labels and runs online evaluation (E3). | ~6–8 min |
| **Stage 6: Methods B & C (GRPO, E4, E5)** | `bash scripts/stage6_grpo.sh` | Trains GRPO models across $\lambda \in \{0, 0.1, 0.3, 0.6\}$ from base (B) and SFT (C) with `<result>` span masking. | ~18–25 min |
| **Stage 7: Experiments E6–E11** | `bash scripts/stage7_experiments.sh` | Runs Cross-Domain Transfer (E6), Calibration (E7), Ablations (E8), Retrieval Quality (E9), Failure Taxonomy (E10), and 7B comparison (E11). | ~15–20 min |
| **Stage 8: Regenerate Figures** | `bash scripts/make_figures.sh` | Regenerates all PNG figures in `results/` directly from the CSV tables. | ~5 sec |
| **Full Pipeline** | `bash scripts/run_all.sh` | Runs all stages end-to-end, regenerates figures, and runs the test suite. | ~75–90 min |

