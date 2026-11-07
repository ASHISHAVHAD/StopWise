# StopWise Project Progress (`PROGRESS.md`)

## Current Status (Updated 2026-11-07 15:04 IST)
- **Currently Running**: **NONE** (Full-scale Stage 5 completed; paused awaiting user confirmation to launch **Stage 6: Full-Scale Methods B & C (GRPO $\lambda$ Sweep) & E4/E5 Evaluation**).

---

## Completed Full-Scale Stages

### Stage 1: Full-Scale Data Splits & Hybrid Index — **COMPLETED**
- Indexed **14,704** unique Wikipedia passages (`BM25 + FAISS + Cross-Encoder`).
- Dev-slice Retrieval Recall@3 = **98.0%** (`results/stage1_data_summary.json`).
- Sandbox infinite-loop kill & network block verified; **12/12** unit & zero-leakage tests passed.

### Stage 2: Full-Scale Agent Trajectory Collection — **COMPLETED**
- Collected **2,130** full multi-step trajectories across `search_qa`, `math`, and `code` (`runs/full/*_runs.jsonl` and `runs/full/*_steps.jsonl`).
- **Overall Success Rate**: **52.82%** (`1,125 / 2,130` trajectories correct, passing the $\ge 25\%$ acceptance check; `results/stage2_runs_summary.json`).

### Stage 3: Full-Scale Cut-Off Labelling (`5` Forced Samples per Step) — **COMPLETED**
- Labelled all **2,130** trajectories across **7,014** intermediate step cut-offs (`35,070` forced-answer evaluations) and saved enriched trajectories to `runs/full/*_labelled.jsonl`.

### Stage 4: Baselines & Problem Size Measurement (E1 & E2) — **COMPLETED**
- Saved [e1_problem_size.csv](results/e1_problem_size.csv), [e2_baselines_curves.csv](results/e2_baselines_curves.csv), [e2_baselines_summary.csv](results/e2_baselines_summary.csv), and [stage4_summary.json](results/stage4_summary.json).
- **E1 Key Findings ($N=200$ test runs per domain)**:
  - `search_qa`: **46.5% ± 6.5%** continue after ready (**1.135 ± 0.238** wasted steps), **33.0% ± 6.0%** stop too early. Untrained accuracy = **32.5% ± 6.2%** (4.21 steps) vs. Hindsight Ceiling = **50.0% ± 6.5%** (3.07 steps).
  - `math`: **65.5% ± 6.5%** continue after ready (**0.960 ± 0.123** wasted steps), **33.5% ± 6.2%** stop too early. Untrained accuracy = **62.5% ± 6.5%** (3.07 steps) vs. Hindsight Ceiling = **67.0% ± 6.3%** (2.11 steps).
  - `code`: **56.5% ± 6.8%** continue after ready (**0.610 ± 0.083** wasted steps), **11.5% ± 4.3%** stop too early. Untrained accuracy = **57.0% ± 6.5%** (3.98 steps) vs. Hindsight Ceiling = **57.5% ± 6.8%** (3.37 steps).

### Stage 5: Method A (LoRA SFT on Cut-Off Labels) & E3 Evaluation — **COMPLETED**
- Built **3,022** balanced 50/50 `ready`/`not_ready` SFT training examples (`search_qa=548`, `math=1,040`, `code=1,434`) and trained `Qwen/Qwen2.5-3B-Instruct` LoRA (`r=16, alpha=32, lr=1e-4, 3 epochs, 1,134 steps`, final train loss `0.0651`) in `checkpoints/full/method_a_sft`.
- **Held-Out Validation Acceptance Check** ([stage5_sft_summary.json](results/stage5_sft_summary.json)):
  - Action accuracy = **72.0%**, Binary Stop/Continue accuracy = **81.0%**, Readiness MAE = **0.185**.
- **Online E3 Evaluation ($N=100$ test tasks per domain)** ([e3_method_a.csv](results/e3_method_a.csv)):
  - `search_qa`: Accuracy = **35.0% ± 9.0%** (vs. 32.5% Untrained), Mean Steps = **2.770 ± 0.190** (34.2% fewer steps than Untrained 4.21), Wasted Steps = **0.780 ± 0.220**.
  - `code`: Accuracy = **57.0% ± 9.5%** (matches Untrained & Best Fixed-Limit 57.0%), Mean Steps = **2.210 ± 0.090** (44.4% fewer steps than Untrained 3.98, saving **1.77 steps** vs. Best Fixed-Limit).
  - `math`: Accuracy = **47.0% ± 10.0%**, Mean Steps = **2.140 ± 0.070** (down from 3.07 steps; aggressive early stopping on math motivates RL verification in Methods B/C).

---

## Stage Checklist
- [x] **Stage 0**: Environment verification, git init, directory layout, `configs/default.yaml`
- [x] **Stage 1 (Full & Smoke)**: Data splits, passage index (14,704 passages, Recall@3 = 0.9800), sandbox, calculator, checkers, 12/12 unit & leakage tests
- [x] **All Smoke Tests (Stages 2–7 + Figures)**: Completed & verified
- [x] **Stage 2 (Full)**: Collect base agent runs (2,130 trajectories, 52.82% overall success rate)
- [x] **Stage 3 (Full)**: Cut-off label builder & hindsight stop computation (7,014 cut-offs, 35,070 forced answers)
- [x] **Stage 4 (Full)**: Baselines & E1/E2 problem measurement (600 test runs, 1,000 bootstrap resamples)
- [x] **Stage 5 (Full)**: Method A (SFT) & E3 evaluation (3,022 examples, 81% val stop acc, online E3 complete)
- [ ] **Stage 6 (Full)**: Methods B and C (GRPO $\lambda$ sweep) & E4/E5 evaluation (awaiting user approval)
- [ ] **Stage 7 (Full)**: Experiments E6, E7, E8, E9, E10, E11
- [ ] **Stage 8**: Regenerate all figures via `scripts/make_figures.sh`, run quality checks, write `REPORT.md`
