# StopWise Project Progress (`PROGRESS.md`)

## Current Status (Updated 2026-11-07 14:00 IST)
- **Currently Running**: **NONE** (Full-scale Stage 3 completed; paused awaiting user confirmation to launch **Stage 4: Baselines & Problem Size Measurement (E1 & E2)**).

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
- Generated [stage3_cutoff_summary.json](results/stage3_cutoff_summary.json), [stage3_ready_by_step.csv](results/stage3_ready_by_step.csv), and [stage3_ready_by_step.png](results/stage3_ready_by_step.png).
- **Per-Split Cut-Off Label Counts & Mean Wasted Steps**:
  - `search_qa_train` (480 runs): **1,743** cut-offs (`421` ready / `1,322` not_ready), mean wasted steps = **1.1646**
  - `math_train` (480 runs): **1,320** cut-offs (`523` ready / `797` not_ready), mean wasted steps = **1.0479**
  - `code_train` (480 runs): **1,859** cut-offs (`769` ready / `1,090` not_ready), mean wasted steps = **1.4229**
  - `search_qa_val` (30 runs): **118** cut-offs (`30` ready / `88` not_ready), mean wasted steps = **1.2333**
  - `math_val` (30 runs): **66** cut-offs (`24` ready / `42` not_ready), mean wasted steps = **0.9000**
  - `code_val` (30 runs): **126** cut-offs (`17` ready / `109` not_ready), mean wasted steps = **0.6333**
  - `search_qa_test` (200 runs): **701** cut-offs (`152` ready / `549` not_ready), mean wasted steps = **1.1350**
  - `math_test` (200 runs): **421** cut-offs (`183` ready / `238` not_ready), mean wasted steps = **0.9600**
  - `code_test` (200 runs): **660** cut-offs (`120` ready / `540` not_ready), mean wasted steps = **0.6100**

---

## Stage Checklist
- [x] **Stage 0**: Environment verification, git init, directory layout, `configs/default.yaml`
- [x] **Stage 1 (Full & Smoke)**: Data splits, passage index (14,704 passages, Recall@3 = 0.9800), sandbox, calculator, checkers, 12/12 unit & leakage tests
- [x] **All Smoke Tests (Stages 2–7 + Figures)**: Completed & verified
- [x] **Stage 2 (Full)**: Collect base agent runs (2,130 trajectories, 52.82% overall success rate)
- [x] **Stage 3 (Full)**: Cut-off label builder & hindsight stop computation (7,014 cut-offs, 35,070 forced answers)
- [ ] **Stage 4 (Full)**: Baselines & E1/E2 problem measurement (awaiting user approval — ~15 seconds)
- [ ] **Stage 5 (Full)**: Method A (SFT) & E3 evaluation
- [ ] **Stage 6 (Full)**: Methods B and C (GRPO $\lambda$ sweep) & E4/E5 evaluation
- [ ] **Stage 7 (Full)**: Experiments E6, E7, E8, E9, E10, E11
- [ ] **Stage 8**: Regenerate all figures via `scripts/make_figures.sh`, run quality checks, write `REPORT.md`
