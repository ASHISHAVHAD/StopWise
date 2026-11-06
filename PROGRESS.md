# StopWise Project Progress (`PROGRESS.md`)

## Current Status (Updated 2026-11-06 21:33 IST)
- **Currently Running**: **NONE** (Full-scale Stage 2 completed; paused awaiting user confirmation to launch **Stage 3: Full-Scale Cut-Off Labelling**).

---

## Completed Full-Scale Stages

### Stage 1: Full-Scale Data Splits & Hybrid Index — **COMPLETED**
- Indexed **14,704** unique Wikipedia passages (`BM25 + FAISS + Cross-Encoder`).
- Dev-slice Retrieval Recall@3 = **98.0%** (`results/stage1_data_summary.json`).
- Sandbox infinite-loop kill & network block verified; **12/12** unit & zero-leakage tests passed.

### Stage 2: Full-Scale Agent Trajectory Collection — **COMPLETED**
- Collected **2,130** full multi-step trajectories across `search_qa`, `math`, and `code` (`runs/full/*_runs.jsonl` and `runs/full/*_steps.jsonl`).
- **Overall Success Rate**: **52.82%** (`1,125 / 2,130` trajectories correct, passing the $\ge 25\%$ acceptance check; `results/stage2_runs_summary.json`).
- **Per-Split Breakdown**:
  - `search_qa_train`: 480 runs (120 tasks $\times 4$), 173 correct (**36.04%**), mean steps = **4.22**
  - `math_train`: 480 runs (120 tasks $\times 4$), 307 correct (**63.96%**), mean steps = **3.54**
  - `code_train`: 480 runs (120 tasks $\times 4$), 298 correct (**62.08%**), mean steps = **4.28**
  - `search_qa_val`: 30 runs, 11 correct (**36.67%**), mean steps = **4.57**
  - `math_val`: 30 runs, 17 correct (**56.67%**), mean steps = **3.20**
  - `code_val`: 30 runs, 15 correct (**50.00%**), mean steps = **4.70**
  - `search_qa_test`: 200 runs (100 tasks $\times 2$), 65 correct (**32.50%**), mean steps = **4.21**
  - `math_test`: 200 runs (100 tasks $\times 2$), 125 correct (**62.50%**), mean steps = **3.07**
  - `code_test`: 200 runs (100 tasks $\times 2$), 114 correct (**57.00%**), mean steps = **3.98**

---

## Stage Checklist
- [x] **Stage 0**: Environment verification, git init, directory layout, `configs/default.yaml`
- [x] **Stage 1 (Full & Smoke)**: Data splits, passage index (14,704 passages, Recall@3 = 0.9800), sandbox, calculator, checkers, 12/12 unit & leakage tests
- [x] **All Smoke Tests (Stages 2–7 + Figures)**: Completed & verified
- [x] **Stage 2 (Full)**: Collect base agent runs (2,130 trajectories, 52.82% overall success rate)
- [ ] **Stage 3 (Full)**: Cut-off label builder & hindsight stop computation (awaiting user approval)
- [ ] **Stage 4 (Full)**: Baselines & E1/E2 problem measurement
- [ ] **Stage 5 (Full)**: Method A (SFT) & E3 evaluation
- [ ] **Stage 6 (Full)**: Methods B and C (GRPO $\lambda$ sweep) & E4/E5 evaluation
- [ ] **Stage 7 (Full)**: Experiments E6, E7, E8, E9, E10, E11
- [ ] **Stage 8**: Regenerate all figures via `scripts/make_figures.sh`, run quality checks, write `REPORT.md`
