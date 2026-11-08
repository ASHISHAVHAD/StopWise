# StopWise Project Progress (`PROGRESS.md`)

## Current Status (Updated 2026-11-08 09:04 IST)
- **Currently Running**: **NONE** (Full-scale Stage 7 completed; paused awaiting user confirmation to launch final **Stage 8: Regenerate Full-Scale Figures, Run Quality Checks, and Write `REPORT.md`**).
## Current Status (Updated 2026-11-08 11:34 IST)
- **Currently Running**: **NONE** — **ALL STAGES (0–8), FULL-SCALE EXPERIMENTS (E1–E11), FIGURES, TESTS, AND `REPORT.md` ARE 100% COMPLETED.**

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

### Stage 6: Methods B & C (GRPO $\lambda$ Sweep) & E4/E5 Evaluation — **COMPLETED**
- Trained all **6** full-scale GRPO models (`80` steps $\times$ `8` group rollouts each = `3,840` training rollouts) in `checkpoints/full/method_b_lam_*` and `checkpoints/full/method_c_lam_*` with `<result>` span masking and KL penalty (`0.01`).
- Evaluated all **6** models online across the `300` test tasks (`1,800` test trajectories) and saved [e4_method_b_lambda_sweep.csv](results/e4_method_b_lambda_sweep.csv), [e5_method_c.csv](results/e5_method_c.csv), and [stage6_grpo_summary.json](results/stage6_grpo_summary.json).
- **E4 Key Findings (Method B GRPO from Base)**:
  - `search_qa`: $\lambda=0.0$ reaches **39.0% ± 9.5%** (4.30 steps); $\lambda=0.3$ reaches **36.0% ± 9.0%** (4.01 steps); $\lambda=0.6$ reaches **35.0% ± 8.5%** (3.96 steps) — all improving over Untrained (32.5%, 4.21 steps).
  - `math`: $\lambda=0.3$ reaches **64.0% ± 9.5%** (3.20 steps) and $\lambda=0.6$ reaches **65.0% ± 9.0%** (**2.97 ± 0.21** steps) — beating Untrained (62.5%, 3.07 steps) and Best Fixed-Limit (62.5%, 2.98 steps).
  - `code`: $\lambda=0.1$ reaches **60.0% ± 10.0%** (3.80 steps) and $\lambda=0.6$ reaches **64.0% ± 9.5%** (**3.63 ± 0.44** steps) — improving +7.0% over Untrained and Best Fixed-Limit (57.0%, 3.98 steps).
  - Step-1 reward hacking (`early_answer_rate`) remained **0.0%–1.0%** across all $\lambda \in \{0.0, 0.1, 0.3, 0.6\}$.
- **E5 Key Findings (Method C: Method A SFT $\rightarrow$ GRPO)**:
  - `code` ($\lambda=0.1$): **58.0% ± 10.0%** accuracy in **2.070 ± 0.055** steps (saving **1.91 steps** / 48% fewer steps than Untrained & Best Fixed-Limit while improving accuracy from 57.0% to 58.0%).
  - `search_qa` ($\lambda=0.1$): **35.0% ± 9.0%** accuracy in **2.100 ± 0.055** steps (50.1% fewer steps than Untrained 4.21, cutting wasted steps from `1.14` to `0.40 ± 0.10`).
  - `math` ($\lambda=0.3$): **46.0% ± 9.5%** accuracy in **2.070 ± 0.055** steps (constrained by the SFT initialization's 2-step horizon vs. Method B's 65.0% at 2.97 steps).

### Stage 7: Full-Scale Experiments E6–E11 — **COMPLETED**
- Saved [e6_transfer.csv](results/e6_transfer.csv), [e7_calibration_summary.csv](results/e7_calibration_summary.csv), [e7_calibration_bins.csv](results/e7_calibration_bins.csv), [e7_threshold_sweep.csv](results/e7_threshold_sweep.csv), [e8_ablations.csv](results/e8_ablations.csv), [e9_retrieval_quality.csv](results/e9_retrieval_quality.csv), [e10_failure_taxonomy.csv](results/e10_failure_taxonomy.csv), [e10_spot_check_30.csv](results/e10_spot_check_30.csv), [e11_model_comparison.csv](results/e11_model_comparison.csv), and [stage7_summary.json](results/stage7_summary.json).
- **E6 (Cross-Domain Transfer)**:
  - `search_qa+math → code`: **53.0% ± 9.5%** in **2.12** steps (retains 93% of in-domain SFT accuracy while cutting steps by 46.7% vs. Untrained 3.98 steps).
  - `search_qa+code → math`: **49.0% ± 10.0%** in **2.24** steps (exceeds in-domain SFT 47.0% in 2.14 steps).
  - `math+code → search_qa`: **28.0% ± 9.0%** in **2.30** steps (cuts steps by 45.4% and wasted steps `1.14 → 0.51` vs. Untrained).
- **E7 (Calibration & ReadinessHead Probe)**:
  - Trained linear `ReadinessHead` achieves **ECE = 0.0769** and **Brier = 0.1639** ($N=2,250$ steps), cutting calibration error by **53.1%** relative to untrained text `<ready>` (`ECE = 0.1640`, `Brier = 0.2471`).
- **E8 (Ablations on `search_qa`)**:
  - Removing `<ready>` output drops Method A accuracy from **35.0% to 31.0%** and doubles premature stops (`12.0% → 26.0%`).
  - Step-count-only effort in Method B (`λ=0.1`) reaches **34.0% ± 9.0%** (`4.10` steps) vs. **30.0%** (`4.30` steps) with combined token+step effort.
- **E9 (Retrieval Quality Sensitivity on `search_qa`)**:
  - Removing cross-encoder reranker drops accuracy from **32.0% to 27.0%** (`2.07 → 2.12` steps); `k=1` drops accuracy to **29.0%** (`2.19` steps); `k=5` maintains **32.0%** at higher step cost (`2.56` steps).
- **E10 (Failure Taxonomy & 30-Example Spot Check)**:
  - Method B ($\lambda=0.6$) achieves the lowest overall failure rate (**45.33%** vs. **49.33%** Untrained), cutting `stopped_too_early` in half (`18.0% → 9.33%`), while Method A/C virtually eliminate `reopened_source` (`4.0% → 0.0%–0.33%`) and `repeated_check` (`14.83% → 2.0%–4.67%`).
- **E11 (Model Scale Comparison on `search_qa`, $N=50$)**:
  - `Qwen2.5-3B-Instruct + StopWise Method C` reaches **44.0% ± 13.0%** accuracy in **2.12** steps (`0.48` wasted steps), outperforming `Qwen2.5-7B-Instruct` Untrained (**36.0% ± 13.0%** in **4.26** steps, `1.90` wasted steps).

### Stage 8: Final Figures, Quality Checks & `REPORT.md` — **COMPLETED**
- Regenerated all 11 full-scale publication-ready figures (`results/fig_e1_problem_size.png` through `results/fig_e11_model_comparison.png` and `results/fig_stage3_ready_by_step.png`) from the CSVs via `bash scripts/make_figures.sh`.
- Verified all 33 CSV files in `results/` are non-empty and valid.
- Verified **12/12** unit and train/test zero-leakage tests pass (`.venv/bin/pytest tests/ -v`).
- Wrote final comprehensive report in [REPORT.md](REPORT.md) covering Abstract, Methodology, Setup, E1–E11 tables/figures/interpretations, explicit evaluation of H1–H5, and Limitations.

---

## Stage Checklist
- [x] **Stage 0**: Environment verification, git init, directory layout, `configs/default.yaml`
- [x] **Stage 1 (Full & Smoke)**: Data splits, passage index (14,704 passages, Recall@3 = 0.9800), sandbox, calculator, checkers, 12/12 unit & leakage tests
- [x] **All Smoke Tests (Stages 2–7 + Figures)**: Completed & verified
- [x] **Stage 2 (Full)**: Collect base agent runs (2,130 trajectories, 52.82% overall success rate)
- [x] **Stage 3 (Full)**: Cut-off label builder & hindsight stop computation (7,014 cut-offs, 35,070 forced answers)
- [x] **Stage 4 (Full)**: Baselines & E1/E2 problem measurement (600 test runs, 1,000 bootstrap resamples)
- [x] **Stage 5 (Full)**: Method A (SFT) & E3 evaluation (3,022 examples, 81% val stop acc, online E3 complete)
- [x] **Stage 6 (Full)**: Methods B and C (GRPO $\lambda$ sweep) & E4/E5 evaluation (6 GRPO models, 1,800 online test trajectories)
- [x] **Stage 7 (Full)**: Experiments E6, E7, E8, E9, E10, E11 (all CSVs and 30-example spot check complete)
- [ ] **Stage 8**: Regenerate all figures via `scripts/make_figures.sh`, run quality checks, write `REPORT.md` (awaiting user approval)
- [x] **Stage 8**: Regenerate all figures via `scripts/make_figures.sh`, run quality checks, write `REPORT.md`
