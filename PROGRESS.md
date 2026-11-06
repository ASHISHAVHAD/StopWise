# StopWise Project Progress (`PROGRESS.md`)

## Current Status (Updated 2026-11-06 20:11 IST)
- **Currently Running**: **NONE** (All smoke-test stages 1–7 and smoke figures are **100% COMPLETED** and committed; paused awaiting user instruction to launch full-scale stages).

---

## Completed Smoke Test Summary (Stages 1–7)

| Stage | Status | Key Acceptance Metrics & Outputs |
| :--- | :--- | :--- |
| **Stage 1: Data & Tools (Full + Smoke)** | **COMPLETED** | 14,704 unique Wikipedia passages indexed (BM25 + FAISS + Cross-Encoder); Dev-slice Recall@3 = **98.0%**; sandbox infinite-loop kill & network block verified; **12/12** unit & zero-leakage tests passed. |
| **Stage 2: Collect Runs (Smoke)** | **COMPLETED** | 138 trajectories across `search_qa`, `math`, and `code`; overall success rate = **60.87%** ($\ge 25\%$ check passed). |
| **Stage 3: Cut-Off Labels (Smoke)** | **COMPLETED** | 351 step cut-offs labelled (3 forced answers/step); mean wasted steps: `search_qa=0.95`, `math=1.00`, `code=0.60`. Saved `results/stage3_smoke_ready_by_step.csv` & `.png`. |
| **Stage 4: Baselines E1 & E2 (Smoke)** | **COMPLETED** | **E1**: 50%–65% of untrained runs continue after ready; 10%–45% stop too early. **E2**: Fixed limits (1..10), Own Decision, Confidence Rule, and Hindsight Best curves saved in `results/smoke_e2_baselines_summary.csv`. |
| **Stage 5: Method A SFT & E3 (Smoke)** | **COMPLETED** | Trained LoRA (`r=16, alpha=32`) on 106 balanced cut-off examples; held-out binary stop accuracy = **79.41%**, action accuracy = **55.88%**. Saved `results/smoke_e3_method_a.csv`. |
| **Stage 6: Methods B & C GRPO (Smoke)** | **COMPLETED** | Trained GRPO for $\lambda \in \{0.0, 0.1, 0.3, 0.6\}$ (Method B) and Method C (SFT $\rightarrow$ GRPO $\lambda=0.1$). Method C achieves **70.0%** accuracy in **2.70** steps on `math` and **70.0%** accuracy in **3.40** steps on `code`. Saved `results/smoke_e4_method_b_lambda_sweep.csv` and `results/smoke_e5_method_c.csv`. |
| **Stage 7: Experiments E6–E10 (Smoke)** | **COMPLETED** | **E6 Transfer**: Out-of-domain SFT reaches 60% on held-out `math` and `code` (saving 0.55 steps vs fixed limit). **E7 Calibration**: Linear `ReadinessHead` reduces ECE from **0.2243** to **0.0762**. **E8 Ablations**: All 5 Method A & B ablations evaluated. **E9 Retrieval**: Tested `k=1, 3, 5` & no-reranker (`mean_steps` adapts from 3.7 at `k=3` to 4.2 at `k=1` and 5.8 at `k=5`). **E10 Failure Taxonomy**: Method C cuts premature-stop errors from 21.7% to 3.3%. |
| **Smoke Figures** | **COMPLETED** | All 10 smoke figures generated from CSVs in `results/smoke_fig_*.png`. |

---

## Runtime Estimates for Full-Scale Stages (Calculated from Smoke Timings + KV-Cache Optimization)

With KV-caching (`use_cache=True`) enabled during rollouts and batched generation (`batch_size=24`):
- **Stage 2 (Full Run Collection)**: ~15–20 minutes (collecting multi-sample train + test runs across all 3 domains).
- **Stage 3 (Full Cut-Off Labelling)**: ~18–22 minutes (forced-answer sampling across all saved trajectories).
- **Stage 4 (Full Baselines E1 & E2)**: ~15 seconds (offline evaluation + 1,000 bootstrap resamples).
- **Stage 5 (Full Method A SFT & E3)**: ~6–8 minutes (3 epochs LoRA SFT + online test evaluation).
- **Stage 6 (Full Methods B & C GRPO & E4/E5)**: ~18–22 minutes (with KV-cache enabled, 4x faster than un-cached smoke rollouts).
- **Stage 7 (Full Experiments E6–E11)**: ~15–18 minutes (Transfer, Calibration, Ablations, Retrieval Quality, Failure Taxonomy, and 7B comparison).
- **Stage 8 (Figure Regeneration & `REPORT.md`)**: ~2 minutes.

---

## Stage Checklist
- [x] **Stage 0**: Environment verification, git init, directory layout, `configs/default.yaml`
- [x] **Stage 1 (Full & Smoke)**: Data splits, passage index (14,704 passages, Recall@3 = 0.9800), sandbox, calculator, checkers, 12/12 unit & leakage tests
- [x] **All Smoke Tests (Stages 2–7 + Figures)**: Completed & verified
- [ ] **Stage 2 (Full)**: Collect base agent runs
- [ ] **Stage 3 (Full)**: Cut-off label builder & hindsight stop computation
- [ ] **Stage 4 (Full)**: Baselines & E1/E2 problem measurement
- [ ] **Stage 5 (Full)**: Method A (SFT) & E3 evaluation
- [ ] **Stage 6 (Full)**: Methods B and C (GRPO $\lambda$ sweep) & E4/E5 evaluation
- [ ] **Stage 7 (Full)**: Experiments E6, E7, E8, E9, E10, E11
- [ ] **Stage 8**: Regenerate all figures via `scripts/make_figures.sh`, run quality checks, write `REPORT.md`
