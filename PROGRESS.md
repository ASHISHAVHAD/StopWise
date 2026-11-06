# StopWise Project Progress (`PROGRESS.md`)

## Current Status
- **Stage 0 (Machine Check & Environment Setup)**: Completed.
  - Verified hardware: 1x NVIDIA RTX 6000 Ada Generation (48 GB VRAM, 0 MiB used by others), 16-core Xeon CPU, 58 GB free RAM, 1.2 TB free disk.
  - Recorded hardware and pinned packages in `results/ENVIRONMENT.md` and `requirements.txt`.
- **Stage 1 (Data and Tools)**: Completed & Acceptance Checks Passed.
  - Downloaded and verified datasets: HotpotQA (90,447 train / 7,405 val), 2WikiMultiHopQA (12,576 val), MuSiQue (2,417 val), GSM8K (7,473 train / 1,319 test), MBPP full (374 train / 90 val / 500 test), HumanEval (164 test).
  - Created deterministic splits (`2000` train / `100` val / `300` test for Search QA and Maths; `723` train / `90` val / `300` test for Code) with zero train/test leakage verified by `tests/test_data_leakage.py`.
  - Built Hybrid BM25 + FAISS (`BAAI/bge-small-en-v1.5`) + Cross-Encoder (`BAAI/bge-reranker-base`) index over 14,704 unique ~100-word Wikipedia passages.
  - **Acceptance Checks**:
    - Retrieval Recall@3 on 100-question dev slice: **0.9800** (98.0% title & passage Recall@3).
    - Sandbox kills infinite loop within 1.5s timeout and blocks network access (`PASSED`).
    - All 12 unit & leakage tests in `pytest tests/` (`PASSED`).
- **Next Step**: Run Stage 2 smoke test (`collect_runs.py --smoke`), estimate runtime, then run full Stage 2.

## Stage Checklist
- [x] **Stage 0**: Environment verification, git init, directory layout, `configs/default.yaml`
- [x] **Stage 1**: Data download/splits, passage index (BM25 + FAISS + Cross-Encoder), sandbox, calculator, answer checkers, unit tests & acceptance checks
- [ ] **Stage 2**: Collect base agent runs (multi-sample + few-shot long budget pass), verify >=25% success rate
- [ ] **Stage 3**: Cut-off label builder & hindsight stop computation, label counts & plot, unit tests
- [ ] **Stage 4**: Baselines (fixed-limit, own-decision, confidence-rule, hindsight-best) & E1/E2 problem measurement
- [ ] **Stage 5**: Method A (SFT on cut-off labels with 50/50 class balance, LoRA r=16, alpha=32) & E3 evaluation
- [ ] **Stage 6**: Methods B and C (GRPO with cost-aware reward, $\lambda \in \{0, 0.1, 0.3, 0.6\}$, `<result>` masking) & E4/E5 evaluation
- [ ] **Stage 7**: Experiments E6 (Transfer), E7 (Calibration), E8 (Ablations), E9 (Retrieval Quality), E10 (Failure Taxonomy), E11 (Optional MoE/Architecture comparison)
- [ ] **Stage 8**: Regenerate all figures via `scripts/make_figures.sh`, verify quality checks, write `REPORT.md` and `README.md`
