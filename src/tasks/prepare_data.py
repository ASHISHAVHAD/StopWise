"""Stage 1: Download datasets, log row counts, create deterministic splits, build FAISS+BM25 index, and run acceptance checks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from src.tasks.code import extract_humaneval_item, extract_mbpp_item
from src.tasks.math import extract_gsm8k_item
from src.tasks.search_qa import extract_2wiki_item, extract_hotpot_item, extract_musique_item
from src.tools.calculator import calc
from src.tools.retriever import HybridRetriever
from src.tools.sandbox import run_code_sandbox
from src.utils.config import load_config, set_seed
from src.utils.io import read_jsonl, setup_logger, write_jsonl


def build_splits_and_index(config_path: str = "configs/default.yaml", smoke: bool = False) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg["project"]["primary_seed"])
    set_seed(seed)
    rng = random.Random(seed)

    data_dir = Path(cfg["project"]["data_dir"])
    splits_dir = data_dir / ("splits_smoke" if smoke else "splits")
    index_dir = data_dir / ("index_smoke" if smoke else "index")
    results_dir = Path(cfg["project"]["output_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("stage1", results_dir / "logs" / "stage1_data_and_tools.log")
    summary_file = results_dir / ("stage1_smoke_summary.json" if smoke else "stage1_data_summary.json")

    if summary_file.exists() and (splits_dir / "search_qa_test.jsonl").exists() and (index_dir / "faiss.index").exists():
        logger.info(f"Stage 1 outputs already exist at {summary_file}; loading cached summary.")
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 1. Load raw datasets and log row counts
    logger.info("Downloading / loading raw datasets from HuggingFace cache...")
    hp_train_raw = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    hp_val_raw = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    wiki2_val_raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    musique_val_raw = load_dataset("dgslibisey/MuSiQue", split="validation")

    gsm_train_raw = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test_raw = load_dataset("openai/gsm8k", "main", split="test")

    mbpp_train_raw = load_dataset("google-research-datasets/mbpp", "full", split="train")
    mbpp_val_raw = load_dataset("google-research-datasets/mbpp", "full", split="validation")
    mbpp_test_raw = load_dataset("google-research-datasets/mbpp", "full", split="test")
    heval_test_raw = load_dataset("openai/openai_humaneval", split="test")

    raw_counts = {
        "hotpot_qa_train": len(hp_train_raw),
        "hotpot_qa_validation": len(hp_val_raw),
        "2wikimultihopqa_validation": len(wiki2_val_raw),
        "musique_validation": len(musique_val_raw),
        "gsm8k_train": len(gsm_train_raw),
        "gsm8k_test": len(gsm_test_raw),
        "mbpp_full_train": len(mbpp_train_raw),
        "mbpp_full_validation": len(mbpp_val_raw),
        "mbpp_full_test": len(mbpp_test_raw),
        "humaneval_test": len(heval_test_raw),
    }
    logger.info(f"Raw dataset row counts: {json.dumps(raw_counts)}")

    if smoke:
        n_train, n_val, n_test = 20, 10, 20
    else:
        n_train, n_val, n_test = 2000, 100, 300

    # --- Domain 1: Search QA ---
    all_passages: List[Dict[str, Any]] = []
    search_train: List[Dict[str, Any]] = []
    for i in range(min(n_train, len(hp_train_raw))):
        t, p_list = extract_hotpot_item(hp_train_raw[i], i, "train")
        search_train.append(t)
        # Keep supporting passages + sample of distractors for training tasks so index stays fast and realistic
        for p in p_list:
            if p["is_supporting"] or (i < 400):
                all_passages.append(p)

    search_val: List[Dict[str, Any]] = []
    for i in range(min(n_val, len(hp_val_raw))):
        t, p_list = extract_hotpot_item(hp_val_raw[i], i, "val")
        search_val.append(t)
        all_passages.extend(p_list)

    # Search QA test: balanced across HotpotQA, 2WikiMultiHopQA, and MuSiQue
    per_source_test = max(1, n_test // 3)
    search_test: List[Dict[str, Any]] = []
    for i in range(n_val, min(n_val + per_source_test, len(hp_val_raw))):
        t, p_list = extract_hotpot_item(hp_val_raw[i], i, "test")
        search_test.append(t)
        all_passages.extend(p_list)
    for i in range(min(per_source_test, len(wiki2_val_raw))):
        t, p_list = extract_2wiki_item(wiki2_val_raw[i], i, "test")
        search_test.append(t)
        all_passages.extend(p_list)
    rem_test = n_test - len(search_test)
    for i in range(min(rem_test, len(musique_val_raw))):
        t, p_list = extract_musique_item(musique_val_raw[i], i, "test")
        search_test.append(t)
        all_passages.extend(p_list)

    # --- Domain 2: Maths (GSM8K) ---
    math_train = [extract_gsm8k_item(gsm_train_raw[i], i, "train") for i in range(min(n_train, len(gsm_train_raw)))]
    math_val = [
        extract_gsm8k_item(gsm_train_raw[len(gsm_train_raw) - 1 - i], i, "val")
        for i in range(min(n_val, 200))
    ]
    math_test = [extract_gsm8k_item(gsm_test_raw[i], i, "test") for i in range(min(n_test, len(gsm_test_raw)))]

    # --- Domain 3: Code (MBPP + HumanEval) ---
    code_val = [extract_mbpp_item(mbpp_val_raw[i], i, "val") for i in range(min(n_val, len(mbpp_val_raw)))]
    code_test: List[Dict[str, Any]] = []
    n_mbpp_test = n_test // 2
    n_heval_test = n_test - n_mbpp_test
    for i in range(min(n_mbpp_test, 150)):
        code_test.append(extract_mbpp_item(mbpp_test_raw[i], i, "test"))
    for i in range(min(n_heval_test, len(heval_test_raw))):
        code_test.append(extract_humaneval_item(heval_test_raw[i], i, "test"))

    reserved_code_questions = {r["question"].strip() for r in code_test} | {r["question"].strip() for r in code_val}
    code_train_base = [
        item
        for i in range(len(mbpp_train_raw))
        if (item := extract_mbpp_item(mbpp_train_raw[i], i, "train"))["question"].strip() not in reserved_code_questions
    ]
    for i in range(150, len(mbpp_test_raw)):
        item = extract_mbpp_item(mbpp_test_raw[i], i + 1000, "train")
        if item["question"].strip() not in reserved_code_questions:
            code_train_base.append(item)
            reserved_code_questions.add(item["question"].strip())
    if smoke:
        code_train = code_train_base[:n_train]
    else:
        code_train = code_train_base

    # Write splits to disk
    splits_map = {
        "search_qa_train": search_train,
        "search_qa_val": search_val,
        "search_qa_test": search_test,
        "math_train": math_train,
        "math_val": math_val,
        "math_test": math_test,
        "code_train": code_train,
        "code_val": code_val,
        "code_test": code_test,
    }
    for name, records in splits_map.items():
        write_jsonl(splits_dir / f"{name}.jsonl", records)
        logger.info(f"Saved split {name}: {len(records)} tasks")

    # Build Hybrid Retriever Index (BM25 + FAISS + CrossEncoder)
    logger.info(f"Building HybridRetriever index over {len(all_passages)} raw passage chunks -> {index_dir}...")
    retriever = HybridRetriever(
        index_dir=index_dir,
        embedding_model_name=cfg["retriever"]["embedding_model"],
        reranker_model_name=cfg["retriever"]["reranker_model"],
        device="cuda",
    )
    retriever.build_or_load(passages=all_passages, force_rebuild=True)
    logger.info(f"Indexed {len(retriever.passages)} unique passage chunks.")

    # Stage 1 Acceptance Check 1: Retrieval Recall@3 on 100-question dev slice
    dev_slice = search_val[: min(100, len(search_val))]
    hits_at_3_title = 0
    hits_at_3_pid = 0
    for q_item in dev_slice:
        top_docs = retriever.search(q_item["question"], k=3, use_reranker=True)
        ret_titles = {d["title"].strip().lower() for d in top_docs}
        ret_pids: set[str] = set()
        for d in top_docs:
            ret_pids.add(d["id"])
            ret_pids.update(d.get("alias_ids", []))
        gold_titles = {t.strip().lower() for t in q_item.get("supporting_titles", [])}
        gold_pids = set(q_item.get("gold_passage_ids", []))

        if gold_titles & ret_titles:
            hits_at_3_title += 1
        if gold_pids & ret_pids:
            hits_at_3_pid += 1

    recall_at_3_title = hits_at_3_title / max(1, len(dev_slice))
    recall_at_3_pid = hits_at_3_pid / max(1, len(dev_slice))
    logger.info(
        f"Stage 1 Acceptance Check - Retrieval Recall@3 on {len(dev_slice)}-question dev slice: "
        f"Title Recall@3 = {recall_at_3_title:.4f}, Passage Recall@3 = {recall_at_3_pid:.4f}"
    )

    # Stage 1 Acceptance Check 2: Sandbox kills infinite loop
    loop_res = run_code_sandbox("while True:\n    pass\n", timeout_seconds=1.5)
    calc_res = calc("12 * 12 + 6")

    summary = {
        "smoke": smoke,
        "raw_dataset_counts": raw_counts,
        "split_counts": {k: len(v) for k, v in splits_map.items()},
        "indexed_unique_passages": len(retriever.passages),
        "acceptance_checks": {
            "dev_slice_size": len(dev_slice),
            "retrieval_recall_at_3_title": round(recall_at_3_title, 4),
            "retrieval_recall_at_3_passage": round(recall_at_3_pid, 4),
            "sandbox_infinite_loop_killed": bool(loop_res["timed_out"] and not loop_res["ok"]),
            "calculator_check_passed": bool(calc_res["ok"] and calc_res["result"] == "150"),
        },
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Stage 1 complete. Summary written to {summary_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    build_splits_and_index(config_path=args.config, smoke=args.smoke)
