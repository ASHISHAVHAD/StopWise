"""Search QA task loader and SQuAD-style Exact Match / Token F1 checker."""

from __future__ import annotations

import collections
import re
import string
from typing import Any, Dict, List, Sequence, Tuple


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace (SQuAD standard)."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> float:
    return float(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def check_search_qa(prediction: str, gold_answers: str | Sequence[str]) -> Dict[str, Any]:
    """Check Search QA prediction against one or more valid gold answers.

    Correct = Exact Match (EM == 1.0). Also computes token F1.
    """
    if isinstance(gold_answers, str):
        golds = [gold_answers]
    else:
        golds = list(gold_answers)

    # Clean common prefix wrappers if model emitted "Final Answer: ..."
    cleaned_pred = str(prediction).strip()
    cleaned_pred = re.sub(r"^(final\s+answer\s*[:\-]\s*)", "", cleaned_pred, flags=re.IGNORECASE).strip()

    best_em = max((compute_exact(g, cleaned_pred) for g in golds), default=0.0)
    best_f1 = max((compute_f1(g, cleaned_pred) for g in golds), default=0.0)
    return {
        "correct": int(best_em == 1.0),
        "em": float(best_em),
        "f1": float(best_f1),
        "normalized_pred": normalize_answer(cleaned_pred),
    }


def chunk_text_words(title: str, text: str, max_words: int = 100) -> List[str]:
    """Split passage text into ~100-word chunks prefixed with title."""
    words = text.strip().split()
    if not words:
        return []
    chunks: List[str] = []
    for i in range(0, len(words), max_words):
        part = " ".join(words[i : i + max_words])
        chunks.append(f"{title}: {part}" if title else part)
    return chunks


def extract_hotpot_item(row: Dict[str, Any], idx: int, split_name: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Convert a HotpotQA row into standardized task dict and passage chunks."""
    q_id = str(row.get("id", f"hotpot_{split_name}_{idx}"))
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    sup_titles = set(row.get("supporting_facts", {}).get("title", []))

    passages: List[Dict[str, str]] = []
    gold_passage_ids: List[str] = []
    context = row.get("context", {})
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])

    for t_idx, (title, sents) in enumerate(zip(titles, sentences_list)):
        full_text = " ".join(s.strip() for s in sents if s.strip())
        chunks = chunk_text_words(title, full_text, max_words=100)
        for c_idx, ch in enumerate(chunks):
            pid = f"hp_{q_id}_{t_idx}_{c_idx}"
            is_sup = title in sup_titles
            passages.append({"id": pid, "title": str(title), "text": ch, "is_supporting": is_sup})
            if is_sup:
                gold_passage_ids.append(pid)

    task = {
        "task_id": f"search_qa_hotpot_{q_id}",
        "domain": "search_qa",
        "source": "hotpot_qa",
        "split": split_name,
        "question": question,
        "answer": answer,
        "aliases": [answer],
        "supporting_titles": list(sup_titles),
        "gold_passage_ids": gold_passage_ids,
    }
    return task, passages


def extract_2wiki_item(row: Dict[str, Any], idx: int, split_name: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Convert a 2WikiMultiHopQA row into standardized task dict and passage chunks."""
    q_id = str(row.get("_id", row.get("id", f"2wiki_{split_name}_{idx}")))
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()

    sup_facts = row.get("supporting_facts", [])
    if isinstance(sup_facts, dict):
        sup_titles = set(sup_facts.get("title", []))
    elif isinstance(sup_facts, list):
        sup_titles = {str(item[0]) for item in sup_facts if isinstance(item, (list, tuple)) and len(item) > 0}
    else:
        sup_titles = set()

    passages: List[Dict[str, str]] = []
    gold_passage_ids: List[str] = []
    context = row.get("context", [])
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        pairs = list(zip(titles, sentences_list))
    elif isinstance(context, list):
        pairs = [(item[0], item[1]) for item in context if isinstance(item, (list, tuple)) and len(item) >= 2]
    else:
        pairs = []

    for t_idx, (title, sents) in enumerate(pairs):
        if isinstance(sents, list):
            full_text = " ".join(str(s).strip() for s in sents if str(s).strip())
        else:
            full_text = str(sents).strip()
        chunks = chunk_text_words(str(title), full_text, max_words=100)
        for c_idx, ch in enumerate(chunks):
            pid = f"2wiki_{q_id}_{t_idx}_{c_idx}"
            is_sup = str(title) in sup_titles
            passages.append({"id": pid, "title": str(title), "text": ch, "is_supporting": is_sup})
            if is_sup:
                gold_passage_ids.append(pid)

    task = {
        "task_id": f"search_qa_2wiki_{q_id}",
        "domain": "search_qa",
        "source": "2wikimultihopqa",
        "split": split_name,
        "question": question,
        "answer": answer,
        "aliases": [answer],
        "supporting_titles": list(sup_titles),
        "gold_passage_ids": gold_passage_ids,
    }
    return task, passages


def extract_musique_item(row: Dict[str, Any], idx: int, split_name: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Convert a MuSiQue row into standardized task dict and passage chunks."""
    q_id = str(row.get("id", f"musique_{split_name}_{idx}"))
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    aliases = [answer] + [str(a) for a in row.get("answer_aliases", []) if str(a).strip()]

    passages: List[Dict[str, str]] = []
    gold_passage_ids: List[str] = []
    sup_titles: List[str] = []
    paragraphs = row.get("paragraphs", [])
    for p_idx, para in enumerate(paragraphs):
        title = str(para.get("title", f"Doc_{p_idx}")).strip()
        text = str(para.get("paragraph_text", "")).strip()
        is_sup = bool(para.get("is_supporting", False))
        if is_sup:
            sup_titles.append(title)
        chunks = chunk_text_words(title, text, max_words=100)
        for c_idx, ch in enumerate(chunks):
            pid = f"mus_{q_id}_{p_idx}_{c_idx}"
            passages.append({"id": pid, "title": title, "text": ch, "is_supporting": is_sup})
            if is_sup:
                gold_passage_ids.append(pid)

    task = {
        "task_id": f"search_qa_musique_{q_id}",
        "domain": "search_qa",
        "source": "musique",
        "split": split_name,
        "question": question,
        "answer": answer,
        "aliases": list(dict.fromkeys(aliases)),
        "supporting_titles": sup_titles,
        "gold_passage_ids": gold_passage_ids,
    }
    return task, passages

