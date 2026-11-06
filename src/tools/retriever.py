"""Hybrid BM25 + FAISS Retriever with Cross-Encoder Re-ranking and passage cache."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def _tokenize_bm25(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """BM25 + dense FAISS search merged via Reciprocal Rank Fusion, then Cross-Encoder re-ranked."""

    def __init__(
        self,
        index_dir: str | Path = "data/index",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
        load_models: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.device = device

        self.passages: List[Dict[str, Any]] = []
        self.id_to_passage: Dict[str, Dict[str, Any]] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None

        self.embedder: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        if load_models:
            self._init_models()

    def _init_models(self) -> None:
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
        if self.reranker is None:
            self.reranker = CrossEncoder(self.reranker_model_name, device=self.device)

    def build_or_load(self, passages: Optional[List[Dict[str, Any]]] = None, force_rebuild: bool = False) -> None:
        """Load cached FAISS + BM25 index from disk if available, otherwise build and save."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        passages_path = self.index_dir / "passages.jsonl"
        bm25_path = self.index_dir / "bm25.pkl"
        faiss_path = self.index_dir / "faiss.index"

        if not force_rebuild and passages_path.exists() and bm25_path.exists() and faiss_path.exists():
            self.passages = []
            with open(passages_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.passages.append(json.loads(line))
            self.id_to_passage = {p["id"]: p for p in self.passages}
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            self.faiss_index = faiss.read_index(str(faiss_path))
            return

        if not passages:
            raise ValueError("Passages must be provided when building index for the first time.")

        # Deduplicate passages by text while keeping all IDs mapped
        seen_texts: Dict[str, Dict[str, Any]] = {}
        deduped: List[Dict[str, Any]] = []
        for p in passages:
            text_key = p["text"].strip()
            if not text_key:
                continue
            if text_key not in seen_texts:
                entry = {
                    "id": str(p["id"]),
                    "title": str(p.get("title", "")),
                    "text": text_key,
                    "alias_ids": [str(p["id"])],
                }
                seen_texts[text_key] = entry
                deduped.append(entry)
            else:
                seen_texts[text_key]["alias_ids"].append(str(p["id"]))

        self.passages = deduped
        self.id_to_passage = {}
        for entry in self.passages:
            self.id_to_passage[entry["id"]] = entry
            for aid in entry.get("alias_ids", []):
                self.id_to_passage[aid] = entry

        with open(passages_path, "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        tokenized_corpus = [_tokenize_bm25(p["text"]) for p in self.passages]
        self.bm25 = BM25Okapi(tokenized_corpus)
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

        self._init_models()
        assert self.embedder is not None
        texts = [p["text"] for p in self.passages]
        embeddings = self.embedder.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.faiss_index = index
        faiss.write_index(index, str(faiss_path))

    def search(
        self,
        query: str,
        k: int = 3,
        bm25_candidates: int = 20,
        dense_candidates: int = 20,
        use_reranker: bool = True,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: BM25 + FAISS merged, then re-ranked by Cross-Encoder."""
        if not self.passages or self.bm25 is None or self.faiss_index is None:
            raise RuntimeError("Retriever index is not loaded. Call build_or_load() first.")

        q = str(query).strip()
        if not q:
            return []

        n_docs = len(self.passages)
        bm25_k = min(bm25_candidates, n_docs)
        dense_k = min(dense_candidates, n_docs)

        # 1. BM25 retrieval
        q_tokens = _tokenize_bm25(q)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:bm25_k]

        # 2. Dense FAISS retrieval
        self._init_models()
        assert self.embedder is not None
        q_emb = self.embedder.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        _, dense_top_idx_mat = self.faiss_index.search(q_emb, dense_k)
        dense_top_idx = dense_top_idx_mat[0]

        # 3. Merge via Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(bm25_top_idx):
            i = int(idx)
            if i >= 0:
                rrf_scores[i] = rrf_scores.get(i, 0.0) + 1.0 / (60.0 + rank + 1.0)
        for rank, idx in enumerate(dense_top_idx):
            i = int(idx)
            if i >= 0:
                rrf_scores[i] = rrf_scores.get(i, 0.0) + 1.0 / (60.0 + rank + 1.0)

        merged_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        candidate_indices = merged_indices[: max(k * 4, 15)]

        # 4. Cross-Encoder Re-ranking
        if use_reranker and self.reranker is not None and len(candidate_indices) > 1:
            pairs = [[q, self.passages[i]["text"]] for i in candidate_indices]
            ce_scores = self.reranker.predict(pairs, show_progress_bar=False)
            ranked = sorted(
                zip(candidate_indices, ce_scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            top_items = ranked[:k]
        else:
            top_items = [(i, rrf_scores[i]) for i in candidate_indices[:k]]

        results: List[Dict[str, Any]] = []
        for idx, score in top_items:
            p = self.passages[idx]
            results.append(
                {
                    "id": p["id"],
                    "title": p.get("title", ""),
                    "text": p["text"],
                    "alias_ids": p.get("alias_ids", [p["id"]]),
                    "score": float(score),
                }
            )
        return results

    def read(self, doc_id: str) -> Dict[str, Any]:
        """Return full passage by ID or title prefix."""
        did = str(doc_id).strip().strip("'\"`")
        if did in self.id_to_passage:
            p = self.id_to_passage[did]
            return {"ok": True, "id": p["id"], "title": p.get("title", ""), "text": p["text"]}
        # Try case-insensitive title match
        did_lower = did.lower()
        for p in self.passages:
            if p.get("title", "").lower() == did_lower or p["id"].lower() == did_lower:
                return {"ok": True, "id": p["id"], "title": p.get("title", ""), "text": p["text"]}
        return {"ok": False, "id": did, "title": "", "text": f"Passage '{did}' not found."}

