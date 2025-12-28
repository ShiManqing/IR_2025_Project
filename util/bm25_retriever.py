# IR_2025_Project/util/bm25_retriever.py
from __future__ import annotations

import csv
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import nlargest
from typing import DefaultDict, Dict, List, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass
class _Bm25Index:
    # doc_idx -> real_doc_id (注意：必须是 str，和 retriever_utils.load_passages 的 key 一致)
    doc_ids: List[str]
    doc_lens: List[int]
    df: Dict[str, int]
    postings: Dict[str, List[Tuple[int, int]]]  # term -> [(doc_internal_idx, tf)]
    avgdl: float
    n_docs: int
    k1: float
    b: float


class Bm25Retriever:
    """
    BM25 检索器（接口对齐你已有的 TFIDF）：
    - build_or_load(force_rebuild=False)
    - retrieve_all(questions, topk=100) -> List[Tuple[List[str], List[float]]]
      注意 doc_id 输出为 str，保证 cal_hit_multi + retriever_utils + qa_validation 能查到。
    """

    def __init__(
        self,
        corpus_tsv_path: str,
        index_pkl_path: str,
        k1: float = 1.2,
        b: float = 0.75,
        min_df: int = 2,
        max_df_ratio: float = 0.3,
    ) -> None:
        self.corpus_tsv_path = corpus_tsv_path
        self.index_pkl_path = index_pkl_path
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self._index: _Bm25Index | None = None

    def build_or_load(self, force_rebuild: bool = False) -> None:
        if (not force_rebuild) and os.path.exists(self.index_pkl_path):
            with open(self.index_pkl_path, "rb") as f:
                self._index = pickle.load(f)
            return

        index = self._build_index()
        os.makedirs(os.path.dirname(self.index_pkl_path), exist_ok=True)
        with open(self.index_pkl_path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._index = index

    def _build_index(self) -> _Bm25Index:
        doc_ids: List[str] = []
        doc_lens: List[int] = []
        df: Dict[str, int] = defaultdict(int)
        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        # 关键：语料是 TSV（老师 retriever_utils 用 delimiter="\t" 读）
        with open(self.corpus_tsv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                # 跳过表头（老师是 if row[0] != "id":）
                if row[0] == "id":
                    continue
                if len(row) < 2:
                    continue

                did = row[0]  # 保持 str
                text = row[1]
                tokens = _tokenize(text)
                if not tokens:
                    continue

                internal_idx = len(doc_ids)
                doc_ids.append(did)
                doc_lens.append(len(tokens))

                tf_counter = Counter(tokens)
                for term, tf in tf_counter.items():
                    postings[term].append((internal_idx, tf))
                    df[term] += 1

        n_docs = len(doc_ids)
        avgdl = (sum(doc_lens) / n_docs) if n_docs > 0 else 0.0

        # 过滤超低频 / 超高频词，减少索引体积
        max_df = int(self.max_df_ratio * n_docs)
        if max_df < 1:
            max_df = 1

        to_delete = [t for t, dfi in df.items() if dfi < self.min_df or dfi > max_df]
        for t in to_delete:
            df.pop(t, None)
            postings.pop(t, None)

        return _Bm25Index(
            doc_ids=doc_ids,
            doc_lens=doc_lens,
            df=dict(df),
            postings=dict(postings),
            avgdl=avgdl,
            n_docs=n_docs,
            k1=self.k1,
            b=self.b,
        )

    def retrieve_all(self, questions: List[str], topk: int = 100) -> List[Tuple[List[str], List[float]]]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build_or_load() first.")
        return [self._retrieve_one(q, topk) for q in questions]

    @staticmethod
    def _idf(dfi: int, n_docs: int) -> float:
        return math.log(1.0 + (n_docs - dfi + 0.5) / (dfi + 0.5))

    def _retrieve_one(self, question: str, topk: int) -> Tuple[List[str], List[float]]:
        idx = self._index
        assert idx is not None

        q_terms = _tokenize(question)
        if not q_terms:
            return ([], [])

        scores: DefaultDict[int, float] = defaultdict(float)
        avgdl = idx.avgdl if idx.avgdl > 0 else 1.0

        for term in q_terms:
            dfi = idx.df.get(term)
            if dfi is None:
                continue

            idf = self._idf(dfi, idx.n_docs)
            for doc_internal_idx, tf in idx.postings.get(term, []):
                dl = idx.doc_lens[doc_internal_idx]
                denom = tf + idx.k1 * (1.0 - idx.b + idx.b * (dl / avgdl))
                scores[doc_internal_idx] += idf * (tf * (idx.k1 + 1.0) / denom)

        if not scores:
            return ([], [])

        top_items = nlargest(topk, scores.items(), key=lambda x: x[1])
        doc_ids = [idx.doc_ids[doc_internal_idx] for doc_internal_idx, _ in top_items]  # <- str
        doc_scores = [float(score) for _, score in top_items]
        return (doc_ids, doc_scores)
