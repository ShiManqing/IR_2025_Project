# IR_2025_Project/util/tfidf_retriever.py
from __future__ import annotations

import csv
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional

import numpy as np

# 说明：sklearn 通常在课程/科研环境里是有的；如果你环境里没有，
# 我再给你一个“纯 Python BM25”的版本替换这个文件即可。
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfIndex:
    doc_ids: List[str]
    vectorizer: TfidfVectorizer
    doc_matrix: "scipy.sparse.csr_matrix"  # noqa: F821


class TfidfRetriever:
    """
    负责：
    1) 建立/加载 TF-IDF 索引
    2) 根据 question 检索 top-k doc_id 和分数
    """

    def __init__(
        self,
        corpus_tsv_path: str,
        index_pkl_path: str,
    ) -> None:
        self.corpus_tsv_path = corpus_tsv_path
        self.index_pkl_path = index_pkl_path
        self._index: Optional[TfidfIndex] = None

    @staticmethod
    def _iter_corpus_tsv(tsv_path: str) -> Iterable[Tuple[str, str]]:
        """
        读取 wiki_webq_corpus.tsv（格式：id, text, title）
        产出 (doc_id, combined_text)
        """
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                if row[0] == "id":
                    continue
                # row: [id, text, title]
                doc_id = row[0]
                text = row[1] if len(row) > 1 else ""
                title = row[2] if len(row) > 2 else ""
                combined = f"{title} {text}".strip()
                yield doc_id, combined

    def build_or_load(self, force_rebuild: bool = False) -> None:
        if (not force_rebuild) and os.path.exists(self.index_pkl_path):
            with open(self.index_pkl_path, "rb") as f:
                self._index = pickle.load(f)
            return

        doc_ids: List[str] = []
        docs: List[str] = []
        for doc_id, combined in self._iter_corpus_tsv(self.corpus_tsv_path):
            doc_ids.append(doc_id)
            docs.append(combined)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=200_000,   # 可调：特征太大内存会涨
            ngram_range=(1, 2),     # unigram + bigram，通常比纯 unigram 稳一点
            dtype=np.float32,
        )
        doc_matrix = vectorizer.fit_transform(docs)

        self._index = TfidfIndex(doc_ids=doc_ids, vectorizer=vectorizer, doc_matrix=doc_matrix)

        os.makedirs(os.path.dirname(self.index_pkl_path), exist_ok=True)
        with open(self.index_pkl_path, "wb") as f:
            pickle.dump(self._index, f, protocol=pickle.HIGHEST_PROTOCOL)

    def retrieve_all(
        self,
        questions: List[str],
        topk: int = 100,
        batch_size: int = 64,
    ) -> List[Tuple[List[str], List[float]]]:
        """
        返回结构必须匹配老师评测脚本：
        List[ ( [doc_id...], [score...] ) ]
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_or_load() first.")

        idx = self._index
        doc_ids = np.array(idx.doc_ids, dtype=object)

        results: List[Tuple[List[str], List[float]]] = []
        n = len(questions)

        for start in range(0, n, batch_size):
            batch_q = questions[start : start + batch_size]
            q_mat = idx.vectorizer.transform(batch_q)

            # cosine 相似度在 TF-IDF L2-normalize 情况下等价于点积
            scores_mat = q_mat @ idx.doc_matrix.T  # (b, num_docs) sparse
            scores_mat = scores_mat.toarray()      # 转 dense：子任务规模一般可接受

            for i in range(scores_mat.shape[0]):
                scores = scores_mat[i]
                k = min(topk, scores.shape[0])

                # argpartition 找 topk，再排序
                top_idx = np.argpartition(-scores, kth=k - 1)[:k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]

                top_doc_ids = doc_ids[top_idx].tolist()
                top_scores = scores[top_idx].astype(np.float32).tolist()

                results.append((top_doc_ids, top_scores))

        return results
