import argparse
import csv
import os
import pickle
from typing import List, Tuple

from util.tfidf_retriever import TfidfRetriever
from util.bm25_retriever import Bm25Retriever  # 如果你类名是 Bm25Retriever，就改成 from util.bm25_retriever import Bm25Retriever


# ========== 1. 读取 WebQ 测试集 ==========
def read_webq_test_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    """webq-test.csv：每行 question \\t [answers]"""
    questions: List[str] = []
    answers: List[List[str]] = []

    # utf-8-sig：兼容可能的 BOM；newline=""：避免 csv 在 Windows 多空行
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            q = row[0]
            a = eval(row[1])  # 与老师 cal_hit_multi.py 一致（作业数据可信）
            questions.append(q)
            answers.append(a)
    return questions, answers


# ========== 2. 读取语料库 doc_id（TSV 第一列） ==========
def load_corpus_docids(corpus_tsv_path: str) -> List[str]:
    """返回语料库每一行的真实 doc_id（字符串），顺序与 TSV 行一致。"""
    docids: List[str] = []
    with open(corpus_tsv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "id":  # 跳过表头（如果有）
                continue
            docids.append(str(row[0]))
    return docids


# ========== 3. 核心：修正/归一化检索结果里的 doc_id ==========
def normalize_top_docs_ids(
    top_docs: List[Tuple[List[object], List[object]]],
    corpus_docids: List[str],
) -> List[Tuple[List[str], List[str]]]:
    """
    让 top_docs 满足 qa_validation.py 的假设：
      - doc_id 必须能作为 load_passages() 读入字典的 key
      - 即：doc_id 必须是语料库 TSV 第一列中出现过的字符串

    兼容 retriever 输出的三种 doc_id：
      A) 真实 doc_id（str/int） -> 直接转成字符串并保留
      B) 行号索引（0-based int）-> 映射到 corpus_docids[idx]
      C) 无法识别/越界 -> 丢弃，避免 KeyError
    """
    docid_set = set(corpus_docids)
    n = len(corpus_docids)

    fixed: List[Tuple[List[str], List[str]]] = []

    for doc_ids, scores in top_docs:
        new_ids: List[str] = []
        new_scores: List[str] = []

        for doc_id, score in zip(doc_ids, scores):
            score_str = str(score)
            doc_id_str = str(doc_id)

            # 情况 A：已经是语料库真实 doc_id
            if doc_id_str in docid_set:
                new_ids.append(doc_id_str)
                new_scores.append(score_str)
                continue

            # 情况 B：可能是行号索引（0-based）
            try:
                idx = int(doc_id)
            except Exception:
                continue

            if 0 <= idx < n:
                real_id = corpus_docids[idx]
                new_ids.append(real_id)
                new_scores.append(score_str)
            # else：越界直接丢弃

        fixed.append((new_ids, new_scores))

    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tfidf", "bm25"], default="tfidf")
    parser.add_argument("--corpus", default="./corpus/wiki_webq_corpus.tsv")
    parser.add_argument("--test_csv", default="./datas/webq-test.csv")
    parser.add_argument("--out_pkl", default="./output/result_str.pkl")
    parser.add_argument("--index_pkl", default="./output/retriever_index.pkl")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    # 1) 读问题（answers 这里只是保持接口一致，不参与检索）
    questions, _ = read_webq_test_csv(args.test_csv)

    # 2) 构建 / 加载检索器
    if args.model == "tfidf":
        retriever = TfidfRetriever(
            corpus_tsv_path=args.corpus,
            index_pkl_path=args.index_pkl,
        )
    else:
        retriever = Bm25Retriever(
            corpus_tsv_path=args.corpus,
            index_pkl_path=args.index_pkl,
        )

    retriever.build_or_load(force_rebuild=args.force_rebuild)

    # 3) 检索
    top_docs = retriever.retrieve_all(questions, topk=args.topk)

    # 4) ✅ 关键修复：将 doc_id 统一成语料库真实 doc_id（字符串）
    corpus_docids = load_corpus_docids(args.corpus)
    top_docs = normalize_top_docs_ids(top_docs, corpus_docids)

    # 5) 保存为老师评测脚本需要的 result_str.pkl
    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(top_docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Saved retrieval results to: {args.out_pkl}")
    print("Next: run `python cal_hit_multi.py` to compute hit@k.")


if __name__ == "__main__":
    main()
