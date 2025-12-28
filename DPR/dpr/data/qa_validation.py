#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for Q&A results validation tasks - Retriver passage validation and Reader predicted answer validation
"""

import collections
import logging
import string
import unicodedata
import zlib
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict

import regex as re

from dpr.data.retriever_data import TableChunk
from dpr.utils.tokenizers import SimpleTokenizer

logger = logging.getLogger(__name__)

# Ensure globals exist in the module namespace (important for Windows multiprocessing spawn)
dpr_all_documents = {}
dpr_all_tables = {}


QAMatchStats = collections.namedtuple("QAMatchStats", ["top_k_hits", "questions_doc_hits"])

QATableMatchStats = collections.namedtuple(
    "QAMatchStats", ["top_k_chunk_hits", "top_k_table_hits", "questions_doc_hits"]
)


def calculate_matches(
    all_docs: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> QAMatchStats:
    logger.info("all_docs size %d", len(all_docs))
    global dpr_all_documents
    dpr_all_documents = all_docs
    logger.info("dpr_all_documents size %d%s", len(dpr_all_documents), type(dpr_all_documents))

    tokenizer = SimpleTokenizer(**{})

    logger.info("Matching answers in top docs...")
    questions_answers_docs = list(zip(answers, closest_docs))

    # ✅ SAFE PATH: no multiprocessing when workers_num <= 1
    if workers_num <= 1:
        scores = [check_answer(qad, tokenizer=tokenizer, match_type=match_type)
                  for qad in questions_answers_docs]
    else:
        # Optional: keep multiprocessing, but initialize globals for workers
        def _init_worker(docs):
            global dpr_all_documents
            dpr_all_documents = docs

        get_score_partial = partial(check_answer, match_type=match_type, tokenizer=tokenizer)
        processes = ProcessPool(processes=workers_num, initializer=_init_worker, initargs=(all_docs,))
        try:
            scores = processes.map(get_score_partial, questions_answers_docs)
        finally:
            processes.close()
            processes.join()

    logger.info("Per question validation results len=%d", len(scores))

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)



def calculate_matches_from_meta(
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
    use_title: bool = False,
    meta_compressed: bool = False,
) -> QAMatchStats:

    tokenizer = SimpleTokenizer(**{})

    logger.info("Matching answers in top docs...")
    questions_answers_docs = list(zip(answers, closest_docs))

    if workers_num <= 1:
        scores = [
            check_answer_from_meta(
                qad,
                tokenizer=tokenizer,
                match_type=match_type,
                use_title=use_title,
                meta_compressed=meta_compressed,
            )
            for qad in questions_answers_docs
        ]
    else:
        get_score_partial = partial(
            check_answer_from_meta,
            match_type=match_type,
            tokenizer=tokenizer,
            use_title=use_title,
            meta_compressed=meta_compressed,
        )
        processes = ProcessPool(processes=workers_num)
        try:
            scores = processes.map(get_score_partial, questions_answers_docs)
        finally:
            processes.close()
            processes.join()

    logger.info("Per question validation results len=%d", len(scores))

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)



def check_answer(questions_answers_docs, tokenizer, match_type) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (doc_ids, doc_scores) = questions_answers_docs

    global dpr_all_documents
    hits = []

    for i, doc_id in enumerate(doc_ids):
        doc = dpr_all_documents[doc_id]
        text = doc[0]

        answer_found = False
        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue
        if match_type == "kilt":
            if has_answer_kilt(answers, text):
                answer_found = True
        elif has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def check_answer_from_meta(
    questions_answers_docs,
    tokenizer,
    match_type,
    meta_body_idx: int = 1,
    meta_title_idx: int = 2,
    use_title: bool = False,
    meta_compressed: bool = False,
) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (docs_meta, doc_scores) = questions_answers_docs

    hits = []

    for i, doc_meta in enumerate(docs_meta):

        text = doc_meta[meta_body_idx]
        title = doc_meta[meta_title_idx] if len(doc_meta) > meta_title_idx else ""
        if meta_compressed:
            text = zlib.decompress(text).decode()
            title = zlib.decompress(title).decode()

        if use_title:
            text = title + " . " + text
        answer_found = False
        if has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def has_answer(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


# function for the reader model answer validation
def exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def calculate_chunked_matches(
    all_docs: Dict[object, TableChunk],
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> QATableMatchStats:
    global dpr_all_documents
    dpr_all_documents = all_docs

    global dpr_all_tables
    dpr_all_tables = {}

    for key, table_chunk in all_docs.items():
        table_str, title, table_id = table_chunk
        table_chunks = dpr_all_tables.get(table_id, [])
        table_chunks.append((table_str, title))
        dpr_all_tables[table_id] = table_chunks

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(processes=workers_num)

    logger.info("Matching answers in top docs...")
    questions_answers_docs = list(zip(answers, closest_docs))

    if workers_num <= 1:
        scores = [check_chunked_docs_answer(qad, tokenizer=tokenizer, match_type=match_type)
                  for qad in questions_answers_docs]
    else:
        get_score_partial = partial(check_chunked_docs_answer, match_type=match_type, tokenizer=tokenizer)
        processes = ProcessPool(processes=workers_num)
        try:
            scores = processes.map(get_score_partial, questions_answers_docs)
        finally:
            processes.close()
            processes.join()


    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    top_k_orig_hits = [0] * n_docs
    for s in scores:
        question_hits, question_orig_doc_hits = s
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

        best_hit = next((i for i, x in enumerate(question_orig_doc_hits) if x), None)
        if best_hit is not None:
            top_k_orig_hits[best_hit:] = [v + 1 for v in top_k_orig_hits[best_hit:]]

    return QATableMatchStats(top_k_hits, top_k_orig_hits, scores)


# -------------------- KILT eval ---------------------------------


def has_answer_kilt(answers, text) -> bool:
    text = normalize_kilt(text)
    for single_answer in answers:
        single_answer = normalize_kilt(single_answer)
        if single_answer in text:
            return True
    return False


# answer normalization
def normalize_kilt(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
