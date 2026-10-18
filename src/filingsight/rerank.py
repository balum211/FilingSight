"""Second-stage rerankers. Each takes (query, candidates) and returns the
candidates re-ordered with a 0..1 relevance score that also drives the
"I don't know" gate.

* ``local``          Runs anywhere with no download. Scores each candidate on
                     (a) its best single sentence/table row vs the query
                     (sentence-level late interaction, so one precise sentence in
                     a long chunk is not diluted), (b) whole-chunk cosine, and
                     (c) coverage of the query's key terms, with numbers, years
                     and capitalised names weighted double.
* ``cross-encoder``  ms-marco-MiniLM-L-6-v2 reads query and chunk together.
                     Needs ``pip install sentence-transformers`` (pulls PyTorch).
* ``llm``            Asks the chat model to grade every candidate 0-10 in one call.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document

from . import config
from .chunking import indexed_text, split_units
from .index import tokenize

RERANKERS = ("none", "local", "cross-encoder", "llm")

# Confidence thresholds per reranker (their scores live on different scales).
THRESHOLDS = {
    "local": config.MIN_CONFIDENCE,
    "cross-encoder": config.MIN_CONFIDENCE_CE,
    "llm": 0.5,
    "none": 0.0,
}

_GENERIC = frozenset(
    "uber lyft company companies 10-k annual report fiscal year filing filings much many total tell describe "
    "explain list main key say says said".split()
)


def key_terms(query: str) -> dict[str, float]:
    """Query terms that a relevant chunk should contain, with weights."""
    weights: dict[str, float] = {}
    for tok in tokenize(query):
        if tok in _GENERIC or len(tok) < 3 and not tok.isdigit():
            continue
        w = 2.0 if any(ch.isdigit() for ch in tok) else 1.0
        weights[tok] = max(weights.get(tok, 0.0), w)
    for name in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s[A-Z][A-Za-z0-9]+)*", query):
        for tok in tokenize(name):
            if tok in weights and tok not in _GENERIC:
                weights[tok] = max(weights[tok], 2.0)
    return weights


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 4 and t.endswith("s") else t


def coverage(query_terms: dict[str, float], text: str) -> float:
    if not query_terms:
        return 0.0
    toks = {_stem(t) for t in tokenize(text)}
    got = sum(w for t, w in query_terms.items() if _stem(t) in toks)
    return got / sum(query_terms.values())


class LocalReranker:
    name = "local"

    def __init__(self, w_sentence: float = 0.45, w_chunk: float = 0.25, w_terms: float = 0.30):
        self.w = (w_sentence, w_chunk, w_terms)

    max_units = 6  # sentences/rows per chunk sent to the embedder (speed cap)

    def _best_units(self, text: str, terms: dict[str, float]) -> list[str]:
        units = split_units(text, with_captions=True) or [text]
        if len(units) <= self.max_units:
            return units
        # keep the units that share the most key terms with the query
        scored = sorted(units, key=lambda u: -coverage(terms, u))
        return scored[: self.max_units]

    def rerank(
        self, query: str, candidates: list[Document], chunk_vectors: np.ndarray | None = None
    ) -> list[tuple[Document, float]]:
        if not candidates:
            return []
        from .embeddings import get_embeddings

        emb = get_embeddings()
        terms = key_terms(query)
        q = emb.embed_array([query])[0]
        units_per_doc = [self._best_units(d.page_content, terms) for d in candidates]
        flat = [u for us in units_per_doc for u in us]
        uvecs = emb.embed_array(flat)
        cvecs = chunk_vectors if chunk_vectors is not None else emb.embed_array([indexed_text(d) for d in candidates])
        scored = []
        pos = 0
        for doc, units, cvec in zip(candidates, units_per_doc, cvecs):
            sims = uvecs[pos : pos + len(units)] @ q
            pos += len(units)
            best_sentence = float(np.max(sims))
            chunk_sim = float(cvec @ q)
            cov = coverage(terms, indexed_text(doc))
            score = self.w[0] * best_sentence + self.w[1] * chunk_sim + self.w[2] * cov
            scored.append((doc, score))
        return sorted(scored, key=lambda t: -t[1])


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str = config.CROSS_ENCODER_MODEL):
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list[Document], chunk_vectors=None) -> list[tuple[Document, float]]:
        if not candidates:
            return []
        logits = self.model.predict([(query, indexed_text(d)) for d in candidates])
        scores = [1.0 / (1.0 + math.exp(-float(x))) for x in logits]
        return sorted(zip(candidates, scores), key=lambda t: -t[1])


class LLMReranker:
    name = "llm"

    def rerank(self, query: str, candidates: list[Document], chunk_vectors=None) -> list[tuple[Document, float]]:
        from .llm import get_chat_model
        from .prompts import RERANK_PROMPT

        if not candidates:
            return []
        passages = "\n\n".join(
            f"[{i}] {d.metadata['header']}\n{d.page_content[:700]}" for i, d in enumerate(candidates)
        )
        msg = get_chat_model(temperature=0).invoke(RERANK_PROMPT.format(question=query, passages=passages))
        grades = _parse_grades(str(msg.content), len(candidates))
        return sorted(zip(candidates, grades), key=lambda t: -t[1])


def _parse_grades(text: str, n: int) -> list[float]:
    grades = [0.0] * n
    try:
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        for k, v in data.items():
            i = int(k)
            if 0 <= i < n:
                grades[i] = max(0.0, min(10.0, float(v))) / 10.0
    except Exception:
        pass
    return grades


class NoReranker:
    name = "none"

    def rerank(self, query: str, candidates: list[Document], chunk_vectors=None) -> list[tuple[Document, float]]:
        # Keep retrieval order; give a flat score so the gate never fires.
        return [(d, 1.0) for d in candidates]


@lru_cache(maxsize=4)
def get_reranker(name: str = config.RERANKER):
    if name == "local":
        return LocalReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker()
    if name == "llm":
        return LLMReranker()
    if name == "none":
        return NoReranker()
    raise ValueError(f"unknown reranker {name!r}; pick one of {RERANKERS}")
