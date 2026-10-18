"""Dense, sparse (BM25) and hybrid retrieval with metadata filters.

Hybrid = Reciprocal Rank Fusion of the dense and BM25 rankings:
    score(d) = sum over retrievers of 1 / (RRF_K + rank(d))
RRF needs no score calibration between cosine similarity and BM25, which is
why it is the usual default for hybrid search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document

from . import config
from .index import SearchIndex, tokenize

MODES = ("dense", "bm25", "hybrid")

_COMPANY_PATTERNS = {
    "Uber": re.compile(r"\buber\b", re.I),
    "Lyft": re.compile(r"\blyft\b", re.I),
}
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|both|each|difference|differ|which (company|one)|higher|lower|more than|less than)\b", re.I)


@dataclass
class QueryPlan:
    """What the query analyser worked out before retrieval."""

    question: str
    companies: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    is_comparison: bool = False
    sub_queries: list[tuple[str, str | None]] = field(default_factory=list)  # (query, company filter)
    out_of_scope_years: list[int] = field(default_factory=list)


def plan_query(question: str) -> QueryPlan:
    companies = [c for c, p in _COMPANY_PATTERNS.items() if p.search(question)]
    years = sorted({int(y) for y in _YEAR.findall(question)})
    plan = QueryPlan(question=question, companies=companies, years=years)
    plan.is_comparison = len(companies) > 1 or (bool(_COMPARE.search(question)) and len(companies) != 1)
    plan.out_of_scope_years = [y for y in years if y not in config.COVERED_YEARS]
    if len(companies) != 1:
        # Multi-query decomposition: one focused retrieval per company, so the
        # bigger filing (Uber) cannot crowd the other one out of the top-k.
        # A question that names no company is ambiguous, so it is answered for
        # every company in the corpus rather than guessed.
        targets = companies or list(config.COMPANIES)
        names = re.compile("|".join(_COMPANY_PATTERNS[c].pattern for c in targets), re.I)
        for c in targets:
            focused = names.sub(c, question) if companies else f"{c}: {question}"
            plan.sub_queries.append((focused, c))
    else:
        plan.sub_queries.append((question, companies[0] if companies else None))
    return plan


def _mask(index: SearchIndex, company: str | None) -> np.ndarray:
    if company is None:
        return np.ones(len(index.docs), dtype=bool)
    return np.array([d.metadata["company"] == company for d in index.docs])


def dense_ranking(index: SearchIndex, query: str, k: int, company: str | None = None) -> list[tuple[int, float]]:
    """Top-k (doc position, cosine) from the Chroma vector store."""
    flt = {"company": company} if company else None
    hits = index.vector_store.similarity_search_with_score(query, k=k, filter=flt)
    pos = {cid: i for i, cid in enumerate(index.ids())}
    # Chroma returns cosine *distance*; convert to similarity.
    return [(pos[d.metadata["chunk_id"]], 1.0 - float(dist)) for d, dist in hits if d.metadata.get("chunk_id") in pos]


def bm25_ranking(index: SearchIndex, query: str, k: int, company: str | None = None) -> list[tuple[int, float]]:
    scores = index.bm25.get_scores(tokenize(query))
    scores = np.where(_mask(index, company), scores, -np.inf)
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]


def rrf_fuse(
    rankings: list[list[tuple[int, float]]],
    k: int,
    weights: tuple[float, ...] | None = None,
    rrf_k: int = config.RRF_K,
) -> list[tuple[int, float]]:
    weights = weights or (1.0,) * len(rankings)
    fused: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, (doc_pos, _) in enumerate(ranking, start=1):
            fused[doc_pos] = fused.get(doc_pos, 0.0) + w / (rrf_k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])[:k]


def search(
    index: SearchIndex,
    query: str,
    mode: str = "hybrid",
    k: int = config.CANDIDATES_K,
    company: str | None = None,
) -> list[tuple[Document, float]]:
    if mode == "dense":
        ranking = dense_ranking(index, query, k, company)
    elif mode == "bm25":
        ranking = bm25_ranking(index, query, k, company)
    elif mode == "hybrid":
        ranking = rrf_fuse(
            [dense_ranking(index, query, k, company), bm25_ranking(index, query, k, company)],
            k,
            weights=(config.DENSE_WEIGHT, config.BM25_WEIGHT),
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; pick one of {MODES}")
    return [(index.docs[i], s) for i, s in ranking]


def retrieve_for_plan(
    index: SearchIndex, plan: QueryPlan, mode: str = "hybrid", k: int = config.CANDIDATES_K
) -> list[tuple[Document, float]]:
    """Run every sub-query and interleave the results (round-robin, de-duplicated)."""
    per_query = [search(index, q, mode, k, company) for q, company in plan.sub_queries]
    seen: set[str] = set()
    merged: list[tuple[Document, float]] = []
    for rank in range(k):
        for results in per_query:
            if rank < len(results):
                doc, score = results[rank]
                cid = doc.metadata["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    merged.append((doc, score))
    return merged


def select_context(ranked: list[tuple[Document, float]], plan: QueryPlan, k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """Pick the final k chunks. For multi-company questions keep a fair share per
    company (round-robin over each company's best chunks), otherwise the reranker
    can fill every slot with the larger filing and the comparison loses a side."""
    companies = [c for _, c in plan.sub_queries if c]
    if len(companies) < 2:
        return ranked[:k]
    queues = {c: [t for t in ranked if t[0].metadata["company"] == c] for c in companies}
    out: list[tuple[Document, float]] = []
    while len(out) < k and any(queues.values()):
        for c in companies:
            if queues[c] and len(out) < k:
                out.append(queues[c].pop(0))
    return out
