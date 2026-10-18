"""Retrieval pipeline shared by the LangGraph app and the evaluation:
plan -> hybrid retrieve (per sub-query) -> rerank -> select context."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.documents import Document

from . import config
from .index import load_index
from .rerank import THRESHOLDS, get_reranker
from .retrieval import QueryPlan, plan_query, retrieve_for_plan, select_context


@dataclass
class RetrievalResult:
    plan: QueryPlan
    candidates: list[Document]
    context: list[tuple[Document, float]]
    top_score: float
    threshold: float
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def confident(self) -> bool:
        return self.top_score >= self.threshold


def run_retrieval(
    question: str,
    strategy: str = config.DEFAULT_CHUNKING,
    mode: str = "hybrid",
    reranker: str = config.RERANKER,
    k: int = config.TOP_K,
    plan: QueryPlan | None = None,
) -> RetrievalResult:
    timings: dict[str, float] = {}
    t = time.perf_counter()
    index = load_index(strategy)
    plan = plan or plan_query(question)
    cands = retrieve_for_plan(index, plan, mode=mode, k=config.CANDIDATES_K)
    docs = [d for d, _ in cands]
    timings["retrieve"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    if reranker == "none":
        ranked = cands
        top = 1.0 if cands else 0.0
    else:
        ranked = get_reranker(reranker).rerank(question, docs, chunk_vectors=index.vectors_for(docs))
        top = ranked[0][1] if ranked else 0.0
    timings["rerank"] = (time.perf_counter() - t) * 1000
    return RetrievalResult(
        plan=plan,
        candidates=docs,
        context=select_context(ranked, plan, k),
        top_score=float(top),
        threshold=THRESHOLDS.get(reranker, 0.0),
        timings_ms=timings,
    )
