"""Retrieval metrics that are fair across chunking strategies.

Gold labels are *evidence phrases* (e.g. "17,455" for Uber's 2021 revenue), not
chunk ids, so the same question set scores fixed and semantic chunks equally.
A question has one or more requirements (one per company for comparisons);
each requirement lists alternative phrase sets, and a chunk satisfies it when
it belongs to the right company and contains every phrase of one alternative.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from .config import EVAL_DIR


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).lower()


def satisfies(doc: Document, requirement: dict) -> bool:
    if requirement.get("company") and doc.metadata.get("company") != requirement["company"]:
        return False
    body = _norm(doc.page_content)
    return any(all(_norm(p) in body for p in alt) for alt in requirement["any"])


@dataclass
class RetrievalScore:
    hit: float  # every requirement satisfied within top-k
    recall: float  # fraction of requirements satisfied within top-k
    rr: float  # reciprocal rank of the first satisfying chunk
    precision: float  # share of top-k chunks that satisfy some requirement


def score_retrieval(docs: list[Document], evidence: list[dict], k: int) -> RetrievalScore:
    top = docs[:k]
    found = [any(satisfies(d, req) for d in top) for req in evidence]
    first = next((i for i, d in enumerate(docs) if any(satisfies(d, r) for r in evidence)), None)
    relevant = sum(1 for d in top if any(satisfies(d, r) for r in evidence))
    return RetrievalScore(
        hit=float(all(found)),
        recall=sum(found) / len(found),
        rr=0.0 if first is None else 1.0 / (first + 1),
        precision=relevant / max(len(top), 1),
    )


def load_questions(path: Path = EVAL_DIR / "questions.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
