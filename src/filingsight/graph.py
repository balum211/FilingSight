"""The answer flow as a LangGraph state machine.

    analyze ──(year outside 2019-2021)──────────────────────────► refuse
       │
       ▼
    retrieve (hybrid per company + rerank)
       │
       ▼
    gate ──(top rerank score < threshold)───────────────────────► refuse
       │
       ▼
    generate (LLM with numbered sources, or extractive fallback)
       │
       ▼
    validate_citations ──(NOT_FOUND)────────────────────────────► refuse
       │   └─(no valid citation, first try)──► generate (stricter)
       ▼
      END

The refusal path is designed first: a wrong number in a financial answer is
worse than "I could not find this".
"""
from __future__ import annotations

import re
import time
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from . import config
from .chunking import split_units
from .pipeline import run_retrieval
from .prompts import ANSWER_SYSTEM, ANSWER_USER, REFUSAL
from .rerank import coverage, key_terms
from .retrieval import plan_query

_CITE = re.compile(r"\[(\d+)\]")
_NUMERIC_Q = re.compile(r"\b(how (many|much)|what (was|were|is)|revenue|loss|ebitda|employees|cash|percent|share)\b", re.I)


class RAGState(TypedDict, total=False):
    question: str
    strategy: str
    reranker: str
    use_llm: bool
    plan: Any
    sources: list[dict]  # numbered context passed to the LLM
    top_score: float
    threshold: float
    answer: str
    citations: list[int]
    route: str  # "answered" | "refused"
    refusal_reason: str
    attempts: int
    generator: str  # "llm" | "extractive"
    timings_ms: dict[str, float]


def _source_dict(i: int, doc, score: float) -> dict:
    m = doc.metadata
    return {
        "n": i,
        "company": m["company"],
        "page": m["page"],
        "item": m["item"],
        "item_title": m["item_title"],
        "header": m["header"],
        "text": doc.page_content,
        "score": round(float(score), 3),
        "chunk_id": m["chunk_id"],
    }


def format_sources(sources: list[dict]) -> str:
    return "\n\n".join(f"[{s['n']}] {s['header']}\n{s['text']}" for s in sources)


# ---------------------------------------------------------------- nodes
def analyze(state: RAGState) -> RAGState:
    plan = plan_query(state["question"])
    out: RAGState = {"plan": plan, "attempts": 0, "timings_ms": {}}
    if plan.out_of_scope_years:
        years = ", ".join(map(str, plan.out_of_scope_years))
        out["route"] = "refused"
        out["refusal_reason"] = (
            f"The corpus is the FY2021 10-Ks, which report {config.COVERED_YEARS[0]}–{config.COVERED_YEARS[-1]}; "
            f"{years} is not covered."
        )
    return out


def retrieve(state: RAGState) -> RAGState:
    res = run_retrieval(
        state["question"],
        strategy=state.get("strategy", config.DEFAULT_CHUNKING),
        reranker=state.get("reranker", config.RERANKER),
        plan=state["plan"],
    )
    sources = [_source_dict(i, d, s) for i, (d, s) in enumerate(res.context, start=1)]
    timings = dict(state.get("timings_ms", {}))
    timings.update(res.timings_ms)
    return {"sources": sources, "top_score": res.top_score, "threshold": res.threshold, "timings_ms": timings}


def gate(state: RAGState) -> RAGState:
    if not state["sources"] or state["top_score"] < state["threshold"]:
        return {
            "route": "refused",
            "refusal_reason": (
                f"Best passage relevance {state['top_score']:.2f} is below the confidence threshold "
                f"{state['threshold']:.2f}, so the filings probably do not answer this."
            ),
        }
    return {}


def extractive_answer(question: str, sources: list[dict], max_sentences: int = 4) -> str:
    """No-LLM fallback: the most relevant sentences/table rows, each cited."""
    from .embeddings import get_embeddings

    terms = key_terms(question)
    units = [(s["n"], u) for s in sources for u in split_units(s["text"], with_captions=True) if len(u) > 25]
    if not units:
        return "NOT_FOUND"
    emb = get_embeddings()
    q = emb.embed_array([question])[0]
    sims = emb.embed_array([u for _, u in units]) @ q
    numeric = bool(_NUMERIC_Q.search(question))
    scores = [
        0.6 * float(sim) + 0.4 * coverage(terms, u) + (0.15 if numeric and any(c.isdigit() for c in u) else 0.0)
        for sim, (_, u) in zip(sims, units)
    ]
    order = list(np.argsort(scores)[::-1])
    company_of = {s["n"]: s["company"] for s in sources}
    companies = list(dict.fromkeys(s["company"] for s in sources))
    # Round-robin over companies so a comparison shows evidence for each side.
    queues = {c: [i for i in order if company_of[units[i][0]] == c] for c in companies}
    picked, used = [], set()
    while len(picked) < max_sentences and any(queues.values()):
        for c in companies:
            while queues[c]:
                n, u = units[queues[c].pop(0)]
                if u not in used:
                    used.add(u)
                    label = f"**{c}:** " if len(companies) > 1 else ""
                    picked.append(f"- {label}{u} [{n}]")
                    break
            if len(picked) >= max_sentences:
                break
    return "Most relevant passages from the filings (no LLM key set, so this is extractive):\n" + "\n".join(picked)


def generate(state: RAGState) -> RAGState:
    t = time.perf_counter()
    use_llm = state.get("use_llm", config.llm_available()) and config.llm_available()
    if use_llm:
        from .llm import get_chat_model

        system = ANSWER_SYSTEM
        if state.get("attempts", 0) > 0:
            system += "\nYour previous answer had no valid [n] citations. Every sentence MUST end with a citation."
        msgs = [
            ("system", system),
            ("user", ANSWER_USER.format(question=state["question"], sources=format_sources(state["sources"]))),
        ]
        answer = str(get_chat_model(temperature=0).invoke(msgs).content).strip()
        generator = "llm"
    else:
        answer = extractive_answer(state["question"], state["sources"])
        generator = "extractive"
    timings = dict(state.get("timings_ms", {}))
    timings["generate"] = (time.perf_counter() - t) * 1000
    return {"answer": answer, "generator": generator, "attempts": state.get("attempts", 0) + 1, "timings_ms": timings}


def validate_citations(state: RAGState) -> RAGState:
    answer = state["answer"]
    if answer.strip().upper().startswith("NOT_FOUND"):
        return {"route": "refused", "refusal_reason": "The model read the retrieved passages and found no answer in them."}
    valid = {s["n"] for s in state["sources"]}
    cited = sorted({int(n) for n in _CITE.findall(answer)})
    good = [n for n in cited if n in valid]
    # Drop citations that point at sources that do not exist.
    cleaned = _CITE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", answer)
    out: RAGState = {"answer": cleaned, "citations": good}
    if good:
        out["route"] = "answered"
    elif state.get("attempts", 0) >= 2:
        out["route"] = "answered"
        out["answer"] = cleaned + "\n\n⚠️ The model did not cite its sources; treat this answer as unverified."
    else:
        out["route"] = "retry"
    return out


def refuse(state: RAGState) -> RAGState:
    return {"answer": REFUSAL, "route": "refused", "citations": []}


# ---------------------------------------------------------------- graph
def _after_analyze(state: RAGState) -> str:
    return "refuse" if state.get("route") == "refused" else "retrieve"


def _after_gate(state: RAGState) -> str:
    return "refuse" if state.get("route") == "refused" else "generate"


def _after_validate(state: RAGState) -> str:
    return {"refused": "refuse", "retry": "generate"}.get(state.get("route", ""), END)


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("analyze", analyze)
    g.add_node("retrieve", retrieve)
    g.add_node("gate", gate)
    g.add_node("generate", generate)
    g.add_node("validate_citations", validate_citations)
    g.add_node("refuse", refuse)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", _after_analyze, {"refuse": "refuse", "retrieve": "retrieve"})
    g.add_edge("retrieve", "gate")
    g.add_conditional_edges("gate", _after_gate, {"refuse": "refuse", "generate": "generate"})
    g.add_edge("generate", "validate_citations")
    g.add_conditional_edges(
        "validate_citations", _after_validate, {"refuse": "refuse", "generate": "generate", END: END}
    )
    g.add_edge("refuse", END)
    return g.compile()


_GRAPH = None


def ask(question: str, **options) -> RAGState:
    """Run one question through the graph. Options: strategy, reranker, use_llm."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    t = time.perf_counter()
    state = _GRAPH.invoke({"question": question, **options})
    state.setdefault("timings_ms", {})["total"] = (time.perf_counter() - t) * 1000
    return state
