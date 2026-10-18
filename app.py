"""FilingSight: chat with the Uber and Lyft FY2021 10-K filings.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from filingsight import config  # noqa: E402
from filingsight.embeddings import EMBEDDING_MODEL_NAME  # noqa: E402
from filingsight.graph import ask  # noqa: E402
from filingsight.index import load_index  # noqa: E402

st.set_page_config(page_title="FilingSight · 10-K RAG", page_icon="📑", layout="wide")

RESULTS = Path(__file__).resolve().parent / "eval" / "results"
SAMPLES = [
    "What was Uber's total revenue in 2021?",
    "How many Active Riders did Lyft have in Q4 2021?",
    "Compare Uber's and Lyft's revenue growth in 2021.",
    "What did Lyft sell to Woven Planet?",
    "How do Uber and Lyft describe the impact of Proposition 22?",
    "What was Uber's revenue in 2023?",
]


@st.cache_resource(show_spinner="Loading the search index (first run only)...")
def warm(strategy: str):
    return len(load_index(strategy).docs)


def reranker_options() -> list[str]:
    opts = ["local", "none"]
    try:
        import sentence_transformers  # noqa: F401

        opts.insert(1, "cross-encoder")
    except ImportError:
        pass
    if config.llm_available():
        opts.append("llm")
    return opts


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📑 FilingSight")
    st.caption("Cited answers from the Uber and Lyft FY2021 Form 10-K filings.")
    strategy = st.radio(
        "Chunking strategy", ["semantic", "fixed"], index=0 if config.DEFAULT_CHUNKING == "semantic" else 1,
        help="semantic = embedding-breakpoint chunks; fixed = 900-char recursive splitter",
    )
    reranker = st.selectbox("Reranker", reranker_options(), help="local needs no download; none shows raw hybrid order")
    llm_on = config.llm_available()
    use_llm = st.toggle("Use LLM for answers", value=llm_on, disabled=not llm_on)
    if llm_on:
        st.success(f"LLM: {config.LLM_MODEL}", icon="✅")
    else:
        st.info("No NEBIUS_API_KEY set, so answers are extractive (top cited passages). Add a key in .env "
                "or Streamlit secrets for written answers.", icon="ℹ️")
    n_chunks = warm(strategy)
    st.divider()
    st.markdown(
        f"**Corpus:** Uber 10-K (150 pp.) + Lyft 10-K (125 pp.), FY2021  \n"
        f"**Chunks:** {n_chunks:,} ({strategy})  \n"
        f"**Embeddings:** {EMBEDDING_MODEL_NAME}  \n"
        f"**Retrieval:** Chroma dense + BM25, RRF fusion, top-{config.TOP_K}"
    )
    if st.button("Clear chat"):
        st.session_state.pop("history", None)

chat_tab, eval_tab, how_tab = st.tabs(["💬 Ask the filings", "📊 Evaluation", "🧭 How it works"])


def md_safe(text: str) -> str:
    """Escape $ so Streamlit does not render dollar amounts as LaTeX."""
    return text.replace("$", "\\$")


def render_answer(state: dict):
    if state["route"] == "refused":
        st.warning(state["answer"], icon="🛑")
        if state.get("refusal_reason"):
            st.caption(f"Why: {state['refusal_reason']}")
    else:
        st.markdown(md_safe(state["answer"]))
    t = state.get("timings_ms", {})
    bits = [f"{k} {v / 1000:.1f}s" for k, v in t.items() if k != "total"]
    meta = f"⏱️ {t.get('total', 0) / 1000:.1f}s ({', '.join(bits)})"
    if state.get("top_score") is not None and state.get("sources"):
        meta += f" · top relevance {state['top_score']:.2f} (gate {state.get('threshold', 0):.2f})"
    if state.get("generator"):
        meta += f" · {state['generator']}"
    st.caption(meta)
    sources = state.get("sources") or []
    if sources:
        cited = set(state.get("citations") or [])
        with st.expander(f"Sources ({len(sources)} retrieved, {len(cited)} cited)"):
            for s in sources:
                mark = "📌 " if s["n"] in cited else ""
                st.markdown(f"{mark}**[{s['n']}] {s['company']} 10-K FY2021, p.{s['page']}** · "
                            f"Item {s['item']} {s['item_title']} · score {s['score']}")
                st.text(s["text"][:1200])


with chat_tab:
    history = st.session_state.setdefault("history", [])
    if not history:
        st.markdown("Try one of these, or type your own question below:")
        cols = st.columns(3)
        for i, q in enumerate(SAMPLES):
            if cols[i % 3].button(q, key=f"sample{i}", use_container_width=True):
                st.session_state["pending"] = q
                st.rerun()
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant", avatar="📑"):
            render_answer(turn["state"])

    question = st.chat_input("Ask about Uber or Lyft's 2021 annual report...") or st.session_state.pop("pending", None)
    if question:
        with st.spinner(f"Retrieving, reranking and answering: {question}"):
            try:
                state = ask(question, strategy=strategy, reranker=reranker, use_llm=use_llm)
            except Exception as exc:  # surface provider errors instead of a stack trace
                st.error(f"Something went wrong: {exc}")
                st.stop()
        history.append({"question": question, "state": state})
        st.rerun()  # redraw so the newest answer sits above the input box

with eval_tab:
    st.subheader("Chunking × retrieval × reranking comparison")
    st.caption("25 labelled questions (20 answerable, 5 unanswerable). Reproduce with "
               "`python eval/run_retrieval_eval.py` and `python eval/run_answer_eval.py`.")
    for name in ("retrieval.md", "answers.md"):
        path = RESULTS / name
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))
        else:
            st.info(f"{name} not generated yet.")

with how_tab:
    st.markdown(
        """
**One-liner:** FilingSight helps retail investors and junior equity analysts answer metric, risk and
strategy questions from the Uber and Lyft FY2021 Form 10-K filings in a cited chat app, targeting
≥ 90% faithfulness and ≥ 90% hit@5 retrieval relevance within an 8-second answer.

```
analyze ──(year outside 2019-2021)──────────────► refuse
   │
retrieve: per-company hybrid (Chroma dense + BM25, RRF) → rerank → fair top-5
   │
gate ──(top relevance < threshold)──────────────► refuse
   │
generate (LLM with numbered sources | extractive fallback)
   │
validate_citations ──(NOT_FOUND)────────────────► refuse
   └──(no valid [n] on first try)──► generate again
```

| Layer | Decision |
|---|---|
| Corpus | Uber and Lyft 10-K for FY2021 (SEC filings, PDF, English, 275 body pages after dropping exhibits) |
| Cleaning | PyMuPDF text, page numbers dropped, table cells re-joined into rows, wrapped lines re-flowed, Item tagged |
| Chunking | fixed 900/150 chars vs semantic (embedding breakpoints, 85th pct, ≤1,100 chars); header prefix per chunk |
| Embedding | all-MiniLM-L6-v2, 384-dim, ONNX on CPU (chunk size matched to its 256-token window) |
| Retrieve | Chroma (cosine) + BM25, weighted RRF, 20 candidates per company, metadata filter by company |
| Rerank | local sentence-level + key-term reranker (or cross-encoder / LLM), company-balanced top-5 |
| Refusal | year-scope rule, confidence gate, `NOT_FOUND` from the LLM, citation validation |
"""
    )
