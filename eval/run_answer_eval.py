"""End-to-end answer evaluation through the LangGraph app.

    python eval/run_answer_eval.py              # uses the LLM if NEBIUS_API_KEY / OPENAI_API_KEY is set
    python eval/run_answer_eval.py --no-llm     # extractive mode, no key needed

Metrics
* refusal accuracy      unanswerable questions answered with the refusal message
* false refusals        answerable questions that were refused
* cited-evidence rate   answered questions where a cited source contains the gold
                        evidence (deterministic groundedness proxy, no judge needed)
* faithfulness          LLM judge: share of claims supported by retrieved sources (LLM mode)
* correctness           LLM judge: agrees with the human reference answer (LLM mode)
* latency               p50 / p95 end to end

Writes eval/results/answers.json and eval/results/answers.md.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.documents import Document  # noqa: E402

from filingsight import config  # noqa: E402
from filingsight.evaluation import load_questions, satisfies  # noqa: E402
from filingsight.graph import ask, format_sources  # noqa: E402
from filingsight.prompts import JUDGE_PROMPT  # noqa: E402

RESULTS = ROOT / "eval" / "results"


def judge(q: dict, state: dict) -> dict:
    from filingsight.llm import get_chat_model

    msg = get_chat_model(temperature=0, model=config.JUDGE_MODEL).invoke(
        JUDGE_PROMPT.format(question=q["question"], reference=q["answer"],
                            sources=format_sources(state.get("sources", [])), answer=state["answer"])
    )
    try:
        return json.loads(re.search(r"\{.*\}", str(msg.content), re.S).group(0))
    except Exception:
        return {"faithfulness": None, "correct": None, "notes": "judge output not parseable"}


def cited_evidence(q: dict, state: dict) -> bool:
    cited = [s for s in state.get("sources", []) if s["n"] in state.get("citations", [])]
    docs = [Document(page_content=s["text"], metadata={"company": s["company"]}) for s in cited]
    return all(any(satisfies(d, req) for d in docs) for req in q["evidence"])


def pct(xs):
    return f"{100 * statistics.mean(xs):.0f}%" if xs else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--strategy", default=config.DEFAULT_CHUNKING)
    ap.add_argument("--reranker", default=config.RERANKER)
    args = ap.parse_args()
    use_llm = config.llm_available() and not args.no_llm

    rows = []
    for q in load_questions():
        state = ask(q["question"], strategy=args.strategy, reranker=args.reranker, use_llm=use_llm)
        row = {
            "id": q["id"], "category": q["category"], "question": q["question"], "reference": q["answer"],
            "answer": state["answer"], "route": state["route"], "refusal_reason": state.get("refusal_reason", ""),
            "citations": state.get("citations", []),
            "sources": [f"[{s['n']}] {s['company']} p.{s['page']} (Item {s['item']})" for s in state.get("sources", [])],
            "latency_ms": round(state["timings_ms"]["total"]),
        }
        if q["evidence"] and state["route"] == "answered":
            row["cited_evidence"] = cited_evidence(q, state)
            if use_llm:
                row.update(judge(q, state))
        rows.append(row)
        print(f"{q['id']} {state['route']:9s} {row['latency_ms']:6d}ms  {state['answer'][:90]!r}")

    ans = [r for r in rows if r["category"] != "unanswerable"]
    una = [r for r in rows if r["category"] == "unanswerable"]
    answered = [r for r in ans if r["route"] == "answered"]
    lat = sorted(r["latency_ms"] for r in rows)
    summary = {
        "generator": "llm:" + config.LLM_MODEL if use_llm else "extractive (no LLM)",
        "strategy": args.strategy, "reranker": args.reranker,
        "refusal_accuracy": statistics.mean(r["route"] == "refused" for r in una),
        "false_refusal_rate": statistics.mean(r["route"] == "refused" for r in ans),
        "cited_evidence_rate": statistics.mean(r["cited_evidence"] for r in answered) if answered else None,
        "p50_latency_ms": statistics.median(lat),
        "p95_latency_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
    }
    faith = [r["faithfulness"] for r in answered if isinstance(r.get("faithfulness"), (int, float))]
    corr = [bool(r["correct"]) for r in answered if r.get("correct") is not None]
    if faith:
        summary["faithfulness"] = statistics.mean(faith)
    if corr:
        summary["correctness"] = statistics.mean(corr)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "answers.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    md = [f"# Answer evaluation ({summary['generator']}, {args.strategy} chunks, {args.reranker} reranker)", "",
          "| Metric | Value |", "|---|---|",
          f"| Unanswerable questions correctly refused | {pct([r['route'] == 'refused' for r in una])} ({len(una)}) |",
          f"| Answerable questions wrongly refused | {pct([r['route'] == 'refused' for r in ans])} ({len(ans)}) |",
          f"| Answered with the gold evidence in a cited source | {pct([r['cited_evidence'] for r in answered])} |"]
    if faith:
        md.append(f"| Faithfulness (LLM judge) | {100 * summary['faithfulness']:.0f}% |")
    if corr:
        md.append(f"| Correct vs reference (LLM judge) | {100 * summary['correctness']:.0f}% |")
    md += [f"| Latency p50 / p95 | {summary['p50_latency_ms'] / 1000:.1f}s / {summary['p95_latency_ms'] / 1000:.1f}s |", "",
           "| Q | Route | Cited evidence | Faithful | Correct | Latency | Answer |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ans_txt = r["answer"].replace("\n", " ").replace("|", "/")[:220]
        md.append(f"| {r['id']} | {r['route']} | {r.get('cited_evidence', '')} | {r.get('faithfulness', '')} | "
                  f"{r.get('correct', '')} | {r['latency_ms'] / 1000:.1f}s | {ans_txt} |")
    (RESULTS / "answers.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
