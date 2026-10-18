"""Chunking x retrieval x reranking comparison on the labelled question set.

No API key needed: this measures retrieval only (the part RAG usually fails at).

    python eval/run_retrieval_eval.py                      # local reranker
    python eval/run_retrieval_eval.py --rerankers none local cross-encoder

Writes eval/results/retrieval.json and eval/results/retrieval.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filingsight import config  # noqa: E402
from filingsight.chunking import STRATEGIES  # noqa: E402
from filingsight.evaluation import load_questions, score_retrieval  # noqa: E402
from filingsight.index import load_index  # noqa: E402
from filingsight.pipeline import run_retrieval  # noqa: E402
from filingsight.rerank import THRESHOLDS  # noqa: E402
from filingsight.retrieval import MODES  # noqa: E402

RESULTS = ROOT / "eval" / "results"


def run_config(strategy: str, mode: str, reranker: str, questions: list[dict], k: int) -> dict:
    load_index(strategy)  # warm up so latency excludes index loading
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        res = run_retrieval(q["question"], strategy=strategy, mode=mode, reranker=reranker, k=10)
        docs = [d for d, _ in res.context]
        top_score = res.top_score if reranker != "none" else None
        ms = (time.perf_counter() - t0) * 1000
        year_rule = bool(res.plan.out_of_scope_years)
        row = {"id": q["id"], "year_rule": year_rule, "category": q["category"], "latency_ms": round(ms, 1), "top_score": top_score,
               "top_pages": [f"{d.metadata['ticker']} p.{d.metadata['page']}" for d in docs[:k]]}
        if q["evidence"]:
            s = score_retrieval(docs, q["evidence"], k)
            row.update(hit=s.hit, recall=s.recall, rr=score_retrieval(docs, q["evidence"], 10).rr, precision=s.precision)
        rows.append(row)

    answerable = [r for r in rows if "hit" in r]
    unanswerable = [r for r in rows if "hit" not in r]
    summary = {
        "strategy": strategy, "mode": mode, "reranker": reranker, "k": k,
        "hit@k": statistics.mean(r["hit"] for r in answerable),
        "recall@k": statistics.mean(r["recall"] for r in answerable),
        "mrr@10": statistics.mean(r["rr"] for r in answerable),
        "precision@k": statistics.mean(r["precision"] for r in answerable),
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in rows),
    }
    if reranker != "none":
        th = THRESHOLDS[reranker]
        summary["threshold"] = th
        summary["refusal_recall"] = statistics.mean(float(r["top_score"] < th) for r in unanswerable)
        summary["false_refusal_rate"] = statistics.mean(float(r["top_score"] < th) for r in answerable)
        summary["refusal_recall_with_year_rule"] = statistics.mean(
            float(r["top_score"] < th or r["year_rule"]) for r in unanswerable)
    return {"summary": summary, "rows": rows}


def fmt(x: float) -> str:
    return f"{x:.2f}"


def to_markdown(results: list[dict], n_answerable: int, n_unanswerable: int) -> str:
    lines = [
        f"Questions: {n_answerable} answerable (with evidence labels) + {n_unanswerable} unanswerable. "
        f"k = {results[0]['summary']['k']}. Candidates per sub-query before reranking: {config.CANDIDATES_K}.",
        "",
        "| Chunking | Retrieval | Reranker | Hit@5 | Recall@5 | MRR@10 | Precision@5 | p50 latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["summary"]
        lines.append(
            f"| {s['strategy']} | {s['mode']} | {s['reranker']} | {fmt(s['hit@k'])} | {fmt(s['recall@k'])} | "
            f"{fmt(s['mrr@10'])} | {fmt(s['precision@k'])} | {s['p50_latency_ms']:.0f} ms |"
        )
    gated = [r["summary"] for r in results if "threshold" in r["summary"]]
    if gated:
        lines += ["", "\"I don't know\" gate (refuse when the top reranked score < threshold):", "",
                  "| Chunking | Retrieval | Reranker | Threshold | Unanswerable refused (score only) | + year-scope rule | Answerable wrongly refused |",
                  "|---|---|---|---|---|---|---|"]
        for s in gated:
            lines.append(f"| {s['strategy']} | {s['mode']} | {s['reranker']} | {s['threshold']} | "
                         f"{s['refusal_recall']:.0%} | {s['refusal_recall_with_year_rule']:.0%} | {s['false_refusal_rate']:.0%} |")
    return "\n".join(lines) + "\n"


def per_question_table(results: list[dict], pick: list[tuple[str, str, str]]) -> str:
    by_key = {(r["summary"]["strategy"], r["summary"]["mode"], r["summary"]["reranker"]): r for r in results}
    chosen = [by_key[p] for p in pick if p in by_key]
    head = "| Q | Category | " + " | ".join(f"{s}/{m}/{rr} (hit, RR@10)" for s, m, rr in pick if (s, m, rr) in by_key) + " |"
    lines = [head, "|" + "---|" * (2 + len(chosen))]
    for i, row in enumerate(chosen[0]["rows"]):
        if "hit" not in row:
            continue
        cells = [f"{'✅' if c['rows'][i]['hit'] else '❌'} {c['rows'][i]['rr']:.2f}" for c in chosen]
        lines.append(f"| {row['id']} | {row['category']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerankers", nargs="+", default=["none", "local"])
    ap.add_argument("--k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    questions = load_questions()
    results = []
    for strategy in STRATEGIES:
        for mode in MODES:
            for reranker in args.rerankers:
                if reranker != "none" and mode != "hybrid":
                    continue  # rerank the production retriever only
                res = run_config(strategy, mode, reranker, questions, args.k)
                results.append(res)
                s = res["summary"]
                print(f"{strategy:9s} {mode:7s} {reranker:13s} hit={s['hit@k']:.2f} recall={s['recall@k']:.2f} "
                      f"mrr={s['mrr@10']:.2f} prec={s['precision@k']:.2f} p50={s['p50_latency_ms']:.0f}ms"
                      + (f" refuse={s['refusal_recall']:.0%} false_refuse={s['false_refusal_rate']:.0%}" if "threshold" in s else ""))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(results, indent=2))
    n_ans = sum(1 for q in questions if q["evidence"])
    md = "# Retrieval evaluation\n\n" + to_markdown(results, n_ans, len(questions) - n_ans)
    best_rr = args.rerankers[-1]
    md += "\n## Per question\n\n" + per_question_table(
        results, [("fixed", "dense", "none"), ("fixed", "hybrid", "none"), ("semantic", "hybrid", "none"),
                  ("fixed", "hybrid", best_rr), ("semantic", "hybrid", best_rr)])
    (RESULTS / "retrieval.md").write_text(md)
    print(f"\nwrote {RESULTS / 'retrieval.md'}")


if __name__ == "__main__":
    main()
