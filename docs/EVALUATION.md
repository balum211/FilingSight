# FilingSight evaluation: chunking comparison and reranking impact

Project 2 asks for (1) two chunking strategies compared on the same queries and (2) a reranking step with its measured improvement. This report covers both, plus the refusal path and a failure analysis. Every number here is reproducible with the commands at the bottom; raw results are in `eval/results/`.

## Test set

25 questions in `data/eval/questions.json`, written by reading the filings:

| Category | Count | Example |
|---|---|---|
| Numeric from a table | 5 | How many Active Riders did Lyft have in Q4 2021? |
| Numeric from prose | 6 | What share of Uber's Gross Bookings was paid by card in 2021? |
| Entity / event | 5 | What did Lyft sell to Woven Planet in 2021? |
| Multi-fact | 1 | Which metro areas generated 23% of Uber's Mobility Gross Bookings? |
| Comparison (both filings) | 3 | Compare Uber's and Lyft's revenue growth in 2021. |
| Ambiguous (no company named) | 1 | What was the net loss in 2021? |
| Unanswerable | 5 | What was Uber's revenue in 2023? Who is the CEO of DoorDash? |

**Labels are evidence phrases, not chunk ids.** For example U01 is relevant if a retrieved Uber chunk contains `17,455` or `$17.5 billion`. That makes the metrics fair across chunkers (fixed and semantic chunks have different ids and boundaries). Comparison questions have one requirement per company, so a hit needs evidence from both filings.

**Metrics:** Hit@5 (all requirements found in the top 5 chunks the LLM sees), Recall@5 (share of requirements found), MRR@10 (1 / rank of the first relevant chunk), Precision@5 (share of the top 5 that is relevant), p50 retrieval latency on a 4-core CPU.

## 1. Chunking strategy comparison

| Chunking | Chunks | Mean size | Retrieval | Hit@5 | Recall@5 | MRR@10 | Precision@5 |
|---|---|---|---|---|---|---|---|
| Fixed 900/150 | 2,104 | 615 chars | dense | 0.65 | 0.70 | 0.55 | 0.28 |
| Semantic (p85 breakpoints, ≤1,100) | 1,767 | 718 chars | dense | **0.85** | **0.88** | **0.61** | **0.32** |
| Fixed | | | BM25 | 0.85 | 0.88 | 0.81 | 0.42 |
| Semantic | | | BM25 | **0.90** | **0.93** | 0.81 | 0.38 |
| Fixed | | | hybrid (RRF) | 0.80 | 0.82 | 0.69 | 0.39 |
| Semantic | | | hybrid (RRF) | **0.85** | **0.88** | **0.74** | **0.40** |
| Fixed | | | hybrid + rerank | 0.90 | 0.93 | 0.83 | **0.46** |
| Semantic | | | hybrid + rerank | **0.95** | **0.97** | **0.87** | 0.45 |

**Semantic chunking wins in every mode**, by +5 to +20 points of hit@5. The biggest gap is dense-only retrieval (+20), where a chunk's embedding is only as good as the topic coherence of its text.

Why, from the per-question results:

* **Tables stay whole.** L02 (Lyft Active Riders, Q4 2021) is missed by fixed chunks in every mode: the 900-char window cuts the table so the `Three Months Ended December 31 | 18,728` row lands in a chunk without the "Active Riders" title. The semantic chunker keeps consecutive table rows together because their embeddings are similar.
* **One topic per chunk.** U08 (the five metro areas) and U02 (MAPCs) are paragraphs embedded in long risk-factor pages. Fixed windows mix them with the neighbouring risk, which dilutes the embedding; semantic boundaries isolate them.
* **Cost:** the semantic chunker embeds every sentence (≈15k units) and took ~4 minutes on CPU versus 45 seconds for fixed. That is why chunks and embeddings are precomputed and committed.

Fixed chunks have slightly higher Precision@5 after reranking (0.46 vs 0.45) because there are more, smaller chunks, so several overlapping windows can all contain the same figure. That is redundancy, not extra information.

## 2. Reranking impact

Reranking takes the hybrid candidates (20 per company) and re-scores them; the top 5 go to the LLM.

| Chunking | Before rerank (hybrid) Hit@5 / MRR | After `local` rerank Hit@5 / MRR | Change |
|---|---|---|---|
| Fixed | 0.80 / 0.69 | 0.90 / 0.83 | **+10 pts hit, +0.14 MRR** |
| Semantic | 0.85 / 0.74 | 0.95 / 0.87 | **+10 pts hit, +0.13 MRR** |

The `local` reranker scores each candidate on its single best sentence or table row (45%), whole-chunk similarity (25%) and coverage of the question's key terms, with numbers and names weighted double (30%). Table rows are scored with their table title attached (`Active Riders | 2021 | 2020 › Three Months Ended December 31 | 18,728`).

What it fixed: U02, U07 and U08 move from rank 4 to 9 to **rank 1** (one precise sentence inside a long chunk no longer gets averaged away), L03 (Lyft Adjusted EBITDA) moves from rank 6 to rank 1, and C03 is now answered for both companies.

Cost: about **2.4 s** extra per question on a 4-core CPU (it embeds up to 6 sentences per candidate). Within the 8 s budget, but it is the slowest step; the cross-encoder option (`FS_RERANKER=cross-encoder`) is the production choice on a machine with PyTorch.

**Company-balanced selection.** The first reranked version *broke* comparison questions: Uber's filing is bigger, so its chunks filled all five slots and Lyft disappeared (C01 and C03 failed). The final top 5 now round-robins between companies for multi-company questions, which fixed C03 and A01.

**Hybrid vs BM25 alone.** On this question set plain BM25 (0.85 to 0.90) beats unweighted hybrid (0.80 to 0.85), because most questions contain exact figures, names and terms of art (MAPCs, Adjusted EBITDA, Woven Planet). Dense retrieval still earns its place: it finds answers BM25 ranks low (L03, Lyft Adjusted EBITDA, is a semantic-dense hit but a BM25 miss), and after reranking, hybrid beats BM25. A BM25 weight of 2.0 in RRF was tried and did not change the reranked result, so standard 1:1 RRF was kept.

## 3. The "I don't know" path

| Check | Catches |
|---|---|
| Year-scope rule (filings report 2019 to 2021) | X01 "Uber's revenue in 2023" |
| Confidence gate: top reranked score < 0.60 | X02 DoorDash CEO (0.36), X03 analyst price target (0.56), X04 Tesla (0.45), X05 Uber NPS (0.58) |
| LLM must reply `NOT_FOUND` when sources lack the answer | anything the gate lets through |
| Citation validation: drop citations to non-existent sources; regenerate once if none | uncited answers |

Results (semantic + hybrid + local rerank): **5/5 unanswerable questions refused, 0/20 answerable questions wrongly refused.**

Honest caveats: the threshold (0.60) was set by looking at this same set's score distribution (lowest answerable 0.65, highest unanswerable 0.62 for the 2023 question, which the year rule catches), so the margin is thin and a held-out set would likely show some errors. X05 is a deliberate trap: Lyft's 10-K mentions Net Promoter Score, but no filing gives a score.

## 4. End-to-end answers

| Metric | Extractive mode (no LLM, measured) |
|---|---|
| Unanswerable correctly refused | 100% (5/5) |
| Answerable wrongly refused | 0% (0/20) |
| Gold evidence inside a cited source | 90% (18/20) |
| Latency p50 / p95 | 3.2 s / 5.9 s |

With an LLM key, `python eval/run_answer_eval.py` adds two LLM-judge metrics per answer: **faithfulness** (share of claims supported by the retrieved sources; target ≥ 90%) and **correctness** against the human reference answer, and writes them into `eval/results/answers.md`.

## 5. Failure analysis (what still goes wrong)

| Q | What happens | Why | Fix to try next |
|---|---|---|---|
| C01 Compare revenue growth | Uber side found (p.57 "Revenue increased $6.3 billion, or 57%"); Lyft's first chunk with the revenue figure ranks 8th overall (4th among Lyft chunks), just outside the top 5 | "growth" is phrased as "increase" in Lyft's MD&A and the Lyft sub-query competes with incentive-program revenue chunks | query expansion (growth → increase, year over year) or a cross-encoder |
| L01 / L02 | Found, but at rank 2 to 4 | the same figure appears in the income statement, MD&A and notes; the reranker prefers the narrative mention | fine for answering; dedupe near-identical figures |
| Fixed chunks, L02 | Table row separated from its title | fixed windows ignore table structure | semantic or table-aware chunking (adopted) |
| Gate margin | 0.03 between lowest answerable and highest unanswerable score | small, same-set tuning | larger held-out set; cross-encoder scores separate better |

## 6. Learnings

1. **Retrieval, not the model, decided quality.** Moving from fixed/dense to semantic/hybrid/rerank took hit@5 from 0.65 to 0.95 without touching the LLM.
2. **Clean the PDF before you chunk.** pypdf produced "rapidlychanged" and one table cell per line; switching to PyMuPDF and rebuilding table rows was the cheapest quality win.
3. **Exact-figure questions need lexical search.** BM25 alone was strong; dense alone missed numbers and terms of art.
4. **Rerankers can break comparisons.** Any top-k cut needs to respect the question's structure (one share per company).
5. **Design the refusal first.** Three independent refusal checks gave 5/5 refusals with no false refusals, and the year rule handles the case the score threshold cannot.
6. **Label with evidence phrases.** It made the chunking comparison fair and the eval reusable when chunk boundaries change.

## Reproduce

```bash
python eval/run_retrieval_eval.py          # sections 1 to 3
python eval/run_answer_eval.py --no-llm    # section 4, extractive
python eval/run_answer_eval.py             # section 4 with LLM + judge (needs NEBIUS_API_KEY)
```
