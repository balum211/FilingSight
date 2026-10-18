# Retrieval evaluation

Questions: 20 answerable (with evidence labels) + 5 unanswerable. k = 5. Candidates per sub-query before reranking: 20.

| Chunking | Retrieval | Reranker | Hit@5 | Recall@5 | MRR@10 | Precision@5 | p50 latency |
|---|---|---|---|---|---|---|---|
| fixed | dense | none | 0.65 | 0.70 | 0.55 | 0.28 | 37 ms |
| fixed | bm25 | none | 0.85 | 0.88 | 0.81 | 0.42 | 3 ms |
| fixed | hybrid | none | 0.80 | 0.82 | 0.69 | 0.39 | 39 ms |
| fixed | hybrid | local | 0.90 | 0.93 | 0.83 | 0.46 | 2401 ms |
| semantic | dense | none | 0.85 | 0.88 | 0.61 | 0.32 | 40 ms |
| semantic | bm25 | none | 0.90 | 0.93 | 0.81 | 0.38 | 2 ms |
| semantic | hybrid | none | 0.85 | 0.88 | 0.74 | 0.40 | 38 ms |
| semantic | hybrid | local | 0.95 | 0.97 | 0.87 | 0.45 | 2404 ms |

"I don't know" gate (refuse when the top reranked score < threshold):

| Chunking | Retrieval | Reranker | Threshold | Unanswerable refused (score only) | + year-scope rule | Answerable wrongly refused |
|---|---|---|---|---|---|---|
| fixed | hybrid | local | 0.6 | 100% | 100% | 0% |
| semantic | hybrid | local | 0.6 | 80% | 100% | 0% |

## Per question

| Q | Category | fixed/dense/none (hit, RR@10) | fixed/hybrid/none (hit, RR@10) | semantic/hybrid/none (hit, RR@10) | fixed/hybrid/local (hit, RR@10) | semantic/hybrid/local (hit, RR@10) |
|---|---|---|---|---|---|---|
| U01 | numeric-table | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| U02 | numeric-table | ❌ 0.00 | ❌ 0.11 | ❌ 0.12 | ✅ 1.00 | ✅ 1.00 |
| U03 | numeric-table | ✅ 0.50 | ✅ 1.00 | ✅ 1.00 | ✅ 0.50 | ✅ 1.00 |
| U04 | numeric-prose | ✅ 0.50 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| U05 | numeric-prose | ✅ 0.50 | ✅ 1.00 | ✅ 0.50 | ✅ 1.00 | ✅ 1.00 |
| U06 | entity | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| U07 | numeric-prose | ❌ 0.00 | ✅ 0.25 | ✅ 0.20 | ✅ 1.00 | ✅ 1.00 |
| U08 | multi-fact | ❌ 0.00 | ❌ 0.12 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| U09 | entity | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| L01 | numeric-table | ✅ 0.20 | ✅ 1.00 | ✅ 1.00 | ✅ 0.50 | ✅ 0.50 |
| L02 | numeric-table | ❌ 0.00 | ❌ 0.00 | ✅ 0.33 | ❌ 0.00 | ✅ 0.25 |
| L03 | numeric-prose | ✅ 0.20 | ✅ 0.20 | ❌ 0.17 | ✅ 1.00 | ✅ 1.00 |
| L04 | entity | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| L05 | numeric-prose | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| L06 | numeric-prose | ❌ 0.14 | ✅ 0.33 | ✅ 0.50 | ✅ 0.50 | ✅ 1.00 |
| L07 | entity | ✅ 0.50 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| C01 | comparison | ❌ 1.00 | ✅ 0.25 | ✅ 0.50 | ❌ 0.20 | ❌ 0.33 |
| C02 | comparison | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| C03 | comparison | ❌ 1.00 | ❌ 1.00 | ❌ 1.00 | ✅ 1.00 | ✅ 0.33 |
| A01 | ambiguous | ✅ 0.50 | ✅ 0.50 | ✅ 0.50 | ✅ 1.00 | ✅ 1.00 |
