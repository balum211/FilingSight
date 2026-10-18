# Prompts, iterations and learnings

## 1. How AI coding tools were used

The project was built with **Claude Code** (an AI coding agent) working from the Week 2 handout and the course repos (Week 2 Session 1 RAG notebooks, Week 3 LangGraph notebooks). The starting prompt was:

> Can you read attached study materials and GitHub codes. I need to do the Week 2 project, the last document has the ask. Can you do the project for me.

The agent then worked through these steps, each checked by running code, not just written:

| Step | What was asked of the agent / what it did |
|---|---|
| 1 | Read the handout; pick a use case and track. Chose Project 2 (Financial Document Intelligence) on Track 2 (LangChain + LangGraph), because it requires the two things RAG projects usually skip: a chunking comparison and a measured reranking step. |
| 2 | Fill in the RAG framework before coding (one-liner, corpus, cleaning, freshness, chunking + embedding, retrieve). |
| 3 | Find a real financial corpus that downloads without an account: Uber and Lyft FY2021 10-Ks. |
| 4 | Build ingestion and inspect the cleaned text of the key MD&A and income-statement pages. |
| 5 | Implement fixed and semantic chunkers, Chroma + BM25 hybrid retrieval, and a reranker. |
| 6 | Write 25 questions with reference answers *by reading the filings*, labelled with evidence phrases; include comparisons, an ambiguous question and 5 unanswerable ones. |
| 7 | Build the evaluation matrix and iterate on the failures (section 3). |
| 8 | Build the LangGraph answer flow with the refusal path first, then the Streamlit chat UI. |
| 9 | Tests (19), CI, docs, screenshots. |

## 2. Prompts inside the app (`src/filingsight/prompts.py`)

### Answer prompt (system)

```
You are FilingSight, a financial research assistant that answers questions using ONLY excerpts
from the Uber and Lyft annual reports (Form 10-K, fiscal year 2021).

Rules:
1. Use only the numbered sources below. Never use outside knowledge, even if you are sure of it.
2. Cite every factual sentence with its source number in square brackets, e.g. [1] or [2][3].
3. Copy numbers exactly as written in the source and keep their units ("in millions", "in thousands").
   If a table is "in thousands", you may also restate the value in millions or billions.
4. If the question names no company, answer for each company separately.
5. For comparisons, give each company's figure with its own citation, then one sentence comparing them.
6. If the sources do not contain the answer, reply with exactly: NOT_FOUND
7. At most 6 sentences. No preamble, no investment advice.
```

User turn: `Question: {question}` + numbered sources, each with a header such as `[2] Lyft 10-K FY2021 | Item 7 MD&A | p.57`.

Why it looks like this:
* `NOT_FOUND` is a machine-checkable contract, so the graph routes to a fixed refusal message instead of trusting the model to phrase "I don't know" consistently.
* Rule 3 exists because Lyft reports tables in thousands and Uber in millions; mixing them is the most likely numeric error.
* Rules 4 and 5 match the retriever's per-company sub-queries.
* If the answer has no valid `[n]` citation, the graph regenerates once with: *"Your previous answer had no valid [n] citations. Every sentence MUST end with a citation."*

### LLM reranker prompt (`FS_RERANKER=llm`)

```
Grade how well each passage answers the question, from 0 (irrelevant) to 10 (contains the exact
answer). Reply with JSON only, mapping passage number to grade, e.g. {"0": 7, "1": 0}.
```

### LLM-judge prompt (`eval/run_answer_eval.py`)

Returns JSON with `faithfulness` (share of the answer's claims supported by the retrieved sources), `correct` (agrees with the human reference answer) and a one-line `notes`.

## 3. Iterations (what broke and what changed)

| # | Problem found | Change | Effect |
|---|---|---|---|
| 1 | pypdf output glued words ("rapidlychanged") and split others ("Sta tements") | Switched to PyMuPDF | Clean prose; BM25 matches real words |
| 2 | Tables came out one cell per line, so "17,455" sat alone with no label | Re-join numeric cells onto the row label; merge lone `$` cells | Rows like `Revenue \| $11,139 \| $17,455 \| 57 %` |
| 3 | Pages after the Exhibit Index (credit agreements, indentures) polluted results | Keep only the 10-K body (Uber p.1-150, Lyft p.1-125) | 545 → 275 pages, less noise |
| 4 | Item detection skipped long headings, so all of MD&A was tagged "Item 6 [Reserved]" | Accept any line that starts with `Item N.` | Correct Item tags for citations |
| 5 | Semantic chunking took ~4 min on CPU | Precompute chunks and embeddings once, commit them (float16) | App starts in seconds |
| 6 | First reranker took ~4 s per question | Embed at most 6 key sentences per candidate; reuse stored chunk vectors | ~2.4 s |
| 7 | Comparison questions lost one company after reranking (Uber filled all 5 slots) | Company-balanced top-5 | C03 and A01 fixed |
| 8 | Ambiguous "What was the net loss?" answered for one company at random | Questions naming no company are split into one sub-query per company | Both companies answered |
| 9 | Table rows scored poorly in the reranker ("Three Months Ended December 31 \| 18,728" has no query words) | Prefix each row with its table title when scoring | MRR 0.80 → 0.86 |
| 10 | Confidence threshold 0.42 refused only 1 of 5 unanswerable questions | Studied the score distribution; set 0.60 and added a year-scope rule | 5/5 refused, 0 false refusals |
| 11 | C01 labels only accepted absolute revenue, but the question asks about growth | Added "57%" / "36%" growth phrases as valid evidence | Fairer label; still a miss on the Lyft side (see failure analysis) |
| 12 | Streamlit rendered `$17,455 ... $11,139` as LaTeX math | Escape `$` before rendering | Figures display correctly |

## 4. Learnings

* Retrieval quality moved from 0.65 to 0.95 hit@5 without changing the LLM: corpus cleaning, chunking and reranking mattered more than the model.
* For financial filings, keyword search is not optional. Exact figures and terms of art (MAPCs, Adjusted EBITDA) are where dense embeddings are weakest.
* Every top-k cut must respect the question's shape; a reranker optimises relevance, not coverage.
* A refusal path built from several independent, simple checks (scope rule, score gate, `NOT_FOUND` contract, citation validation) is easier to trust and test than one clever prompt.
* Evidence-phrase labels let you compare chunkers fairly and keep the eval valid when chunk boundaries change.
