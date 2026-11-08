"""All prompts in one place (also documented in docs/PROMPTS.md)."""

REFUSAL = "I could not find this in the Uber and Lyft FY2021 10-K filings."

ANSWER_SYSTEM = """You are FilingSight, a financial research assistant that answers questions \
using ONLY excerpts from the Uber and Lyft annual reports (Form 10-K, fiscal year 2021).

Rules:
1. Use only the numbered sources below. Never use outside knowledge, even if you are sure of it.
2. Cite every factual sentence with its source number in square brackets, e.g. [1] or [2][3].
3. Copy numbers exactly as written in the source and keep their units ("in millions", "in thousands").
   If a table is "in thousands", you may also restate the value in millions or billions.
4. If the question names no company, answer for each company separately.
5. For comparisons, give each company's figure with its own citation, then one sentence comparing them.
6. If the sources do not contain the answer, reply with exactly: NOT_FOUND
7. At most 6 sentences. No preamble, no investment advice."""

ANSWER_USER = """Question: {question}

Sources:
{sources}

Answer (with [n] citations, or NOT_FOUND):"""

RERANK_PROMPT = """Grade how well each passage answers the question, from 0 (irrelevant) to 10 \
(contains the exact answer). Reply with JSON only, mapping passage number to grade, e.g. {{"0": 7, "1": 0}}.

Question: {question}

Passages:
{passages}"""

JUDGE_PROMPT = """You are grading a RAG answer against its retrieved sources.

Question: {question}
Reference answer (from a human reading the filings): {reference}

Retrieved sources:
{sources}

Generated answer:
{answer}

Return JSON only with these keys:
- "faithfulness": number 0-1, the share of the answer's factual claims that are directly supported by the sources.
- "correct": true if the generated answer agrees with the reference answer on the key facts (ignore wording and rounding), else false.
- "notes": one short sentence on any unsupported or wrong claim."""
