"""All prompts in one place (also documented in docs/PROMPTS.md)."""

REFUSAL = "I could not find this in the Uber and Lyft FY2021 10-K filings."

# Written for a small local model (Qwen2.5-0.5B-Instruct). Small models over-use an
# "I don't know" escape hatch, so refusals are decided before generation (year rule and
# confidence gate) and the prompt asks only for a short, cited answer.
ANSWER_SYSTEM = """You answer questions about the Uber and Lyft annual reports for 2021 (Form 10-K).
Use only the numbered sources. Write 2 to 4 short sentences.
Copy every number exactly as written, with its unit.
End every sentence with the number of the source it came from, for example [1] or [2].
If the question is about both companies, give one sentence for each company."""

ANSWER_USER = """Sources:
{sources}

Question: {question}

Answer in 2 to 4 sentences, each ending with a source number like [1]:"""

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
