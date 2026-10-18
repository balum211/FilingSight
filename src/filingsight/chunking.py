"""Two chunking strategies, compared in the evaluation.

* ``fixed``    RecursiveCharacterTextSplitter, 900 chars with 150 overlap.
               Size-bounded and predictable; ignores meaning.
* ``semantic`` Split into sentences / table rows, embed each one, and cut where
               the cosine distance between neighbours jumps above the 85th
               percentile (the same idea as LangChain's SemanticChunker). Chunks
               are then capped at 1,100 chars so they still fit the 256-token
               MiniLM window, and tiny fragments are merged forward.

Both strategies chunk page by page so every chunk keeps an exact page citation,
and both prepend a short header ("Uber 10-K FY2021 | Item 7 MD&A | p.53") to the
text that gets embedded and keyword-indexed. The header lets "Lyft revenue"
match Lyft chunks even when the paragraph itself only says "we".
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

STRATEGIES = ("fixed", "semantic")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"$•])")


def chunk_header(meta: dict) -> str:
    return f"{meta['company']} {meta['form']} FY{meta['fiscal_year']} | Item {meta['item']} {meta['item_title']} | p.{meta['page']}"


def _finalise(chunks: list[Document], strategy: str) -> list[Document]:
    for i, c in enumerate(chunks):
        m = c.metadata
        m["strategy"] = strategy
        m["chunk_id"] = f"{strategy}-{m['ticker']}-p{m['page']}-{i}"
        m["header"] = chunk_header(m)
    return chunks


def fixed_chunks(pages: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.FIXED_CHUNK_SIZE,
        chunk_overlap=config.FIXED_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return _finalise(splitter.split_documents(list(pages)), "fixed")


def split_units(text: str, with_captions: bool = False) -> list[str]:
    """Sentences for prose, whole lines for table rows and headings.

    With ``with_captions`` a table row is prefixed by the closest non-row line
    above it (usually the table title), so "Three Months Ended December 31 |
    18,728" becomes "Active Riders | 2021 | 2020 › Three Months Ended December 31 | 18,728".
    """
    units: list[str] = []
    caption = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if " | " in line and with_captions and caption:
            units.append(f"{caption} › {line}")
        elif " | " in line or len(line) < 120:
            units.append(line)
            if len(line) < 120 and (" | " not in line or not caption):
                caption = line
        else:
            units.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
    return units


def semantic_breakpoints(vectors: np.ndarray, percentile: float) -> list[int]:
    """Indices i where a new chunk should start before unit i."""
    if len(vectors) < 3:
        return []
    sims = np.sum(vectors[:-1] * vectors[1:], axis=1)
    distances = 1.0 - sims
    threshold = np.percentile(distances, percentile)
    return [i + 1 for i, d in enumerate(distances) if d > threshold]


def group_units(units: list[str], breaks: list[int], max_chars: int, min_chars: int) -> list[str]:
    groups: list[list[str]] = [[]]
    size = 0
    bset = set(breaks)
    for i, u in enumerate(units):
        if groups[-1] and (i in bset or size + len(u) > max_chars) and size >= min_chars:
            groups.append([])
            size = 0
        elif groups[-1] and size + len(u) > max_chars * 1.5:
            groups.append([])
            size = 0
        groups[-1].append(u)
        size += len(u) + 1
    # merge a tiny tail into the previous group
    if len(groups) > 1 and sum(len(u) for u in groups[-1]) < min_chars:
        tail = groups.pop()
        groups[-1].extend(tail)
    return ["\n".join(g) for g in groups if g]


def semantic_chunks(pages: Iterable[Document], embeddings=None) -> list[Document]:
    from .embeddings import get_embeddings

    emb = embeddings or get_embeddings()
    pages = list(pages)
    page_units = [split_units(p.page_content) for p in pages]
    flat = [u for units in page_units for u in units]
    vectors = emb.embed_array(flat) if flat else np.zeros((0, 384))
    out: list[Document] = []
    pos = 0
    for page, units in zip(pages, page_units):
        vecs = vectors[pos : pos + len(units)]
        pos += len(units)
        breaks = semantic_breakpoints(vecs, config.SEMANTIC_BREAKPOINT_PERCENTILE)
        for text in group_units(units, breaks, config.SEMANTIC_MAX_CHARS, config.SEMANTIC_MIN_CHARS):
            out.append(Document(page_content=text, metadata=dict(page.metadata)))
    return _finalise(out, "semantic")


def make_chunks(strategy: str, pages: Iterable[Document]) -> list[Document]:
    if strategy == "fixed":
        return fixed_chunks(pages)
    if strategy == "semantic":
        return semantic_chunks(pages)
    raise ValueError(f"unknown chunking strategy {strategy!r}; pick one of {STRATEGIES}")


def indexed_text(doc: Document) -> str:
    """What gets embedded and BM25-indexed: header + chunk body."""
    return f"{doc.metadata['header']}\n{doc.page_content}"
