"""Build, save and load the search indexes.

``python -m scripts.build_index`` (or ``build_and_save``) runs ingestion,
chunking and embedding once and writes ``data/processed/<strategy>_chunks.jsonl``
plus ``<strategy>_emb.npy``. The app and the evaluation load those files into a
Chroma collection (dense) and a BM25 index (sparse) in a few seconds, so nobody
has to wait for the semantic chunker or re-embed 2,000 chunks on start-up.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from . import config
from .chunking import STRATEGIES, indexed_text, make_chunks

PROCESSED_DIR = config.DATA_DIR / "processed"
MIN_CHUNK_CHARS = 40

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or our the their this to was "
    "were what when where which who why with we you they than that these those per during about".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. Keeps figures like 17,455 and 49.2% intact."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _paths(strategy: str, root: Path = PROCESSED_DIR) -> tuple[Path, Path]:
    return root / f"{strategy}_chunks.jsonl", root / f"{strategy}_emb.npy"


def build_and_save(strategy: str, root: Path = PROCESSED_DIR) -> int:
    from .embeddings import get_embeddings
    from .ingest import load_corpus

    chunks = [c for c in make_chunks(strategy, load_corpus()) if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    vectors = get_embeddings().embed_array([indexed_text(c) for c in chunks])
    root.mkdir(parents=True, exist_ok=True)
    chunks_path, emb_path = _paths(strategy, root)
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")
    np.save(emb_path, vectors.astype(np.float16))  # float16 halves the repo size; recall is unaffected
    return len(chunks)


@dataclass
class SearchIndex:
    """Everything retrieval needs for one chunking strategy."""

    strategy: str
    docs: list[Document]
    vectors: np.ndarray  # (n, 384), unit-normalised
    bm25: BM25Okapi
    vector_store: object  # langchain_chroma.Chroma

    def ids(self) -> list[str]:
        return [d.metadata["chunk_id"] for d in self.docs]

    def vectors_for(self, docs: list[Document]) -> np.ndarray:
        if not hasattr(self, "_pos"):
            self._pos = {cid: i for i, cid in enumerate(self.ids())}
        return self.vectors[[self._pos[d.metadata["chunk_id"]] for d in docs]]


def _load_chunks(strategy: str, root: Path) -> tuple[list[Document], np.ndarray]:
    chunks_path, emb_path = _paths(strategy, root)
    if not chunks_path.exists():
        build_and_save(strategy, root)
    docs = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(page_content=row["text"], metadata=row["metadata"]))
    vectors = np.load(emb_path).astype(np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    return docs, vectors


def _chroma_store(strategy: str, docs: list[Document], vectors: np.ndarray):
    """Load precomputed vectors into an in-memory Chroma collection."""
    import chromadb
    from langchain_chroma import Chroma

    from .embeddings import get_embeddings

    client = chromadb.EphemeralClient()
    name = f"filingsight_{strategy}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    store = Chroma(
        client=client,
        collection_name=name,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    batch = 2000
    for i in range(0, len(docs), batch):
        part = docs[i : i + batch]
        store._collection.add(
            ids=[d.metadata["chunk_id"] for d in part],
            embeddings=vectors[i : i + batch].tolist(),
            documents=[d.page_content for d in part],
            metadatas=[{k: v for k, v in d.metadata.items() if isinstance(v, (str, int, float, bool))} for d in part],
        )
    return store


@lru_cache(maxsize=len(STRATEGIES))
def load_index(strategy: str, root: Path = PROCESSED_DIR) -> SearchIndex:
    docs, vectors = _load_chunks(strategy, root)
    bm25 = BM25Okapi([tokenize(indexed_text(d)) for d in docs])
    store = _chroma_store(strategy, docs, vectors)
    return SearchIndex(strategy=strategy, docs=docs, vectors=vectors, bm25=bm25, vector_store=store)
