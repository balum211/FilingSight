"""Embedding model.

all-MiniLM-L6-v2 (384-dim) run through ONNX Runtime via Chroma's bundled
loader: free, CPU-only, no PyTorch, and no API key, so indexing and the whole
retrieval evaluation are reproducible on any laptop or on Streamlit Cloud.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2 (ONNX, 384-dim)"


class MiniLMEmbeddings(Embeddings):
    """LangChain ``Embeddings`` adapter around Chroma's ONNX MiniLM."""

    def __init__(self, batch_size: int = 64):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2()
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._fn(texts[i : i + self.batch_size]))
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0].tolist()

    def embed_array(self, texts: list[str]) -> np.ndarray:
        """Unit-normalised vectors as a numpy array (used by the reranker and chunker)."""
        return self._embed(list(texts))


@lru_cache(maxsize=1)
def get_embeddings() -> MiniLMEmbeddings:
    return MiniLMEmbeddings()
