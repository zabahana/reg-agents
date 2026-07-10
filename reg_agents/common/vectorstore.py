"""Minimal vector store.

Local dev: in-process FAISS (CPU). On the GPU demo this is swapped for Milvus
with the NVIDIA cuVS backend for GPU-accelerated ANN search -- same interface,
so the agents don't change. The GPU path is documented in k8s/.

To keep local setup dependency-light, if FAISS or embeddings are unavailable we
fall back to an exact numpy cosine search, so the demo always runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from reg_agents.common.embeddings import embed_query, embed_texts


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchHit:
    document: Document
    score: float


class VectorStore:
    """Tiny cosine-similarity store with a FAISS fast path."""

    def __init__(self) -> None:
        self._docs: List[Document] = []
        self._vectors: List[List[float]] = []
        self._faiss_index = None
        self._dim: Optional[int] = None

    def add(self, docs: List[Document]) -> None:
        vectors = embed_texts([d.text for d in docs])
        self._docs.extend(docs)
        self._vectors.extend(vectors)
        self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        try:
            import faiss  # type: ignore
            import numpy as np

            if not self._vectors:
                return
            arr = np.array(self._vectors, dtype="float32")
            faiss.normalize_L2(arr)
            self._dim = arr.shape[1]
            index = faiss.IndexFlatIP(self._dim)
            index.add(arr)
            self._faiss_index = index
        except Exception:
            # numpy-only fallback below
            self._faiss_index = None

    def search(self, query: str, k: int = 4) -> List[SearchHit]:
        if not self._docs:
            return []
        q = embed_query(query)
        if self._faiss_index is not None:
            import numpy as np

            qa = np.array([q], dtype="float32")
            import faiss  # type: ignore

            faiss.normalize_L2(qa)
            scores, idxs = self._faiss_index.search(qa, min(k, len(self._docs)))
            return [
                SearchHit(self._docs[i], float(scores[0][rank]))
                for rank, i in enumerate(idxs[0])
                if i >= 0
            ]
        return self._numpy_search(q, k)

    def _numpy_search(self, q: List[float], k: int) -> List[SearchHit]:
        def cos(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb + 1e-9)

        scored = [(cos(q, v), d) for v, d in zip(self._vectors, self._docs)]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [SearchHit(d, s) for s, d in scored[:k]]
