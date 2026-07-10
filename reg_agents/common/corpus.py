"""Load and index the regulatory corpus.

Documents live in data/regulations/*.md. We chunk on markdown headings so each
chunk is a coherent regulatory clause. Retrieval tries semantic search
(embeddings + vector store) and transparently falls back to lexical overlap so
the system runs with zero API keys during local development.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import List, Optional

from reg_agents.common.vectorstore import Document, SearchHit, VectorStore

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "regulations"
)


def _chunk_markdown(text: str, source: str) -> List[Document]:
    chunks: List[Document] = []
    current_heading = source
    buffer: List[str] = []
    idx = 0

    def flush() -> None:
        nonlocal buffer, idx
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(
                Document(
                    id=f"{source}#{idx}",
                    text=f"{current_heading}\n{body}",
                    metadata={"source": source, "heading": current_heading},
                )
            )
            idx += 1
        buffer = []

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            flush()
            current_heading = line.lstrip("#").strip()
        else:
            buffer.append(line)
    flush()
    return chunks


@lru_cache
def load_regulations(data_dir: Optional[str] = None) -> List[Document]:
    directory = data_dir or _DATA_DIR
    docs: List[Document] = []
    if not os.path.isdir(directory):
        return docs
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as fh:
            docs.extend(_chunk_markdown(fh.read(), name))
    return docs


def lexical_search(docs: List[Document], query: str, k: int = 4) -> List[SearchHit]:
    q_terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
    scored = []
    for d in docs:
        d_terms = re.findall(r"[a-z0-9]+", d.text.lower())
        if not d_terms:
            continue
        overlap = sum(1 for t in d_terms if t in q_terms)
        score = overlap / (len(q_terms) + 1)
        if score > 0:
            scored.append(SearchHit(d, score))
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:k]


class RegulationRetriever:
    """Semantic search with a lexical fallback.

    The vector store is built lazily on first search, and only when an
    embedding API key is actually configured. This keeps server startup instant
    and avoids hanging on network calls during local/offline runs.
    """

    def __init__(self) -> None:
        self.docs = load_regulations()
        self._store: Optional[VectorStore] = None
        self._store_ready = False

    @staticmethod
    def _embeddings_configured() -> bool:
        from reg_agents.config import get_settings

        s = get_settings()
        if s.embedding_provider == "nemo":
            return bool(s.nim_api_key)
        return bool(s.openai_api_key)

    def _ensure_store(self) -> None:
        if self._store_ready:
            return
        self._store_ready = True  # only try once
        if not self._embeddings_configured():
            return
        try:
            store = VectorStore()
            store.add(self.docs)
            self._store = store
        except Exception:
            self._store = None  # embeddings failed -> lexical fallback

    def search(self, query: str, k: int = 4) -> List[SearchHit]:
        self._ensure_store()
        if self._store is not None:
            try:
                return self._store.search(query, k)
            except Exception:
                pass
        return lexical_search(self.docs, query, k)
