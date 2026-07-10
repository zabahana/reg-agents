"""Embeddings client.

Local dev uses OpenAI text embeddings. The demo uses NVIDIA NeMo Retriever
embedding NIMs (e.g. nvidia/nv-embedqa-e5-v5), which are also served through an
OpenAI-compatible /embeddings endpoint -- so, again, one client, config-driven.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from openai import OpenAI

from reg_agents.config import get_settings


def _build_client() -> Tuple[OpenAI, str]:
    s = get_settings()
    common = {"timeout": s.request_timeout, "max_retries": 1}
    if s.embedding_provider == "nemo":
        client = OpenAI(base_url=s.nemo_embed_base_url, api_key=s.nim_api_key or "not-needed", **common)
        return client, s.nemo_embed_model
    client = OpenAI(base_url=s.openai_base_url, api_key=s.openai_api_key or "not-needed", **common)
    return client, s.openai_embed_model


_cached: Optional[Tuple[OpenAI, str]] = None


def embed_texts(texts: List[str]) -> List[List[float]]:
    global _cached
    if _cached is None:
        _cached = _build_client()
    client, model = _cached
    # NeMo Retriever embedding NIMs accept an input_type hint (query vs passage).
    kwargs = {}
    s = get_settings()
    if s.embedding_provider == "nemo":
        kwargs["extra_body"] = {"input_type": "passage", "truncate": "END"}
    resp = client.embeddings.create(model=model, input=texts, **kwargs)
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> List[float]:
    s = get_settings()
    if s.embedding_provider == "nemo":
        global _cached
        if _cached is None:
            _cached = _build_client()
        client, model = _cached
        resp = client.embeddings.create(
            model=model, input=[text], extra_body={"input_type": "query", "truncate": "END"}
        )
        return resp.data[0].embedding
    return embed_texts([text])[0]
