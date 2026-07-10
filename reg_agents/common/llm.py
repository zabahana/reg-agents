"""Provider-agnostic LLM client.

NVIDIA NIM exposes an OpenAI-compatible Chat Completions API, so a single
`openai.OpenAI` client talks to both OpenAI (local dev) and NIM (the demo /
GCP GPU) with only a base_url + api_key + model change.

    Migrating from OpenAI to NVIDIA NIM in this codebase = flip LLM_PROVIDER.

That is a deliberate talking point for the Solutions Architect demo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from reg_agents.config import get_settings

_client: Optional[OpenAI] = None


def _build_client() -> Tuple[OpenAI, str]:
    s = get_settings()
    if s.llm_provider == "nim":
        client = OpenAI(base_url=s.nim_base_url, api_key=s.nim_api_key or "not-needed")
        return client, s.nim_model
    client = OpenAI(base_url=s.openai_base_url, api_key=s.openai_api_key or "not-needed")
    return client, s.openai_model


def get_client() -> Tuple[OpenAI, str]:
    global _client
    client, model = _build_client()
    _client = client
    return client, model


def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> str:
    """Single-shot chat completion returning the assistant text."""
    client, model = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def system_user(system: str, user: str, **kwargs: Any) -> str:
    return chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
