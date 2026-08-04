"""Provider-agnostic LLM client.

NVIDIA NIM exposes an OpenAI-compatible Chat Completions API, so a single
`openai.OpenAI` client talks to both OpenAI (local dev) and NIM (the demo /
GCP GPU) with only a base_url + api_key + model change.

    Migrating from OpenAI to NVIDIA NIM in this codebase = flip LLM_PROVIDER.

Two usage modes:

- **Pipeline mode** (default): `chat()` / `system_user()` use the provider
  selected by LLM_PROVIDER — this is what the agents and MCP servers run on.
- **Judge-panel mode**: pass `provider="openai"` or `provider="nim"`
  explicitly to address a specific provider regardless of LLM_PROVIDER.
  Both providers are first-class judges (OpenAI is NOT a fallback for NIM);
  the dual-judge agreement study (`scripts/judge_agreement_study.py`) runs
  them side by side against the stage-1 logistic-regression gate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from reg_agents.config import get_settings

PROVIDERS = ("openai", "nim")

_clients: Dict[str, Tuple[OpenAI, str]] = {}


def _build_client(provider: str) -> Tuple[OpenAI, str]:
    s = get_settings()
    # Explicit short timeout + limited retries so a stalled call fails fast
    # instead of hanging on the SDK's very long default timeout.
    common = {"timeout": s.request_timeout, "max_retries": 1}
    if provider == "nim":
        client = OpenAI(base_url=s.nim_base_url, api_key=s.nim_api_key or "not-needed", **common)
        return client, s.nim_model
    client = OpenAI(base_url=s.openai_base_url, api_key=s.openai_api_key or "not-needed", **common)
    return client, s.openai_model


def get_client(provider: Optional[str] = None) -> Tuple[OpenAI, str]:
    """Client + model for `provider`, or the LLM_PROVIDER default."""
    provider = provider or get_settings().llm_provider
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r}; expected one of {PROVIDERS}")
    if provider not in _clients:
        _clients[provider] = _build_client(provider)
    return _clients[provider]


def available_judges() -> List[str]:
    """Providers with credentials configured — the judge panel."""
    s = get_settings()
    out = []
    if s.nim_api_key:
        out.append("nim")
    if s.openai_api_key:
        out.append("openai")
    return out


def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Single-shot chat completion returning the assistant text."""
    client, model = get_client(provider)
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
