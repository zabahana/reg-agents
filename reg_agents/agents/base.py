"""Shared helpers for A2A agents."""

from __future__ import annotations

import os

from reg_agents.common import llm


def reason(system: str, user: str, fallback: str = "", **kwargs) -> str:
    """LLM reasoning with a graceful fallback.

    If no model/key is configured (local smoke test), we return `fallback` so
    the multi-agent pipeline still produces end-to-end output.
    """
    try:
        return llm.system_user(system, user, **kwargs)
    except Exception as exc:  # noqa: BLE001
        note = f"[LLM unavailable: {exc}]"
        return f"{note}\n\n{fallback}" if fallback else note


def run(app_path: str, default_port: int) -> None:
    """uvicorn entrypoint used by each agent's __main__."""
    import uvicorn

    port = int(os.getenv("PORT", str(default_port)))
    uvicorn.run(app_path, host="0.0.0.0", port=port, log_level="info")
