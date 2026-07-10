"""Retriever Agent (A2A server on :8101).

Skill: given a natural-language question about regulatory requirements, search
the regulatory knowledge base (via the regulations MCP server) and synthesize a
grounded, citation-bearing answer with the LLM (NIM/OpenAI).
"""

from __future__ import annotations

from typing import Any, Dict

from reg_agents.agents.base import reason, run
from reg_agents.common.a2a import (
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Task,
    TextPart,
    build_a2a_app,
)
from reg_agents.common.mcp_client import call_tool
from reg_agents.config import get_settings

CARD = AgentCard(
    name="Retriever Agent",
    description="Retrieves and synthesizes relevant banking regulations for a query.",
    url="http://localhost:8101",
    skills=[
        AgentSkill(
            id="regulatory-search",
            name="Regulatory search",
            description="Semantic search over SR 11-7, ECOA, FCRA, UDAAP and more.",
            tags=["rag", "regulations", "nemo-retriever"],
        )
    ],
)

_SYS = (
    "You are a banking regulatory research assistant. Using ONLY the provided "
    "regulation excerpts, answer the question. Cite each point with its source "
    "file and heading in [brackets]. If the excerpts do not cover something, say so."
)


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    query = message.as_text()
    settings = get_settings()
    try:
        excerpts = call_tool(settings.regulations_mcp_url, "search_regulations",
                             {"query": query, "k": 5})
    except Exception as exc:  # noqa: BLE001
        excerpts = f"[regulations MCP unavailable: {exc}]"

    answer = reason(
        _SYS,
        f"Question:\n{query}\n\nRegulation excerpts (JSON):\n{excerpts}",
        fallback=f"Retrieved excerpts:\n{excerpts}",
    )
    return Task(
        artifacts=[Artifact(name="regulatory_context", parts=[TextPart(text=answer)])],
        metadata={"retrieved": excerpts[:2000]},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.retriever_agent:app", 8101)
