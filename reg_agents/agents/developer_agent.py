"""Developer Agent (A2A server on :8105) — FIRST line of defense.

Skill: develop a model. Runs a candidate-model bake-off via the modeling MCP
server, selects a champion against a documented primary metric, and writes a
**Model Development Document** (the model card an MRM function would receive):
purpose, data, candidate models considered, selection rationale, performance,
assumptions/limitations, intended use, and a proposed monitoring plan.

Input (A2A message text): a task_id, e.g. "fraud".
"""

from __future__ import annotations

import json
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
    name="Developer Agent",
    description="First-line model developer: trains candidates, selects a champion, documents it.",
    url="http://localhost:8105",
    skills=[
        AgentSkill(
            id="model-development",
            name="Model development",
            description="Run a model bake-off and produce a model development document.",
            tags=["first-line", "data-science", "model-development"],
        )
    ],
)

_SYS = (
    "You are a first-line model developer (Senior Data Scientist) documenting a "
    "newly built model for Model Risk Management under SR 11-7 / OCC 2011-12. "
    "Given a candidate-model bake-off leaderboard and the selected champion, write "
    "a MODEL DEVELOPMENT DOCUMENT with these sections: 1) Purpose & Intended Use, "
    "2) Data & Features, 3) Candidate Models Considered (compare the leaderboard "
    "and justify the champion against the documented primary metric; acknowledge "
    "any metric where a challenger was stronger), 4) Champion Performance, "
    "5) Assumptions & Limitations, 6) Proposed Ongoing Monitoring. Be concrete and "
    "reference the actual metrics. Use markdown headings."
)


def develop(task_id: str) -> Dict[str, Any]:
    """Run the bake-off (via MCP) and synthesize the development document."""
    settings = get_settings()
    try:
        bakeoff_raw = call_tool(settings.modeling_mcp_url, "run_model_bakeoff",
                                {"task_id": task_id})
        bakeoff = json.loads(bakeoff_raw)
    except Exception as exc:  # noqa: BLE001
        bakeoff_raw = json.dumps({"error": f"modeling MCP unavailable: {exc}"})
        bakeoff = {}

    doc = reason(
        _SYS,
        f"MODEL BAKE-OFF RESULT (JSON):\n{bakeoff_raw}",
        fallback=f"Bake-off result:\n{bakeoff_raw}",
        max_tokens=1600,
    )
    return {"bakeoff": bakeoff, "bakeoff_raw": bakeoff_raw, "document": doc}


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    task_id = (metadata.get("task_id") or message.as_text() or "fraud").strip()
    result = develop(task_id)
    champion = (result.get("bakeoff") or {}).get("champion", {})
    return Task(
        artifacts=[
            Artifact(name="model_development_document",
                     parts=[TextPart(text=result["document"])]),
            Artifact(name="bakeoff_leaderboard", parts=[TextPart(text=result["bakeoff_raw"])]),
        ],
        metadata={"task_id": task_id, "champion": champion.get("model", "")},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.developer_agent:app", 8105)
