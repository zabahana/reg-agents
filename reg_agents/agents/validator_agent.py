"""Validator Agent (A2A server on :8106) — SECOND line of defense.

Skill: independent model validation ("effective challenge") of a freshly
developed model. Consumes the Developer agent's model development document and
bake-off leaderboard, pulls relevant rules from the regulations MCP server, and
writes an independent **Validation Report** with a formal disposition.

Distinct from `validation_agent.py` (which reviews an already-registered model
from the inventory); this one validates a model coming out of development.

Input (A2A message text): the model development document + leaderboard.
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
    name="Validator Agent",
    description="Second-line independent validation of a developed model (effective challenge).",
    url="http://localhost:8106",
    skills=[
        AgentSkill(
            id="independent-validation",
            name="Independent validation",
            description="Effective challenge + SR 11-7 validation report with disposition.",
            tags=["second-line", "model-risk", "sr11-7", "validation"],
        )
    ],
)

_SYS = (
    "You are an INDEPENDENT model validator (VP, Model Risk Management) — the "
    "second line of defense — performing effective challenge on a model produced "
    "by the development team. You did not build the model. Given the development "
    "document, the candidate leaderboard, and relevant regulations, write a "
    "VALIDATION REPORT with: 1) Scope & Materials Reviewed, 2) Conceptual "
    "Soundness, 3) Effective Challenge of Model Selection (was the champion chosen "
    "on an appropriate primary metric given class imbalance? critique it and note "
    "if a challenger was better on PR-AUC/recall), 4) Data & Outcomes Analysis, "
    "5) Fair-Lending / Consumer-Protection Review, 6) Findings (severity-ranked) "
    "with required remediations, and 7) a DISPOSITION: Approve / Approve with "
    "Conditions / Reject. Cite regulation sources in [brackets]. Use markdown."
)


def validate(task_id: str, development_document: str, leaderboard: str) -> str:
    settings = get_settings()
    try:
        rules = call_tool(
            settings.regulations_mcp_url, "search_regulations",
            {"query": (
                "independent model validation, effective challenge, benchmarking, "
                "outcomes analysis and fair-lending review requirements under "
                f"SR 11-7 / OCC 2011-12 for a {task_id} model"), "k": 6},
        )
    except Exception as exc:  # noqa: BLE001
        rules = f"[regulations MCP unavailable: {exc}]"

    return reason(
        _SYS,
        f"MODEL DEVELOPMENT DOCUMENT:\n{development_document}\n\n"
        f"CANDIDATE LEADERBOARD (JSON):\n{leaderboard}\n\n"
        f"RELEVANT REGULATIONS:\n{rules}",
        fallback=f"Development document:\n{development_document}\n\nRules:\n{rules}",
        max_tokens=1700,
    )


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    task_id = (metadata.get("task_id") or "fraud").strip()
    dev_doc = metadata.get("development_document") or message.as_text()
    leaderboard = metadata.get("leaderboard") or ""
    report = validate(task_id, dev_doc, leaderboard)
    return Task(
        artifacts=[Artifact(name="validation_report", parts=[TextPart(text=report)])],
        metadata={"task_id": task_id},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.validator_agent:app", 8106)
