"""Audit Agent (A2A server on :8107) — THIRD line of defense.

Skill: internal audit of the model governance *process*. Consumes the
development document and the independent validation report, checks them against
SR 11-7 governance and documentation expectations, and writes an **Audit
Report** with an audit opinion and issue log. Audit does not re-validate the
math; it assesses whether the control framework operated effectively.

Input (A2A message text): development document + validation report.
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
    name="Audit Agent",
    description="Third-line internal audit of the model governance process.",
    url="http://localhost:8107",
    skills=[
        AgentSkill(
            id="internal-audit",
            name="Internal audit",
            description="Audit MRM process controls and issue an audit opinion.",
            tags=["third-line", "internal-audit", "governance", "sr11-7"],
        )
    ],
)

_SYS = (
    "You are INTERNAL AUDIT (third line of defense) reviewing the model risk "
    "management process — NOT re-validating the model math. Given the model "
    "development document, the independent validation report, and governance "
    "regulations, assess whether controls operated effectively and write an AUDIT "
    "REPORT with: 1) Objective & Scope, 2) Process Walkthrough (development → "
    "independent validation → approval), 3) Control Assessment (independence of "
    "validation from development, documentation completeness, evidence of "
    "effective challenge, issue tracking, approval before deployment), 4) Audit "
    "Findings & Issues (rated High/Medium/Low with owners), 5) an AUDIT OPINION "
    "(Satisfactory / Needs Improvement / Unsatisfactory) with justification. "
    "Cite governance requirements in [brackets]. Use markdown."
)


def audit(task_id: str, development_document: str, validation_report: str) -> str:
    settings = get_settings()
    try:
        rules = call_tool(
            settings.regulations_mcp_url, "search_regulations",
            {"query": (
                "model risk management governance, roles and responsibilities, "
                "board and senior management oversight, documentation, internal "
                "audit role and independence under SR 11-7 / OCC 2011-12"), "k": 6},
        )
    except Exception as exc:  # noqa: BLE001
        rules = f"[regulations MCP unavailable: {exc}]"

    return reason(
        _SYS,
        f"MODEL DEVELOPMENT DOCUMENT:\n{development_document}\n\n"
        f"INDEPENDENT VALIDATION REPORT:\n{validation_report}\n\n"
        f"GOVERNANCE REGULATIONS:\n{rules}",
        fallback=f"Validation report:\n{validation_report}\n\nRules:\n{rules}",
        max_tokens=1600,
    )


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    task_id = (metadata.get("task_id") or "fraud").strip()
    dev_doc = metadata.get("development_document") or ""
    val_report = metadata.get("validation_report") or message.as_text()
    report = audit(task_id, dev_doc, val_report)
    return Task(
        artifacts=[Artifact(name="audit_report", parts=[TextPart(text=report)])],
        metadata={"task_id": task_id},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.audit_agent:app", 8107)
