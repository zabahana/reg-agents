"""Report Agent (A2A server on :8104).

Skill: assemble an audit-ready governance report from upstream artifacts
(validation findings, fraud analysis, regulatory context). Produces the kind of
SR 11-7 documentation an MRM function would file.
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

CARD = AgentCard(
    name="Report Agent",
    description="Generates audit-ready model governance reports.",
    url="http://localhost:8104",
    skills=[
        AgentSkill(
            id="governance-report",
            name="Governance report",
            description="Compose SR 11-7-style documentation from findings.",
            tags=["reporting", "governance", "audit"],
        )
    ],
)

_SYS = (
    "You are a model governance lead. Compose a concise, audit-ready report with "
    "these sections: Executive Summary, Model Overview, Validation Findings, "
    "Fraud/Performance Analysis, Regulatory Mapping, Open Gaps & Remediation, "
    "and an overall Risk Rating (Low/Medium/High). Use markdown headings."
)


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    material = message.as_text()
    report = reason(
        _SYS,
        f"Source material to synthesize into the report:\n\n{material}",
        fallback=f"# Governance Report (template)\n\n{material}",
        max_tokens=1800,
    )
    return Task(artifacts=[Artifact(name="governance_report", parts=[TextPart(text=report)])])


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.report_agent:app", 8104)
