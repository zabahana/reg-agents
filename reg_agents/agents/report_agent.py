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
    "You are a senior model-governance lead (PhD econometrics, 20 years in "
    "model risk at large banks) assembling an audit-ready report an examiner "
    "could rely on. Compose these sections: Executive Summary, Model Overview, "
    "Validation Findings, a Performance Analysis section, Regulatory Mapping, "
    "Open Gaps & Remediation, and an overall Risk Rating (Low/Medium/High) "
    "with a one-sentence justification. Title the performance section to match "
    "the analysis in the source material: use 'Fraud Analysis' when it contains "
    "transaction-level fraud scoring, otherwise 'Performance & Monitoring'. "
    "Describe the model exactly as the source material characterizes it — "
    "never assume or invent a model type, purpose, or numbers; quote reported "
    "metrics verbatim. Where the material "
    "supports it, render findings and remediations as markdown tables "
    "(columns: #, Severity, Finding, Remediation, Owner) and the regulatory "
    "mapping as a table (Regulation, Relevance, Evidence). Use markdown "
    "headings."
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
