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
from reg_agents.common.mrm_document_instructions import (
    GOVERNANCE_MDD_VAL_BRIDGE,
    MDD_SECTION_INSTRUCTIONS,
    VALIDATION_SECTION_INSTRUCTIONS,
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
    "could rely on.\n\n"
    f"{GOVERNANCE_MDD_VAL_BRIDGE}\n\n"
    "Compose these sections, in order, with those exact titles:\n"
    "1. Executive Summary\n"
    "2. Model Overview (summarize MDD coverage: purpose, data, split protocol, "
    "bake-off, champion, threshold — mirror the MDD section checklist)\n"
    "3. Validation Findings (mirror the independent validation section "
    "checklist; include disposition)\n"
    "4. Performance Analysis\n"
    "5. Regulatory Mapping\n"
    "6. Open Gaps & Remediation\n"
    "7. Risk Rating (Low/Medium/High) with a one-sentence justification\n\n"
    "MDD section checklist (use when summarizing development evidence):\n"
    f"{MDD_SECTION_INSTRUCTIONS}\n\n"
    "Validation section checklist (use when summarizing 2nd-line evidence):\n"
    f"{VALIDATION_SECTION_INSTRUCTIONS}\n\n"
    "Describe the model exactly as the source material characterizes it — "
    "never invent a model type, purpose, or numbers; quote reported metrics "
    "verbatim. Prefer markdown tables for findings "
    "(#: Severity | Finding | Remediation | Owner) and regulatory mapping "
    "(Regulation | Relevance | Evidence)."
)


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    material = message.as_text()
    report = reason(
        _SYS,
        f"Source material to synthesize into the report:\n\n{material}",
        fallback=f"# Governance Report (template)\n\n{material}",
        max_tokens=2400,
    )
    return Task(artifacts=[Artifact(name="governance_report", parts=[TextPart(text=report)])])


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.report_agent:app", 8104)
