"""Validation Agent (A2A server on :8102).

Skill: assess a model in the registry against SR 11-7 model-risk expectations.
Pulls model metadata + documentation from the model-registry MCP server, pulls
relevant rules from the regulations MCP server, then produces structured
findings (conceptual soundness, data, testing, monitoring, gaps).

Input (A2A message text): a model_id, e.g. "FRAUD-XGB-GNN-001".
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
    name="Validation Agent",
    description="Validates a model against SR 11-7 using registry docs + regulations.",
    url="http://localhost:8102",
    skills=[
        AgentSkill(
            id="sr11-7-validation",
            name="SR 11-7 validation",
            description="Independent model validation findings for MRM.",
            tags=["model-risk", "sr11-7", "governance"],
        )
    ],
)

_SYS = (
    "You are a senior independent model validator — PhD in econometrics, 20 "
    "years validating tier-1 bank models (credit scorecards, CCAR/PPNR, fraud, "
    "AML, and now GenAI/agentic systems). You write with the measured, "
    "evidence-first voice of someone who has defended findings to the Fed and "
    "the OCC: precise statistical language, no hedging, no marketing. Write "
    "SR 11-7-aligned validation findings covering: (1) conceptual soundness — "
    "is the methodology fit for purpose, and what are the maintained "
    "assumptions; (2) data quality and representativeness, including label "
    "provenance; (3) developmental evidence and outcomes analysis — interrogate "
    "the metrics (discrimination vs calibration, class imbalance, stability), "
    "quoting reported figures verbatim; (4) ongoing monitoring adequacy "
    "(drift/PSI, guardrails, alerting); (5) fair-lending/ECOA and consumer- "
    "protection exposure. Number every finding with severity (High/Medium/Low), "
    "give a concrete remediation and owner for each, and cite regulation "
    "sources in [brackets]. Close with a disposition (Approve / Approve with "
    "Conditions / Reject) and the specific conditions."
)


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    model_id = (metadata.get("model_id") or message.as_text()).strip()
    settings = get_settings()

    try:
        meta = call_tool(settings.model_registry_mcp_url, "get_model_metadata",
                         {"model_id": model_id})
        docs = call_tool(settings.model_registry_mcp_url, "get_model_documentation",
                         {"model_id": model_id})
    except Exception as exc:  # noqa: BLE001
        meta, docs = f"[model-registry MCP unavailable: {exc}]", ""

    try:
        rules = call_tool(
            settings.regulations_mcp_url, "search_regulations",
            {"query": f"model risk management validation requirements for {model_id}", "k": 5},
        )
    except Exception as exc:  # noqa: BLE001
        rules = f"[regulations MCP unavailable: {exc}]"

    findings = reason(
        _SYS,
        f"MODEL METADATA:\n{meta}\n\nMODEL DOCUMENTATION:\n{docs}\n\n"
        f"RELEVANT REGULATIONS:\n{rules}",
        fallback=f"Model metadata:\n{meta}\n\nRelevant rules:\n{rules}",
        max_tokens=1400,
    )
    return Task(
        artifacts=[Artifact(name="validation_findings", parts=[TextPart(text=findings)])],
        metadata={"model_id": model_id},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.validation_agent:app", 8102)
