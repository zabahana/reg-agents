"""Orchestrator Agent (A2A server on :8100).

Coordinates the specialist agents over A2A to run an end-to-end model
governance review:

    Validation Agent  ─┐
    Fraud Agent       ─┼─►  Report Agent  ─►  audit-ready report
    Retriever Agent   ─┘

Exposes its own A2A skill ("governance-review") so it too is composable, and a
`run_review()` function used by the CLI and Streamlit UI.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from reg_agents.agents.base import run
from reg_agents.common.a2a import (
    A2AClient,
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Task,
    TextPart,
    build_a2a_app,
)
from reg_agents.config import get_settings

CARD = AgentCard(
    name="Governance Orchestrator",
    description="Runs an end-to-end SR 11-7 model governance review across agents.",
    url="http://localhost:8100",
    skills=[
        AgentSkill(
            id="governance-review",
            name="Governance review",
            description="Validate a model, analyze fraud performance, produce a report.",
            tags=["orchestration", "a2a", "governance"],
        )
    ],
)


_REG_QUERY = (
    "What SR 11-7 model risk management requirements and fair-lending / "
    "consumer-protection rules (ECOA, FCRA, UDAAP) apply to validating, "
    "explaining, and monitoring an AI-based card transaction fraud model?"
)


def run_review(model_id: str, transaction: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Fan out to specialist agents over A2A, then compose a report."""
    s = get_settings()
    validation = A2AClient(s.validation_agent_url)
    fraud = A2AClient(s.fraud_agent_url)
    retriever = A2AClient(s.retriever_agent_url)
    reporter = A2AClient(s.report_agent_url)

    steps: Dict[str, str] = {}

    steps["validation_findings"] = _safe(
        lambda: validation.send_text(model_id, {"model_id": model_id}),
        "validation agent",
    )
    steps["fraud_analysis"] = _safe(
        lambda: fraud.send_text("score this transaction", {"transaction": transaction or {}}),
        "fraud agent",
    )
    steps["regulatory_context"] = _safe(
        lambda: retriever.send_text(_REG_QUERY),
        "retriever agent",
    )

    combined = (
        f"MODEL ID: {model_id}\n\n"
        f"## Validation Findings\n{steps['validation_findings']}\n\n"
        f"## Fraud / Performance Analysis\n{steps['fraud_analysis']}\n\n"
        f"## Regulatory Context\n{steps['regulatory_context']}\n"
    )
    steps["report"] = _safe(lambda: reporter.send_text(combined), "report agent")
    return steps


def run_validation_review(model_id: str) -> Dict[str, str]:
    """Validation / governance report for a model (no transaction involved).

    Second-line model validation: validation findings + regulatory context,
    composed into an audit-ready report.
    """
    s = get_settings()
    validation = A2AClient(s.validation_agent_url)
    retriever = A2AClient(s.retriever_agent_url)
    reporter = A2AClient(s.report_agent_url)

    steps: Dict[str, str] = {}
    steps["validation_findings"] = _safe(
        lambda: validation.send_text(model_id, {"model_id": model_id}),
        "validation agent",
    )
    steps["regulatory_context"] = _safe(
        lambda: retriever.send_text(_REG_QUERY),
        "retriever agent",
    )
    combined = (
        f"MODEL ID: {model_id}\n\n"
        f"## Validation Findings\n{steps['validation_findings']}\n\n"
        f"## Regulatory Context\n{steps['regulatory_context']}\n"
    )
    steps["report"] = _safe(lambda: reporter.send_text(combined), "report agent")
    return steps


def run_fraud_monitoring(transaction: Dict[str, Any]) -> Dict[str, str]:
    """Real-time fraud monitoring: score one transaction and explain it.

    Returns the raw model output JSON (`fraud_score`) and the analyst-facing
    `fraud_explanation`.
    """
    s = get_settings()
    fraud = A2AClient(s.fraud_agent_url)
    out: Dict[str, str] = {"fraud_score": "", "fraud_explanation": ""}
    try:
        task = fraud.send("score this transaction", {"transaction": transaction or {}})
        for a in task.artifacts:
            if a.name == "fraud_score":
                out["fraud_score"] = a.as_text()
            elif a.name == "fraud_explanation":
                out["fraud_explanation"] = a.as_text()
        if not out["fraud_score"] and task.artifacts:
            out["fraud_score"] = task.artifacts[0].as_text()
        if not out["fraud_explanation"] and len(task.artifacts) > 1:
            out["fraud_explanation"] = task.artifacts[-1].as_text()
    except Exception as exc:  # noqa: BLE001
        out["fraud_explanation"] = f"[fraud agent unavailable: {exc}]"
    return out


def _safe(fn, label: str) -> str:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return f"[{label} unavailable: {exc}]"


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    model_id = (metadata.get("model_id") or message.as_text() or "FRAUD-XGB-GNN-001").strip()
    transaction = metadata.get("transaction")
    result = run_review(model_id, transaction)
    return Task(
        artifacts=[
            Artifact(name="governance_report", parts=[TextPart(text=result["report"])]),
            Artifact(name="trace", parts=[TextPart(text=json.dumps(result, indent=2)[:6000])]),
        ],
        metadata={"model_id": model_id},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.orchestrator:app", 8100)
