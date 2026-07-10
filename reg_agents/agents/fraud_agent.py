"""Fraud Agent (A2A server on :8103).

Skill: score a card transaction for fraud (via the fraud MCP server, which
proxies NVIDIA Triton in the demo) and produce an analyst-ready explanation
including the regulatory lens (e.g. adverse-action / UDAAP considerations).

Input (A2A metadata): a `transaction` dict, or free text describing one.
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
    name="Fraud Agent",
    description="Scores transactions (Triton) and explains decisions for analysts.",
    url="http://localhost:8103",
    skills=[
        AgentSkill(
            id="fraud-scoring",
            name="Fraud scoring + explanation",
            description="Real-time fraud scoring with a human-readable rationale.",
            tags=["fraud", "triton", "xgboost", "gnn"],
        )
    ],
)

_SYS = (
    "You are a fraud analyst copilot. Given a model fraud score and the "
    "transaction features, explain the decision in 3-4 sentences for an ops "
    "analyst, note the top risk drivers, and flag any consumer-protection "
    "considerations (e.g. documenting reasons if a legitimate customer is "
    "declined). Do not invent features."
)

_DEFAULT_TXN = {"amount": 4200.0, "is_foreign": True, "merchant_risk": 0.6,
                "hour": 2, "velocity_24h": 9}


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    txn = metadata.get("transaction")
    if not isinstance(txn, dict):
        txn = _DEFAULT_TXN
    settings = get_settings()

    try:
        score_json = call_tool(settings.fraud_mcp_url, "score_transaction", txn)
    except Exception as exc:  # noqa: BLE001
        score_json = json.dumps({"error": f"fraud MCP unavailable: {exc}"})

    explanation = reason(
        _SYS,
        f"Fraud model output (JSON):\n{score_json}",
        fallback=f"Fraud model output:\n{score_json}",
    )
    return Task(
        artifacts=[
            Artifact(name="fraud_score", parts=[TextPart(text=score_json)]),
            Artifact(name="fraud_explanation", parts=[TextPart(text=explanation)]),
        ],
        metadata={"transaction": txn},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.fraud_agent:app", 8103)
