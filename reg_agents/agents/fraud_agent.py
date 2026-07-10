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

# --- Model-behavior metrics (scraped from /metrics; power the Grafana panels
# and the block-rate / guardrail alerts). Best-effort so tests run without the
# client installed. ---
try:
    from prometheus_client import Counter, Histogram

    _DECISIONS = Counter(
        "fraud_decisions_total", "Fraud decisions by outcome and backend",
        ["decision", "backend_kind"],
    )
    _PROB = Histogram(
        "fraud_probability", "Fraud probability score distribution",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    _GUARDRAIL = Counter(
        "fraud_guardrail_triggered_total", "Model guardrail triggers by rule",
        ["rule"],
    )
except Exception:  # noqa: BLE001
    _DECISIONS = _PROB = _GUARDRAIL = None


def _record_metrics(score_json: str) -> None:
    if _DECISIONS is None:
        return
    try:
        data = json.loads(score_json)
    except Exception:  # noqa: BLE001
        return
    if not isinstance(data, dict) or "fraud_probability" not in data:
        return
    backend = str(data.get("backend", ""))
    backend_kind = "triton" if backend.startswith("triton") else "heuristic"
    decision = str(data.get("decision", "UNKNOWN"))
    _DECISIONS.labels(decision=decision, backend_kind=backend_kind).inc()
    prob = data.get("fraud_probability")
    if isinstance(prob, (int, float)):
        _PROB.observe(float(prob))
    for rule in data.get("guardrails", []) or []:
        _GUARDRAIL.labels(rule=str(rule)).inc()


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    txn = metadata.get("transaction")
    if not isinstance(txn, dict):
        txn = _DEFAULT_TXN
    settings = get_settings()

    try:
        score_json = call_tool(settings.fraud_mcp_url, "score_transaction", txn)
    except Exception as exc:  # noqa: BLE001
        score_json = json.dumps({"error": f"fraud MCP unavailable: {exc}"})

    _record_metrics(score_json)

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
