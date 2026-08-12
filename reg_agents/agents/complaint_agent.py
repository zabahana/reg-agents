"""Complaint Agent (A2A server on :8110).

Skill: classify a consumer-complaint narrative into the 24-category regulation
taxonomy via the complaint MCP server (two-stage: binary gate, then RAG + LLM
with few-shot examples and citations), assess whether the complaint signals a
systemic control failure (risk intelligence), and produce a compliance-analyst
summary covering both the label and the control-risk read.

Input (A2A metadata): `narrative` string, or the message text itself.
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
    name="Complaint Agent",
    description="Classifies consumer complaints into 24 regulation categories "
                "with RAG citations, then assesses systemic control-failure "
                "risk (CFPB-trained two-stage model + risk intelligence).",
    url="http://localhost:8110",
    skills=[
        AgentSkill(
            id="complaint-classification",
            name="Complaint → regulation classification",
            description="Two-stage classification (binary gate + RAG/LLM "
                        "labeling) with citations from the policy corpus.",
            tags=["complaints", "udaap", "rag", "nlp", "cfpb"],
        ),
        AgentSkill(
            id="risk-intelligence",
            name="Systemic control-failure risk intelligence",
            description="Elevates classification into a control-risk read: "
                        "systemic signal, control domain, prior-case "
                        "similarity, local explanation, recommended action.",
            tags=["risk", "controls", "systemic", "orm", "complaints"],
        ),
    ],
)

_SYS = (
    "You are a bank compliance operations copilot. Given the JSON output of a "
    "complaint-classification model (stage-1 regulatory gate + stage-2 "
    "regulation label with citation) AND its risk_intelligence and hitl "
    "blocks, write 4-5 sentences for a complaints / operational-risk analyst:\n"
    "1) State the assigned regulation category and why.\n"
    "2) State the systemic_signal (none / isolated / moderate / elevated) and "
    "the implicated control_domain — classification says what the complaint "
    "is; risk intelligence asks whether it signals a systemic control "
    "failure.\n"
    "3) If hitl.required is true, say the case is queued for human review "
    "(approve / override / escalate); otherwise note auto-route with optional "
    "override.\n"
    "4) Recommend routing using recommended_action.\n"
    "If a citation is present, quote its key phrase. If the complaint was "
    "gated non-regulatory at stage 1 (mode stage1_gate), say the gate found "
    "no regulatory nexus, note systemic_signal=none, and recommend service "
    "recovery — never mention null fields or JSON internals. Do not invent "
    "facts."
)

# Per-classification metrics for Grafana (label + mode), best-effort.
try:
    from prometheus_client import Counter

    _CLASSIFICATIONS = Counter(
        "complaint_classifications_total",
        "Complaint classifications by regulation label and stage-2 mode",
        ["label", "mode"],
    )
    _RISK_SIGNALS = Counter(
        "complaint_risk_signals_total",
        "Complaint risk-intelligence systemic signals",
        ["systemic_signal", "control_domain"],
    )
    _HITL_PENDING = Counter(
        "complaint_hitl_pending_total",
        "Complaints auto-queued for human-in-the-loop review",
        ["required"],
    )
except Exception:  # noqa: BLE001
    _CLASSIFICATIONS = None
    _RISK_SIGNALS = None
    _HITL_PENDING = None


def _record_metrics(result_json: str) -> None:
    try:
        data = json.loads(result_json)
    except Exception:  # noqa: BLE001
        return
    if _CLASSIFICATIONS is not None:
        try:
            s2 = data.get("stage2", {})
            _CLASSIFICATIONS.labels(
                label=str(s2.get("label", "UNKNOWN")),
                mode=str(s2.get("mode", "unknown")),
            ).inc()
        except Exception:  # noqa: BLE001
            pass
    if _RISK_SIGNALS is not None:
        try:
            risk = data.get("risk_intelligence") or {}
            _RISK_SIGNALS.labels(
                systemic_signal=str(risk.get("systemic_signal", "unknown")),
                control_domain=str(risk.get("control_domain", "unknown"))[:60],
            ).inc()
        except Exception:  # noqa: BLE001
            pass
    if _HITL_PENDING is not None:
        try:
            hitl = data.get("hitl") or {}
            _HITL_PENDING.labels(
                required=str(bool(hitl.get("required"))).lower(),
            ).inc()
        except Exception:  # noqa: BLE001
            pass


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    narrative = metadata.get("narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        narrative = " ".join(
            p.text for p in message.parts if getattr(p, "text", None)
        ).strip()
    settings = get_settings()

    try:
        result_json = call_tool(
            settings.complaint_mcp_url, "classify_complaint",
            {"narrative": narrative},
        )
    except Exception:  # noqa: BLE001 - MCP down: classify in-process
        try:
            from reg_agents.common import complaints as C

            result_json = json.dumps(C.classify_complaint(narrative), indent=2)
        except Exception as exc2:  # noqa: BLE001
            result_json = json.dumps({"error": f"classification failed: {exc2}"})

    _record_metrics(result_json)

    risk_json = "{}"
    try:
        risk_json = json.dumps(
            json.loads(result_json).get("risk_intelligence", {}), indent=2,
        )
    except Exception:  # noqa: BLE001
        pass

    summary = reason(
        _SYS,
        f"Classification + risk intelligence (JSON):\n{result_json}",
        fallback=f"Classification output:\n{result_json}",
    )
    return Task(
        artifacts=[
            Artifact(name="complaint_classification", parts=[TextPart(text=result_json)]),
            Artifact(name="risk_intelligence", parts=[TextPart(text=risk_json)]),
            Artifact(name="analyst_summary", parts=[TextPart(text=summary)]),
        ],
        metadata={"narrative": narrative[:400]},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.complaint_agent:app", 8110)
