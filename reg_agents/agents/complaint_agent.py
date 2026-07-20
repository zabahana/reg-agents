"""Complaint Agent (A2A server on :8110).

Skill: classify a consumer-complaint narrative into the 24-category regulation
taxonomy via the complaint MCP server (two-stage: binary gate, then RAG + LLM
with few-shot examples and citations), and produce a compliance-analyst
summary of the classification.

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
                "with RAG citations (CFPB-trained two-stage model).",
    url="http://localhost:8110",
    skills=[
        AgentSkill(
            id="complaint-classification",
            name="Complaint → regulation classification",
            description="Two-stage classification (binary gate + RAG/LLM "
                        "labeling) with citations from the policy corpus.",
            tags=["complaints", "udaap", "rag", "nlp", "cfpb"],
        )
    ],
)

_SYS = (
    "You are a bank compliance operations copilot. Given the JSON output of a "
    "complaint-classification model (stage-1 regulatory gate + stage-2 "
    "regulation label with citation), write 3-4 sentences for a complaints "
    "analyst: state the assigned category and why, and recommend the routing "
    "(e.g. compliance review queue vs standard service recovery). If a "
    "citation is present, quote its key phrase. If the complaint was gated "
    "non-regulatory at stage 1 (mode stage1_gate), simply say the gate found "
    "no regulatory nexus and recommend service recovery — no citation exists "
    "on that path, so never mention citations, null fields, or JSON internals. "
    "Do not invent facts."
)

# Per-classification metrics for Grafana (label + mode), best-effort.
try:
    from prometheus_client import Counter

    _CLASSIFICATIONS = Counter(
        "complaint_classifications_total",
        "Complaint classifications by regulation label and stage-2 mode",
        ["label", "mode"],
    )
except Exception:  # noqa: BLE001
    _CLASSIFICATIONS = None


def _record_metrics(result_json: str) -> None:
    if _CLASSIFICATIONS is None:
        return
    try:
        data = json.loads(result_json)
        s2 = data.get("stage2", {})
        _CLASSIFICATIONS.labels(
            label=str(s2.get("label", "UNKNOWN")),
            mode=str(s2.get("mode", "unknown")),
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

    summary = reason(
        _SYS,
        f"Classification output (JSON):\n{result_json}",
        fallback=f"Classification output:\n{result_json}",
    )
    return Task(
        artifacts=[
            Artifact(name="complaint_classification", parts=[TextPart(text=result_json)]),
            Artifact(name="analyst_summary", parts=[TextPart(text=summary)]),
        ],
        metadata={"narrative": narrative[:400]},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.complaint_agent:app", 8110)
