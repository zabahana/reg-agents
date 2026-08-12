"""MCP server: complaint → regulation classification tools.

Fronts the two-stage complaint model (reg_agents/common/complaints.py):
stage 1 = TF-IDF binary gate (local or Triton ``complaint_stage1``);
stage 2 = RAG + LLM with citations; risk intelligence; NeMo/native
guardrails; and HITL disposition tools for analyst approve/override/escalate.

Run:  python -m reg_agents.mcp_servers.complaint_server  (PORT default 9105)
"""

# NOTE: no `from __future__ import annotations` here -- FastMCP introspects real
# annotation types when registering tools, and stringized annotations break it.

import json
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("complaints", host="0.0.0.0", port=int(os.getenv("PORT", "9105")))


@mcp.tool()
def classify_complaint(narrative: str, use_llm: bool = True) -> str:
    """Classify a consumer-complaint narrative.

    Stage 1 (local sklearn or Triton) decides regulatory vs not; stage 2
    assigns a regulation category with citation; risk_intelligence assesses
    systemic control failure; hitl flags pending analyst review. Returns JSON.
    """
    from reg_agents.common import complaints as C

    result = C.classify_complaint(narrative, use_llm=use_llm)
    return json.dumps(result, indent=2)


@mcp.tool()
def assess_risk_intelligence(narrative: str, use_llm: bool = True) -> str:
    """Assess whether a complaint signals a systemic control failure.

    Runs the two-stage classifier as the model anchor, then returns only the
    risk_intelligence block (JSON).
    """
    from reg_agents.common import complaints as C

    result = C.classify_complaint(narrative, use_llm=use_llm)
    return json.dumps(result.get("risk_intelligence", {}), indent=2)


@mcp.tool()
def submit_hitl_decision(
    narrative: str,
    disposition: str,
    analyst: str = "analyst",
    override_label: str = "",
    rationale: str = "",
    classification_json: str = "",
) -> str:
    """Record a human-in-the-loop disposition for a complaint.

    disposition: approve | override | escalate.
    For override, pass override_label as a taxonomy code (e.g. FCRA_ACCURACY).
    If classification_json is empty, the narrative is re-classified first.
    """
    from reg_agents.common import complaints as C
    from reg_agents.common import hitl as H

    if classification_json.strip():
        classification = json.loads(classification_json)
    else:
        classification = C.classify_complaint(narrative, use_llm=False)
    record = H.submit_decision(
        narrative=narrative,
        classification=classification,
        disposition=disposition,
        analyst=analyst,
        override_label=override_label,
        rationale=rationale,
    )
    return json.dumps(record, indent=2)


@mcp.tool()
def list_hitl_decisions(limit: int = 25) -> str:
    """List recent HITL dispositions from the audit log (JSON)."""
    from reg_agents.common import hitl as H

    return json.dumps(
        {"counts": H.decision_counts(), "decisions": H.list_decisions(limit)},
        indent=2,
    )


@mcp.tool()
def list_regulation_taxonomy() -> str:
    """List the 24 regulation categories the model can assign (JSON)."""
    from reg_agents.common import complaints as C

    return json.dumps(
        [
            {"label": r.label, "name": r.name, "description": r.description}
            for r in C.REGULATIONS.values()
        ],
        indent=2,
    )


@mcp.tool()
def sample_complaints(n: int = 5) -> str:
    """Return n random real complaints from the curated CFPB dataset (JSON)."""
    from reg_agents.common import complaints as C

    df = C.load_complaints().sample(n=min(int(n), 25))
    return json.dumps(
        df[["complaint_id", "product", "issue", "narrative"]].to_dict("records"),
        indent=2,
    )


@mcp.tool()
def get_model_metrics() -> str:
    """Return the committed evaluation metrics for the complaint model (JSON)."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "docs", "complaint_model", "metrics.json",
    )
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return json.dumps({"error": "metrics.json not generated yet"})


if __name__ == "__main__":
    mcp.run(transport="sse")
