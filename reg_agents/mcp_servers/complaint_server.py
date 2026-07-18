"""MCP server: complaint → regulation classification tools.

Fronts the two-stage complaint model (reg_agents/common/complaints.py):
stage 1 = TF-IDF binary "regulatory or not" champion; stage 2 = RAG over the
regulation corpus + LLM reasoning with few-shot examples, returning a label
from the 24-category taxonomy plus a cited excerpt.

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

    Stage 1 decides regulatory vs not; if regulatory, stage 2 assigns one of
    the 24 regulation categories with a confidence, rationale, and a citation
    from the retrieved policy corpus. Returns JSON.
    """
    from reg_agents.common import complaints as C

    result = C.classify_complaint(narrative, use_llm=use_llm)
    return json.dumps(result, indent=2)


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
