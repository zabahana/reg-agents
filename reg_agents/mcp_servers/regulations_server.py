"""MCP server: regulatory knowledge base.

Exposes semantic search over banking regulations (SR 11-7, ECOA, FCRA, UDAAP,
etc.). Backed by NeMo Retriever embeddings + vector store in the demo; lexical
fallback locally. Runs over SSE so it is reachable from containers/k8s.

Run:  python -m reg_agents.mcp_servers.regulations_server  (PORT env, default 9101)
"""

# NOTE: no `from __future__ import annotations` -- FastMCP needs real annotation
# types when registering tools.

import json
import os

from mcp.server.fastmcp import FastMCP

from reg_agents.common.corpus import RegulationRetriever

mcp = FastMCP("regulations", host="0.0.0.0", port=int(os.getenv("PORT", "9101")))
_retriever = RegulationRetriever()


@mcp.tool()
def search_regulations(query: str, k: int = 4) -> str:
    """Search banking regulations for clauses relevant to a query.

    Returns a JSON list of {source, heading, score, text} passages.
    """
    hits = _retriever.search(query, k)
    payload = [
        {
            "source": h.document.metadata.get("source", ""),
            "heading": h.document.metadata.get("heading", ""),
            "score": round(h.score, 4),
            "text": h.document.text,
        }
        for h in hits
    ]
    return json.dumps(payload, indent=2)


@mcp.tool()
def list_regulation_sources() -> str:
    """List the regulation documents available in the knowledge base."""
    sources = sorted({d.metadata.get("source", "") for d in _retriever.docs})
    return json.dumps(sources)


if __name__ == "__main__":
    mcp.run(transport="sse")
