"""MCP server: model registry.

Stands in for a bank's Model Risk Management (MRM) inventory. Exposes model
metadata + documentation (model cards) so the validation agent can assess a
model against SR 11-7 expectations.

Run:  python -m reg_agents.mcp_servers.model_registry_server  (PORT default 9102)
"""

# NOTE: no `from __future__ import annotations` -- FastMCP needs real annotation
# types when registering tools.

import json
import os

from mcp.server.fastmcp import FastMCP

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models"
)

mcp = FastMCP("model-registry", host="0.0.0.0", port=int(os.getenv("PORT", "9102")))

# A tiny in-memory inventory. In production this is a governance system (e.g.
# an MRM tool) fronted by MCP.
_INVENTORY = {
    "FRAUD-XGB-GNN-001": {
        "model_id": "FRAUD-XGB-GNN-001",
        "name": "Card Transaction Fraud Detector",
        "type": "GNN-enhanced XGBoost",
        "owner": "Payments Risk",
        "tier": "1 (high risk)",
        "use": "Real-time card transaction fraud scoring",
        "protected_class_features": "none (age/zip excluded)",
        "doc_file": "model_card_fraud.md",
    },
    "CREDIT-LGD-014": {
        "model_id": "CREDIT-LGD-014",
        "name": "Small Business LGD Model",
        "type": "Gradient Boosted Trees",
        "owner": "Credit Risk",
        "tier": "1 (high risk)",
        "use": "Loss Given Default for CCAR",
        "protected_class_features": "reviewed under ECOA",
        "doc_file": "model_card_credit.md",
    },
}


@mcp.tool()
def list_models() -> str:
    """List all models in the registry (id, name, tier, use)."""
    return json.dumps(
        [
            {k: m[k] for k in ("model_id", "name", "tier", "use")}
            for m in _INVENTORY.values()
        ],
        indent=2,
    )


@mcp.tool()
def get_model_metadata(model_id: str) -> str:
    """Return structured metadata for a model id."""
    model = _INVENTORY.get(model_id)
    if not model:
        return json.dumps({"error": f"unknown model_id: {model_id}"})
    return json.dumps(model, indent=2)


@mcp.tool()
def get_model_documentation(model_id: str) -> str:
    """Return the full model-card documentation text for a model id."""
    model = _INVENTORY.get(model_id)
    if not model:
        return json.dumps({"error": f"unknown model_id: {model_id}"})
    path = os.path.join(_DATA_DIR, model["doc_file"])
    if not os.path.isfile(path):
        return json.dumps({"error": f"documentation not found for {model_id}"})
    with open(path, encoding="utf-8") as fh:
        return fh.read()


if __name__ == "__main__":
    mcp.run(transport="sse")
