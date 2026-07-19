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
    "CREDIT-PD-007": {
        "model_id": "CREDIT-PD-007",
        "name": "Consumer Card PD Origination Scorecard",
        "type": "Logistic Regression + Gradient Boosting ensemble",
        "owner": "Credit Risk",
        "tier": "1 (high risk)",
        "use": "Probability of default at origination / underwriting",
        "protected_class_features": "excluded; disparate-impact tested (ECOA)",
        "doc_file": "model_card_pd_scorecard.md",
    },
    "AML-TM-021": {
        "model_id": "AML-TM-021",
        "name": "AML Transaction Monitoring Model",
        "type": "Rules + Isolation Forest + GNN typologies",
        "owner": "Financial Crimes",
        "tier": "1 (high risk)",
        "use": "Detect suspicious activity for SAR triage (BSA/AML)",
        "protected_class_features": "n/a (entity/transaction features)",
        "doc_file": "model_card_aml_monitoring.md",
    },
    "CMPL-REG-24": {
        "model_id": "CMPL-REG-24",
        "name": "Complaint → Regulation Classifier (two-stage, CFPB data)",
        "type": "TF-IDF gate (logistic/XGBoost) + RAG/LLM labeler (NIM)",
        "owner": "Regulatory Intelligence",
        "tier": "2 (medium risk)",
        "use": "Classify consumer complaints into a 24-category regulation taxonomy with citations",
        "protected_class_features": "none (free-text narrative only)",
        "doc_file": "model_card_complaint_reg24.md",
    },
    "GENAI-COMPLAINT-030": {
        "model_id": "GENAI-COMPLAINT-030",
        "name": "GenAI Complaint Classification & Regulatory Mapping",
        "type": "LLM (RAG + fine-tuned classifier)",
        "owner": "Regulatory Intelligence",
        "tier": "2 (medium risk)",
        "use": "Classify consumer complaints and map to federal regulations",
        "protected_class_features": "n/a (free-text complaints)",
        "doc_file": "model_card_genai_complaint.md",
    },
    "PPNR-CARD-009": {
        "model_id": "PPNR-CARD-009",
        "name": "Credit Card PPNR Forecasting Model",
        "type": "Econometric time-series (ARIMAX) + LSTM overlay",
        "owner": "Balance Sheet Analytics",
        "tier": "1 (high risk)",
        "use": "Pre-provision net revenue forecasting for CCAR",
        "protected_class_features": "n/a (portfolio aggregates)",
        "doc_file": "model_card_ppnr_forecast.md",
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
