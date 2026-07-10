"""MCP server: fraud transaction scoring.

Fronts the fraud model. In the GPU demo this proxies to an NVIDIA Triton
Inference Server endpoint (GNN-enhanced XGBoost, mirroring NVIDIA's financial
fraud AI Blueprint). Locally, with no TRITON_URL set, it uses a transparent
heuristic so the agent pipeline is always demonstrable.

Run:  python -m reg_agents.mcp_servers.fraud_server  (PORT default 9103)
"""

# NOTE: no `from __future__ import annotations` here -- FastMCP introspects real
# annotation types when registering tools, and stringized annotations break it.

import json
import os

import httpx
from mcp.server.fastmcp import FastMCP

from reg_agents.config import get_settings

mcp = FastMCP("fraud", host="0.0.0.0", port=int(os.getenv("PORT", "9103")))


def _heuristic_score(amount: float, is_foreign: bool, merchant_risk: float,
                     hour: int, velocity_24h: int) -> float:
    """Interpretable stand-in for the GNN+XGBoost model."""
    score = 0.0
    score += min(amount / 5000.0, 1.0) * 0.35
    score += 0.2 if is_foreign else 0.0
    score += max(0.0, min(merchant_risk, 1.0)) * 0.25
    score += 0.1 if (hour < 6 or hour >= 23) else 0.0
    score += min(velocity_24h / 20.0, 1.0) * 0.1
    return round(min(score, 0.99), 4)


@mcp.tool()
def score_transaction(
    amount: float,
    is_foreign: bool = False,
    merchant_risk: float = 0.1,
    hour: int = 12,
    velocity_24h: int = 1,
) -> str:
    """Score a card transaction for fraud risk.

    Returns JSON: {fraud_probability, decision, backend, features}.
    """
    settings = get_settings()
    backend = "heuristic-local"
    prob = _heuristic_score(amount, is_foreign, merchant_risk, hour, velocity_24h)

    if settings.triton_url:
        try:
            # Triton HTTP inference (KServe v2 protocol). Feature vector order
            # must match the deployed model's config.pbtxt.
            features = [amount, float(is_foreign), merchant_risk, float(hour), float(velocity_24h)]
            payload = {
                "inputs": [
                    {"name": "input__0", "shape": [1, len(features)],
                     "datatype": "FP32", "data": features}
                ]
            }
            url = f"{settings.triton_url.rstrip('/')}/v2/models/fraud_xgb_gnn/infer"
            with httpx.Client(timeout=10.0) as c:
                r = c.post(url, json=payload)
                r.raise_for_status()
                out = r.json()
                prob = round(float(out["outputs"][0]["data"][0]), 4)
                backend = "triton-gpu"
        except Exception as exc:  # noqa: BLE001 - fall back but report why
            backend = f"heuristic-local (triton error: {exc})"

    decision = "BLOCK" if prob >= 0.7 else "REVIEW" if prob >= 0.4 else "APPROVE"
    return json.dumps(
        {
            "fraud_probability": prob,
            "decision": decision,
            "backend": backend,
            "features": {
                "amount": amount,
                "is_foreign": is_foreign,
                "merchant_risk": merchant_risk,
                "hour": hour,
                "velocity_24h": velocity_24h,
            },
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="sse")
