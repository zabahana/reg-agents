"""Fraud scoring heuristic tests (no Triton, no keys required)."""

import json

from reg_agents.mcp_servers.fraud_server import _heuristic_score, score_transaction


def test_low_risk_transaction_approved():
    result = json.loads(score_transaction(amount=20.0, is_foreign=False,
                                           merchant_risk=0.02, hour=12, velocity_24h=1))
    assert result["decision"] == "APPROVE"
    assert result["fraud_probability"] < 0.4


def test_high_risk_transaction_flagged():
    result = json.loads(score_transaction(amount=9000.0, is_foreign=True,
                                          merchant_risk=0.9, hour=3, velocity_24h=15))
    assert result["decision"] in {"REVIEW", "BLOCK"}
    assert result["fraud_probability"] >= 0.4


def test_score_monotonic_in_amount():
    low = _heuristic_score(100, False, 0.1, 12, 1)
    high = _heuristic_score(5000, False, 0.1, 12, 1)
    assert high > low
