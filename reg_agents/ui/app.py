"""Streamlit demo UI for the reg-agents governance review.

    streamlit run reg_agents/ui/app.py

Shows the multi-agent flow and renders the final audit-ready report.
"""

from __future__ import annotations

import os
import sys

# `streamlit run` puts this file's directory on sys.path, not the repo root, so
# make the reg_agents package importable regardless of how the UI is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json  # noqa: E402

import streamlit as st  # noqa: E402

from reg_agents.agents.orchestrator import (  # noqa: E402
    run_fraud_monitoring,
    run_validation_review,
)
from reg_agents.config import get_settings  # noqa: E402

st.set_page_config(page_title="reg-agents — Model Governance", layout="wide")

settings = get_settings()

st.title("reg-agents — Regulatory Intelligence & Model Governance")
st.caption(
    f"LLM provider: **{settings.llm_provider}** ({settings.active_model})  ·  "
    f"Embeddings: **{settings.embedding_provider}**  ·  "
    f"Vector backend: **{settings.vector_backend}**  ·  "
    f"Triton: **{settings.triton_url or 'local heuristic'}**"
)

with st.sidebar:
    st.header("Operations")

    with st.form("validation_form"):
        st.subheader("① Model validation")
        st.caption("Second-line validation report for a model (SR 11-7).")
        model_id = st.selectbox("Model", ["FRAUD-XGB-GNN-001", "CREDIT-LGD-014"])
        validation_go = st.form_submit_button("Run validation review", type="primary")

    st.divider()

    with st.form("fraud_form"):
        st.subheader("② Fraud monitoring")
        st.caption("Real-time scoring of a single transaction.")
        amount = st.number_input("Amount", value=4200.0, step=100.0)
        is_foreign = st.checkbox("Cross-border", value=True)
        merchant_risk = st.slider("Merchant risk", 0.0, 1.0, 0.6, 0.05)
        hour = st.slider("Hour of day", 0, 23, 2)
        velocity = st.slider("24h velocity", 0, 30, 9)
        fraud_go = st.form_submit_button("Run fraud check", type="primary")


def _render_validation(model_id: str) -> None:
    with st.spinner("Agents working (validation → retrieval → report)…"):
        result = run_validation_review(model_id)
    st.subheader(f"Model validation — {model_id}")
    tab_report, tab_val, tab_reg = st.tabs(
        ["Final Report", "Validation Findings", "Regulatory Context"]
    )
    with tab_report:
        st.markdown(result["report"])
    with tab_val:
        st.markdown(result["validation_findings"])
    with tab_reg:
        st.markdown(result["regulatory_context"])


def _render_fraud(txn: dict) -> None:
    with st.spinner("Scoring transaction (fraud model → explanation)…"):
        result = run_fraud_monitoring(txn)

    st.subheader("Fraud monitoring")
    try:
        score = json.loads(result["fraud_score"])
    except Exception:  # noqa: BLE001
        score = {}

    decision = str(score.get("decision", "—"))
    prob = score.get("fraud_probability")
    backend = str(score.get("backend", "—"))
    guardrails = score.get("guardrails") or []

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{prob:.1%}" if isinstance(prob, (int, float)) else "—")
    c2.metric("Decision", decision)
    c3.metric("Backend", backend.split(" ")[0])

    banner = {"BLOCK": st.error, "REVIEW": st.warning, "APPROVE": st.success}.get(decision, st.info)
    banner(f"Decision: **{decision}**  (probability {prob:.1%})"
           if isinstance(prob, (int, float)) else f"Decision: **{decision}**")

    if guardrails:
        st.warning("Guardrails triggered: " + ", ".join(guardrails))

    st.caption(f"Transaction: {json.dumps(txn)}")
    st.markdown("#### Analyst explanation")
    st.markdown(result["fraud_explanation"] or "_No explanation returned._")

    with st.expander("Raw model output (JSON)"):
        st.code(result["fraud_score"] or "{}", language="json")


if validation_go:
    _render_validation(model_id)
elif fraud_go:
    _render_fraud({
        "amount": amount,
        "is_foreign": is_foreign,
        "merchant_risk": merchant_risk,
        "hour": hour,
        "velocity_24h": velocity,
    })
else:
    st.info(
        "Pick an operation in the sidebar:\n\n"
        "- **① Model validation** — generate a second-line validation report for a model.\n"
        "- **② Fraud monitoring** — score a single transaction in real time."
    )
