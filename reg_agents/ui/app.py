"""Streamlit demo UI for the reg-agents governance review.

    streamlit run reg_agents/ui/app.py

Shows the multi-agent flow and renders the final audit-ready report.
"""

from __future__ import annotations

import streamlit as st

from reg_agents.agents.orchestrator import run_review
from reg_agents.config import get_settings

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
    st.header("Review inputs")
    model_id = st.selectbox("Model", ["FRAUD-XGB-GNN-001", "CREDIT-LGD-014"])
    st.subheader("Sample transaction")
    amount = st.number_input("Amount", value=4200.0, step=100.0)
    is_foreign = st.checkbox("Cross-border", value=True)
    merchant_risk = st.slider("Merchant risk", 0.0, 1.0, 0.6, 0.05)
    hour = st.slider("Hour of day", 0, 23, 2)
    velocity = st.slider("24h velocity", 0, 30, 9)
    go = st.button("Run governance review", type="primary")

if go:
    txn = {
        "amount": amount,
        "is_foreign": is_foreign,
        "merchant_risk": merchant_risk,
        "hour": hour,
        "velocity_24h": velocity,
    }
    with st.spinner("Agents working (validation → fraud → retrieval → report)…"):
        result = run_review(model_id, txn)

    tab_report, tab_val, tab_fraud, tab_reg = st.tabs(
        ["Final Report", "Validation", "Fraud Analysis", "Regulatory Context"]
    )
    with tab_report:
        st.markdown(result["report"])
    with tab_val:
        st.markdown(result["validation_findings"])
    with tab_fraud:
        st.markdown(result["fraud_analysis"])
    with tab_reg:
        st.markdown(result["regulatory_context"])
else:
    st.info("Configure inputs in the sidebar and click **Run governance review**.")
