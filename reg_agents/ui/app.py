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
    run_complaint_classification,
    run_fraud_monitoring,
    run_validation_review,
)
from reg_agents.config import get_settings  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_COMPLAINT_DOCS = os.path.join(_ROOT, "docs", "complaint_model")


@st.cache_data
def _load_complaint_samples(n: int = 40):
    """Random real CFPB complaints for the picker (cached per session)."""
    import pandas as pd

    path = os.path.join(_ROOT, "data", "complaints", "cfpb_complaints.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).sample(n=n, random_state=7)
    return df[["complaint_id", "product", "issue", "narrative"]].reset_index(drop=True)


@st.cache_data
def _load_complaint_metrics():
    path = os.path.join(_COMPLAINT_DOCS, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

st.set_page_config(page_title="reg-agents — Model Governance", layout="wide")

settings = get_settings()

st.title("reg-agents — Regulatory Intelligence & Model Governance")
st.caption(
    f"LLM provider: **{settings.llm_provider}** ({settings.active_model})  ·  "
    f"Embeddings: **{settings.embedding_provider}**  ·  "
    f"Vector backend: **{settings.vector_backend}**  ·  "
    f"Fraud serving: **{'Triton (GPU)' if settings.triton_url else 'local heuristic'}**"
)

with st.sidebar:
    st.header("Operations")

    with st.form("validation_form"):
        st.subheader("① Model validation")
        st.caption("Second-line validation report for a model (SR 11-7).")
        model_id = st.selectbox(
            "Model",
            [
                "CMPL-REG-24",          # complaint → regulation classifier (real CFPB data)
                "FRAUD-XGB-GNN-001",
                "CREDIT-LGD-014",
                "CREDIT-PD-007",
                "AML-TM-021",
                "GENAI-COMPLAINT-030",
                "PPNR-CARD-009",
            ],
        )
        validation_go = st.form_submit_button("Run validation review", type="primary")

    st.divider()

    with st.form("complaint_form"):
        st.subheader("② Complaint classification")
        st.caption("Two-stage model: regulatory gate → RAG+LLM label "
                   "(1 of 24 regulations) with citation. Real CFPB data.")
        samples = _load_complaint_samples()
        complaint_text = ""
        if samples is not None:
            options = [
                f"{r.complaint_id} · {r.product[:34]} · {r.issue[:38]}"
                for r in samples.itertuples()
            ]
            picked = st.selectbox("Pick a real CFPB complaint", options)
            complaint_text = str(samples.iloc[options.index(picked)]["narrative"])
        custom = st.text_area("…or paste a complaint narrative", height=90)
        complaint_go = st.form_submit_button("Classify complaint", type="primary")
        if custom.strip():
            complaint_text = custom.strip()

    st.divider()

    with st.form("fraud_form"):
        st.subheader("③ Fraud monitoring")
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


def _render_complaint(narrative: str) -> None:
    with st.spinner("Classifying (stage 1 gate → stage 2 RAG + LLM)…"):
        result = run_complaint_classification(narrative)

    st.subheader("Complaint classification")
    try:
        data = json.loads(result["classification"])
    except Exception:  # noqa: BLE001
        data = {}
    s1, s2 = data.get("stage1", {}), data.get("stage2", {})

    c1, c2, c3 = st.columns(3)
    p1 = s1.get("probability")
    c1.metric("Stage 1 — regulatory?",
              "YES" if s1.get("is_regulatory") else "NO",
              f"p = {p1:.2f}" if isinstance(p1, (int, float)) else None)
    c2.metric("Regulation label", str(s2.get("label", "—")))
    conf = s2.get("confidence")
    c3.metric("Stage-2 confidence",
              f"{conf:.0%}" if isinstance(conf, (int, float)) else "—")

    if s2.get("label") and s2.get("label") != "NON_REGULATORY":
        st.error(f"**{s2.get('regulation_name', s2.get('label'))}** — "
                 f"{s2.get('regulation_description', '')}")
    else:
        st.success("Non-regulatory — route to standard service recovery.")

    if s2.get("rationale"):
        st.markdown(f"**Model rationale:** {s2['rationale']}")

    citation = s2.get("citation")
    if citation:
        st.markdown("#### Citation (retrieved from the regulation corpus)")
        st.info(f"**{citation.get('source', '')} — {citation.get('heading', '')}**\n\n"
                f"{citation.get('text', '')}")

    if result.get("summary"):
        st.markdown("#### Analyst summary")
        st.markdown(result["summary"])

    with st.expander("Complaint narrative"):
        st.write(narrative)
    with st.expander("Raw model output (JSON)"):
        st.code(result["classification"] or "{}", language="json")

    metrics = _load_complaint_metrics()
    if metrics:
        st.markdown("#### Model accuracy (from the committed validation run)")
        lb = metrics["stage1"]["leaderboard"][0]
        s2m = metrics["stage2"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stage 1 PR-AUC", lb["pr_auc"])
        m2.metric("Stage 1 F1", lb["f1"])
        m3.metric("Stage 2 family accuracy", s2m["family_accuracy"])
        m4.metric("Stage 2 macro-F1", s2m["macro_f1"])
        fig = os.path.join(_COMPLAINT_DOCS, "figures", "stage2_recall.png")
        if os.path.exists(fig):
            with st.expander("Per-category recall (validation figure)"):
                st.image(fig)
        st.caption("Full documentation: `docs/complaint_model/` — development "
                   "document + independent validation report (MD + PDF).")


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
elif complaint_go and complaint_text.strip():
    _render_complaint(complaint_text.strip())
else:
    st.info(
        "Pick an operation in the sidebar:\n\n"
        "- **① Model validation** — generate a second-line validation report for a model.\n"
        "- **② Complaint classification** — assign a real CFPB complaint to one of "
        "24 regulation categories with a RAG citation.\n"
        "- **③ Fraud monitoring** — score a single transaction in real time."
    )
