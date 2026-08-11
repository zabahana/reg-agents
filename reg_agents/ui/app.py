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
        st.subheader("② Complaint classification + risk intelligence")
        st.caption("Model anchor: regulatory gate → RAG+LLM label "
                   "(1 of 24 regulations) with citation. Then risk "
                   "intelligence: does it signal a systemic control failure? "
                   "Real CFPB data.")
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
        complaint_go = st.form_submit_button(
            "Classify + assess risk", type="primary")
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

    st.divider()

    with st.form("batch_form"):
        st.subheader("④ Batch scoring (ingestion)")
        st.caption("Upload a CSV of complaints — or trigger the reserved 5% "
                   "scoring holdout — through classification + risk "
                   "intelligence. Output CSV includes systemic_signal, "
                   "risk_score, and control_domain.")
        uploaded = st.file_uploader(
            "CSV with a `narrative` / `complaint` / `text` column "
            "(optional `complaint_id`)",
            type="csv",
        )
        use_holdout = st.checkbox(
            "No file — score the reserved 5% holdout instead", value=True)
        batch_limit = st.slider("Max rows to score", 5, 200, 25, 5)
        batch_llm = st.checkbox("Use LLM for stage-2 / risk hypothesis", value=True)
        batch_go = st.form_submit_button("Score batch", type="primary")


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
    with st.spinner("Classifying (stage 1 → stage 2) + risk intelligence…"):
        result = run_complaint_classification(narrative)

    st.subheader("Complaint classification")
    try:
        data = json.loads(result["classification"])
    except Exception:  # noqa: BLE001
        data = {}
    s1, s2 = data.get("stage1", {}), data.get("stage2", {})
    risk = data.get("risk_intelligence") or {}
    if not risk and result.get("risk"):
        try:
            risk = json.loads(result["risk"])
        except Exception:  # noqa: BLE001
            risk = {}

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

    # --- Risk intelligence (elevates classification → control risk) ---
    st.markdown("#### Risk intelligence — systemic control failure?")
    st.caption(
        "Classification says **what** the complaint is. Risk intelligence asks "
        "whether it signals a **systemic control failure**."
    )
    signal = str(risk.get("systemic_signal", "—"))
    rscore = risk.get("score")
    r1, r2, r3 = st.columns(3)
    r1.metric("Systemic signal", signal.upper() if signal else "—")
    r2.metric("Risk score",
              f"{rscore:.0%}" if isinstance(rscore, (int, float)) else "—")
    r3.metric("Control domain", str(risk.get("control_domain", "—")))

    banner = {
        "elevated": st.error,
        "moderate": st.warning,
        "isolated": st.info,
        "none": st.success,
    }.get(signal, st.info)
    failure = risk.get("failure_mode") or ""
    action = risk.get("recommended_action") or ""
    banner(
        (f"**{failure}**\n\n" if failure else "")
        + (f"{action}" if action else f"Signal: {signal}")
    )

    drivers = risk.get("drivers") or []
    if drivers:
        st.markdown("**Drivers**")
        for d in drivers:
            st.markdown(f"- {d}")

    hyp = risk.get("hypothesis") or {}
    if isinstance(hyp, dict) and hyp.get("hypothesis"):
        st.markdown("**Control-failure hypothesis**")
        st.markdown(hyp["hypothesis"])
        if hyp.get("control_test"):
            st.caption(f"Suggested control test: {hyp['control_test']}")

    similars = risk.get("similar_prior_cases") or []
    if similars:
        st.markdown("**Similar prior cases**")
        for c in similars:
            st.markdown(
                f"- `{c.get('complaint_id')}` · {c.get('label')} · "
                f"sim={c.get('similarity')} · {c.get('product', '')}\n\n"
                f"  _{c.get('excerpt', '')}_"
            )

    expl = risk.get("local_explanation") or {}
    pos = expl.get("top_positive") or []
    if pos:
        with st.expander(
            f"Local explanation — top TF-IDF terms "
            f"(stage-1 {expl.get('model') or 'champion'})"
        ):
            st.dataframe(
                [{"term": p["term"], "contribution": p["contribution"]} for p in pos],
                use_container_width=True,
                hide_index=True,
            )

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
        fig_cm = os.path.join(_COMPLAINT_DOCS, "figures", "stage1_confusion.png")
        fig_rec = os.path.join(_COMPLAINT_DOCS, "figures", "stage2_recall.png")
        if os.path.exists(fig_cm) or os.path.exists(fig_rec):
            with st.expander("Validation figures (confusion matrix / per-class recall)"):
                cols = st.columns(2)
                if os.path.exists(fig_cm):
                    cols[0].image(fig_cm, caption="Stage-1 confusion matrix")
                if os.path.exists(fig_rec):
                    cols[1].image(fig_rec, caption="Stage-2 per-category recall")
        st.caption("Full documentation: `docs/complaint_model/` — development "
                   "document + independent validation report (MD + PDF). "
                   "Model anchor: TF-IDF + logistic/XGBoost gate (imbalance "
                   "treatment, validation-tuned threshold), macro-F1 / "
                   "per-class recall, local explanation, prior-case similarity.")


def _render_batch(uploaded_file, use_holdout: bool, limit: int,
                  use_llm: bool) -> None:
    import pandas as pd

    from reg_agents.common import complaints as C

    st.subheader("Batch scoring — classification + risk intelligence")
    if uploaded_file is not None:
        try:
            batch = pd.read_csv(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the uploaded CSV: {exc}")
            return
        source = f"uploaded file `{uploaded_file.name}`"
    elif use_holdout:
        batch = C.scoring_holdout()
        source = ("reserved 5% scoring holdout — carved out **before** the "
                  "80/10/10 modeling split, never seen in training, "
                  "validation, test, or threshold tuning")
    else:
        st.warning("Upload a CSV or tick the holdout option.")
        return

    try:
        C._resolve_text_column(batch)
    except ValueError as exc:
        st.error(str(exc))
        return

    total_available = len(batch)
    batch = batch.head(limit)
    st.caption(f"Source: {source} · scoring {len(batch)} of "
               f"{total_available} rows · LLM: {'on' if use_llm else 'off'}")

    bar = st.progress(0.0, text="Scoring…")

    def progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Scoring… {done}/{total}")

    scored = C.score_batch(batch, use_llm=use_llm, progress=progress)
    bar.empty()

    n_reg = int(scored["is_regulatory"].sum())
    n_elev = int((scored["systemic_signal"] == "elevated").sum()) if "systemic_signal" in scored else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Complaints scored", len(scored))
    c2.metric("Flagged regulatory", f"{n_reg} ({n_reg / max(len(scored), 1):.0%})")
    c3.metric("Elevated systemic risk", n_elev)
    c4.metric("Stage-2 mode", scored["mode"].mode().iloc[0] if len(scored) else "—")

    display_cols = [
        "complaint_id", "complaint", "score", "is_regulatory",
        "label", "confidence", "systemic_signal", "risk_score",
        "control_domain", "llm_reasoning",
    ]
    display_cols = [c for c in display_cols if c in scored.columns]
    st.dataframe(
        scored[display_cols],
        use_container_width=True,
        column_config={
            "complaint": st.column_config.TextColumn(width="medium"),
            "score": st.column_config.NumberColumn(
                "score P(regulatory)", format="%.4f"),
            "risk_score": st.column_config.NumberColumn(
                "risk score", format="%.2f"),
            "llm_reasoning": st.column_config.TextColumn(width="large"),
        },
    )

    st.download_button(
        "Download scored CSV",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_complaints.csv",
        mime="text/csv",
        type="primary",
    )

    with st.expander("Regulation label mix"):
        st.bar_chart(scored["label"].value_counts())
    if "systemic_signal" in scored.columns:
        with st.expander("Systemic-signal mix"):
            st.bar_chart(scored["systemic_signal"].value_counts())


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
elif batch_go:
    _render_batch(uploaded, use_holdout, batch_limit, batch_llm)
else:
    st.info(
        "Pick an operation in the sidebar:\n\n"
        "- **① Model validation** — generate a second-line validation report for a model.\n"
        "- **② Complaint classification + risk intelligence** — assign a CFPB "
        "complaint to a regulation category, then ask whether it signals a "
        "systemic control failure.\n"
        "- **③ Fraud monitoring** — score a single transaction in real time.\n"
        "- **④ Batch scoring (ingestion)** — upload a complaint CSV (or trigger the "
        "reserved 5% holdout) and download scored CSV with risk signals."
    )
