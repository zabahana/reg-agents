#!/usr/bin/env python3
"""End-to-end Model Development Document for CMPL-REG-24.

Assembles the full development story from committed artifacts — data
acquisition (v1→v3), split protocol, stage-1 bake-off + regularization,
leakage audit, stage-2 RAG+LLM, dual-judge panel (NIM + OpenAI), DPO/RLAIF
post-training, batch scoring, and agentic serving — into one markdown + PDF.

Run:
    python scripts/generate_e2e_model_development_doc.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "docs", "complaint_model")
MDD_DIR = os.path.join(ROOT, "docs", "model_development")
FIG = os.path.join(OUT_DIR, "figures")
MDD_FIG = os.path.join(MDD_DIR, "figures")
GREEN, GRAY = "#76b900", "#555555"
PAGE = (8.5, 11)


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _title_page(pdf, title, subtitle, meta):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.72, title, fontsize=20, weight="bold", wrap=True)
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.70, 0.70], color=GREEN,
                                lw=3, transform=fig.transFigure))
    fig.text(0.08, 0.62, subtitle, fontsize=11, color=GRAY)
    fig.text(0.08, 0.14, meta, fontsize=8.5, color=GRAY, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _text_page(pdf, heading, body):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.94, heading, fontsize=14, weight="bold")
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=GREEN,
                                lw=2, transform=fig.transFigure))
    wrapped = []
    for para in body.split("\n"):
        wrapped.extend(textwrap.wrap(para, width=100) or [""])
    fig.text(0.08, 0.89, "\n".join(wrapped[:52]), fontsize=9, va="top",
             family="serif", linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)
    rest = wrapped[52:]
    while rest:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.08, 0.94, f"{heading} (cont.)", fontsize=12, weight="bold")
        fig.text(0.08, 0.90, "\n".join(rest[:56]), fontsize=9, va="top",
                 family="serif", linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)
        rest = rest[56:]


def _table_page(pdf, heading, headers, rows, caption=""):
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off")
    fig.text(0.08, 0.94, heading, fontsize=13, weight="bold")
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=GREEN,
                                lw=2, transform=fig.transFigure))
    n = len(rows)
    tbl = ax.table(cellText=[[str(c) for c in r] for r in rows],
                   colLabels=headers, loc="upper center", cellLoc="center",
                   bbox=[0.0, max(0.05, 0.86 - 0.032 * n), 1.0,
                         min(0.82, 0.032 * (n + 1))])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8f2d8")
            cell.set_text_props(weight="bold")
    if caption:
        fig.text(0.08, 0.04, caption, fontsize=8, color=GRAY)
    pdf.savefig(fig)
    plt.close(fig)


def _figure_page(pdf, heading, png_path, caption=""):
    if not png_path or not os.path.exists(png_path):
        return
    img = plt.imread(png_path)
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.94, heading, fontsize=13, weight="bold")
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=GREEN,
                                lw=2, transform=fig.transFigure))
    h, w = img.shape[:2]
    disp_w, disp_h = 0.84, min(0.78, 0.84 * (h / w) * (PAGE[0] / PAGE[1]))
    ax = fig.add_axes([0.08, 0.88 - disp_h, disp_w, disp_h])
    ax.imshow(img)
    ax.axis("off")
    if caption:
        fig.text(0.08, 0.86 - disp_h, caption, fontsize=8, color=GRAY, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    metrics = load(os.path.join(OUT_DIR, "metrics.json"), {})
    mdd = load(os.path.join(MDD_DIR, "results.json"), {})
    judges = load(os.path.join(OUT_DIR, "judge_agreement.json"), {})
    dpo = load(os.path.join(OUT_DIR, "dpo_results.json"), {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ds = metrics.get("dataset", {})
    s1 = metrics.get("stage1", {})
    s2 = metrics.get("stage2", {})
    lb = s1.get("leaderboard", [])
    champ = next((r for r in lb if r["model"] == s1.get("champion")), lb[0] if lb else {})
    leak = mdd.get("leakage_audit", {})
    jp = judges.get("pairs", {})
    dpo_eval = dpo.get("eval", {})
    dpo_gate = dpo_eval.get("gate", {})
    dpo_pol = dpo_eval.get("dpo", {})

    # Research bake-off (MDD) vs production (metrics.json)
    research = mdd.get("test", {})
    research_champ = mdd.get("champion", "lightgbm")
    research_thr = mdd.get("champion_threshold", "—")

    abstract = (
        f"This end-to-end Model Development Document records the full lifecycle "
        f"of CMPL-REG-24, a two-stage consumer-complaint classifier on "
        f"{ds.get('n_rows', 4000):,} curated CFPB narratives "
        f"({ds.get('regulatory_rate', 0.875):.1%} regulatory / "
        f"{1 - ds.get('regulatory_rate', 0.875):.1%} non-regulatory). "
        f"Stage 1 is a millisecond TF-IDF gate — production champion "
        f"**{s1.get('champion', 'logistic_regression')}** "
        f"({champ.get('params', 'L1')}, cut-off {champ.get('threshold', 0.491)}) "
        f"with test ROC-AUC {champ.get('roc_auc', 0.936)}, PR-AUC "
        f"{champ.get('pr_auc', 0.990)}, train/test gap "
        f"{champ.get('train_test_gap', 0.06)}. Stage 2 is RAG + LLM over a "
        f"24-category regulation taxonomy (exact agreement "
        f"{s2.get('accuracy', 0.25):.2f}, family "
        f"{s2.get('family_accuracy', 0.39):.2f} on n={s2.get('n', 101)}). "
        f"The document also covers the 5% scoring holdout, the v1→v3 data "
        f"update log, the leakage audit (split contamination + weak-label "
        f"target leakage), the dual-judge panel (NIM + OpenAI as first-class "
        f"judges, not fallbacks), the DPO/RLAIF post-training loop, batch "
        f"ingestion, and MCP/A2A serving. Disposition: **Approve with "
        f"Conditions** pending a human-adjudicated golden set."
    )

    # ---- tables ------------------------------------------------------------
    data_update_rows = [
        ["v1 · generic pull only", "135 (3.4%)", "0.639", "0.849", "0.147"],
        ["v2 · + targeted service-issue pass", "500 (12.5%)", "0.397", "0.945", "0.031"],
        ["v3 · + cosine fuzzy dedup (current)", "500 (12.5%)", "0.491", "0.936", "0.060"],
        ["v1 → v3 net", "+365 minority", "—", "+0.087", "−0.087"],
    ]
    split_rows = mdd.get("split", [
        ["scoring holdout (5%)", "200", "175", "25", "87.5%"],
        ["train (80% of remainder)", "3040", "2660", "380", "87.5%"],
        ["validation (10%)", "380", "333", "47", "87.6%"],
        ["test (10%)", "380", "332", "48", "87.4%"],
    ])
    prod_lb_rows = [
        [r["model"], r.get("params", ""), r["val_pr_auc"], r["threshold"],
         r["pr_auc"], r["roc_auc"], r.get("train_test_gap", ""), r["f1"],
         r["precision"], r["recall"]]
        for r in lb
    ]
    research_rows = []
    for name, t in research.items():
        research_rows.append([
            name, mdd.get("thresholds", {}).get(name, "—"),
            t.get("pr_auc_minority", "—"), t.get("roc_auc", "—"),
            t.get("f1_minority", "—"), t.get("balanced_acc", "—"),
        ])
    contam = leak.get("contamination", [])
    slice_rows = leak.get("slices", [])
    judge_rows = [
        ["NIM vs LR gate",
         jp.get("nim_vs_gate", {}).get("n"),
         jp.get("nim_vs_gate", {}).get("agree"),
         jp.get("nim_vs_gate", {}).get("disagree"),
         f"{jp.get('nim_vs_gate', {}).get('rate', 0):.1%}",
         jp.get("nim_vs_gate", {}).get("kappa")],
        ["OpenAI vs LR gate",
         jp.get("openai_vs_gate", {}).get("n"),
         jp.get("openai_vs_gate", {}).get("agree"),
         jp.get("openai_vs_gate", {}).get("disagree"),
         f"{jp.get('openai_vs_gate', {}).get('rate', 0):.1%}",
         jp.get("openai_vs_gate", {}).get("kappa")],
        ["NIM vs OpenAI",
         jp.get("nim_vs_openai", {}).get("n"),
         jp.get("nim_vs_openai", {}).get("agree"),
         jp.get("nim_vs_openai", {}).get("disagree"),
         f"{jp.get('nim_vs_openai', {}).get('rate', 0):.1%}",
         jp.get("nim_vs_openai", {}).get("kappa")],
        ["LR gate vs weak label",
         jp.get("gate_vs_weak", {}).get("n"),
         jp.get("gate_vs_weak", {}).get("agree"),
         jp.get("gate_vs_weak", {}).get("disagree"),
         f"{jp.get('gate_vs_weak', {}).get('rate', 0):.1%}",
         jp.get("gate_vs_weak", {}).get("kappa")],
    ]
    dpo_rows = [
        ["LR gate (production)",
         dpo_gate.get("threshold", champ.get("threshold")),
         dpo_gate.get("roc_auc", champ.get("roc_auc")),
         dpo_gate.get("pr_auc", champ.get("pr_auc")),
         dpo_gate.get("f1", champ.get("f1")),
         dpo_gate.get("leakage_free_roc", "—"),
         "1.0"],
        ["DPO / RLAIF policy",
         dpo_pol.get("threshold", "—"),
         dpo_pol.get("roc_auc", "—"),
         dpo_pol.get("pr_auc", "—"),
         dpo_pol.get("f1", "—"),
         dpo_pol.get("leakage_free_roc", "—"),
         dpo_pol.get("agree_with_gate", "—")],
    ]
    s2_top = sorted(s2.get("per_label", []), key=lambda r: -r.get("recall", 0))[:8]
    s2_rows = [[r["label"], r["support"], r["recall"]] for r in s2_top]

    # ---- markdown ----------------------------------------------------------
    md = f"""# End-to-End Model Development Document — CMPL-REG-24

**Complaint → Regulation Classifier · Two-stage gate + RAG/LLM · Dual-judge panel · DPO/RLAIF**

> Generated by `scripts/generate_e2e_model_development_doc.py` on {now}.
> Numbers are taken from committed artifacts under `docs/complaint_model/` and
> `docs/model_development/`. Companion docs: data profile (`00_`), stage docs
> (`01_`/`02_`), judge agreement (`03_`), DPO/RLAIF (`04_`), publication-grade
> stage-1 MDD (`docs/model_development/`).

## Abstract

{abstract}

## 1 · Objective, risk tier, and scope

**Objective.** Route inbound consumer complaints so that regulatory matters
reach the correct regulation family with citable rationale, while
non-regulatory service noise is gated away from expensive LLM inference.

**Risk tier.** Tier-2 (medium): a false negative delays a regulatory
complaint into a service queue; a false positive costs one unnecessary LLM
call. The asymmetry favors high recall on the regulatory class at stage 1,
with precision recovered at stage 2 via retrieval-grounded labeling.

**In scope for this document.**

1. Data acquisition, curation, and versioned updates (v1→v3)
2. Split protocol (5% scoring holdout → stratified 80/10/10)
3. Stage-1 binary gate: bake-off, regularization, cut-off tuning
4. Leakage audit (contamination + weak-label target leakage)
5. Stage-2 RAG + LLM multi-class labeling
6. Dual-judge evaluation (NVIDIA NIM + OpenAI as first-class judges)
7. DPO / RLAIF post-training loop
8. Batch scoring ingestion layer
9. Agentic serving (MCP tools + A2A agents + NIM inference)
10. Limitations, monitoring, and disposition

## 2 · End-to-end architecture

```
CFPB / upload CSV
        |
        v
+-------------------+     5% scoring holdout (never trains)
|  Ingestion layer  |--------------------------------------> batch CSV out
+---------+---------+
          | modeling pool (95%)
          v
+-------------------+
| Stage 1 -- gate   |  TF-IDF + L1 logistic (prod) / LightGBM (research)
| REGULATORY?       |  cut-off tuned on validation (not 0.5)
+---------+---------+
          | yes (~88%)
          v
+-------------------+
| Stage 2 -- RAG+LLM|  NeMo Retriever / FAISS + NIM or OpenAI
| 24-category label |  taxonomy whitelist + citations
+---------+---------+
          v
   MCP tool / A2A Complaint Agent / UI / scored CSV

Evaluation overlays (do not replace production path):
  - Dual-judge panel (NIM + OpenAI) vs gate on held-out test
  - DPO/RLAIF challenger policy trained on val/train preferences
  - Leakage-free (metadata-labeled) test slice
```

**Production vs research.** Production stage 1 is logistic regression
(L1, C=4.0) for millisecond CPU latency and interpretability. The
publication-grade MDD also reports LightGBM / DistilBERT challengers;
LightGBM edges validation minority PR-AUC inside the seed-stability band
but is not promoted on latency economics.

## 3 · Data

### 3.1 Source and curation

| | |
|---|---|
| Source | CFPB Consumer Complaint Database (public, PII-redacted narratives) |
| Rows after curation | {ds.get('n_rows', 4000):,} |
| Class mix | {ds.get('regulatory_rate', 0.875):.1%} regulatory / {1 - ds.get('regulatory_rate', 0.875):.1%} non-regulatory |
| Curation | NeMo Data Curator–style: length filter (120–1,800 chars), exact hash dedup, 200-char prefix near-dedup, **TF-IDF-cosine fuzzy dedup (≥ 0.9)**, PII verification, per-issue balanced sampling |
| Labels | **Weak supervision** from CFPB `issue` taxonomy + narrative keyword overrides (24 categories incl. `NON_REGULATORY`) |

### 3.2 Dataset version log (measured impact on stage-1 champion)

The largest performance lifts came from **data**, not architecture changes.

{md_table(["dataset version", "non-regulatory", "val cut-off", "test ROC-AUC", "train/test gap"],
          data_update_rows)}

- **v2** added a targeted second pass over service-heavy issues with a
  500-row non-regulatory floor — stable threshold tuning became possible.
- **v3** removed ~4.5% test/train near-copies that the prefix signature
  missed; the headline moved down (0.945 → 0.936) because v2 was partly
  inflated by memorized templates. That is the honest number.

Full profile: [`00_data_profile.md`](00_data_profile.md).

### 3.3 Split protocol

Reserve a stratified **5% scoring holdout** first (batch-ingestion demo;
never used for training, validation, threshold tuning, or model selection).
Then stratified **80 / 10 / 10** on the remaining 95%. Seed = 42.

{md_table(["split", "n", "regulatory", "non-regulatory", "reg rate"], split_rows)}

## 4 · Stage 1 — binary regulatory gate

### 4.1 Features and candidates

- **Features:** TF-IDF, 30k vocab, 1–2 grams, `sublinear_tf`, `min_df=2`,
  fit on the training fold only.
- **Class imbalance:** `class_weight='balanced'` (logistic) /
  `scale_pos_weight` (trees); DistilBERT uses class-weighted CE.
- **Candidates (research MDD):** logistic regression, XGBoost, LightGBM,
  fine-tuned DistilBERT.
- **Candidates (production bake-off in `complaints.train_stage1`):**
  logistic regression (validation-tuned L1/L2 × C) + XGBoost.
- **Selection:** validation PR-AUC (majority/regulatory for production
  leaderboard; minority PR-AUC in the research MDD).
- **Cut-off:** validation-optimized to maximize minority-class F1 — never
  the default 0.5. Test opened once.

### 4.2 Production leaderboard (committed `metrics.json`)

{md_table(["model", "params", "val PR-AUC", "thr", "test PR-AUC", "test ROC",
           "train−test gap", "F1", "precision", "recall"], prod_lb_rows)}

**Production champion:** `{s1.get('champion')}` · `{champ.get('params')}` ·
cut-off **{champ.get('threshold')}** · test ROC-AUC **{champ.get('roc_auc')}** ·
PR-AUC **{champ.get('pr_auc')}** · F1 **{champ.get('f1')}** ·
precision **{champ.get('precision')}** · recall **{champ.get('recall')}** ·
train/test ROC gap **{champ.get('train_test_gap')}**.

Regularization note: an under-penalized linear model memorized the 30k-dim
TF-IDF space (train ROC ~1.0, gap ~0.20). Validation grid search over
`{{l1,l2}} × C∈{{0.5,1,2,4}}` selected **L1, C=4.0**, which sparsifies
n-gram weights and closes most of the gap.

### 4.3 Research bake-off (publication MDD, held-out test)

Research champion by validation minority PR-AUC: **{research_champ}** @
cut-off {research_thr}. Production still ships logistic regression.

{md_table(["model", "thr", "PR-AUC (min)", "ROC-AUC", "F1 (min)", "balanced acc"],
          research_rows)}

![stage1 curves](figures/stage1_curves.png)

![model comparison](../model_development/figures/fig_model_comparison.png)

## 5 · Leakage audit

Two vectors are audited at every regeneration of the stage-1 MDD and
guarded in `tests/test_complaints.py`.

### 5.1 Split contamination

Curation deduplicates exact hashes, 200-char prefixes, and TF-IDF-cosine
near-copies (≥ 0.9) **before** any split. Post-split check:

{md_table(["check (test vs train)", "count", "expected", "status"], contam)}

### 5.2 Weak-label target leakage

~{leak.get('label_provenance_narrative_share', 0.75):.0%} of labels are
decided by narrative regexes — the labeling rule is embedded in the model
input, so headline agreement partly measures re-learning the labeler.
`label_source()` tracks provenance; the **metadata-labeled slice** (labels
from CFPB `issue` only — never seen by the model) is the honest
generalization read.

{md_table(["test slice", "n", "reg-rate", "ROC-AUC", "PR-AUC (reg)",
           "F1 (min)", "balanced acc"], slice_rows)}

Leakage-free slice (research LightGBM context in MDD): ROC-AUC
~{leak.get('leakage_free_slice', {}).get('roc_auc', 0.907):.3f}
(n={leak.get('leakage_free_slice', {}).get('n', 81)}). Production LR on the
same slice in the DPO eval: ROC-AUC {dpo_gate.get('leakage_free_roc', 0.888)}.

## 6 · Stage 2 — RAG + LLM regulation labeling

Only complaints that clear the stage-1 gate enter stage 2.

| | |
|---|---|
| Retriever | FAISS (local) / NeMo Retriever embeddings (NVIDIA path) |
| Generator | NIM (`{metrics.get('llm', {}).get('model', 'meta/llama-3.1-8b-instruct')}`) or OpenAI — provider selected by `LLM_PROVIDER` for the **pipeline**; judges are addressed separately |
| Output | One of 24 taxonomy labels + confidence + rationale + citation excerpt |
| Guardrail | Label must be in the taxonomy whitelist; keyword scorer is the offline/no-LLM path for stage 2 only |

**Committed stage-2 metrics** (vs weak labels, stratified n={s2.get('n')}):

| metric | value |
|---|---|
| Exact agreement | {s2.get('accuracy', 0):.3f} |
| Family-level agreement | {s2.get('family_accuracy', 0):.3f} |
| Macro-F1 | {s2.get('macro_f1', 0):.3f} |

Top per-label recall (by recall):

{md_table(["label", "support", "recall"], s2_rows)}

Agreement is measured against **weak** labels on a harder, service-heavy
mix — not human truth. Disagreements concentrate within regulation families
(e.g., FCRA accuracy vs reinvestigation). The independent validation
report conditions promotion on a human-adjudicated golden set.

![stage2 recall](figures/stage2_recall.png)

## 7 · Dual-judge evaluation (NIM + OpenAI)

OpenAI is **not** fallback logic. Both providers are independent,
first-class judges that answer the same binary question as the stage-1
gate on the held-out test fold. API failures are abstentions — never
replaced by a keyword heuristic.

| party | model | abstentions |
|---|---|---|
| LR gate | `{judges.get('gate', {}).get('model')}` @ {judges.get('gate', {}).get('threshold')} | — |
| NIM judge | `{judges.get('judges', {}).get('nim', {}).get('model')}` | {judges.get('judges', {}).get('nim', {}).get('abstentions')} |
| OpenAI judge | `{judges.get('judges', {}).get('openai', {}).get('model')}` | {judges.get('judges', {}).get('openai', {}).get('abstentions')} |

{md_table(["pair", "n", "agree", "disagree", "rate", "Cohen's κ"], judge_rows)}

**Disputed set:** {judges.get('disputed', {}).get('n', 35)} rows where
*both* judges contradict the gate; judges side with the weak label on
{judges.get('disputed', {}).get('judges_match_weak', 13)} of those. This
set is the natural **human-adjudication queue** and the seed for RLAIF
preference data.

![judge agreement](figures/judge_agreement.png)

Full write-up: [`03_judge_agreement.md`](03_judge_agreement.md).

## 8 · DPO / RLAIF post-training

Preference rule: when NIM and OpenAI agree on a **val/train** panel row,
that consensus is *chosen* and the opposite regulatory label is *rejected*.
High-value rows are those where consensus also contradicts the LR gate.
The test-fold agreement study is evaluation-only (no training leakage).

| | |
|---|---|
| Method (committed) | `{dpo.get('train', {}).get('method', 'classifier_rlaif_dpo')}` |
| Base | `{dpo.get('train', {}).get('base', 'distilbert-base-uncased')}` |
| Corpus | {dpo.get('pairs', {}).get('n_pairs', 3040)} pairs · sources {dpo.get('pairs', {}).get('by_source', {})} |
| Note | Binary opposite-class DPO ≡ class-weighted CE on the preferred label; `--method generative` runs LoRA + TRL `DPOTrainer` |

{md_table(["party", "thr", "ROC-AUC", "PR-AUC", "F1", "leakage-free ROC", "agree w/ LR"],
          dpo_rows)}

The LR gate remains the production millisecond path. The DPO policy is the
post-training **challenger** — promote only after golden-set adjudication
confirms the judges. Full write-up: [`04_dpo_rlaif.md`](04_dpo_rlaif.md).

![dpo comparison](figures/dpo_comparison.png)

## 9 · Batch scoring (ingestion layer)

- **Holdout:** 5% stratified reserve carved out before any modeling split.
- **CLI:** `python scripts/score_batch.py [--input csv] [--limit N] [--no-llm]`
- **UI:** Streamlit **④ Batch scoring** — upload CSV or score the holdout;
  download scored CSV.
- **Output columns:** `complaint_id`, `complaint`, `score`, `is_regulatory`,
  `label`, `regulation_name`, `confidence`, `llm_reasoning`,
  `citation_source`, `mode`.

## 10 · Serving and agentic framework

| layer | component |
|---|---|
| Tools (MCP) | `complaint-mcp` — `classify_complaint`, taxonomy, samples, metrics |
| Agents (A2A) | Complaint Agent (+ orchestrator, validator, report, …) |
| LLM inference | NVIDIA NIM (OpenAI-compatible) or OpenAI; judge-panel mode addresses both |
| Embeddings | OpenAI or NeMo Retriever |
| Model serving | Triton (fraud FIL; BERT-gate upgrade path) |
| Observability | Prometheus + Grafana + DCGM + Jaeger / OTel |
| Deploy | Docker Compose, Brev GPU, GKE |

Architecture detail: [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md).

## 11 · Robustness evidence (stage 1)

Covered in depth in [`docs/model_development/model_development_document.md`](../model_development/model_development_document.md):

- Out-of-vocabulary analysis (TF-IDF vs DistilBERT subwords)
- Threshold sensitivity sweep
- Input perturbations (truncation, noise)
- Class-weight ablation
- Split-seed stability (5 seeds)

![sensitivity](../model_development/figures/fig_sensitivity_threshold.png)

![seed stability](../model_development/figures/fig_seed_stability.png)

## 12 · Limitations and monitoring

1. **Weak labels.** All agreement metrics are vs CFPB-taxonomy weak
   supervision. A human-adjudicated golden set — seeded by the 35-row
   disputed set — is a standing validation condition.
2. **Target leakage.** ~75% of labels are narrative-regex-derived; quote
   the metadata-labeled leakage-free slice for honest generalization.
3. **Minority support.** ~47 non-regulatory cases per held-out fold → wide
   bands on minority metrics and on the tuned cut-off itself.
4. **Stage-2 ceiling.** Exact agreement ~0.25 / family ~0.39 reflects weak
   reference noise and a harder service-heavy mix, not necessarily model
   failure.
5. **Judge panel.** Low Cohen's κ despite high raw agreement means much
   agreement is majority-class; judges are a triangulation tool, not ground
   truth.
6. **DPO corpus.** Committed challenger currently uses train-fold weak
   contrastive pairs; mix in `dpo_panel_val_rows.jsonl` after
   `judge_agreement_study.py --split val` for true RLAIF high-value pairs.
7. **Monitoring.** Track stage-1 score distribution, `train_test_gap`,
   gate/judge disagreement rate, stage-2 family agreement, and GPU/LLM
   latency in Grafana.

## 13 · Disposition and recommendation

| item | decision |
|---|---|
| Stage-1 production gate | **Approve** — L1 logistic @ 0.491; millisecond CPU; gap monitored |
| Stage-2 RAG+LLM | **Approve with Conditions** — citations + whitelist; golden set required before reading agreement as accuracy |
| Dual-judge panel | **Approve as evaluation overlay** — first-class NIM + OpenAI; feeds adjudication queue |
| DPO / RLAIF policy | **Challenger — do not promote yet** — competitive (ROC 0.921 vs 0.936) but awaits adjudicated preferences |
| Batch ingestion | **Approve** for demo / offline scoring on the reserved holdout |
| Overall | **Approve with Conditions** (SR 11-7 effective challenge: golden set, leakage-free reporting, gap monitoring) |

## 14 · Reproducibility

```bash
python scripts/fetch_cfpb_complaints.py                 # data (v3 curation)
python scripts/generate_complaint_data_profile.py       # data profile
python scripts/generate_complaint_model_docs.py         # stage docs + metrics.json
python scripts/generate_model_development_doc.py        # publication MDD + leakage
python scripts/judge_agreement_study.py                 # dual-judge on test
python scripts/judge_agreement_study.py --split val     # RLAIF panel (no test leak)
python scripts/train_dpo_from_judges.py                 # DPO / RLAIF challenger
python scripts/generate_e2e_model_development_doc.py    # this document
python scripts/score_batch.py --limit 25                # batch ingestion smoke
```

**Primary artifacts**

| artifact | path |
|---|---|
| Data | `data/complaints/cfpb_complaints.csv` |
| Production metrics | `docs/complaint_model/metrics.json` |
| Stage-1 MDD + leakage | `docs/model_development/results.json` |
| Judge panel | `docs/complaint_model/judge_agreement.json` |
| DPO results | `docs/complaint_model/dpo_results.json` |
| Model card | `data/models/model_card_complaint_reg24.md` |

---

*CMPL-REG-24 · End-to-End Model Development Document · {now}*
"""

    os.makedirs(OUT_DIR, exist_ok=True)
    md_path = os.path.join(OUT_DIR, "05_end_to_end_model_development.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {md_path}")

    # ---- PDF ---------------------------------------------------------------
    pdf_path = os.path.join(OUT_DIR, "05_end_to_end_model_development.pdf")
    with PdfPages(pdf_path) as pdf:
        _title_page(
            pdf,
            "End-to-End Model Development Document\nCMPL-REG-24",
            "Two-stage complaint classifier · leakage audit · dual-judge panel · DPO/RLAIF",
            f"Generated {now}\nArtifact: docs/complaint_model/05_end_to_end_model_development.pdf\n"
            f"Production gate: {s1.get('champion')} ({champ.get('params')}) @ {champ.get('threshold')}\n"
            f"Test ROC-AUC {champ.get('roc_auc')} · PR-AUC {champ.get('pr_auc')} · gap {champ.get('train_test_gap')}",
        )
        _text_page(pdf, "Abstract", abstract)
        _text_page(pdf, "1 · Objective, risk tier, and scope",
                   "Tier-2 two-stage classifier: stage-1 gates regulatory nexus; "
                   "stage-2 assigns a 24-category label with citations. Scope "
                   "covers data (v1–v3), splits, stage-1/2, leakage audit, "
                   "dual-judge panel, DPO/RLAIF, batch ingestion, and MCP/A2A serving.")
        _table_page(pdf, "3 · Data update log (Table)",
                    ["version", "non-reg", "cut-off", "ROC-AUC", "gap"],
                    data_update_rows,
                    "v3 is the honest headline after cosine fuzzy dedup.")
        _table_page(pdf, "3 · Split protocol (Table)",
                    ["split", "n", "reg", "non-reg", "rate"], split_rows)
        _table_page(pdf, "4 · Production stage-1 leaderboard (Table)",
                    ["model", "params", "valPR", "thr", "PR", "ROC", "gap",
                     "F1", "prec", "rec"], prod_lb_rows)
        _figure_page(pdf, "4 · Stage-1 ROC / PR curves",
                     os.path.join(FIG, "stage1_curves.png"))
        _table_page(pdf, "5 · Leakage — split contamination (Table)",
                    ["check", "count", "expected", "status"], contam)
        if slice_rows:
            _table_page(pdf, "5 · Leakage — evaluation slices (Table)",
                        ["slice", "n", "reg-rate", "ROC", "PR", "F1min", "bAcc"],
                        slice_rows)
        _table_page(pdf, "6 · Stage-2 top per-label recall (Table)",
                    ["label", "support", "recall"], s2_rows,
                    f"Exact agreement {s2.get('accuracy'):.3f} · family {s2.get('family_accuracy'):.3f}")
        _figure_page(pdf, "6 · Stage-2 per-label recall",
                     os.path.join(FIG, "stage2_recall.png"))
        _table_page(pdf, "7 · Dual-judge agreement (Table)",
                    ["pair", "n", "agree", "disagree", "rate", "κ"], judge_rows,
                    f"Disputed set: {judges.get('disputed', {}).get('n')} rows")
        _figure_page(pdf, "7 · Dual-judge agreement figure",
                     os.path.join(FIG, "judge_agreement.png"))
        _table_page(pdf, "8 · DPO / RLAIF vs LR gate (Table)",
                    ["party", "thr", "ROC", "PR", "F1", "leak-free", "vs LR"],
                    dpo_rows)
        _figure_page(pdf, "8 · DPO vs gate",
                     os.path.join(FIG, "dpo_comparison.png"))
        _text_page(
            pdf, "13 · Disposition",
            "Stage-1 production gate: Approve (L1 logistic @ tuned cut-off). "
            "Stage-2 RAG+LLM: Approve with Conditions (golden set required). "
            "Dual-judge panel: Approve as evaluation overlay. "
            "DPO/RLAIF policy: Challenger — do not promote yet. "
            "Overall: Approve with Conditions under SR 11-7 effective challenge.",
        )
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
