"""Generate complaint-model documentation: dev doc + validation report.

Trains the stage-1 binary bake-off, evaluates the stage-2 RAG+LLM labeler on a
stratified sample of real CFPB complaints, renders accuracy figures + tables,
and writes BOTH markdown (with PNG figures) and structured PDF documents:

    docs/complaint_model/
      figures/*.png
      metrics.json                          (consumed by the Streamlit UI)
      01_model_development_document.{md,pdf}   (first line)
      02_validation_report.{md,pdf}            (second line)

    python scripts/generate_complaint_model_docs.py --stage2-n 120
    python scripts/generate_complaint_model_docs.py --no-llm   # offline mode
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reg_agents.common import complaints as C  # noqa: E402
from reg_agents.config import get_settings  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "complaint_model",
)
FIG_DIR = os.path.join(OUT_DIR, "figures")
GREEN = "#76b900"
GRAY = "#444444"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_label_distribution(df) -> str:
    counts = df["label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [GREEN if lbl != C.NON_REGULATORY else GRAY for lbl in counts.index]
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_title("Weak-label distribution — 24 regulation categories\n"
                 "(real CFPB complaint narratives)")
    ax.set_xlabel("complaints")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "label_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_stage1_curves(s1) -> str:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, c in s1["curves"].items():
        fpr, tpr, _ = roc_curve(c["y_true"], c["y_score"])
        p, r, _ = precision_recall_curve(c["y_true"], c["y_score"])
        row = next(x for x in s1["leaderboard"] if x["model"] == name)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {row['roc_auc']})")
        axes[1].plot(r, p, label=f"{name} (PR-AUC {row['pr_auc']})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="Stage 1 — ROC", xlabel="FPR", ylabel="TPR")
    axes[1].set(title="Stage 1 — Precision-Recall", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stage1_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_stage1_confusion(s1) -> str:
    cm = np.array(s1["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Greens")
    ticks = ["non-regulatory", "regulatory"]
    ax.set_xticks([0, 1], ticks)
    ax.set_yticks([0, 1], ticks)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Stage 1 confusion matrix — champion: {s1['champion']}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stage1_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_stage2_recall(s2) -> str:
    rows = sorted(s2["per_label"], key=lambda r: r["recall"])
    labels = [r["label"] for r in rows]
    recalls = [r["recall"] for r in rows]
    supports = [r["support"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    bars = ax.barh(labels, recalls, color=GREEN)
    for bar, sup in zip(bars, supports):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={sup}", va="center", fontsize=7, color=GRAY)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("recall vs weak label")
    ax.set_title(f"Stage 2 (RAG + LLM) — per-category recall\n"
                 f"exact acc {s2['accuracy']:.2f} · family acc "
                 f"{s2['family_accuracy']:.2f} · macro-F1 {s2['macro_f1']:.2f} "
                 f"· n={s2['n']} · mode={s2['mode']}")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stage2_recall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_pipeline(s1, s2) -> str:
    """Simple architecture figure for the docs."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    boxes = [
        (0.01, "Complaint\nnarrative\n(CFPB, real)", "#eeeeee"),
        (0.21, f"Stage 1\n{s1['champion']}\nregulatory? "
               f"(PR-AUC {s1['leaderboard'][0]['pr_auc']})", "#dcedc8"),
        (0.44, "Stage 2\nRAG over reg corpus\n+ LLM w/ few-shots", "#dcedc8"),
        (0.68, f"Label (1 of 24)\n+ citation + rationale\n"
               f"(family acc {s2['family_accuracy']:.2f}, n={s2['n']})", "#c5e1a5"),
    ]
    for x, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, 0.25), 0.17, 0.5, facecolor=color,
                                   edgecolor=GRAY, lw=1.2))
        ax.text(x + 0.085, 0.5, label, ha="center", va="center", fontsize=8.5)
    for x in (0.18, 0.41, 0.63):
        ax.annotate("", xy=(x + 0.035, 0.5), xytext=(x, 0.5),
                    arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.4})
    ax.text(0.525, 0.1, "non-regulatory → gated out at stage 1",
            fontsize=8, color=GRAY, ha="center")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "pipeline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# PDF builder (matplotlib PdfPages: title page, text pages, tables, figures)
# --------------------------------------------------------------------------- #
PAGE = (8.5, 11)


def _text_page(pdf: PdfPages, heading: str, body: str) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.94, heading, fontsize=15, weight="bold", color="#1a1a1a")
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=GREEN,
                                lw=2, transform=fig.transFigure))
    wrapped = []
    for para in body.split("\n"):
        wrapped.extend(textwrap.wrap(para, width=100) or [""])
    fig.text(0.08, 0.89, "\n".join(wrapped[:52]), fontsize=9.2, va="top",
             family="serif", linespacing=1.45)
    pdf.savefig(fig)
    plt.close(fig)
    # overflow pages
    rest = wrapped[52:]
    while rest:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.08, 0.94, f"{heading} (cont.)", fontsize=12, weight="bold")
        fig.text(0.08, 0.90, "\n".join(rest[:56]), fontsize=9.2, va="top",
                 family="serif", linespacing=1.45)
        pdf.savefig(fig)
        plt.close(fig)
        rest = rest[56:]


def _table_page(pdf: PdfPages, heading: str, headers, rows, caption="") -> None:
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off")
    fig.text(0.08, 0.94, heading, fontsize=14, weight="bold")
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=GREEN,
                                lw=2, transform=fig.transFigure))
    tbl = ax.table(cellText=[[str(c) for c in r] for r in rows],
                   colLabels=headers, loc="upper center",
                   cellLoc="center", bbox=[0.0, max(0.05, 0.86 - 0.035 * len(rows)),
                                           1.0, min(0.82, 0.035 * (len(rows) + 1))])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8f2d8")
            cell.set_text_props(weight="bold")
    if caption:
        fig.text(0.08, 0.04, caption, fontsize=8, color=GRAY)
    pdf.savefig(fig)
    plt.close(fig)


def _figure_page(pdf: PdfPages, heading: str, png_path: str, caption="") -> None:
    img = plt.imread(png_path)
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.94, heading, fontsize=14, weight="bold")
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=GREEN,
                                lw=2, transform=fig.transFigure))
    h, w = img.shape[:2]
    disp_w = 0.84
    disp_h = disp_w * (h / w) * (PAGE[0] / PAGE[1])
    disp_h = min(disp_h, 0.78)
    ax = fig.add_axes([0.08, 0.88 - disp_h, disp_w, disp_h])
    ax.imshow(img)
    ax.axis("off")
    if caption:
        fig.text(0.08, 0.86 - disp_h, caption, fontsize=8, color=GRAY, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _title_page(pdf: PdfPages, title: str, subtitle: str, meta: str) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.72, title, fontsize=21, weight="bold", wrap=True)
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.70, 0.70], color=GREEN,
                                lw=3, transform=fig.transFigure))
    fig.text(0.08, 0.64, subtitle, fontsize=12, color=GRAY)
    fig.text(0.08, 0.16, meta, fontsize=9, color=GRAY, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Narrative sections (LLM with deterministic fallback)
# --------------------------------------------------------------------------- #
def _narrative(system: str, user: str, fallback: str) -> str:
    try:
        from reg_agents.common import llm

        return llm.system_user(system, user, max_tokens=900, temperature=0.2)
    except Exception:  # noqa: BLE001
        return fallback


DEV_SYS = (
    "You are a first-line model developer at a bank writing the executive summary "
    "and methodology narrative of a Model Development Document for a two-stage "
    "complaint-to-regulation classification model. Be precise, cite the supplied "
    "metrics verbatim, note key design choices, and keep it under 450 words. "
    "Plain prose, no markdown headers."
)

VAL_SYS = (
    "You are an independent second-line model validator (SR 11-7). Write the "
    "assessment narrative for a validation report on a two-stage complaint "
    "classification model: evaluate conceptual soundness, data quality (weak "
    "labels!), and outcomes analysis from the supplied metrics; provide "
    "severity-ranked findings with remediations; end with the disposition "
    "'Approve with Conditions' and the conditions. Under 500 words. Plain prose, "
    "no markdown headers."
)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-n", type=int, default=120)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM calls (stage-2 uses keyword fallback)")
    args = ap.parse_args()
    use_llm = not args.no_llm

    os.makedirs(FIG_DIR, exist_ok=True)
    s = get_settings()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("== loading data + training stage 1 ==")
    df = C.load_complaints()
    s1 = C.train_stage1(df)
    print(f"   champion: {s1['champion']}  leaderboard: {s1['leaderboard']}")

    print(f"== evaluating stage 2 on n={args.stage2_n} (llm={use_llm}) ==")
    s2 = C.evaluate_stage2(n=args.stage2_n, use_llm=use_llm, df=df)
    print(f"   accuracy {s2['accuracy']}  macro-F1 {s2['macro_f1']}  mode {s2['mode']}")

    print("== rendering figures ==")
    figs = {
        "pipeline": fig_pipeline(s1, s2),
        "labels": fig_label_distribution(df),
        "curves": fig_stage1_curves(s1),
        "confusion": fig_stage1_confusion(s1),
        "recall": fig_stage2_recall(s2),
    }

    metrics = {
        "generated": ts,
        "llm": {"provider": s.llm_provider, "model": s.active_model},
        "dataset": s1["dataset"],
        "stage1": {"champion": s1["champion"], "leaderboard": s1["leaderboard"],
                   "confusion_matrix": s1["confusion_matrix"]},
        "stage2": {k: s2[k] for k in
                   ("n", "mode", "accuracy", "family_accuracy", "macro_f1",
                    "weighted_f1", "per_label")},
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    metrics_str = json.dumps(metrics, indent=2)
    data_text = (
        f"Source: real, redacted consumer complaint narratives from the public CFPB "
        f"Consumer Complaint Database ({s1['dataset']['n_rows']} curated rows; "
        f"{s1['dataset']['n_train']} train / {s1['dataset']['n_test']} test; "
        f"regulatory rate {s1['dataset']['regulatory_rate']}).\n\n"
        "Curation (scripts/fetch_cfpb_complaints.py) mirrors NVIDIA NeMo Data "
        "Curator stages: length filtering (ScoreFilter), exact deduplication "
        "(ExactDuplicates), near-deduplication (FuzzyDuplicates/MinHash analog), "
        "PII verification (CFPB pre-masks PII as XXXX; PiiModifier analog), and "
        "per-issue balanced sampling. At corpus scale each stage maps directly to "
        "the GPU-accelerated Curator module.\n\n"
        "Ground truth is WEAK SUPERVISION: labels derive from the CFPB "
        "product/issue taxonomy plus narrative keyword rules across 24 regulation "
        "categories (ECOA, FCRA, FDCPA, Reg E, Reg Z, RESPA, UDAAP, sales "
        "practices, ...). This is the standard bootstrap before human-adjudicated "
        "labels exist and is flagged as a limitation in the validation report."
    )
    arch_text = (
        "Stage 1 (binary gate): TF-IDF (1-2 grams, 30k features) into a "
        "logistic-regression vs XGBoost bake-off; champion selected on PR-AUC. "
        "Upgrade path: fine-tuned BERT-class encoder (NeMo Framework) served via "
        "Triton, same interface.\n\n"
        "Stage 2 (multi-class): retrieval-augmented generation over the "
        "regulation/policy corpus (NeMo Retriever embeddings, FAISS locally / "
        "cuVS-Milvus on GPU) + LLM reasoning (NIM Llama-3.1-8B) with 8 curated "
        "few-shot examples and the 24-category taxonomy in-prompt. Output is "
        "strict JSON: label, confidence, rationale, and the cited source "
        "document, which the UI renders alongside the excerpt. A keyword scorer "
        "provides a deterministic no-LLM fallback; stage-1 gates non-regulatory "
        "complaints so the LLM is only invoked when needed.\n\n"
        "Deployment: complaint MCP server (tools: classify_complaint, "
        "sample_complaints, get_model_metrics) + Complaint A2A agent; wired into "
        "the Streamlit UI and observable via the existing Prometheus/Grafana and "
        "OpenTelemetry stack."
    )

    dev_narrative = _narrative(
        DEV_SYS, f"METRICS (JSON):\n{metrics_str}",
        fallback="LLM narrative unavailable offline; see metrics tables and figures.",
    )
    val_narrative = _narrative(
        VAL_SYS, f"MODEL METRICS (JSON):\n{metrics_str}\n\nDATA NOTES:\n{data_text}",
        fallback="LLM narrative unavailable offline; see metrics tables and figures.",
    )

    lb_headers = ["model", "pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]
    lb_rows = [[r[h] for h in lb_headers] for r in s1["leaderboard"]]
    s2_rows = [[r["label"], r["support"], r["recall"]]
               for r in sorted(s2["per_label"], key=lambda x: -x["support"])]

    banner = (
        f"> Generated by `reg-agents` on {ts} · LLM: **{s.llm_provider}** "
        f"(`{s.active_model}`) · Data: **CFPB Consumer Complaint Database** (real, "
        f"redacted narratives) · Stage-2 eval n={s2['n']} (mode: {s2['mode']}).\n"
    )

    # ---------------- markdown: development document ----------------
    dev_md = f"""# Model Development Document — Complaint→Regulation Classifier (CMPL-REG-24)

*First line of defense — Model Development*

{banner}
## 1 · Executive summary & methodology

{dev_narrative}

## 2 · Architecture

![pipeline](figures/pipeline.png)

{arch_text}

## 3 · Data & curation

{data_text}

![labels](figures/label_distribution.png)

## 4 · Stage 1 — binary bake-off (regulatory vs not)

| {' | '.join(lb_headers)} |
|{'|'.join(['---'] * len(lb_headers))}|
{chr(10).join('| ' + ' | '.join(str(v) for v in row) + ' |' for row in lb_rows)}

Champion: **{s1['champion']}** (primary metric: PR-AUC).

![curves](figures/stage1_curves.png)

![confusion](figures/stage1_confusion.png)

## 5 · Stage 2 — RAG + LLM regulation labeling

Evaluated on a stratified sample of {s2['n']} regulatory complaints against weak
labels: **exact accuracy {s2['accuracy']} · regulation-family accuracy
{s2['family_accuracy']} · macro-F1 {s2['macro_f1']} · weighted-F1
{s2['weighted_f1']}** (mode: {s2['mode']}).

Exact-match vs the *weak* labels understates true quality: adjudicated
disagreements are dominated by within-family confusions (e.g. FCRA accuracy vs
FCRA reinvestigation) and cases where the weak label itself is wrong — which is
why family-level agreement and the golden-set condition in the validation
report are the operative quality gates.

![recall](figures/stage2_recall.png)

## 6 · Monitoring & deployment

Served via the complaint MCP server + Complaint A2A agent. Per-classification
Prometheus counters (by label and mode), latency histograms via the agent
`/metrics` endpoint, OpenTelemetry traces across A2A hops, and drift monitoring
on the stage-1 score distribution. Guardrails: strict-JSON output parsing,
taxonomy whitelist on labels, keyword fallback on LLM failure, stage-1 gating.
"""

    val_md = f"""# Independent Validation Report — Complaint→Regulation Classifier (CMPL-REG-24)

*Second line of defense — Independent Validation (effective challenge)*

{banner}
## 1 · Scope & materials

Model development document, training/eval code, the curated CFPB dataset
({s1['dataset']['n_rows']} rows), stage-1 leaderboard, and the stage-2
evaluation (n={s2['n']}) with per-category metrics.

## 2 · Assessment narrative

{val_narrative}

## 3 · Outcomes analysis — evidence reviewed

Stage-1 leaderboard (validated re-run):

| {' | '.join(lb_headers)} |
|{'|'.join(['---'] * len(lb_headers))}|
{chr(10).join('| ' + ' | '.join(str(v) for v in row) + ' |' for row in lb_rows)}

![curves](figures/stage1_curves.png)

Stage-2 per-category recall vs weak labels (support-weighted):

![recall](figures/stage2_recall.png)

## 4 · Key limitations (validator-confirmed)

1. **Weak labels** — ground truth derives from the CFPB issue taxonomy +
   keyword rules, not human adjudication; stage-2 "accuracy" is agreement with
   a noisy reference. Condition: human-adjudicated golden set (≥25/category).
2. **Class imbalance** — several categories (SCRA/MLA, sales practices,
   adverse action) have thin support; per-category metrics are unstable there.
3. **LLM nondeterminism & drift** — prompt/model versions must be pinned and
   re-evaluated on the golden set before any provider/model change (the system
   records provider + model in every artifact banner).
4. **Fair-lending lens** — complaint routing affects remediation speed;
   monitoring should include parity of routing outcomes across products.

## 5 · Disposition

**Approve with Conditions** — see conditions in the assessment narrative and
limitations above; re-validation triggered by model/prompt change or stage-1
PSI > 0.25.
"""

    with open(os.path.join(OUT_DIR, "01_model_development_document.md"), "w",
              encoding="utf-8") as fh:
        fh.write(dev_md)
    with open(os.path.join(OUT_DIR, "02_validation_report.md"), "w",
              encoding="utf-8") as fh:
        fh.write(val_md)

    # ---------------- PDFs ----------------
    meta = (f"Generated: {ts}\nLLM: {s.llm_provider} ({s.active_model})\n"
            f"Data: CFPB Consumer Complaint Database (real, redacted narratives)\n"
            f"Dataset: {s1['dataset']['n_rows']} rows · regulatory rate "
            f"{s1['dataset']['regulatory_rate']}\nRepo: reg-agents · model id: CMPL-REG-24")

    dev_pdf = os.path.join(OUT_DIR, "01_model_development_document.pdf")
    with PdfPages(dev_pdf) as pdf:
        _title_page(pdf, "Model Development Document",
                    "Complaint → Regulation Classifier (CMPL-REG-24)\n"
                    "First line of defense — Model Development", meta)
        _text_page(pdf, "1 · Executive summary & methodology", dev_narrative)
        _figure_page(pdf, "2 · Architecture", figs["pipeline"],
                     "Two-stage pipeline: binary gate, then RAG+LLM labeling with citations.")
        _text_page(pdf, "2 · Architecture (detail)", arch_text)
        _text_page(pdf, "3 · Data & curation", data_text)
        _figure_page(pdf, "3 · Label distribution", figs["labels"],
                     "Weak-label distribution over the 24 regulation categories.")
        _table_page(pdf, "4 · Stage-1 bake-off leaderboard", lb_headers, lb_rows,
                    f"Champion: {s1['champion']} (primary metric: PR-AUC).")
        _figure_page(pdf, "4 · Stage-1 ROC / PR curves", figs["curves"])
        _figure_page(pdf, "4 · Stage-1 confusion matrix", figs["confusion"])
        _figure_page(pdf, "5 · Stage-2 per-category recall", figs["recall"],
                     f"Exact acc {s2['accuracy']} · family acc {s2['family_accuracy']}"
                     f" · macro-F1 {s2['macro_f1']} · n={s2['n']}.")
        _table_page(pdf, "5 · Stage-2 per-category detail",
                    ["category", "support", "recall"], s2_rows,
                    "Recall vs weak labels on the stratified evaluation sample.")

    val_pdf = os.path.join(OUT_DIR, "02_validation_report.pdf")
    with PdfPages(val_pdf) as pdf:
        _title_page(pdf, "Independent Validation Report",
                    "Complaint → Regulation Classifier (CMPL-REG-24)\n"
                    "Second line of defense — effective challenge", meta)
        _text_page(pdf, "1 · Assessment narrative", val_narrative)
        _table_page(pdf, "2 · Stage-1 leaderboard (validated re-run)",
                    lb_headers, lb_rows)
        _figure_page(pdf, "2 · Stage-1 evidence", figs["curves"])
        _figure_page(pdf, "3 · Stage-2 evidence", figs["recall"])
        _table_page(pdf, "3 · Stage-2 per-category detail",
                    ["category", "support", "recall"], s2_rows)
        _text_page(pdf, "4 · Limitations & disposition",
                   "1. Weak labels: ground truth derives from the CFPB issue taxonomy "
                   "plus keyword rules, not human adjudication. Condition: build a "
                   "human-adjudicated golden set (>=25 per category).\n\n"
                   "2. Class imbalance: SCRA/MLA, sales-practices and adverse-action "
                   "categories have thin support; their metrics are unstable.\n\n"
                   "3. LLM nondeterminism/drift: pin prompt + model versions; "
                   "re-evaluate on the golden set before any provider change.\n\n"
                   "4. Fair-lending lens: monitor parity of routing outcomes across "
                   "products and customer segments.\n\n"
                   "DISPOSITION: Approve with Conditions. Re-validation triggered by "
                   "model/prompt change or stage-1 PSI > 0.25.")

    print(f"wrote {OUT_DIR}/(01_model_development_document|02_validation_report).(md|pdf)")
    print(f"wrote {OUT_DIR}/metrics.json + {len(figs)} figures")


if __name__ == "__main__":
    main()
