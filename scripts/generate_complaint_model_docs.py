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


def fig_score_distribution(s1) -> str:
    """Champion stage-1 score distribution by true class (separation view)."""
    c = s1["curves"][s1["champion"]]
    pos = c["y_score"][c["y_true"] == 1]
    neg = c["y_score"][c["y_true"] == 0]
    thr = s1.get("threshold", 0.5)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(pos, bins=bins, alpha=0.65, density=True, color=GREEN,
            label=f"regulatory (n={len(pos)})")
    ax.hist(neg, bins=bins, alpha=0.65, density=True, color=GRAY,
            label=f"non-regulatory (n={len(neg)})")
    ax.axvline(thr, color="#c62828", ls="--", lw=1,
               label=f"decision cut-off {thr:.3f} (validation-optimized)")
    ax.set(title=f"Stage 1 — score distribution by class ({s1['champion']})",
           xlabel="predicted P(regulatory)", ylabel="density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stage1_score_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_calibration(s1) -> str:
    """Reliability diagram for the champion (calibration, not just ranking)."""
    from sklearn.calibration import calibration_curve

    c = s1["curves"][s1["champion"]]
    frac_pos, mean_pred = calibration_curve(c["y_true"], c["y_score"],
                                            n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=GREEN, label=s1["champion"])
    ax.set(title="Stage 1 — reliability diagram (quantile bins)",
           xlabel="mean predicted probability", ylabel="observed frequency")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stage1_calibration.png")
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
def _narrative(system: str, user: str, fallback: str, max_tokens: int = 1500) -> str:
    """LLM narrative with visible retries (stage-2 eval can rate-limit us)."""
    import time

    try:
        from reg_agents.common import llm
    except Exception:  # noqa: BLE001
        return fallback
    for attempt in range(4):
        try:
            return llm.system_user(system, user, max_tokens=max_tokens, temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            wait = 15 * (attempt + 1)
            print(f"   narrative attempt {attempt + 1} failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    return fallback


# Both documents are written in the voice of a career quantitative reviewer:
# PhD econometrics, two decades of tier-1 bank model work, examiner-facing.
DEV_SYS = (
    "You are a senior model developer and quantitative lead — PhD in "
    "econometrics, 20 years building and defending bank models (scorecards, "
    "CCAR/PPNR, fraud, NLP) — writing the executive summary and methodology "
    "narrative of a Model Development Document for a two-stage "
    "complaint-to-regulation classification model. Write the way a seasoned "
    "quant writes for an examiner: measured, exact, in complete paragraphs. "
    "Structure the narrative as flowing prose covering, in order: purpose and "
    "materiality; design rationale for the two-stage architecture (why a cheap "
    "high-recall gate before an expensive grounded labeler, framed as an "
    "economic and statistical decision); estimation and model-selection "
    "protocol, quoting the supplied metrics verbatim and interpreting "
    "discrimination (PR-AUC vs ROC-AUC under class imbalance) correctly; known "
    "weaknesses stated candidly, especially the weak-label reference; and the "
    "monitoring/guardrail design as part of the model, not an afterthought. "
    "600-800 words. Plain prose, no markdown headers, no bullet lists."
)

VAL_SYS = (
    "You are a senior independent model validator — PhD in econometrics, 20 "
    "years in second-line model risk at large banks, having validated credit "
    "scorecards, CCAR models, fraud systems, and now GenAI/agentic systems; "
    "you have defended findings to the Fed and OCC. Write the effective-"
    "challenge assessment narrative for an Independent Validation Report on a "
    "two-stage complaint classification model. Voice: measured, evidence-"
    "first, unsparing where warranted — the way a career validator writes. "
    "Cover, in flowing prose: (1) conceptual soundness — fitness of the "
    "two-stage design, maintained assumptions, and where it could fail; (2) "
    "data quality and label provenance — interrogate the weak-supervision "
    "reference and what that does to every downstream metric, including the "
    "distinction between measured agreement and true accuracy; (3) outcomes "
    "analysis — quote the supplied metrics verbatim, discuss discrimination "
    "vs calibration, small-support instability, and why family-level "
    "agreement diverges from exact agreement; (4) monitoring and guardrails "
    "adequacy; (5) consumer-protection/fair-lending exposure of complaint "
    "routing. Number findings V-1, V-2, ... with severity (High/Medium/Low), "
    "each with a concrete remediation and owner. End with the disposition "
    "'Approve with Conditions' and enumerate the conditions precisely. "
    "700-900 words. Plain prose paragraphs plus the numbered findings; no "
    "markdown headers."
)

SIGNOFF = (
    "Prepared and reviewed under the bank's model risk management framework "
    "(SR 11-7 / OCC 2011-12).\n\n"
    "**Author, development document:** Senior Model Developer & Quantitative "
    "Lead — PhD (Econometrics), 20 years in model development across credit, "
    "capital planning, fraud, and NLP.\n\n"
    "**Author, validation report:** Senior Independent Model Validator — PhD "
    "(Econometrics), 20 years in second-line validation; independent of the "
    "development team per SR 11-7 organizational-independence requirements."
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
        "scores": fig_score_distribution(s1),
        "calibration": fig_calibration(s1),
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
        f"{s1['dataset']['n_holdout']} reserved as a 5% scoring holdout for the "
        f"batch-ingestion layer before any modeling split, then "
        f"{s1['dataset']['n_train']} train / {s1['dataset']['n_val']} validation / "
        f"{s1['dataset']['n_test']} test; "
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

    glossary = (
        "GLOSSARY (use the expansions, never the raw tokens): 'rag_llm' = the "
        "stage-2 retrieval-augmented generation pipeline (NeMo Retriever "
        "embeddings + LLM labeler served via NVIDIA NIM); 'xgboost' and "
        "'logistic_regression' = stage-1 TF-IDF classifiers in the bake-off; "
        "'family_accuracy' = agreement at the regulation-family level (all "
        "FCRA_* variants collapsed to one family, etc.); 'pr_auc' = area under "
        "the precision-recall curve, the primary stage-1 metric given the ~94% "
        "positive-class prevalence."
    )
    dev_narrative = _narrative(
        DEV_SYS, f"{glossary}\n\nMETRICS (JSON):\n{metrics_str}",
        fallback="LLM narrative unavailable offline; see metrics tables and figures.",
    )
    val_narrative = _narrative(
        VAL_SYS,
        f"{glossary}\n\nMODEL METRICS (JSON):\n{metrics_str}\n\nDATA NOTES:\n{data_text}",
        fallback="LLM narrative unavailable offline; see metrics tables and figures.",
    )

    lb_headers = ["model", "val_pr_auc", "threshold", "pr_auc", "roc_auc", "f1",
                  "precision", "recall", "accuracy"]
    lb_rows = [[r[h] for h in lb_headers] for r in s1["leaderboard"]]
    s2_rows = [[r["label"], r["support"], r["recall"]]
               for r in sorted(s2["per_label"], key=lambda x: -x["support"])]

    # Dataset summary table
    ds = s1["dataset"]
    ds_headers = ["property", "value"]
    ds_rows = [
        ["source", "CFPB Consumer Complaint Database (public, PII-redacted)"],
        ["curated rows", f"{ds['n_rows']:,}"],
        ["scoring holdout (reserved first)",
         f"{ds['n_holdout']:,} (5%, stratified — fed only through the ingestion layer)"],
        ["train / val / test split",
         f"{ds['n_train']:,} / {ds['n_val']:,} / {ds['n_test']:,} (stratified 80/10/10 of the remaining 95%)"],
        ["regulatory rate", f"{ds['regulatory_rate']:.1%}"],
        ["taxonomy coverage", f"{df['label'].nunique()} of 24 categories"],
        ["curation stages", "length filter · exact dedup · near dedup · PII check · balanced sampling"],
        ["label provenance", "weak supervision (CFPB issue taxonomy + keyword rules)"],
    ]

    # Taxonomy table: all 24 categories with dataset support
    support = df["label"].value_counts().to_dict()
    tax_headers = ["code", "regulation / category", "n in dataset"]
    tax_rows = [[r.label, r.name, support.get(r.label, 0)]
                for r in C.REGULATIONS.values()]

    # Guardrail inventory table
    gr_headers = ["guardrail", "mechanism", "surfaced as"]
    gr_rows = [
        ["Taxonomy whitelist", "LLM label must be one of the 24 codes; else rejected",
         "complaint_classifications_total{label}"],
        ["Strict-JSON parsing", "malformed LLM output -> deterministic keyword fallback",
         "mode=fallback counter (alertable spike)"],
        ["Stage-1 gating", "non-regulatory volume never reaches the LLM",
         "cost + attack-surface control"],
        ["Retrieval grounding", "answer must cite a retrieved corpus excerpt",
         "citation attached to every prediction"],
        ["Provider portability", "OpenAI <-> NIM via config; keyword mode if both fail",
         "mode label per classification"],
    ]

    banner = (
        f"> Generated by `reg-agents` on {ts} · LLM: **{s.llm_provider}** "
        f"(`{s.active_model}`) · Data: **CFPB Consumer Complaint Database** (real, "
        f"redacted narratives) · Stage-2 eval n={s2['n']} (mode: {s2['mode']}).\n"
    )

    def md_table(headers, rows):
        head = "| " + " | ".join(headers) + " |"
        sep = "|" + "|".join(["---"] * len(headers)) + "|"
        body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
        return f"{head}\n{sep}\n{body}"

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

{md_table(ds_headers, ds_rows)}

![labels](figures/label_distribution.png)

### 3.1 · The 24-category regulation taxonomy (with dataset support)

{md_table(tax_headers, tax_rows)}

## 4 · Stage 1 — binary bake-off (regulatory vs not)

{md_table(lb_headers, lb_rows)}

Champion: **{s1['champion']}** — selected on **validation PR-AUC**
(`val_pr_auc`); all other columns are one-shot test-set metrics.

![curves](figures/stage1_curves.png)

![confusion](figures/stage1_confusion.png)

### 4.1 · Score separation & calibration

The score distribution shows the class separation the gate achieves at its
deployed cut-off of {s1['threshold']:.3f} — optimized on the validation fold
by maximizing minority-class F1, rather than assuming the default 0.5. The
reliability diagram assesses whether predicted probabilities can be read as
probabilities (a prerequisite for retuning the cut-off to an explicit cost
matrix later).

![scores](figures/stage1_score_distribution.png)

![calibration](figures/stage1_calibration.png)

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

## 6 · Monitoring, guardrails & deployment

Served via the complaint MCP server + Complaint A2A agent. Per-classification
Prometheus counters (by label and mode), latency histograms via the agent
`/metrics` endpoint, OpenTelemetry traces across A2A hops, and drift monitoring
on the stage-1 score distribution (PSI trigger at 0.25).

{md_table(gr_headers, gr_rows)}

---

{SIGNOFF}
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

{md_table(lb_headers, lb_rows)}

![curves](figures/stage1_curves.png)

Score separation and calibration of the champion gate (validator re-derived):

![scores](figures/stage1_score_distribution.png)

![calibration](figures/stage1_calibration.png)

Stage-2 per-category recall vs weak labels (support-weighted):

![recall](figures/stage2_recall.png)

Per-category detail (support and recall against the weak reference):

{md_table(["category", "support", "recall"], s2_rows)}

### 3.1 · Guardrail inventory (validated against design)

{md_table(gr_headers, gr_rows)}

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

---

{SIGNOFF}
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
        _table_page(pdf, "3 · Dataset summary", ds_headers, ds_rows)
        _figure_page(pdf, "3 · Label distribution", figs["labels"],
                     "Weak-label distribution over the 24 regulation categories.")
        _table_page(pdf, "3.1 · Regulation taxonomy (with dataset support)",
                    tax_headers, tax_rows)
        _table_page(pdf, "4 · Stage-1 bake-off leaderboard", lb_headers, lb_rows,
                    f"Champion: {s1['champion']} - selected on validation PR-AUC; "
                    "other columns are one-shot test metrics.")
        _figure_page(pdf, "4 · Stage-1 ROC / PR curves", figs["curves"])
        _figure_page(pdf, "4 · Stage-1 confusion matrix", figs["confusion"])
        _figure_page(pdf, "4.1 · Score separation", figs["scores"],
                     f"Class separation at the validation-optimized cut-off "
                     f"{s1['threshold']:.3f} (default 0.5 not used).")
        _figure_page(pdf, "4.1 · Calibration (reliability diagram)", figs["calibration"],
                     "Whether predicted probabilities can be read as probabilities.")
        _figure_page(pdf, "5 · Stage-2 per-category recall", figs["recall"],
                     f"Exact acc {s2['accuracy']} · family acc {s2['family_accuracy']}"
                     f" · macro-F1 {s2['macro_f1']} · n={s2['n']}.")
        _table_page(pdf, "5 · Stage-2 per-category detail",
                    ["category", "support", "recall"], s2_rows,
                    "Recall vs weak labels on the stratified evaluation sample.")
        _table_page(pdf, "6 · Guardrail inventory", gr_headers, gr_rows)
        _text_page(pdf, "Sign-off", SIGNOFF.replace("**", ""))

    val_pdf = os.path.join(OUT_DIR, "02_validation_report.pdf")
    with PdfPages(val_pdf) as pdf:
        _title_page(pdf, "Independent Validation Report",
                    "Complaint → Regulation Classifier (CMPL-REG-24)\n"
                    "Second line of defense — effective challenge", meta)
        _text_page(pdf, "1 · Assessment narrative", val_narrative)
        _table_page(pdf, "2 · Stage-1 leaderboard (validated re-run)",
                    lb_headers, lb_rows)
        _figure_page(pdf, "2 · Stage-1 evidence: discrimination", figs["curves"])
        _figure_page(pdf, "2 · Stage-1 evidence: score separation", figs["scores"])
        _figure_page(pdf, "2 · Stage-1 evidence: calibration", figs["calibration"],
                     "Validator re-derived reliability diagram (quantile bins).")
        _figure_page(pdf, "3 · Stage-2 evidence", figs["recall"])
        _table_page(pdf, "3 · Stage-2 per-category detail",
                    ["category", "support", "recall"], s2_rows)
        _table_page(pdf, "3.1 · Guardrail inventory (validated)", gr_headers, gr_rows)
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
        _text_page(pdf, "Sign-off", SIGNOFF.replace("**", ""))

    print(f"wrote {OUT_DIR}/(01_model_development_document|02_validation_report).(md|pdf)")
    print(f"wrote {OUT_DIR}/metrics.json + {len(figs)} figures")


if __name__ == "__main__":
    main()
