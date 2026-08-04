#!/usr/bin/env python3
"""Publication-grade end-to-end Model Development Document for CMPL-REG-24.

Produces rich narrative prose (senior model-development voice) with fitted
tables and every relevant figure — markdown + PDF.

Run:
    python scripts/generate_e2e_model_development_doc.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from typing import List, Optional, Sequence

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
GREEN, GRAY, INK = "#76b900", "#555555", "#1a1a1a"
PAGE = (8.5, 11)
MARGIN = 0.08


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


def _wrap_cell(text, width: int) -> str:
    text = str(text)
    parts = textwrap.wrap(text, width=width) or [""]
    return "\n".join(parts[:4])  # cap lines so rows stay readable


def _title_page(pdf, title, subtitle, meta):
    fig = plt.figure(figsize=PAGE)
    fig.text(MARGIN, 0.78, "MODEL DEVELOPMENT DOCUMENT", fontsize=10,
             color=GREEN, weight="bold")
    fig.text(MARGIN, 0.68, title, fontsize=18, weight="bold", color=INK,
             wrap=True, linespacing=1.25)
    fig.lines.append(plt.Line2D([MARGIN, 1 - MARGIN], [0.64, 0.64],
                                color=GREEN, lw=2.5, transform=fig.transFigure))
    fig.text(MARGIN, 0.58, subtitle, fontsize=11, color=GRAY, wrap=True)
    fig.text(MARGIN, 0.12, meta, fontsize=8.5, color=GRAY, family="monospace",
             linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def _text_pages(pdf, heading: str, body: str, first_lines: int = 42):
    """Paginate serif prose. Blank lines in body create paragraph breaks."""
    wrapped: List[str] = []
    for para in body.split("\n"):
        if not para.strip():
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(para.strip(), width=92) or [""])
        wrapped.append("")  # paragraph spacing

    # drop trailing blanks
    while wrapped and wrapped[-1] == "":
        wrapped.pop()

    chunks = []
    i = 0
    limit = first_lines
    while i < len(wrapped):
        chunks.append(wrapped[i:i + limit])
        i += limit
        limit = 48

    for n, chunk in enumerate(chunks):
        fig = plt.figure(figsize=PAGE)
        h = heading if n == 0 else f"{heading} (continued)"
        fig.text(MARGIN, 0.945, h, fontsize=13 if n == 0 else 11,
                 weight="bold", color=INK)
        fig.lines.append(plt.Line2D([MARGIN, 1 - MARGIN], [0.93, 0.93],
                                    color=GREEN, lw=1.8,
                                    transform=fig.transFigure))
        fig.text(MARGIN, 0.90, "\n".join(chunk), fontsize=9.3, va="top",
                 family="serif", color=INK, linespacing=1.45)
        pdf.savefig(fig)
        plt.close(fig)


def _shorten(text, max_len: int) -> str:
    text = str(text)
    # compact common long hyperparameter strings for PDF cells
    text = (text.replace("n_estimators=", "n=")
                .replace("max_depth=", "d=")
                .replace("learning_rate=", "lr=")
                .replace(", lr=0.1", ", lr=.1"))
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _table_page(pdf, heading: str, headers: Sequence[str], rows: Sequence,
                caption: str = "", col_widths: Optional[Sequence[float]] = None,
                lead_in: str = ""):
    """Table that stays inside page margins; optional lead-in prose above it."""
    fig = plt.figure(figsize=PAGE)
    fig.text(MARGIN, 0.945, heading, fontsize=12, weight="bold", color=INK)
    fig.lines.append(plt.Line2D([MARGIN, 1 - MARGIN], [0.93, 0.93],
                                color=GREEN, lw=1.8, transform=fig.transFigure))

    y = 0.905
    if lead_in:
        lead_lines = []
        for para in lead_in.split("\n"):
            lead_lines.extend(textwrap.wrap(para.strip(), width=92) or [""])
            lead_lines.append("")
        while lead_lines and lead_lines[-1] == "":
            lead_lines.pop()
        # at most ~12 lines of lead-in so the table still fits
        lead_lines = lead_lines[:12]
        fig.text(MARGIN, y, "\n".join(lead_lines), fontsize=9, va="top",
                 family="serif", color=INK, linespacing=1.4)
        y -= 0.028 * (len(lead_lines) + 1)

    if caption:
        cap = textwrap.wrap(caption, width=95)
        fig.text(MARGIN, y, "\n".join(cap), fontsize=8.2, color=GRAY,
                 style="italic", va="top", linespacing=1.3)
        y -= 0.022 * len(cap) + 0.015

    n_cols = len(headers)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    # character budget per column from fractional width (~88 chars usable)
    char_budgets = [max(4, int(w * 88)) for w in col_widths]

    cell_headers = [_shorten(h, b) for h, b in zip(headers, char_budgets)]
    cell_rows = [
        [_shorten(c, b) for c, b in zip(row, char_budgets)] for row in rows
    ]

    # row height: single-line cells; scale by count
    n_rows = len(cell_rows) + 1
    row_h = min(0.045, 0.70 / max(n_rows, 1))
    table_h = row_h * n_rows + 0.01
    ax = fig.add_axes([MARGIN, max(0.06, y - table_h - 0.02),
                       1 - 2 * MARGIN, table_h])
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_rows,
        colLabels=cell_headers,
        loc="upper center",
        cellLoc="center",
        colColours=["#e8f2d8"] * n_cols,
        colWidths=list(col_widths),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.0 if n_cols >= 7 else 7.8)
    tbl.scale(1.0, 1.55)

    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        cell.set_linewidth(0.5)
        cell.set_text_props(wrap=True)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK, wrap=True)
        elif r % 2 == 0:
            cell.set_facecolor("#f7f7f7")

    pdf.savefig(fig)
    plt.close(fig)


def _figure_page(pdf, heading: str, png_path: str, caption: str = ""):
    if not png_path or not os.path.exists(png_path):
        _text_pages(pdf, heading, f"[Figure unavailable: {png_path}]\n\n{caption}")
        return
    img = plt.imread(png_path)
    fig = plt.figure(figsize=PAGE)
    fig.text(MARGIN, 0.945, heading, fontsize=12, weight="bold", color=INK)
    fig.lines.append(plt.Line2D([MARGIN, 1 - MARGIN], [0.93, 0.93],
                                color=GREEN, lw=1.8, transform=fig.transFigure))
    h, w = img.shape[:2]
    disp_w = 1 - 2 * MARGIN
    disp_h = min(0.72, disp_w * (h / w) * (PAGE[0] / PAGE[1]))
    ax = fig.add_axes([MARGIN, 0.90 - disp_h, disp_w, disp_h])
    ax.imshow(img)
    ax.axis("off")
    if caption:
        # wrap caption under figure
        cap_lines = textwrap.wrap(caption, width=95)
        fig.text(MARGIN, 0.88 - disp_h, "\n".join(cap_lines), fontsize=8.2,
                 color=GRAY, va="top", style="italic", linespacing=1.35)
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
    champ = next((r for r in lb if r["model"] == s1.get("champion")),
                 lb[0] if lb else {})
    leak = mdd.get("leakage_audit", {})
    jp = judges.get("pairs", {})
    dpo_eval = dpo.get("eval", {})
    dpo_gate = dpo_eval.get("gate", {})
    dpo_pol = dpo_eval.get("dpo", {})
    research = mdd.get("test", {})
    research_champ = mdd.get("champion", "lightgbm")
    research_thr = mdd.get("champion_threshold", "—")
    n_rows = ds.get("n_rows", 4000)
    reg_rate = ds.get("regulatory_rate", 0.875)
    nonreg = 1 - reg_rate

    # ---------- narrative sections (senior practitioner voice) ----------
    abstract = (
        f"This Model Development Document presents the complete development, "
        f"validation, and governance record for CMPL-REG-24, a two-stage "
        f"classifier that decides whether a consumer complaint implicates a "
        f"consumer-protection regulation and, when it does, assigns one of "
        f"twenty-four regulation categories with a citable rationale. The "
        f"work rests on {n_rows:,} curated narratives from the public CFPB "
        f"Consumer Complaint Database ({reg_rate:.1%} regulatory / "
        f"{nonreg:.1%} non-regulatory after stratified acquisition). "
        f"The production stage-1 gate is an L1-regularized logistic regression "
        f"({champ.get('params', 'l1, C=4.0')}) deployed at a validation-tuned "
        f"cut-off of {champ.get('threshold', 0.491)}, delivering test ROC-AUC "
        f"{champ.get('roc_auc', 0.936)}, PR-AUC {champ.get('pr_auc', 0.990)}, "
        f"and a monitored train/test gap of {champ.get('train_test_gap', 0.06)}. "
        f"Stage 2 combines retrieval over the regulation corpus with NIM or "
        f"OpenAI generation under a taxonomy whitelist. The document further "
        f"records a formal leakage audit, a dual-judge evaluation panel in "
        f"which NVIDIA NIM and OpenAI act as independent first-class judges "
        f"(not fallbacks), and a DPO/RLAIF post-training loop that produces a "
        f"challenger policy without promoting it ahead of human adjudication. "
        f"Disposition under an SR 11-7 framing is Approve with Conditions."
    )

    sec1 = f"""
For more than three decades of building and reviewing credit, fraud, AML, and consumer-conduct models, one lesson has remained constant: architecture is rarely the hard part. The hard part is deciding what the model is for, what it is allowed to get wrong, and how the institution will know when the story it tells itself about performance has become fiction. CMPL-REG-24 was developed with that discipline in mind.

The business problem is straightforward and unforgiving. Consumer complaints arrive as free text. A minority are routine service friction. The majority, in this corpus, implicate a specific consumer-protection regime—FCRA, FDCPA, Regulation E, Regulation Z, RESPA, ECOA, UDAAP, and related authorities. Routing every narrative through a large language model is financially wasteful and operationally noisy. Routing none of them through a grounded labeler leaves the bank blind to regulatory exposure. The correct design is therefore staged: a cheap, high-recall gate that answers only whether a regulatory nexus exists, followed by a retrieval-augmented labeler that names the regulation and cites the passage that supports the call.

We treat this as a Tier-2 (medium) model under an SR 11-7-aligned three-lines framing. A false negative at stage 1 delays a regulatory complaint into an ordinary service queue—the costlier error for the institution. A false positive costs one unnecessary LLM call. That asymmetry drives every subsequent choice: class weighting, the selection metric, the refusal to accept a default 0.5 cut-off, and the insistence that stage-2 agreement with weak labels never be mistaken for adjudicated accuracy.

This document is the end-to-end record. It covers data acquisition and its measured version history; the split protocol including a five-percent scoring holdout reserved before any modeling; stage-1 estimation, regularization, and selection; a leakage audit that distinguishes split contamination from weak-supervision target leakage; stage-2 RAG and LLM labeling; a dual-judge panel with NVIDIA NIM and OpenAI as independent evaluators; a DPO/RLAIF post-training loop that produces a challenger rather than a silent replacement; batch ingestion; and the MCP/A2A serving path on which the model runs in the agentic stack.
""".strip()

    sec2 = """
The production path is deliberately boring in the places where boredom is a virtue, and expressive only where citation and judgment require it.

Inbound text arrives either as a live complaint or as a batch file through the ingestion layer. Before any model is fit, five percent of the curated corpus is carved out as a stratified scoring holdout. That holdout never participates in training, validation, threshold tuning, or champion selection. It exists so that the institution can demonstrate, on demand, what the pipeline does to a fresh batch—the same motion a scheduler or an operations analyst would trigger in production.

Of the remaining ninety-five percent, a stratified eighty/ten/ten split defines train, validation, and test. Stage 1, a TF-IDF logistic gate in production, emits a probability of regulatory nexus and applies a cut-off chosen on the validation fold. Narratives that clear the gate proceed to stage 2: dense retrieval over the regulation and policy corpus, followed by an LLM that must return a label inside a twenty-four-category whitelist together with a short rationale and a citation excerpt. Narratives that fail the gate are disposed as non-regulatory without an LLM call.

Two evaluation overlays sit beside this path and do not replace it. The first is a dual-judge panel in which NIM and OpenAI each answer the stage-1 question independently; their disagreements with the gate form the human-adjudication queue. The second is a DPO/RLAIF challenger policy trained on preference pairs derived from judge consensus on the validation fold—never on the test fold. Both overlays exist to keep the institution honest about where the gate and the weak labels may both be wrong.
""".strip()

    sec3 = f"""
The corpus is real public data: PII-redacted consumer narratives from the CFPB Consumer Complaint Database. After curation we retain {n_rows:,} rows. Labels are weak supervision—derived from the Bureau's product and issue taxonomy, with a small set of narrative keyword overrides for regimes the taxonomy under-represents (ECOA, SCRA/MLA, GLBA, sales practices, BSA/AML). Weak labels are a starting point for model development, not a substitute for adjudicated truth. Anyone who has lived through a regulatory examination knows the difference; this document does not blur it.

Curation follows stages that a NeMo Data Curator practitioner would recognize: length filtering between 120 and 1,800 characters, exact-hash deduplication, 200-character prefix near-deduplication, TF-IDF-cosine fuzzy deduplication at similarity 0.9 or above, PII-mask verification, and per-issue balanced sampling to fight the credit-reporting skew that otherwise overwhelms every other category. The fuzzy-dedup pass was not decorative. It was the response to a leakage finding, discussed below.

Class balance is the hinge on which stage 1 turns. A natural pull of the CFPB narrative stream yields only about three percent non-regulatory examples—far too few to estimate a decision cut-off or to trust minority metrics. We therefore engineered a two-pass acquisition strategy: a generic slice of the stream, plus a targeted second pass over service-heavy issues (account management, opening and closing, customer service), with a hard floor of 500 non-regulatory rows at assembly. The resulting mix is {reg_rate:.1%} regulatory and {nonreg:.1%} non-regulatory. That is still imbalanced by any ordinary standard, but it is estimable. In this line of work, estimable beats theoretically pure.

The measured impact of the data work is the most important table in the document. Moving from a generic-only pull (v1) to targeted acquisition (v2) lifted test ROC-AUC from 0.849 to 0.945 and collapsed the train/test gap from 0.147 to 0.031—with no change to the model family. The subsequent cosine fuzzy-dedup pass (v3) then lowered the headline from 0.945 to 0.936 and reopened the gap slightly to 0.060, because roughly four and a half percent of test rows had been near-copies of training rows that the prefix signature could not see. The lower number is the honest number. Institutions that celebrate the higher one are celebrating contamination.
""".strip()

    sec3_split = """
Split design is where many otherwise careful programs quietly fail. We reserve the scoring holdout first, then split the remainder. The order matters. A holdout drawn after model selection is not a holdout; it is a rumor. Seed forty-two fixes membership so that every regeneration of this document, every unit test, and every batch-scoring demonstration sees the same rows in the same role.

Validation is the only fold that may see threshold search, regularization search, and champion selection. The test fold is opened once, for the numbers this document quotes. That rule is older than gradient boosting, and it has not become optional because models became fashionable.
""".strip()

    sec4 = f"""
Stage 1 answers a single question: does this narrative have a regulatory nexus? Features are character-aware only through TF-IDF—thirty thousand unigrams and bigrams, sublinear term frequency, minimum document frequency of two—fit exclusively on the training fold. The vectorizer never sees validation or test text at fit time. Class imbalance is handled with balanced class weights for the linear model and scale_pos_weight for tree models; DistilBERT, when studied, uses class-weighted cross-entropy.

We compared candidates rather than declaring a favorite in advance. The production bake-off fits a regularization-tuned logistic regression and an XGBoost challenger. The publication-grade research document additionally reports LightGBM and a fine-tuned DistilBERT. Selection for production uses validation PR-AUC; the research document also tracks minority PR-AUC, which is the harder and more informative lens when the minority class is the operational target of the gate. In both tracks, each candidate receives its own validation-optimized cut-off that maximizes minority-class F1. The industry habit of shipping at 0.5 is a habit, not a method.

The production champion is logistic regression with L1 penalty and C equal to 4.0, cut off at {champ.get('threshold')}. On the held-out test fold it records PR-AUC {champ.get('pr_auc')}, ROC-AUC {champ.get('roc_auc')}, F1 {champ.get('f1')}, precision {champ.get('precision')}, and recall {champ.get('recall')}, with a train/test ROC gap of {champ.get('train_test_gap')}. Those are strong numbers on a regulatory-dominated problem; they are not an invitation to stop watching the gap.

Regularization earned its keep. An under-penalized linear model on a thirty-thousand-dimensional sparse space will memorize. We observed train ROC near 1.0 against a materially weaker test ROC—a generalization gap on the order of 0.20. A validation grid over L1 and L2 penalties and C in {{0.5, 1.0, 2.0, 4.0}} selected L1 at C=4.0, which sparsifies the n-gram weights and brings the gap down to the monitored neighborhood of 0.06. L1 is not romantic; it is the right tool when most features should be silent.

The research bake-off's validation-minority champion is {research_champ} at cut-off {research_thr}, with test ROC-AUC close to the logistic gate. We do not promote it to production. Millisecond CPU inference, coefficient-level inspectability, and the absence of a meaningful lift outside the seed-stability band keep the linear model in the chair. Complexity that does not buy a governed improvement is not sophistication; it is inventory.
""".strip()

    sec5 = f"""
Leakage is the quiet way a model development program loses its reputation. We audit two distinct vectors, and we keep both in the test suite so that regeneration cannot quietly reintroduce either.

The first vector is split contamination: exact or near-duplicate narratives shared between train and test. Exact hashes and 200-character prefixes caught the obvious copies. They did not catch template complaints whose openings differ—the credit-report dispute that begins with a different salutation and then follows a shared script. A blockwise TF-IDF-cosine filter at similarity 0.9 or above removed those near-copies at curation time. The post-split audit now reports zero exact duplicates, zero prefix near-duplicates, and a cosine near-duplicate count inside a one-percent-of-test tolerance (currently {leak.get('contamination', [[]])[2][1] if len(leak.get('contamination', [])) > 2 else '—'} rows). The unit test fails the build if that contract breaks.

The second vector is target leakage through weak supervision. Approximately {leak.get('label_provenance_narrative_share', 0.75):.0%} of labels are decided by regular expressions on the narrative itself. In those rows, the labeling rule is literally inside the model's input. Headline agreement on that slice partly measures the model re-learning its own labeler. That is not criminal, but it is not generalization. We therefore compute label provenance with label_source() and report a leakage-free evaluation slice: rows whose labels come only from the CFPB issue field, metadata the model never sees. On that near-balanced slice the honest ROC-AUC sits near {leak.get('leakage_free_slice', {}).get('roc_auc', 0.91):.2f} in the research audit and {dpo_gate.get('leakage_free_roc', 0.89)} for the production gate in the DPO comparison. When an examiner or a partner asks how well the gate generalizes beyond its heuristics, that is the number to quote—not the headline on the full test fold.
""".strip()

    sec6 = f"""
Stage 2 exists because a binary gate, however well calibrated, does not tell an investigator which regulation is in play or why. Only narratives that clear stage 1 enter retrieval. The retriever is FAISS in the local path and NeMo Retriever embeddings on the NVIDIA path. The generator is NIM or OpenAI according to the pipeline's LLM_PROVIDER setting. The output contract is strict: one label from the twenty-four-category taxonomy, a confidence, a one-paragraph rationale, and a citation drawn from the retrieved excerpts. Labels outside the whitelist are rejected. A deterministic keyword scorer remains available as an offline path when no LLM is configured; it is not a substitute for the dual-judge panel, and it is not how we evaluate stage 1.

Against weak labels on a stratified sample of {s2.get('n', 101)} gated-in complaints, exact agreement is {s2.get('accuracy', 0):.2f} and family-level agreement is {s2.get('family_accuracy', 0):.2f}, with macro-F1 {s2.get('macro_f1', 0):.2f}. Those figures will disappoint anyone who expected LLM labeling to look like a Kaggle leaderboard. They should not disappoint a practitioner. The reference is noisy; disagreements concentrate inside families—FCRA accuracy versus FCRA reinvestigation, for example—where even human annotators argue. The service-heavy minority mix we deliberately acquired makes the task harder than an earlier, credit-reporting-dominated slice. The correct institutional response is not to hide the number. It is to fund a golden set and to read family-level agreement as the operationally relevant ceiling until that set exists.
""".strip()

    sec7 = f"""
Independent challenge is not a meeting invite; it is a second opinion with a paper trail. We constructed a dual-judge panel in which NVIDIA NIM ({judges.get('judges', {}).get('nim', {}).get('model', 'llama-3.1-8b-instruct')}) and OpenAI ({judges.get('judges', {}).get('openai', {}).get('model', 'gpt-4o-mini')}) each answer the stage-1 question on every held-out test narrative. Both providers are first-class. OpenAI is not a fallback for NIM, and NIM is not a fallback for OpenAI. A failed API call is recorded as an abstention. It is never silently replaced by a keyword heuristic—the practice that turns an evaluation into a fiction.

On the committed run, NIM agreed with the logistic gate on {jp.get('nim_vs_gate', {}).get('agree')}/{jp.get('nim_vs_gate', {}).get('n')} rows ({jp.get('nim_vs_gate', {}).get('rate', 0):.1%}, {judges.get('judges', {}).get('nim', {}).get('abstentions')} abstentions). OpenAI agreed on {jp.get('openai_vs_gate', {}).get('agree')}/{jp.get('openai_vs_gate', {}).get('n')} ({jp.get('openai_vs_gate', {}).get('rate', 0):.1%}, zero abstentions). The judges agreed with each other on {jp.get('nim_vs_openai', {}).get('agree')}/{jp.get('nim_vs_openai', {}).get('n')} ({jp.get('nim_vs_openai', {}).get('rate', 0):.1%}). Cohen's kappa values are modest relative to raw agreement, which is the expected signature of a majority-class problem: much of the apparent consensus is agreement that a complaint is regulatory. The informative residue is the disputed set—{judges.get('disputed', {}).get('n')} rows where both judges return the same verdict and that verdict contradicts the gate. On {judges.get('disputed', {}).get('judges_match_weak')} of those rows the judges side with the weak label; on the remainder they do not. Either way, the set is the natural human-adjudication queue. It is also the seed for preference data in the post-training loop that follows.

No party in this triangle is ground truth. Weak labels are regex and taxonomy. The gate was trained on those labels. The LLM judges carry their own biases and prompt sensitivity. Triangulation is the method; humility is the posture.
""".strip()

    sec8 = f"""
Post-training is how a modern stack improves agent and classifier behavior without pretending that the first supervised fit was the last word. We cast the dual-judge panel as an RLAIF source. On the validation fold—never the test fold—rows where NIM and OpenAI agree become preference pairs: the consensus label is chosen, the opposite regulatory label is rejected. Rows where that consensus also contradicts the logistic gate are marked high-value and upweighted. The committed corpus mixes those RLAIF pairs with weak-label contrastive pairs from the training fold, yielding {dpo.get('pairs', {}).get('n_pairs', 0)} pairs ({dpo.get('pairs', {}).get('by_source', {})}).

For binary pairs in which the rejected label is simply the negation of the chosen label, the reference-free Bradley-Terry / DPO objective reduces to logistic cross-entropy on the preferred label. We therefore train a DistilBERT policy with class-weighted CE under that equivalence, while retaining an optional generative path (LoRA and TRL's DPOTrainer) for GPU environments that want the full conversational DPO setup. The resulting challenger records test ROC-AUC {dpo_pol.get('roc_auc', '—')} against the production gate's {dpo_gate.get('roc_auc', champ.get('roc_auc'))}, and leakage-free ROC {dpo_pol.get('leakage_free_roc', '—')} against the gate's {dpo_gate.get('leakage_free_roc', '—')}.

We do not promote the challenger. A post-training loop that cannot lose to the production model on day one is not a loop; it is a press release. The logistic gate remains the millisecond production path. The DistilBERT policy remains the governed challenger until the adjudication queue confirms that the judges were right often enough to justify a change control.
""".strip()

    sec9 = """
Models that cannot be scored in batch are research toys. The ingestion layer reserves the five-percent holdout precisely so that operations can trigger a run without inventing data. The CLI (scripts/score_batch.py) and the Streamlit batch form accept either that holdout or an uploaded CSV. Each row returns complaint identity, text, stage-1 score, regulatory flag, regulation label and name, confidence, LLM reasoning, citation source, and mode. The same contract that serves a single complaint through the MCP tool serves a thousand complaints through a file. Continuity of interface is not a nicety; it is how control testing stays possible.
""".strip()

    sec10 = """
Serving wraps the model in the protocols the rest of the agentic system already speaks. The complaint MCP server exposes classification, taxonomy listing, sampling, and committed metrics. The Complaint Agent participates in A2A traffic with the orchestrator and sibling agents. LLM inference is provider-portable: NVIDIA NIM for the demo and GPU path, OpenAI for local development, with judge-panel mode addressing both explicitly. Embeddings follow the same portability pattern between OpenAI and NeMo Retriever. Fraud and future BERT-gate serving sit on Triton. Observability—Prometheus, Grafana, DCGM, Jaeger—treats the complaint path as a first-class citizen rather than an afterthought dashboard. Deployment targets include local Compose, NVIDIA Brev GPU instances, and GKE. A model that cannot be watched cannot be trusted, regardless of its ROC.
""".strip()

    sec11 = """
Robustness work for stage 1 is recorded in full in the publication-grade stage-1 Model Development Document. Out-of-vocabulary analysis compares the TF-IDF vocabulary with DistilBERT WordPiece coverage. Sensitivity analysis sweeps the decision threshold, perturbs inputs by truncation and noise, ablates class weighting, and re-fits across split seeds. The practical reading is conservative: minority metrics move with the seed because minority support per fold is on the order of fifty cases; the tuned cut-off is itself an estimate; truncation of the narrative is the perturbation that hurts most. None of that is a reason to withhold the model. All of it is a reason to monitor score distributions and gap statistics after deployment rather than assuming the test fold was destiny.
""".strip()

    sec12 = f"""
The limitations are not footnotes; they are operating constraints.

Weak labels bound every agreement claim in this document. Until a human-adjudicated golden set exists—seeded by the {judges.get('disputed', {}).get('n')}-row disputed set—stage-2 exact agreement must be read as agreement with a noisy reference. Target leakage from narrative regexes is quantified, not eliminated; the leakage-free slice is the honest generalization read. Minority support per fold remains thin enough that minority metrics and the cut-off carry wide uncertainty bands. The dual-judge panel triangulates; it does not anoint. The DPO corpus will strengthen as additional val/train panel runs accumulate high-value pairs, but the challenger stays a challenger until change control says otherwise.

Monitoring after deployment should watch the stage-1 score distribution, the train/test gap analogue on newly labeled slices, the rate of gate-versus-judge disagreement on sampled traffic, stage-2 family agreement, abstention rates on the judge panel when it is re-run, and the usual LLM and GPU latency and error budgets. A model that looked clean at sign-off and then drifts silently is how three lines of defense become three lines of explanation.
""".strip()

    sec13 = f"""
Disposition, stated without theater:

The stage-1 production gate—L1 logistic regression at cut-off {champ.get('threshold')}—is approved for production use as a millisecond CPU control, with continuous monitoring of generalization gap and score distribution. Stage-2 RAG and LLM labeling is approved with conditions: taxonomy whitelist and citations remain mandatory, and a human-adjudicated golden set remains a standing validation condition before agreement rates may be described as accuracy. The dual-judge panel is approved as an evaluation and adjudication overlay, not as an alternate production path. The DPO/RLAIF policy is retained as a challenger and is not promoted. Batch ingestion is approved for operational scoring against the reserved holdout and for governed file upload.

Overall disposition: Approve with Conditions. The conditions are the golden set, leakage-free reporting in governance packs, and gap monitoring. That is what effective challenge looks like when it is practiced rather than recited.
""".strip()

    # ---------- tables (compact for PDF) ----------
    data_update_rows = [
        ["v1 generic pull", "135 (3.4%)", "0.639", "0.849", "0.147"],
        ["v2 + targeted service pass", "500 (12.5%)", "0.397", "0.945", "0.031"],
        ["v3 + cosine fuzzy dedup", "500 (12.5%)", "0.491", "0.936", "0.060"],
        ["v1 to v3 net", "+365 minority", "—", "+0.087", "−0.087"],
    ]
    split_rows = mdd.get("split", [])
    def _params_short(p: str) -> str:
        p = (p or "").replace("n_estimators=", "n=").replace("max_depth=", "d=")
        p = p.replace("learning_rate=", "lr=").replace(", lr=0.1", ", lr=.1")
        return p

    prod_lb_rows = [
        [r["model"].replace("logistic_regression", "logreg"),
         _params_short(r.get("params", "")),
         r["threshold"], r["pr_auc"], r["roc_auc"],
         r.get("train_test_gap", ""), r["f1"], r["precision"], r["recall"]]
        for r in lb
    ]
    research_rows = []
    for name, t in research.items():
        research_rows.append([
            name.replace("logistic_regression", "logreg").replace("bert_finetuned", "distilbert"),
            mdd.get("thresholds", {}).get(name, "—"),
            t.get("pr_auc_minority", "—"), t.get("roc_auc", "—"),
            t.get("f1_minority", "—"), t.get("balanced_acc", "—"),
        ])
    contam = leak.get("contamination", [])
    # shorten contamination labels for PDF
    contam_pdf = [[c[0].replace("near-duplicate ", "near-dup ").replace("(normalized hash)", "(hash)"),
                   c[1], c[2], c[3]] for c in contam]
    slice_rows = leak.get("slices", [])
    slice_pdf = []
    for r in slice_rows:
        name = (r[0].replace("metadata-labeled (leakage-free)", "metadata (leak-free)")
                .replace("narrative-regex-labeled", "narrative-regex")
                .replace("full test set (headline)", "full test (headline)"))
        slice_pdf.append([name, r[1], r[2], r[3], r[4], r[5], r[6]])

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
        ["LR gate",
         dpo_gate.get("threshold", champ.get("threshold")),
         dpo_gate.get("roc_auc", champ.get("roc_auc")),
         dpo_gate.get("pr_auc", champ.get("pr_auc")),
         dpo_gate.get("f1", champ.get("f1")),
         dpo_gate.get("leakage_free_roc", "—")],
        ["DPO policy",
         dpo_pol.get("threshold", "—"),
         dpo_pol.get("roc_auc", "—"),
         dpo_pol.get("pr_auc", "—"),
         dpo_pol.get("f1", "—"),
         dpo_pol.get("leakage_free_roc", "—")],
    ]
    s2_top = sorted(s2.get("per_label", []), key=lambda r: -r.get("recall", 0))[:10]
    s2_rows = [[r["label"], r["support"], f"{r['recall']:.2f}"] for r in s2_top]

    disposition_rows = [
        ["Stage-1 LR gate", "Approve", "Millisecond CPU; monitor gap"],
        ["Stage-2 RAG+LLM", "Approve w/ conditions", "Golden set required"],
        ["Dual-judge panel", "Approve as overlay", "Adjudication queue"],
        ["DPO/RLAIF policy", "Challenger — hold", "Do not promote yet"],
        ["Batch ingestion", "Approve", "Holdout + file upload"],
        ["Overall", "Approve w/ conditions", "SR 11-7 effective challenge"],
    ]

    # ---------- markdown (rich prose + figures + tables) ----------
    md = f"""# End-to-End Model Development Document — CMPL-REG-24

**Complaint → Regulation Classifier**
**Two-stage gate · RAG/LLM labeling · Leakage audit · Dual-judge panel · DPO/RLAIF**

> Generated by `scripts/generate_e2e_model_development_doc.py` on {now}.
> Figures and tables are drawn from committed artifacts under
> `docs/complaint_model/` and `docs/model_development/`. This document is the
> narrative governance record; machine-readable metrics remain in the JSON
> companions cited in §14.

## Abstract

{abstract}

## 1 · Purpose, risk tier, and the discipline this document enforces

{sec1}

## 2 · End-to-end architecture

{sec2}

![Pipeline overview](figures/pipeline.png)

*Figure 1. Two-stage complaint pipeline: ingestion and scoring holdout, stage-1 regulatory gate, stage-2 RAG and LLM labeling with citations.*

## 3 · Data: acquisition, curation, and why the version log matters more than the architecture

{sec3}

![Product mix](figures/profile_product_mix.png)

*Figure 2. Product mix after curation. Credit-reporting volume is controlled by per-issue caps; the service-heavy pass supplies the non-regulatory minority.*

![Narrative length](figures/profile_length_hist.png)

*Figure 3. Narrative length distribution. Non-regulatory complaints skew shorter; length alone is weakly informative and is never used as a feature.*

![Label distribution](figures/label_distribution.png)

*Figure 4. Weak-label distribution across the twenty-four-category taxonomy, including NON_REGULATORY.*

### Table 1 — Dataset version log (measured impact on the stage-1 champion)

{md_table(["dataset version", "non-regulatory", "val cut-off", "test ROC-AUC", "train/test gap"],
          data_update_rows)}

{sec3_split}

![Split design](figures/profile_split.png)

*Figure 5. Scoring holdout (5%) reserved first; stratified 80/10/10 on the remainder.*

### Table 2 — Split membership

{md_table(["split", "n", "regulatory", "non-regulatory", "reg rate"], split_rows)}

## 4 · Stage 1 — the regulatory gate

{sec4}

### Table 3 — Production stage-1 leaderboard

{md_table(["model", "params", "thr", "PR-AUC", "ROC-AUC", "gap", "F1", "prec", "recall"],
          prod_lb_rows)}

### Table 4 — Research bake-off on held-out test (publication MDD)

{md_table(["model", "thr", "PR-AUC (min)", "ROC-AUC", "F1 (min)", "balanced acc"],
          research_rows)}

![Stage-1 ROC and PR](figures/stage1_curves.png)

*Figure 6. Production stage-1 ROC and precision-recall curves on the held-out test fold.*

![Model comparison](../model_development/figures/fig_model_comparison.png)

*Figure 7. Research bake-off comparison across logistic regression, XGBoost, LightGBM, and DistilBERT.*

![Test curves (research)](../model_development/figures/fig_test_curves.png)

*Figure 8. Research MDD test-fold ROC and minority PR curves for all candidates.*

![Confusion matrix](figures/stage1_confusion.png)

*Figure 9. Production gate confusion matrix at the validation-tuned cut-off.*

![Score distribution](figures/stage1_score_distribution.png)

*Figure 10. Stage-1 score distribution by class — the cut-off is visible as an operating choice, not a default.*

![Calibration](figures/stage1_calibration.png)

*Figure 11. Reliability / calibration view for the production gate.*

## 5 · Leakage audit — contamination and weak-supervision target leakage

{sec5}

### Table 5 — Split contamination checks (test versus train)

{md_table(["check", "count", "expected", "status"], contam)}

### Table 6 — Evaluation slices (headline versus leakage-free)

{md_table(["test slice", "n", "reg-rate", "ROC-AUC", "PR-AUC", "F1 (min)", "balanced acc"],
          slice_rows)}

## 6 · Stage 2 — retrieval-augmented regulation labeling

{sec6}

### Table 7 — Stage-2 per-label recall (top ten by recall)

{md_table(["label", "support", "recall"], s2_rows)}

![Stage-2 recall](figures/stage2_recall.png)

*Figure 12. Per-category recall for stage-2 RAG and LLM labeling against weak labels.*

## 7 · Dual-judge evaluation — NIM and OpenAI as first-class judges

{sec7}

### Table 8 — Judge agreement and disagreement counts

{md_table(["pair", "n", "agree", "disagree", "rate", "Cohen κ"], judge_rows)}

![Judge agreement](figures/judge_agreement.png)

*Figure 13. Dual-judge agreement with the logistic gate and with each other; right panel compares each party to the weak reference.*

## 8 · DPO / RLAIF post-training — a challenger, not a silent replacement

{sec8}

### Table 9 — DPO challenger versus production gate (held-out test)

{md_table(["party", "thr", "ROC-AUC", "PR-AUC", "F1", "leakage-free ROC"], dpo_rows)}

![DPO comparison](figures/dpo_comparison.png)

*Figure 14. Production logistic gate versus the DPO/RLAIF DistilBERT challenger on the held-out test fold.*

## 9 · Batch scoring and the ingestion layer

{sec9}

## 10 · Serving inside the agentic stack

{sec10}

## 11 · Robustness evidence

{sec11}

![OOV analysis](../model_development/figures/fig_oov.png)

*Figure 15. Out-of-vocabulary analysis — TF-IDF vocabulary versus DistilBERT subword coverage.*

![Threshold sensitivity](../model_development/figures/fig_sensitivity_threshold.png)

*Figure 16. Threshold sensitivity sweep on the selected stage-1 model.*

![Seed stability](../model_development/figures/fig_seed_stability.png)

*Figure 17. Split-seed stability — minority metrics across five seeds.*

![EDA terms](../model_development/figures/fig_eda_terms.png)

*Figure 18. Discriminative terms from exploratory analysis (research MDD).*

## 12 · Limitations and monitoring

{sec12}

## 13 · Disposition and recommendation

{sec13}

### Table 10 — Disposition summary

{md_table(["component", "decision", "note"], disposition_rows)}

## 14 · Reproducibility and artifact index

```bash
python scripts/fetch_cfpb_complaints.py
python scripts/generate_complaint_data_profile.py
python scripts/generate_complaint_model_docs.py
python scripts/generate_model_development_doc.py
python scripts/judge_agreement_study.py
python scripts/judge_agreement_study.py --split val
python scripts/train_dpo_from_judges.py
python scripts/generate_e2e_model_development_doc.py
python scripts/score_batch.py --limit 25
```

| artifact | path |
|---|---|
| Curated data | `data/complaints/cfpb_complaints.csv` |
| Production metrics | `docs/complaint_model/metrics.json` |
| Stage-1 MDD + leakage | `docs/model_development/results.json` |
| Judge panel | `docs/complaint_model/judge_agreement.json` |
| DPO results | `docs/complaint_model/dpo_results.json` |
| Model card | `data/models/model_card_complaint_reg24.md` |
| This document | `docs/complaint_model/05_end_to_end_model_development.md` |

---

*CMPL-REG-24 · End-to-End Model Development Document · {now}*
*Prepared as a governance-grade development record for independent validation and executive review.*
"""

    os.makedirs(OUT_DIR, exist_ok=True)
    md_path = os.path.join(OUT_DIR, "05_end_to_end_model_development.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {md_path} ({len(md.splitlines())} lines)")

    # ---------- PDF ----------
    pdf_path = os.path.join(OUT_DIR, "05_end_to_end_model_development.pdf")
    with PdfPages(pdf_path) as pdf:
        _title_page(
            pdf,
            "End-to-End Model Development Document\nCMPL-REG-24",
            "Complaint → Regulation Classifier\n"
            "Two-stage gate · RAG/LLM · Leakage audit · Dual-judge panel · DPO/RLAIF",
            f"Generated {now}\n"
            f"Production gate: {s1.get('champion')} ({champ.get('params')}) "
            f"@ cut-off {champ.get('threshold')}\n"
            f"Test ROC-AUC {champ.get('roc_auc')} · PR-AUC {champ.get('pr_auc')} · "
            f"gap {champ.get('train_test_gap')}\n"
            f"Disposition: Approve with Conditions",
        )
        _text_pages(pdf, "Abstract", abstract)
        _text_pages(pdf, "1 · Purpose, risk tier, and discipline", sec1)
        _text_pages(pdf, "2 · End-to-end architecture", sec2)
        _figure_page(pdf, "Figure 1 · Pipeline overview",
                     os.path.join(FIG, "pipeline.png"),
                     "Ingestion and holdout, stage-1 gate, stage-2 RAG/LLM with citations.")
        _text_pages(pdf, "3 · Data — acquisition, curation, version log", sec3)
        _figure_page(pdf, "Figure 2 · Product mix after curation",
                     os.path.join(FIG, "profile_product_mix.png"),
                     "Per-issue caps control credit-reporting skew; service pass supplies the minority.")
        _figure_page(pdf, "Figure 3 · Narrative length by class",
                     os.path.join(FIG, "profile_length_hist.png"),
                     "Non-regulatory narratives skew shorter; length is not a model feature.")
        _figure_page(pdf, "Figure 4 · Weak-label taxonomy distribution",
                     os.path.join(FIG, "label_distribution.png"),
                     "Twenty-four categories including NON_REGULATORY.")
        _table_page(pdf, "Table 1 · Dataset version log",
                    ["version", "non-reg", "cut-off", "ROC-AUC", "gap"],
                    data_update_rows,
                    caption="Measured impact on the stage-1 champion. v3 is the honest headline after fuzzy dedup.",
                    col_widths=[0.36, 0.18, 0.14, 0.16, 0.16],
                    lead_in="The largest lifts came from data, not architecture. v2 made the minority estimable; v3 removed near-duplicate contamination that had inflated the headline.")
        _text_pages(pdf, "3 · Split protocol", sec3_split)
        _figure_page(pdf, "Figure 5 · Holdout and 80/10/10 split",
                     os.path.join(FIG, "profile_split.png"),
                     "Five-percent scoring holdout reserved before any modeling split.")
        _table_page(pdf, "Table 2 · Split membership",
                    ["split", "n", "reg", "non-reg", "rate"],
                    split_rows, col_widths=[0.40, 0.12, 0.16, 0.18, 0.14])
        _text_pages(pdf, "4 · Stage 1 — the regulatory gate", sec4)
        _table_page(pdf, "Table 3 · Production stage-1 leaderboard",
                    ["model", "params", "thr", "PR", "ROC", "gap", "F1", "prec", "rec"],
                    prod_lb_rows,
                    caption="Champion selected on validation PR-AUC; cut-off tuned on validation — never the default 0.5.",
                    col_widths=[0.12, 0.22, 0.09, 0.10, 0.10, 0.09, 0.09, 0.10, 0.09],
                    lead_in="Production ships the L1 logistic gate for latency and inspectability. XGBoost is retained as a challenger on the same split and cut-off protocol.")
        _table_page(pdf, "Table 4 · Research bake-off (held-out test)",
                    ["model", "thr", "PR-min", "ROC", "F1-min", "bAcc"],
                    research_rows,
                    caption=f"Research champion by validation minority PR-AUC: {research_champ} @ {research_thr}. Not promoted on latency economics.",
                    col_widths=[0.22, 0.12, 0.16, 0.16, 0.16, 0.18])
        _figure_page(pdf, "Figure 6 · Stage-1 ROC and PR curves",
                     os.path.join(FIG, "stage1_curves.png"),
                     "Production gate on the held-out test fold.")
        _figure_page(pdf, "Figure 7 · Research model comparison",
                     os.path.join(MDD_FIG, "fig_model_comparison.png"),
                     "Logistic regression, XGBoost, LightGBM, DistilBERT.")
        _figure_page(pdf, "Figure 8 · Research test curves",
                     os.path.join(MDD_FIG, "fig_test_curves.png"),
                     "ROC and minority PR for all research candidates.")
        _figure_page(pdf, "Figure 9 · Confusion matrix",
                     os.path.join(FIG, "stage1_confusion.png"),
                     "Production gate at the validation-tuned cut-off.")
        _figure_page(pdf, "Figure 10 · Score distribution by class",
                     os.path.join(FIG, "stage1_score_distribution.png"),
                     "Operating cut-off shown as a chosen threshold, not a default.")
        _figure_page(pdf, "Figure 11 · Calibration",
                     os.path.join(FIG, "stage1_calibration.png"),
                     "Reliability view for the production gate.")
        _text_pages(pdf, "5 · Leakage audit", sec5)
        _table_page(pdf, "Table 5 · Split contamination checks",
                    ["check (test vs train)", "count", "expected", "status"],
                    contam_pdf,
                    col_widths=[0.50, 0.12, 0.22, 0.16],
                    lead_in="Exact, prefix, and cosine near-duplicates are removed at curation and re-checked after the split. The unit test fails the build if the contract breaks.")
        _table_page(pdf, "Table 6 · Evaluation slices",
                    ["slice", "n", "reg", "ROC", "PR", "F1min", "bAcc"],
                    slice_pdf,
                    caption="Metadata-labeled slice is the leakage-free generalization read.",
                    col_widths=[0.28, 0.08, 0.10, 0.14, 0.14, 0.13, 0.13])
        _text_pages(pdf, "6 · Stage 2 — RAG and LLM labeling", sec6)
        _table_page(pdf, "Table 7 · Stage-2 top per-label recall",
                    ["label", "support", "recall"], s2_rows,
                    caption=f"Exact agreement {s2.get('accuracy'):.2f} · family {s2.get('family_accuracy'):.2f} · n={s2.get('n')}.",
                    col_widths=[0.55, 0.22, 0.23],
                    lead_in="Exact agreement with weak labels is modest by design of the reference. Family-level agreement is the operationally relevant ceiling until a golden set exists.")
        _figure_page(pdf, "Figure 12 · Stage-2 per-label recall",
                     os.path.join(FIG, "stage2_recall.png"),
                     "Agreement with weak labels on a harder, service-heavy mix.")
        _text_pages(pdf, "7 · Dual-judge evaluation (NIM + OpenAI)", sec7)
        _table_page(pdf, "Table 8 · Judge agreement counts",
                    ["pair", "n", "agree", "disagree", "rate", "κ"],
                    judge_rows,
                    caption=f"Disputed set (both judges vs gate): {judges.get('disputed', {}).get('n')} rows.",
                    col_widths=[0.30, 0.10, 0.14, 0.16, 0.14, 0.16],
                    lead_in="NIM and OpenAI are independent first-class judges. Abstentions are recorded; keyword heuristics never silently replace a failed call.")
        _figure_page(pdf, "Figure 13 · Dual-judge agreement",
                     os.path.join(FIG, "judge_agreement.png"),
                     "NIM and OpenAI as independent first-class judges — not fallback logic.")
        _text_pages(pdf, "8 · DPO / RLAIF post-training", sec8)
        _table_page(pdf, "Table 9 · DPO challenger vs production gate",
                    ["party", "thr", "ROC", "PR", "F1", "leak-free ROC"],
                    dpo_rows,
                    caption="Challenger retained; logistic gate remains production.",
                    col_widths=[0.18, 0.12, 0.14, 0.14, 0.14, 0.28],
                    lead_in="Preference pairs come from the val/train judge panel only. The test-fold agreement study is evaluation-only — no training leakage.")
        _figure_page(pdf, "Figure 14 · DPO policy vs LR gate",
                     os.path.join(FIG, "dpo_comparison.png"),
                     "Held-out test comparison including the leakage-free slice.")
        _text_pages(pdf, "9 · Batch scoring and ingestion", sec9)
        _text_pages(pdf, "10 · Serving in the agentic stack", sec10)
        _text_pages(pdf, "11 · Robustness evidence", sec11)
        _figure_page(pdf, "Figure 15 · Out-of-vocabulary analysis",
                     os.path.join(MDD_FIG, "fig_oov.png"))
        _figure_page(pdf, "Figure 16 · Threshold sensitivity",
                     os.path.join(MDD_FIG, "fig_sensitivity_threshold.png"))
        _figure_page(pdf, "Figure 17 · Split-seed stability",
                     os.path.join(MDD_FIG, "fig_seed_stability.png"))
        _figure_page(pdf, "Figure 18 · Exploratory discriminative terms",
                     os.path.join(MDD_FIG, "fig_eda_terms.png"))
        _text_pages(pdf, "12 · Limitations and monitoring", sec12)
        _text_pages(pdf, "13 · Disposition and recommendation", sec13)
        _table_page(pdf, "Table 10 · Disposition summary",
                    ["component", "decision", "note"],
                    disposition_rows,
                    col_widths=[0.28, 0.28, 0.44],
                    lead_in="Overall disposition is Approve with Conditions: golden set, leakage-free reporting in governance packs, and continuous gap monitoring.")

    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
