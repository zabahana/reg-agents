# Complaint → Regulation Classifier (CMPL-REG-24) — model documentation

The third model in the agentic system: classifies **real consumer complaint
narratives** (CFPB Consumer Complaint Database) into a **24-category regulation
taxonomy** (UDAAP, sales practices, FCRA, FDCPA, Reg E, Reg Z, RESPA, BSA/AML,
…) using a two-stage architecture:

1. **Stage 1 — binary gate:** TF-IDF + logistic-regression/XGBoost bake-off
   answers *"is this complaint regulatory at all?"* (champion picked on
   validation PR-AUC over an 80/10/10 split of the modeling pool — a 5%
   scoring holdout is reserved first for the batch-ingestion layer — and
   deployed at a validation-optimized decision cut-off rather than the
   default 0.5).
2. **Stage 2 — RAG + LLM:** if regulatory, retrieval over the regulation/policy
   corpus + LLM reasoning with few-shot examples assigns the category and
   returns a **citation** from the retrieved excerpts plus a rationale. A
   keyword scorer is the deterministic no-LLM fallback.

## Documents

| Artifact | Markdown | PDF |
|---|---|---|
| Data Profile & Processing | [`00_data_profile.md`](00_data_profile.md) | [`00_data_profile.pdf`](00_data_profile.pdf) |
| Model Development Document (1st line) | [`01_model_development_document.md`](01_model_development_document.md) | [`01_model_development_document.pdf`](01_model_development_document.pdf) |
| Independent Validation Report (2nd line) | [`02_validation_report.md`](02_validation_report.md) | [`02_validation_report.pdf`](02_validation_report.pdf) |

Both include accuracy **figures** (ROC/PR curves, confusion matrix, per-category
recall, label distribution) and **tables** (bake-off leaderboard, per-category
support/recall). Machine-readable metrics: [`metrics.json`](metrics.json)
(consumed by the Streamlit UI's complaint panel).

The **publication-grade Model Development Document** for the stage-1 gate —
EDA, 80/10/10 split, four-model bake-off (incl. fine-tuned DistilBERT),
validation-optimized decision cut-off, OOV and sensitivity analyses, with
fitted-model artifacts — lives in
[`docs/model_development/`](../model_development/README.md).

## Data

`data/complaints/cfpb_complaints.csv` — 4,000 curated, real, PII-redacted
narratives. The curation pass (`scripts/fetch_cfpb_complaints.py`) mirrors
**NVIDIA NeMo Data Curator** stages: length filtering, exact + near
deduplication, PII verification, balanced sampling. Ground truth is **weak
supervision** from the CFPB product/issue taxonomy + keyword rules — flagged as
a limitation with a golden-set condition in the validation report.

## Regenerate

```bash
python scripts/fetch_cfpb_complaints.py             # refresh data (network)
python scripts/generate_complaint_data_profile.py   # data profile + quality checks
python scripts/generate_complaint_model_docs.py     # retrain + re-evaluate + re-render
python scripts/generate_complaint_model_docs.py --no-llm   # offline (keyword fallback)
```

## Serving

- MCP tools: `classify_complaint`, `list_regulation_taxonomy`,
  `sample_complaints`, `get_model_metrics` (`complaint-mcp`, :9105)
- A2A agent: **Complaint Agent** (:8110) — classification + analyst summary,
  Prometheus counter `complaint_classifications_total{label,mode}`
- UI: sidebar **② Complaint classification** — pick a real CFPB complaint,
  see the label, citation, rationale, and the committed accuracy metrics.
- Batch ingestion: `python scripts/score_batch.py` (or UI **④ Batch
  scoring**) scores a CSV — default: the reserved 5% holdout — and emits
  `complaint_id, complaint, score, label, llm_reasoning` per row
  (sample: `data/scoring/sample_scored_holdout.csv`).
