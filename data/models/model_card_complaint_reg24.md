# Model Card — Complaint → Regulation Classifier (CMPL-REG-24)

## Purpose
Classifies inbound consumer-complaint narratives against a 24-category federal
regulation taxonomy (UDAAP, sales practices, FCRA, FDCPA, Reg E/EFTA, Reg
Z/TILA, RESPA, TISA/Reg DD, Reg CC, BSA/AML, GLBA, ECOA/Reg B, SCRA/MLA, …) to
route complaints with a regulatory nexus to compliance review and surface
regulatory risk trends in real time.

## Methodology
Two-stage architecture, deployed as an MCP tool server + A2A agent:

1. **Stage 1 — binary regulatory gate.** TF-IDF features (1–2 grams, 30k
   vocabulary) into a logistic-regression vs XGBoost bake-off over a
   stratified 80/10/10 train/validation/test split (applied after reserving
   a stratified 5% scoring holdout for the batch-ingestion layer —
   `scripts/score_batch.py` / UI upload); champion selected on
   validation PR-AUC and deployed at a validation-optimized decision cut-off
   (maximizing minority-class F1) rather than the default 0.5. Millisecond
   CPU inference; gates non-regulatory service complaints away from the
   LLM path.
2. **Stage 2 — RAG + LLM regulation labeling.** Retrieval over the
   regulation/policy corpus (NeMo Retriever embeddings; FAISS locally,
   cuVS/Milvus at scale) grounds an LLM (NVIDIA NIM, Llama-3.1-8B) prompted
   with the full 24-category taxonomy, disambiguation rules, and 15 curated
   few-shot examples. Output is strict JSON: label, confidence, rationale, and
   the cited source excerpt. A deterministic keyword scorer serves as the
   no-LLM fallback (`mode=fallback`).

## Training / Grounding Data
4,000 curated, PII-redacted narratives from the public **CFPB Consumer
Complaint Database** (real data; all 24 categories covered). Curation mirrors
NeMo Data Curator stages: length filter, exact + near deduplication, PII
verification, per-issue balanced sampling. Ground truth is **weak
supervision** derived from the CFPB product/issue taxonomy plus narrative
keyword rules — a documented limitation (see Validation).

## Features / Inputs
Free-text complaint narrative (truncated to 1,800 chars); retrieved regulation
passages at stage 2. No demographic or protected-class attributes are used.

## Performance (committed validation run — docs/complaint_model/metrics.json)
- **Stage 1 (champion: logistic regression, cut-off 0.788 tuned on
  validation):** test PR-AUC 0.991 · ROC-AUC 0.797 · F1 0.95 · precision
  0.97 · recall 0.94 (one-shot held-out test, n=380; 80/10/10 stratified
  split after the 5% scoring-holdout reserve).
- **Stage 2 (vs weak labels, stratified n=115):** exact agreement 0.35;
  **regulation-family agreement 0.54**; macro-F1 0.29. Disagreements
  concentrate within regulation families (e.g., FCRA accuracy vs FCRA
  reinvestigation) where the weak reference itself is noisy.
- Full evidence with figures/tables: `docs/complaint_model/` (development
  document + independent validation report, MD + PDF).

## Guardrails & Monitoring
- Taxonomy whitelist on LLM output (invalid label → rejected).
- Strict-JSON parsing with deterministic keyword fallback, labeled by mode.
- Stage-1 gating limits LLM exposure and cost.
- Prometheus counter `complaint_classifications_total{label,mode}` (a
  `mode=fallback` spike alerts LLM-path degradation); agent latency/error via
  `/metrics`; per-request OpenTelemetry traces across A2A/MCP hops.

## Known Limitations
- **Weak labels**: stage-2 "accuracy" measures agreement with a noisy
  reference, not human truth. Condition: human-adjudicated golden set
  (≥25/category) before decision-grade use.
- Thin support in several categories (SCRA/MLA, sales practices, adverse
  action) — per-category metrics unstable.
- LLM nondeterminism/drift: prompt + model versions pinned; re-evaluation
  required on any provider/model change.
- Complaint routing affects remediation speed; routing-outcome parity should
  be monitored across products.

## Governance Status
Independent validation completed; disposition **Approve with Conditions**
(golden set; PSI monitoring on the stage-1 score distribution; re-validation
on model/prompt change).
