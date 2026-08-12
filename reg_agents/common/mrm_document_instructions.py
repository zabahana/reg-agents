"""Shared SR 11-7 document-section instructions for agent system prompts.

Used by Developer, Validator, Validation, Report, and Audit agents so every
lifecycle / governance path asks for the same substantial Model Development
Document (MDD) and Independent Validation Report structure.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Model Development Document — first line
# --------------------------------------------------------------------------- #
MDD_SECTION_INSTRUCTIONS = """
Write a MODEL DEVELOPMENT DOCUMENT with these exact markdown headings and the
required content under each. Do not skip sections. Quote metrics verbatim from
the bake-off / source material; never invent numbers.

## 1. Purpose & Intended Use
- Business problem, decision the model supports, and who consumes the score/label.
- In-scope / out-of-scope use cases; explicit non-uses.
- Risk tier rationale (e.g. Tier-2 medium) and consumer-impact asymmetry
  (false negative vs false positive cost).

## 2. Data Profile & Label Provenance
- Source systems, window, volume, class balance.
- Label definition and provenance (supervised / weak labels / human adjudication).
- Known data limitations (PII masking, selection bias, weak-label noise).

## 3. Experimental Design & Split Protocol
- State explicitly that scoring holdout (if any) is reserved FIRST, then
  stratified train / validation / test on the modeling pool.
- State that all feature treatments (TF-IDF, encoders, scalers, tokenizers)
  are fit on the **training fold only**; validation and test are transform-only.
- Document seed, stratification target, and that the test fold is one-shot.

## 4. Candidate Models Considered (Bake-off)
- Table or structured list of every candidate (algorithm + representation).
- Hyperparameters and imbalance treatment (class_weight, scale_pos_weight, etc.).
- Primary selection metric on the **validation** fold (e.g. PR-AUC or minority
  PR-AUC under imbalance) and any secondary metrics.
- Acknowledge where a challenger beat the champion on a secondary metric.

## 5. Champion Selection & Decision Threshold
- Name the champion and why it won under the documented primary metric.
- Document validation-optimized decision cut-off (never assume 0.5) and the
  objective used (e.g. minority F1).
- Latency, interpretability, and operational constraints that affected promotion.

## 6. Champion Performance (Test / Holdout)
- Discrimination and calibration metrics on the held-out test fold only.
- Confusion matrix / per-class recall where applicable; train–test gap.
- For multi-stage systems: stage-1 vs stage-2 metrics separately.

## 7. Assumptions, Limitations & Challenger Path
- Maintained assumptions; failure modes; upgrade path (e.g. DistilBERT / Triton).
- What was *not* tried (e.g. Word2Vec) if material to the narrative.

## 8. Serving, Controls & Ongoing Monitoring Plan
- Serving path (in-process, Triton, NIM / TensorRT-LLM).
- Guardrails (input/output / NeMo Guardrails), HITL approve-override-escalate,
  drift/PSI or volume monitors, alert owners.
- Proposed periodic revalidation triggers.

## 9. Developer Attestation
- One short paragraph: developer name/role, date, attestation that the document
  matches the committed bake-off and that test was not used for selection.
""".strip()


# --------------------------------------------------------------------------- #
# Independent Validation Report — second line
# --------------------------------------------------------------------------- #
VALIDATION_SECTION_INSTRUCTIONS = """
Write an INDEPENDENT VALIDATION REPORT with these exact markdown headings.
You did **not** build the model. Perform effective challenge. Quote reported
metrics verbatim; never invent figures. Cite regulations in [brackets].

## 1. Scope & Materials Reviewed
- Model id / task, materials (MDD, leaderboard, data profile, metrics.json).
- Validation standards applied (SR 11-7 / OCC 2011-12 and any consumer-protection
  rules relevant to the use case).

## 2. Conceptual Soundness
- Is the methodology fit for purpose given the decision and risk tier?
- Maintained assumptions; alternative approaches that should have been considered.

## 3. Effective Challenge of Model Selection
- Was the champion chosen on an appropriate **primary metric** given class
  imbalance and business cost asymmetry?
- Critique threshold selection (val-optimized vs default 0.5).
- Note if a challenger was stronger on PR-AUC, minority recall, or calibration.
- Confirm split-before-treatment and that test was not used for selection.

## 4. Data Quality, Labels & Leakage
- Representativeness; label provenance / weak-label risk.
- Leakage audit: train/test contamination, target leakage, curation controls.

## 5. Outcomes Analysis
- Interrogate discrimination vs calibration; stability; per-class / family metrics.
- For multi-stage models: gate vs labeler performance and error concentration.

## 6. Fair-Lending / Consumer-Protection Review
- ECOA / UDAAP / FCRA (or relevant) exposure; adverse-impact considerations.
- Whether monitoring and HITL controls are adequate for the consumer impact.

## 7. Ongoing Monitoring, Guardrails & HITL Adequacy
- Drift/PSI, volume, Triton/NIM health, NeMo/native guardrails, HITL queue.
- Gaps in production controls relative to SR 11-7 ongoing monitoring expectations.

## 8. Findings (Severity-Ranked)
- Numbered findings with Severity (High/Medium/Low), Remediation, Owner, Due date.
- Prefer a markdown table: # | Severity | Finding | Remediation | Owner.

## 9. Disposition
- Exactly one of: **Approve** / **Approve with Conditions** / **Reject**.
- List specific conditions if conditional; required evidence before production use.
""".strip()


# --------------------------------------------------------------------------- #
# Governance / audit packaging — short reminders reused in report & audit
# --------------------------------------------------------------------------- #
GOVERNANCE_MDD_VAL_BRIDGE = """
When source material includes both a Model Development Document and a Validation
Report, keep their roles distinct in your synthesis:
- MDD (1st line) = how the model was built, selected, and is intended to be used.
- Validation (2nd line) = independent effective challenge and disposition.
Never conflate developer claims with validator conclusions. Prefer the
validator's disposition when they conflict, and surface the conflict explicitly.
""".strip()
