# Model Development — Stage-1 Regulatory Gate (CMPL-REG-24)

Publication-grade **Model Development Document** for the stage-1 binary gate
of the complaint → regulation classifier, plus every artifact needed to
reproduce or audit it. All numbers are recomputed from
`data/complaints/cfpb_complaints.csv` at generation time.

## Contents

| Artifact | What it is |
|---|---|
| [`model_development_document.md`](model_development_document.md) | The document (markdown, figures inline) |
| [`model_development_document.pdf`](model_development_document.pdf) | Same document as a paginated PDF |
| [`results.json`](results.json) | Machine-readable metrics: leaderboard, OOV, sensitivity, ablation, seed stability |
| `figures/` | All numbered figures (EDA, curves, comparison, OOV, sensitivity) |
| `artifacts/tfidf_vectorizer.joblib` | Train-fitted TF-IDF featurizer (30k, 1–2 gram) |
| `artifacts/model_*.joblib` | Fitted candidates — champion + challengers |
| `artifacts/split_indices.json` | Exact 80/10/10 train/validation/test membership (seed 42) |
| `artifacts/environment.txt` | Package versions used for the committed run |

## Protocol covered by the document

1. **Exploratory analysis** — class balance, narrative length by class,
   regulatory rate by product, most discriminative terms.
2. **Stratified 80/10/10 split** — test fold split off first, touched once.
3. **Four minority-balanced candidates** — logistic regression, XGBoost,
   LightGBM, fine-tuned DistilBERT (class-weighted loss).
4. **Selection on validation minority PR-AUC** — with a 96.6% majority class,
   majority PR-AUC saturates; the minority class is where models differ.
5. **Validation-optimized decision cut-off** — each candidate's threshold on
   P(regulatory) maximizes minority F1 on the validation fold; the default
   0.5 is never assumed.
6. **Out-of-vocabulary analysis** — TF-IDF vocabulary exposure vs BERT
   subwords; error rate by OOV quartile.
7. **Sensitivity analysis** — decision-threshold sweep, input perturbations,
   class-weight ablation, split-seed stability.

The production gate (`reg_agents/common/complaints.py`) follows the same
split-and-select protocol at container start; companion governance documents
(data profile, development document, independent validation report) live in
[`docs/complaint_model/`](../complaint_model/README.md).

## Regenerate

```bash
pip install lightgbm torch transformers   # research extras
python scripts/generate_model_development_doc.py                # full run
python scripts/generate_model_development_doc.py --skip-bert    # CPU-light
```

Note: fine-tuned DistilBERT weights (~256 MB) are intentionally not
committed; rerun the script to reproduce them.
