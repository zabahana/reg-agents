# Model Card — Card Transaction Fraud Detector (FRAUD-XGB-GNN-001)

## Purpose
Real-time fraud scoring for card-present and card-not-present transactions.
Outputs a fraud probability used to APPROVE / REVIEW / BLOCK a transaction.

## Methodology
GNN-enhanced XGBoost. A GraphSAGE GNN produces node embeddings over the
cardholder–merchant–device transaction graph; embeddings are concatenated with
tabular features and scored by an XGBoost classifier. Mirrors the NVIDIA
financial fraud detection AI Blueprint (RAPIDS + cuGraph + XGBoost + Triton).

## Training Data
18 months of labeled transactions (~240M rows). Labels from confirmed fraud +
chargebacks. Class imbalance ~0.3% positive; handled via scale_pos_weight and
focal-style reweighting. Trained on 8x GPUs with RAPIDS cuDF preprocessing.

## Features
Amount, merchant category & risk, transaction hour, 24h velocity, cross-border
flag, device reputation, graph embeddings. Age and ZIP are excluded to reduce
fair-lending proxy risk.

## Performance
AUC 0.971 on out-of-time test; at the BLOCK threshold (0.70), precision 0.62 /
recall 0.55. P95 inference latency 8 ms on Triton (dynamic batching, FP16).

## Known Limitations
- Graph embeddings refreshed nightly; intraday graph drift not captured.
- Performance degrades for new merchant categories with sparse history.
- Threshold tuning is region-agnostic; per-region calibration is a known gap.

## Monitoring
PSI on key features weekly; AUC and precision/recall tracked daily against
labeled feedback. No automated drift-triggered retraining yet (manual review).

## Documentation Gaps (self-reported)
- Outcomes analysis limited to 6 months; longer backtest pending.
- Disparate-impact testing across customer segments not yet documented.
- Analyst override logging exists but is not reconciled into monitoring.
