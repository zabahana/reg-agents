# Model Card — AML Transaction Monitoring Model (AML-TM-021)

## Purpose
Detects potentially suspicious activity across deposit and payment channels to
generate alerts for investigator triage and potential SAR filing (BSA/AML).

## Methodology
Hybrid: deterministic typology rules (structuring, rapid movement of funds)
plus an Isolation Forest for anomaly detection, plus a GNN over the
counterparty graph to surface layering typologies. Alert scores prioritize
investigator queues.

## Training / Tuning Data
Historical alerts, investigation dispositions, and confirmed SARs over 5 years.
Above-the-line / below-the-line (ATL/BTL) sampling used for threshold tuning.

## Features
Transaction amount/frequency, channel, geography risk, counterparty graph
features, velocity, structuring indicators, sanctions-screening flags.

## Performance
Productive-alert rate improved vs. rules-only baseline; false-positive rate
reduced ~22%. Coverage assessment maps rules to known typologies.

## Known Limitations
- GNN typology coverage is incomplete for trade-based money laundering.
- Threshold calibration is enterprise-wide, not segment-specific.

## Monitoring
Monthly productive/false-positive tracking; periodic ATL/BTL testing; threshold
changes require Financial Crimes governance approval.

## Documentation Gaps (self-reported)
- BTL testing cadence not fully documented.
- Sanctions fuzzy-match configuration validation is pending review.
