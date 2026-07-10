# Model Card — Consumer Card PD Origination Scorecard (CREDIT-PD-007)

## Purpose
Estimates probability of default (PD) at origination for consumer credit-card
applications; drives approve/decline and line-assignment decisions.

## Methodology
Two-stage: a regularized logistic regression for interpretability and reason
codes, blended with a gradient-boosting model for lift. Final score calibrated
to a 24-month default definition. Reason codes derived from the logistic layer
to support ECOA adverse-action notices.

## Training Data
7 years of application + bureau + performance data (~12M applications). Reject
inference applied for declined populations. Bureau attributes (FICO,
VantageScore, trade-line features) integrated under FCRA.

## Features
Bureau scores, utilization, delinquency history, inquiry velocity, income
proxies, tenure. Protected-class attributes excluded. Geography features
constrained and reviewed for proxy risk.

## Performance
KS 0.42, AUC 0.81 out-of-time. Calibration within tolerance across score bands.
Stable PSI over the last 4 quarters.

## Fair-Lending Testing
Disparate-impact analysis using BISG proxies across race/ethnicity and sex;
no statistically significant adverse disparity at the approved threshold. A
less-discriminatory-alternative (LDA) search was performed on feature sets.

## Known Limitations
- Reject inference introduces assumption risk in thin-file segments.
- Alternative-data features have shorter performance history.

## Monitoring
Quarterly performance + PSI; annual fair-lending revalidation; adverse-action
reason-code fidelity checks.

## Documentation Gaps (self-reported)
- LDA search documentation is partial for the boosting layer.
- Explanation-fidelity validation for the blended score is outstanding.
