# Model Card — Small Business LGD Model (CREDIT-LGD-014)

## Purpose
Estimates Loss Given Default for small-business loans; feeds CCAR stress-test
loss projections and economic capital.

## Methodology
Gradient boosted trees on facility- and borrower-level features, calibrated to
observed workout recoveries. Macro overlays applied under CCAR scenarios.

## Training Data
2009–2024 workout/recovery data. Downturn period included to capture stress
dynamics. Known sparsity in certain industry segments.

## Features
Collateral type & coverage, seniority, industry, utilization, time-in-default,
macro variables (unemployment, GDP). Protected-class attributes excluded;
fair-lending review performed under ECOA.

## Performance
Out-of-time RMSE within tolerance; segment-level bias observed for CRE-heavy
facilities (over-prediction of recovery).

## Known Limitations
- Thin data for newer industry segments.
- Recovery timing assumptions static across scenarios.

## Monitoring
Quarterly backtesting vs realized recoveries; PSI on macro inputs.

## Documentation Gaps (self-reported)
- Scenario sensitivity analysis incomplete.
- Benchmark model comparison outstanding.
