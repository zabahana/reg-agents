# Model Card — Credit Card PPNR Forecasting Model (PPNR-CARD-009)

## Purpose
Forecasts pre-provision net revenue (PPNR) components for the credit-card
portfolio (net interest income, late fees, interchange) under CCAR scenarios.

## Methodology
Econometric ARIMAX models linking PPNR components to macro drivers
(unemployment, rates, consumer spending), with an LSTM overlay to capture
nonlinear dynamics. Management overlays applied where data is limited.

## Training Data
15 years of monthly portfolio and macroeconomic data spanning at least one
downturn; supervisory scenario variables for projection.

## Features / Drivers
Unemployment, policy rate, consumer spending indices, balance/active trends,
seasonality.

## Performance
Backtest MAPE within tolerance on out-of-time windows; scenario responses
directionally consistent with economic intuition.

## Known Limitations
- LSTM overlay reduces interpretability; overlay contribution must be bounded.
- Structural breaks (e.g., policy shifts) challenge historical relationships.

## Monitoring
Quarterly backtesting vs. actuals; scenario sensitivity review; overlay
governance and documentation.

## Documentation Gaps (self-reported)
- Sensitivity analysis across all supervisory scenarios is incomplete.
- Benchmark (challenger) model comparison outstanding.
