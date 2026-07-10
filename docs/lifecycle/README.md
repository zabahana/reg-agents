# Model-development lifecycle (three lines of defense)

These artifacts show `reg-agents` running a **typical model development
process** end to end, mapped to the banking **three lines of defense** for model
risk management (SR 11-7 / OCC 2011-12):

```
Developer Agent  ─►  Validator Agent  ─►  Audit Agent
(1st line)           (2nd line)           (3rd line)
trains candidates,   independent          audits the process
selects a champion,  "effective           & controls, issues
writes model doc     challenge",          an audit opinion
                     validation report
```

The Developer agent runs a **real scikit-learn bake-off** (multiple candidate
models trained + evaluated on a hold-out), selects a champion against a
documented primary metric (ROC-AUC), and keeps the challengers for benchmarking.
The Validator independently critiques that choice, and Audit assesses whether the
governance process operated effectively. On GPU the bake-off maps to RAPIDS
cuML / XGBoost; the LLM reasoning maps from OpenAI to NVIDIA NIM with one env var.

Generate deterministically (no server needed):

```bash
python scripts/generate_lifecycle.py --task fraud     # one task
python scripts/generate_lifecycle.py --all            # every task
# or run through the live A2A stack:
python scripts/lifecycle_run.py --task fraud
```

The same pipeline runs unchanged across different model types — only the dataset
and task definition differ (`reg_agents/common/modeling.py`). That the champion
comes out *different* per task (a tree for fraud, logistic regression for credit)
is the point: the bake-off + three-lines process generalizes.

## `fraud/` — Card Transaction Fraud Detection

| # | Artifact | Owner (line of defense) |
| --- | --- | --- |
| 00 | [Model bake-off leaderboard](fraud/00_model_bakeoff_leaderboard.md) | Developer (1st) |
| 01 | [Model development document](fraud/01_model_development_document.md) | Developer (1st) |
| 02 | [Independent validation report](fraud/02_independent_validation_report.md) | Validator (2nd) |
| 03 | [Internal audit report](fraud/03_internal_audit_report.md) | Audit (3rd) |

> Note the built-in "effective challenge": the champion is picked on ROC-AUC, but
> the independent validator flags that a challenger scored higher on PR-AUC /
> recall under class imbalance and returns an **Approve-with-Conditions**
> disposition — exactly the kind of finding a real second line would raise.

## `credit/` — Consumer Credit Default (PD) Scorecard

| # | Artifact | Owner (line of defense) |
| --- | --- | --- |
| 00 | [Model bake-off leaderboard](credit/00_model_bakeoff_leaderboard.md) | Developer (1st) |
| 01 | [Model development document](credit/01_model_development_document.md) | Developer (1st) |
| 02 | [Independent validation report](credit/02_independent_validation_report.md) | Validator (2nd) |
| 03 | [Internal audit report](credit/03_internal_audit_report.md) | Audit (3rd) |

> A second task proves the bake-off generalizes: here **logistic regression** wins
> (ROC-AUC ~0.90) — a transparent, adverse-action-explainable model, which is
> exactly what you'd want to defend under **ECOA/Reg B and FCRA**. Protected-class
> attributes and proxies (age, ZIP) are deliberately excluded from the features,
> and the validator's fair-lending review calls out the remaining disparate-impact
> testing gap.
