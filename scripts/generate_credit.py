"""Generate a synthetic consumer-credit dataset for the PD (default) model.

Deterministic (seeded) so runs are reproducible. Labels come from a latent
default propensity driven by the same features the scorecard sees, with noise,
so the data is learnable but not trivial. Stands in for the RAPIDS/cuDF GPU ETL
step in the real pipeline.

NOTE (fair lending): protected-class attributes and their close proxies (age,
ZIP, sex, race) are deliberately **excluded** from the feature set — a point the
Validator/Audit agents pick up under ECOA/Reg B and FCRA.

    python scripts/generate_credit.py --rows 1200 --default-rate 0.15
"""

from __future__ import annotations

import argparse
import csv
import os
import random

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "credit", "sample_credit.csv",
)


def _latent_default_prob(dti, utilization, delinq_2yr, inquiries_6m,
                         annual_income, emp_length_yrs, is_homeowner):
    p = 0.05
    p += min(dti / 0.6, 1.0) * 0.30
    p += utilization * 0.25
    p += min(delinq_2yr / 5.0, 1.0) * 0.25
    p += min(inquiries_6m / 10.0, 1.0) * 0.10
    p -= min(annual_income / 150000.0, 1.0) * 0.18
    p -= min(emp_length_yrs / 10.0, 1.0) * 0.08
    p -= 0.05 if is_homeowner else 0.0
    return min(max(p, 0.01), 0.97)


def generate(rows: int, default_rate: float, seed: int = 42):
    rng = random.Random(seed)
    records = []
    for i in range(1, rows + 1):
        # A "stressed" borrower cohort skews the risk drivers; the rest are prime.
        stressed = rng.random() < default_rate * 2.2
        if stressed:
            dti = round(min(max(rng.gauss(0.42, 0.12), 0.05), 0.95), 3)
            utilization = round(min(max(rng.gauss(0.72, 0.18), 0.0), 1.0), 3)
            delinq_2yr = rng.randint(0, 5)
            inquiries_6m = rng.randint(2, 12)
            annual_income = round(abs(rng.gauss(48000, 18000)) + 12000, 0)
            emp_length_yrs = rng.randint(0, 4)
        else:
            dti = round(min(max(rng.gauss(0.22, 0.09), 0.02), 0.9), 3)
            utilization = round(min(max(rng.gauss(0.28, 0.16), 0.0), 1.0), 3)
            delinq_2yr = rng.choices([0, 1, 2], weights=[0.82, 0.14, 0.04])[0]
            inquiries_6m = rng.randint(0, 4)
            annual_income = round(abs(rng.gauss(96000, 32000)) + 25000, 0)
            emp_length_yrs = rng.randint(1, 15)
        loan_amount = round(abs(rng.gauss(18000, 12000)) + 1000, 0)
        num_open_accounts = rng.randint(2, 20)
        is_homeowner = rng.random() < (0.35 if stressed else 0.62)

        p = _latent_default_prob(dti, utilization, delinq_2yr, inquiries_6m,
                                 annual_income, emp_length_yrs, is_homeowner)
        default = 1 if (rng.random() < p * (0.95 if stressed else 0.55)) else 0
        records.append({
            "loan_id": f"L{200000 + i}",
            "dti": dti,
            "utilization": utilization,
            "delinq_2yr": delinq_2yr,
            "inquiries_6m": inquiries_6m,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "emp_length_yrs": emp_length_yrs,
            "num_open_accounts": num_open_accounts,
            "is_homeowner": str(is_homeowner).lower(),
            "default": default,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--default-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = generate(args.rows, args.default_rate, args.seed)
    fields = ["loan_id", "dti", "utilization", "delinq_2yr", "inquiries_6m",
              "annual_income", "loan_amount", "emp_length_yrs",
              "num_open_accounts", "is_homeowner", "default"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    defaults = sum(r["default"] for r in records)
    print(f"wrote {len(records)} rows to {OUT} ({defaults} default, "
          f"{defaults / len(records) * 100:.1f}%)")


if __name__ == "__main__":
    main()
