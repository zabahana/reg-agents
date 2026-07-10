"""Generate a synthetic card-transaction dataset for the fraud model.

Deterministic (seeded) so runs are reproducible. Labels are generated from a
latent fraud propensity that mirrors the scoring features, with noise, so the
dataset is learnable but not trivial. This stands in for the RAPIDS/cuDF GPU
ETL step in the real pipeline.

    python scripts/generate_transactions.py --rows 1000 --fraud-rate 0.03
"""

from __future__ import annotations

import argparse
import csv
import os
import random

MERCHANT_CATEGORIES = [
    ("grocery", 0.05), ("gas", 0.08), ("restaurant", 0.07), ("retail", 0.10),
    ("electronics", 0.22), ("travel", 0.28), ("gambling", 0.55),
    ("crypto", 0.70), ("wire", 0.60), ("gift_card", 0.65), ("utilities", 0.04),
]

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "transactions", "sample_transactions.csv",
)


def _latent_fraud_prob(amount, is_foreign, merchant_risk, hour, velocity):
    p = 0.0
    p += min(amount / 5000.0, 1.0) * 0.35
    p += 0.20 if is_foreign else 0.0
    p += merchant_risk * 0.25
    p += 0.10 if (hour < 6 or hour >= 23) else 0.0
    p += min(velocity / 20.0, 1.0) * 0.10
    return min(p, 0.99)


def generate(rows: int, fraud_rate: float, seed: int = 42):
    rng = random.Random(seed)
    records = []
    for i in range(1, rows + 1):
        cat, base_risk = rng.choice(MERCHANT_CATEGORIES)
        merchant_risk = round(min(max(rng.gauss(base_risk, 0.08), 0.0), 1.0), 3)
        # Fraudulent transactions skew to higher amounts/velocity/odd hours.
        is_fraudster = rng.random() < fraud_rate
        if is_fraudster:
            amount = round(abs(rng.gauss(3500, 2500)) + 200, 2)
            hour = rng.choice([0, 1, 2, 3, 4, 23])
            velocity = rng.randint(6, 25)
            is_foreign = rng.random() < 0.55
        else:
            amount = round(abs(rng.gauss(120, 200)) + 5, 2)
            hour = rng.randint(6, 22)
            velocity = rng.randint(1, 6)
            is_foreign = rng.random() < 0.08
        p = _latent_fraud_prob(amount, is_foreign, merchant_risk, hour, velocity)
        # Label from latent propensity with noise (so it is learnable, not exact).
        label = 1 if (rng.random() < p * (0.9 if is_fraudster else 0.5)) else 0
        records.append({
            "txn_id": f"T{100000 + i}",
            "amount": amount,
            "is_foreign": str(is_foreign).lower(),
            "merchant_category": cat,
            "merchant_risk": merchant_risk,
            "hour": hour,
            "velocity_24h": velocity,
            "label": label,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--fraud-rate", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = generate(args.rows, args.fraud_rate, args.seed)
    fields = ["txn_id", "amount", "is_foreign", "merchant_category",
              "merchant_risk", "hour", "velocity_24h", "label"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    frauds = sum(r["label"] for r in records)
    print(f"wrote {len(records)} rows to {OUT} ({frauds} fraud, "
          f"{frauds / len(records) * 100:.1f}%)")


if __name__ == "__main__":
    main()
