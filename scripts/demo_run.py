"""CLI demo: run an end-to-end governance review through the orchestrator.

Assumes the stack is running (scripts/run_local.sh) OR calls the orchestrator
in-process if agents are reachable. Prints the final report.

    python scripts/demo_run.py --model FRAUD-XGB-GNN-001
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reg_agents.agents.orchestrator import run_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model governance review.")
    parser.add_argument("--model", default="FRAUD-XGB-GNN-001", help="model_id to review")
    parser.add_argument("--amount", type=float, default=4200.0)
    parser.add_argument("--foreign", action="store_true")
    parser.add_argument("--merchant-risk", type=float, default=0.6)
    parser.add_argument("--hour", type=int, default=2)
    parser.add_argument("--velocity", type=int, default=9)
    parser.add_argument("--json", action="store_true", help="print full trace as JSON")
    args = parser.parse_args()

    txn = {
        "amount": args.amount,
        "is_foreign": args.foreign,
        "merchant_risk": args.merchant_risk,
        "hour": args.hour,
        "velocity_24h": args.velocity,
    }
    result = run_review(args.model, txn)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 78)
    print(f"GOVERNANCE REVIEW — {args.model}")
    print("=" * 78)
    print("\n--- Validation Findings ---\n")
    print(result["validation_findings"])
    print("\n--- Fraud / Performance Analysis ---\n")
    print(result["fraud_analysis"])
    print("\n--- Regulatory Context ---\n")
    print(result["regulatory_context"])
    print("\n" + "=" * 78)
    print("FINAL REPORT")
    print("=" * 78 + "\n")
    print(result["report"])


if __name__ == "__main__":
    main()
