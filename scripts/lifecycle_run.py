"""CLI: run the model-development lifecycle through the live A2A stack.

Requires the stack to be running (scripts/run_local.sh). Fans out
Developer -> Validator -> Audit via the lifecycle orchestrator and prints the
three documents. For deterministic committed artifacts, use
scripts/generate_lifecycle.py instead.

    python scripts/lifecycle_run.py --task fraud
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reg_agents.agents.lifecycle_orchestrator import run_lifecycle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the model-development lifecycle.")
    ap.add_argument("--task", default="fraud", help="task_id, e.g. 'fraud'")
    ap.add_argument("--json", action="store_true", help="print full result as JSON")
    args = ap.parse_args()

    result = run_lifecycle(args.task)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    for title, key in [
        ("MODEL DEVELOPMENT DOCUMENT (1st line)", "model_development_document"),
        ("INDEPENDENT VALIDATION REPORT (2nd line)", "validation_report"),
        ("INTERNAL AUDIT REPORT (3rd line)", "audit_report"),
    ]:
        print("=" * 78)
        print(title)
        print("=" * 78 + "\n")
        print(result.get(key, "[missing]"))
        print()


if __name__ == "__main__":
    main()
