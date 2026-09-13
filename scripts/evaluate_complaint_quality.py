#!/usr/bin/env python3
"""Evaluate the live complaint path against the reserved weak-label holdout.

This is a controlled serving evaluation: run it once per NIM/TensorRT-LLM
profile with the same limit, prompt configuration, and model version. Results
are agreement with the CFPB-derived weak reference—not human-adjudicated
accuracy. Use a golden set before making a production quality claim.

Examples:
  python scripts/evaluate_complaint_quality.py --profile fp16-trtllm --limit 20
  NIM_BASE_URL=http://nim-int8:8000/v1 \
    python scripts/evaluate_complaint_quality.py --profile int8-trtllm --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reg_agents.common import complaints as C  # noqa: E402
from reg_agents.config import get_settings  # noqa: E402

OUT_DIR = ROOT / "docs" / "optimization" / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True,
                        help="Name of the deployed engine/profile under test.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Holdout records to evaluate; use the same value per profile.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive.")

    settings = get_settings()
    holdout = C.scoring_holdout().head(args.limit)
    expected_gate = list(holdout["is_regulatory"].astype(bool))
    predicted_gate: list[bool] = []
    expected_label: list[str] = []
    predicted_label: list[str] = []
    llm_expected: list[str] = []
    llm_predicted: list[str] = []
    modes: Counter[str] = Counter()
    rows = []

    started = time.perf_counter()
    for position, (_, row) in enumerate(holdout.iterrows(), 1):
        narrative = str(row["narrative"])
        stage1 = C.classify_binary(narrative)
        predicted_gate.append(bool(stage1["is_regulatory"]))
        stage2 = {"mode": "stage1_gate", "label": C.NON_REGULATORY}
        if stage1["is_regulatory"]:
            stage2 = C.classify_regulation(narrative, use_llm=True)
        mode = str(stage2["mode"])
        modes[mode] += 1
        expected = str(row["label"])
        predicted = str(stage2["label"])
        expected_label.append(expected)
        predicted_label.append(predicted)
        if expected != C.NON_REGULATORY and mode == "rag_llm":
            llm_expected.append(expected)
            llm_predicted.append(predicted)
        rows.append({
            "complaint_id": int(row["complaint_id"]),
            "reference_label": expected,
            "predicted_label": predicted,
            "stage1_expected_regulatory": bool(row["is_regulatory"]),
            "stage1_predicted_regulatory": bool(stage1["is_regulatory"]),
            "stage2_mode": mode,
            "stage2_confidence": stage2.get("confidence"),
        })
        print(f"{position}/{len(holdout)} {mode}: {predicted}", flush=True)

    def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "weak_label_agreement_not_human_adjudicated_accuracy",
        "profile": args.profile,
        "provider": settings.llm_provider,
        "model": settings.nim_model if settings.llm_provider == "nim" else settings.openai_model,
        "endpoint": settings.nim_base_url if settings.llm_provider == "nim" else settings.openai_base_url,
        "n_records": len(holdout),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "stage1": binary_metrics(expected_gate, predicted_gate),
        "stage2": {
            "mode_counts": dict(modes),
            "rag_llm_coverage": round(modes["rag_llm"] / len(holdout), 4),
            "end_to_end_exact_agreement": round(
                float(accuracy_score(expected_label, predicted_label)), 4
            ),
            "rag_llm_exact_agreement": (
                round(float(accuracy_score(llm_expected, llm_predicted)), 4)
                if llm_expected else None
            ),
            "rag_llm_evaluated_rows": len(llm_expected),
        },
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else OUT_DIR / (
        f"quality-{args.profile}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stage2"], indent=2))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
