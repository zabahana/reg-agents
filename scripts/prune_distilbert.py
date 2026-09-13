#!/usr/bin/env python3
"""Create and evaluate a globally magnitude-pruned DistilBERT challenger.

This is deliberately an experiment, not a replacement for the serving gate.
It measures quality and latency before a compressed model can be considered
for promotion. It requires the optional DPO artifacts plus torch/transformers.

Example:
  python scripts/prune_distilbert.py --amount 0.30 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "complaint_model" / "artifacts" / "dpo_policy"
OUT_DIR = ROOT / "docs" / "optimization" / "results"
PRUNED_DIR = ROOT / "docs" / "optimization" / "artifacts"
sys.path.insert(0, str(ROOT))
from reg_agents.common import complaints as C  # noqa: E402


def model_size_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1_000_000, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", type=float, default=0.30, help="Fraction of Linear weights to prune.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    if not 0 < args.amount < 1:
        raise SystemExit("--amount must be between 0 and 1.")
    if not ARTIFACTS.exists():
        raise SystemExit(f"Missing DPO artifact: {ARTIFACTS}. Run train_dpo_from_judges.py first.")

    # The repository's triton/ directory can shadow torch's triton package.
    saved_path = list(sys.path)
    sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != str(ROOT)]
    try:
        import torch
        import torch.nn.utils.prune as prune
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    finally:
        sys.path = saved_path

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; pass --device cpu.")
    device = args.device
    model = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS)
    parameters = [(module, "weight") for module in model.modules() if isinstance(module, torch.nn.Linear)]
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=args.amount)
    for module, name in parameters:
        prune.remove(module, name)  # materialize zeros for deployment/export

    output_artifact = PRUNED_DIR / f"distilbert-global-l1-{args.amount:.0%}"
    output_artifact.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_artifact)
    tokenizer.save_pretrained(output_artifact)

    _x_tr, _x_va, x_test, _y_tr, _y_va, y_test = C.split_stage1(C.load_complaints())
    texts = list(x_test)
    scores: list[float] = []
    elapsed_s = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[i:i + args.batch_size], truncation=True, padding=True,
                max_length=args.max_length, return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(**encoded).logits
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_s += time.perf_counter() - started
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())

    predictions = np.asarray(scores) >= 0.5
    zero_weights = sum(int((module.weight == 0).sum().item()) for module, _ in parameters)
    total_weights = sum(module.weight.numel() for module, _ in parameters)
    result = {
        "experiment": "global_magnitude_pruning",
        "base_artifact": str(ARTIFACTS.relative_to(ROOT)),
        "requested_pruning_fraction": args.amount,
        "realized_linear_sparsity": round(zero_weights / total_weights, 4),
        "device": device,
        "test_rows": len(y_test),
        "metrics": {
            "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
            "f1": round(float(f1_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions)), 4),
            "recall": round(float(recall_score(y_test, predictions)), 4),
            "inference_ms_per_row": round(1_000 * elapsed_s / len(texts), 3),
        },
        # Dense zero weights do not guarantee a smaller serialized model or faster
        # kernels; this prevents an unsupported serving claim.
        "artifact_size_mb_before_pruning": model_size_mb(ARTIFACTS),
        "artifact_size_mb_after_pruning": model_size_mb(output_artifact),
        "pruned_artifact": str(output_artifact.relative_to(ROOT)),
        "note": "Export to sparse/structured runtime and benchmark before claiming latency or memory gains.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"distilbert-pruning-{args.amount:.0%}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
