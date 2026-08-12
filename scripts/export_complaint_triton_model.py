"""Export the stage-1 complaint gate for Triton's Python backend.

Writes:

    triton/model_repository/complaint_stage1/
      config.pbtxt          (committed)
      1/
        model.py            (committed)
        artifacts.joblib    (generated: vectorizer + champion + threshold)
        meta.json           (generated)

    python scripts/export_complaint_triton_model.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(
    ROOT, "triton", "model_repository", "complaint_stage1", "1",
)


def main() -> None:
    from reg_agents.common import complaints as C

    print("Training stage-1 gate (train-fitted TF-IDF only)…")
    s1 = C.train_stage1()
    champion = s1["champion"]
    artifacts = {
        "vectorizer": s1["vectorizer"],
        "model": s1["models"][champion],
        "threshold": float(s1["threshold"]),
        "champion": champion,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        import joblib
    except ImportError:
        import pickle as joblib  # noqa: N813 — duck-typed dump API unused
        sys.stderr.write("joblib not installed; cannot export artifacts\n")
        raise SystemExit(1)

    art_path = os.path.join(OUT_DIR, "artifacts.joblib")
    joblib.dump(artifacts, art_path)
    meta = {
        "champion": champion,
        "threshold": float(s1["threshold"]),
        "leaderboard": s1.get("leaderboard", []),
        "note": (
            "TF-IDF vectorizer fitted on the training fold only; "
            "val/test never seen at fit time."
        ),
    }
    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {art_path}")
    print(f"wrote {meta_path}")
    print(f"champion={champion} threshold={s1['threshold']}")


if __name__ == "__main__":
    main()
