"""Train + export the fraud model for Triton's FIL backend.

Produces a Triton model repository at `triton/model_repository/fraud_xgb_gnn/`:

    fraud_xgb_gnn/
      config.pbtxt          # FIL backend config (committed)
      1/
        xgboost.json        # the trained XGBoost model (generated here)

The feature vector order MUST match what the fraud MCP server sends to Triton
(`reg_agents/mcp_servers/fraud_server.py`):

    [amount, is_foreign, merchant_risk, hour, velocity_24h]

Deploy: upload this directory to a GCS bucket (or PVC) mounted at /models in the
Triton pod. See k8s/README.md.

    python scripts/export_triton_model.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "transactions", "sample_transactions.csv")
MODEL_DIR = os.path.join(ROOT, "triton", "model_repository", "fraud_xgb_gnn")

# Order MUST match fraud_server.py's Triton payload.
FEATURES = ["amount", "is_foreign", "merchant_risk", "hour", "velocity_24h"]

CONFIG_PBTXT = """# Triton FIL backend config for the fraud XGBoost model.
# Feature order must match reg_agents/mcp_servers/fraud_server.py:
#   [amount, is_foreign, merchant_risk, hour, velocity_24h]
backend: "fil"
max_batch_size: 8192
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 5 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]
instance_group [{ kind: KIND_GPU }]
parameters [
  { key: "model_type"    value: { string_value: "xgboost_json" } },
  # output_class=false → return the raw model output. For an XGBoost
  # binary:logistic model that is the positive-class probability (0..1),
  # which is exactly what fraud_server reads at outputs[0].data[0].
  { key: "output_class"  value: { string_value: "false" } },
  { key: "storage_type"  value: { string_value: "AUTO" } }
]
dynamic_batching { max_queue_delay_microseconds: 1000 }
"""


def main() -> None:
    # config.pbtxt never depends on xgboost, so write it first.
    os.makedirs(MODEL_DIR, exist_ok=True)
    config_path = os.path.join(MODEL_DIR, "config.pbtxt")
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(CONFIG_PBTXT)
    print(f"wrote {config_path}")

    try:
        import xgboost as xgb
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"\nCould not import xgboost: {exc}\n\n"
            "xgboost needs an OpenMP runtime. On Linux (Docker image / CI / GKE) "
            "it works out of the box. On macOS run `brew install libomp`, OR "
            "generate the model inside the app container:\n\n"
            "  docker run --rm -v \"$PWD/triton:/app/triton\" reg-agents:latest \\\n"
            "    python scripts/export_triton_model.py\n\n"
            "config.pbtxt was still written; only the model file is missing.\n"
        )
        raise SystemExit(1)

    df = pd.read_csv(CSV)
    x = pd.DataFrame(index=df.index)
    x["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    x["is_foreign"] = (
        df["is_foreign"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    )
    x["merchant_risk"] = pd.to_numeric(df["merchant_risk"], errors="coerce").fillna(0.0)
    x["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0.0)
    x["velocity_24h"] = pd.to_numeric(df["velocity_24h"], errors="coerce").fillna(0.0)
    x = x[FEATURES]
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    clf.fit(x, y)
    auc = clf.score(x, y)

    version_dir = os.path.join(MODEL_DIR, "1")
    os.makedirs(version_dir, exist_ok=True)
    model_path = os.path.join(version_dir, "xgboost.json")
    clf.get_booster().save_model(model_path)

    print(f"wrote {model_path}")
    print(f"features (order matters): {FEATURES}")
    print(f"train accuracy: {auc:.4f}")
    print("\nDeploy: upload triton/model_repository/ to GCS (or a PVC) mounted at "
          "/models in the Triton pod. See k8s/README.md.")


if __name__ == "__main__":
    main()
