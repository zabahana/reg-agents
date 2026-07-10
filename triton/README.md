# Triton model repository — fraud model

Triton serves the fraud model from this repository via the **FIL backend**
(Forest Inference Library — XGBoost / LightGBM / sklearn & cuML forests).

```
model_repository/
  fraud_xgb_gnn/
    config.pbtxt        # FIL backend config (committed)
    1/
      xgboost.json      # trained model — GENERATED, not committed
```

## Generate the model

```bash
# On Linux / CI / GKE (has libgomp) or macOS with `brew install libomp`:
python scripts/export_triton_model.py

# Or generate inside the app image (no local OpenMP needed):
docker run --rm -v "$PWD/triton:/app/triton" reg-agents:latest \
  python scripts/export_triton_model.py
```

This trains an XGBoost classifier on `data/transactions/sample_transactions.csv`
and writes `1/xgboost.json`. The feature order **must** match what the fraud MCP
server sends (`reg_agents/mcp_servers/fraud_server.py`):

```
[amount, is_foreign, merchant_risk, hour, velocity_24h]
```

## Serve it

- **Local (CPU) Triton for testing:**
  ```bash
  docker run --rm -p 8000:8000 -p 8002:8002 \
    -v "$PWD/triton/model_repository:/models" \
    nvcr.io/nvidia/tritonserver:24.08-py3 \
    tritonserver --model-repository=/models
  # then point the app at it:
  export TRITON_URL=http://localhost:8000
  ```
  (config.pbtxt requests a GPU instance group; for CPU-only local testing change
  `instance_group` kind to `KIND_CPU`.)

- **GKE:** upload `model_repository/` to a GCS bucket and mount it at `/models`
  in the Triton pod (gcsfuse CSI driver) or copy into a PVC. See
  [`../k8s/README.md`](../k8s/README.md).

Until a model is served, the fraud MCP server automatically falls back to its
transparent local heuristic and reports `backend: heuristic-local`.

## Metrics

Triton exposes Prometheus metrics on port **8002** at `/metrics` (inference
count, queue/compute latency, success/failure). These are scraped by the
`ServiceMonitor` in [`../k8s/monitoring/`](../k8s/monitoring/) and shown in
Grafana.
