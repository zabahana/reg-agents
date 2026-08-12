# Triton model repository — fraud + complaint stage-1

Triton serves two models from this repository:

```
model_repository/
  fraud_xgb_gnn/          # FIL backend — fraud XGBoost
    config.pbtxt
    1/xgboost.json        # GENERATED
  complaint_stage1/       # Python backend — CMPL-REG-24 gate
    config.pbtxt
    1/
      model.py            # committed
      artifacts.joblib    # GENERATED (vectorizer + champion + threshold)
      meta.json           # GENERATED
```

## Generate models

```bash
# Fraud (FIL / XGBoost)
python scripts/export_triton_model.py

# Complaint stage-1 (TF-IDF + logistic/XGBoost champion, train-fitted only)
python scripts/export_complaint_triton_model.py

# Or inside the app image:
docker run --rm -v "$PWD/triton:/app/triton" reg-agents:latest \
  python scripts/export_triton_model.py
docker run --rm -v "$PWD/triton:/app/triton" -v "$PWD/data:/app/data" \
  reg-agents:latest python scripts/export_complaint_triton_model.py
```

Fraud feature order must match `fraud_server.py`:
`[amount, is_foreign, merchant_risk, hour, velocity_24h]`.

Complaint stage-1 accepts a raw **NARRATIVE** string and returns
**PROBABILITY** / **THRESHOLD**. The TF-IDF vectorizer inside the artifact was
fitted on the training fold only.

## Serve

```bash
docker run --rm -p 8000:8000 -p 8002:8002 \
  -v "$PWD/triton/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models

export TRITON_URL=http://localhost:8000
```

On Brev / GPU compose (`docker-compose.gpu.yml`) both `fraud-mcp` and
`complaint-mcp` receive `TRITON_URL=http://triton:8000`. Without Triton, each
MCP server falls back locally (`heuristic-local` / `local-sklearn`).

## Metrics

Triton Prometheus metrics on port **8002** (`/metrics`) are scraped by the
compose / GKE monitoring stack.
