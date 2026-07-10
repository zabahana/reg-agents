# Run reg-agents on NVIDIA GPUs with Brev (no hyperscaler)

[NVIDIA Brev](https://brev.nvidia.com) gives you a self-serve GPU VM (L40S /
A100 / H100…) in ~2 minutes. Combined with **hosted NIM** for the LLM +
embeddings, the whole demo runs on **NVIDIA infrastructure end to end** — no GCP,
AWS, or Azure account needed.

```
  hosted NIM (build.nvidia.com)  ── LLM + embeddings ─────────┐
                                                              ▼
  Brev GPU VM:  Triton (fraud model, GPU) ◀── agents + MCP + UI + Grafana
```

- **LLM/embeddings:** hosted NIM — set `NIM_API_KEY` (see below).
- **Fraud model:** Triton on the Brev GPU via `docker-compose.gpu.yml`.
- **Everything else** (agents, MCP servers, UI, Prometheus, Grafana): CPU
  containers on the same VM.

---

## 1. Get a GPU instance

**Option A — CLI**
```bash
# Install + log in (browser OAuth, or `brev login --token` for headless)
brew install brevdev/homebrew-brev/brev || pip install brev
brev login

# One L40S is plenty for the FIL fraud model. See `brev create --help` /
# the GPU Types catalog for other options (A100, H100, ...).
brev create reg-agents --gpu "nebius.l40sx1.pcie"
brev shell reg-agents
```

**Option B — Console** ([brev.nvidia.com](https://brev.nvidia.com))
Create Instance → pick a GPU (L40S/A100/H100) → base image with Docker + CUDA →
expose ports **8501, 8000, 8002, 3000** → Create, then SSH in.

The Ubuntu + Docker + NVIDIA Container Toolkit stack is preinstalled on Brev VM
images, so `--gpus`/`deploy.resources.devices` works out of the box.

---

## 2. Clone + configure

```bash
git clone https://github.com/zabahana/reg-agents.git && cd reg-agents

# Hosted NIM key from build.nvidia.com (nvapi-...)
cp .env.example .env
cat >> .env <<'EOF'
LLM_PROVIDER=nim
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_MODEL=meta/llama-3.1-8b-instruct
EMBEDDING_PROVIDER=nemo
NEMO_EMBED_BASE_URL=https://integrate.api.nvidia.com/v1
NEMO_EMBED_MODEL=nvidia/nv-embedqa-e5-v5
NIM_API_KEY=nvapi-REPLACE_ME
EOF
```

---

## 3. Build the image + generate the Triton model

```bash
docker compose build

# Trains the fraud XGBoost model and writes triton/model_repository/.../xgboost.json
# (runs inside the Linux app image — libgomp is present, no libomp hassle).
docker compose run --rm --no-deps -v "$PWD/triton:/app/triton" fraud-mcp \
  python scripts/export_triton_model.py
```

---

## 4. Bring up the full stack with GPU Triton + monitoring

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --profile monitoring up -d

docker compose logs -f triton   # wait for "Started HTTPService ... 8000"
```

Now reachable on the Brev VM's public host (ports you exposed):

| Service    | Port | URL                          |
|------------|------|------------------------------|
| UI         | 8501 | http://<brev-host>:8501      |
| Triton     | 8000 | http://<brev-host>:8000/v2/health/ready |
| Triton metrics | 8002 | http://<brev-host>:8002/metrics |
| Grafana    | 3000 | http://<brev-host>:3000  (admin / reg-agents) |

Brev shows the public URL / SSH port-forward for each exposed port in the
console and via `brev ports`.

---

## 5. Verify Triton is really serving (not the heuristic)

```bash
# Ask the fraud MCP server through the fraud agent, or hit Triton directly:
curl -s http://localhost:8000/v2/models/fraud_xgb_gnn/ready -o /dev/null -w "%{http_code}\n"

curl -s http://localhost:8000/v2/models/fraud_xgb_gnn/infer -d '{
  "inputs":[{"name":"input__0","shape":[1,5],"datatype":"FP32",
             "data":[5000, 1, 0.9, 3, 12]}]}' | python -m json.tool
```

Run the demos and watch Grafana populate (request rate, p95 latency, Triton
inference/compute latency):
```bash
docker compose exec orchestrator python scripts/demo_run.py
docker compose exec lifecycle-orchestrator python scripts/lifecycle_run.py --task fraud
```
The fraud MCP response `backend` should now read **`triton-gpu`**.

---

## 6. Stop / save cost

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile monitoring down
brev stop reg-agents     # stop billing; `brev start` to resume
brev delete reg-agents   # tear down entirely
```

---

## Optional: package it as a Launchable

In the Brev console → **Launchables → Create Launchable**:
- **Container mode**, point at this repo (`docker-compose.yml` + `docker-compose.gpu.yml`).
- GPU: L40S (or A100/H100).
- Expose ports **8501, 8000, 8002, 3000**.
- Add `NIM_API_KEY` as an environment variable.

Generate the link and anyone can spin up the identical GPU demo in one click —
a clean "reproducible NVIDIA-native deployment" story for the walkthrough.
