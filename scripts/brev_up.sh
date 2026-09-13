#!/usr/bin/env bash
# One-shot bring-up for an NVIDIA Brev GPU VM (or any host with the NVIDIA
# Container Toolkit): build the image, train+export the Triton fraud model,
# then start the full stack with GPU Triton + DCGM + Prometheus + Grafana.
#
#   ./scripts/brev_up.sh            # build + export + up -d
#   ./scripts/brev_up.sh --no-build # skip docker build (reuse existing image)
#
# Requires a .env with NIM_API_KEY (hosted NIM). See brev/README.md.
set -euo pipefail

cd "$(dirname "$0")/.."

# Some pre-release Docker Compose builds crash with "concurrent map writes"
# while recreating a large dependency graph. Serialize this launch by default;
# operators may override it when their Compose version is known-good.
export COMPOSE_PARALLEL_LIMIT="${COMPOSE_PARALLEL_LIMIT:-1}"

BASE="docker-compose.yml"
GPU="docker-compose.gpu.yml"
COMPOSE=(docker compose -f "$BASE" -f "$GPU" --profile monitoring)

if [ ! -f .env ]; then
  echo "!! No .env found. Copy .env.example to .env and set NIM_API_KEY." >&2
  exit 1
fi
if ! grep -q '^NIM_API_KEY=nvapi' .env 2>/dev/null; then
  echo "!! Warning: NIM_API_KEY doesn't look set in .env (LLM calls will fall back)." >&2
fi

if [ "${1:-}" != "--no-build" ]; then
  echo "==> Building app image..."
  docker compose build
fi

echo "==> Generating Triton fraud model (config.pbtxt + xgboost.json)..."
docker compose run --rm --no-deps -v "$PWD/triton:/app/triton" fraud-mcp \
  python scripts/export_triton_model.py

echo "==> Generating complaint stage-1 Triton artifact..."
docker compose run --rm --no-deps -v "$PWD/triton:/app/triton" \
  -v "$PWD/data:/app/data" complaint-mcp \
  python scripts/export_complaint_triton_model.py

echo "==> Starting stack (GPU Triton + DCGM + Prometheus + Grafana)..."
"${COMPOSE[@]}" up -d

echo "==> Waiting for Triton to become ready..."
triton_ready=0
for _ in $(seq 1 40); do
  if curl -fsS localhost:8000/v2/health/ready >/dev/null 2>&1; then
    echo "    Triton ready."
    triton_ready=1
    break
  fi
  sleep 5
done

if [ "$triton_ready" -ne 1 ]; then
  echo "!! Triton was not ready after 200 seconds. Recent logs:" >&2
  "${COMPOSE[@]}" logs --tail 100 triton >&2
  exit 1
fi

cat <<'EOF'

==> Up. Endpoints (bind these to your Brev-exposed ports):
    UI            :8501
    Triton HTTP   :8000   (/v2/health/ready)
    Triton metrics:8002   (/metrics)
    GPU metrics   :9400   (DCGM /metrics)
    Prometheus    :9090   (/alerts shows the guardrails)
    Alertmanager  :9093   (routed guardrail alerts; set Slack webhook in monitoring/alertmanager.yml)
    Grafana       :3000   (admin / reg-agents) -> "reg-agents — agents, model & GPU"
    Jaeger        :16686  (A2A distributed traces per governance/lifecycle run)

    Generate traffic:
      docker compose exec orchestrator python scripts/demo_run.py
      docker compose exec lifecycle-orchestrator python scripts/lifecycle_run.py --task fraud
      docker compose exec lifecycle-orchestrator python scripts/lifecycle_run.py --task credit

    Tear down:
      docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile monitoring down
EOF
