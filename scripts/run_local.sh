#!/usr/bin/env bash
# Launch the full reg-agents stack locally (no Docker required).
# MCP tool servers (SSE) + A2A agents, each in the background.
# Logs go to .run/*.log ; PIDs to .run/pids so `scripts/stop_local.sh` can stop them.
set -euo pipefail
cd "$(dirname "$0")/.."

# Resolve a Python interpreter. Prefer the project venv so this works whether or
# not the venv is activated; override with PYTHON=... if needed.
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PY="$(pwd)/.venv/bin/python"
  elif command -v python  >/dev/null 2>&1; then PY="python"
  elif command -v python3 >/dev/null 2>&1; then PY="python3"
  else
    echo "ERROR: no Python found. Create the venv first (see README Quickstart)." >&2
    exit 1
  fi
fi
echo "using python: $PY"

mkdir -p .run
: > .run/pids

start () {
  local name="$1"; local module="$2"; local port="$3"
  echo "starting $name on :$port"
  PORT="$port" "$PY" -m "$module" > ".run/${name}.log" 2>&1 &
  echo "$! $name" >> .run/pids
}

start_app () {
  local name="$1"; local app="$2"; local port="$3"
  echo "starting $name on :$port"
  PORT="$port" "$PY" -m uvicorn "$app" --host 0.0.0.0 --port "$port" \
    > ".run/${name}.log" 2>&1 &
  echo "$! $name" >> .run/pids
}

# MCP tool servers
start regulations_mcp   reg_agents.mcp_servers.regulations_server     9101
start model_registry_mcp reg_agents.mcp_servers.model_registry_server 9102
start fraud_mcp         reg_agents.mcp_servers.fraud_server           9103
start modeling_mcp      reg_agents.mcp_servers.modeling_server        9104
start complaint_mcp     reg_agents.mcp_servers.complaint_server       9105

sleep 2

# A2A agents — governance review
start_app retriever_agent  reg_agents.agents.retriever_agent:app  8101
start_app validation_agent reg_agents.agents.validation_agent:app 8102
start_app fraud_agent      reg_agents.agents.fraud_agent:app      8103
start_app report_agent     reg_agents.agents.report_agent:app     8104
start_app orchestrator     reg_agents.agents.orchestrator:app     8100

# A2A agents — model-development lifecycle (three lines of defense)
start_app developer_agent  reg_agents.agents.developer_agent:app  8105
start_app validator_agent  reg_agents.agents.validator_agent:app  8106
start_app audit_agent      reg_agents.agents.audit_agent:app      8107
start_app lifecycle        reg_agents.agents.lifecycle_orchestrator:app 8108

# A2A agent — complaint → regulation classification (third model)
start_app complaint_agent  reg_agents.agents.complaint_agent:app  8110

# Wait until the A2A agents actually accept connections (they import heavy libs
# like scikit-learn/faiss and take a few seconds), so callers can run demos
# immediately after this script returns instead of hitting "Connection refused".
wait_ready () {
  local ports="8100 8101 8102 8103 8104 8105 8106 8107 8108 8110"
  echo -n "waiting for agents to be ready"
  for _ in $(seq 1 60); do
    local all_ok=1
    for p in $ports; do
      curl -fs -m 2 "http://localhost:${p}/health" >/dev/null 2>&1 || all_ok=0
    done
    if [[ "$all_ok" == "1" ]]; then echo " — ready."; return 0; fi
    echo -n "."; sleep 1
  done
  echo " — timed out; check .run/*.log"
  return 1
}
wait_ready || true

echo
echo "All services started. Logs in .run/. Try:"
echo "  python scripts/demo_run.py --model FRAUD-XGB-GNN-001"
echo "  python scripts/lifecycle_run.py --task fraud"
echo "  streamlit run reg_agents/ui/app.py"
echo "Stop everything with: scripts/stop_local.sh"
