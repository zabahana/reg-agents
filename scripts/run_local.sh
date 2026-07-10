#!/usr/bin/env bash
# Launch the full reg-agents stack locally (no Docker required).
# MCP tool servers (SSE) + A2A agents, each in the background.
# Logs go to .run/*.log ; PIDs to .run/pids so `scripts/stop_local.sh` can stop them.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p .run
: > .run/pids

start () {
  local name="$1"; local module="$2"; local port="$3"
  echo "starting $name on :$port"
  PORT="$port" python -m "$module" > ".run/${name}.log" 2>&1 &
  echo "$! $name" >> .run/pids
}

start_app () {
  local name="$1"; local app="$2"; local port="$3"
  echo "starting $name on :$port"
  PORT="$port" python -m uvicorn "$app" --host 0.0.0.0 --port "$port" \
    > ".run/${name}.log" 2>&1 &
  echo "$! $name" >> .run/pids
}

# MCP tool servers
start regulations_mcp   reg_agents.mcp_servers.regulations_server     9101
start model_registry_mcp reg_agents.mcp_servers.model_registry_server 9102
start fraud_mcp         reg_agents.mcp_servers.fraud_server           9103
start modeling_mcp      reg_agents.mcp_servers.modeling_server        9104

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

echo
echo "All services started. Logs in .run/. Try:"
echo "  python scripts/demo_run.py --model FRAUD-XGB-GNN-001"
echo "  python scripts/lifecycle_run.py --task fraud"
echo "  streamlit run reg_agents/ui/app.py"
echo "Stop everything with: scripts/stop_local.sh"
