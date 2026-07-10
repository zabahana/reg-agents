# Monitoring — Prometheus + Grafana (kube-prometheus-stack)

This wires the whole stack into **Prometheus** (metrics) + **Grafana**
(dashboards) using the community `kube-prometheus-stack` Helm chart.

## What gets scraped

| Target            | Source of metrics                          | Port / path        |
|-------------------|--------------------------------------------|--------------------|
| A2A agents        | `prometheus-fastapi-instrumentator`        | `http` `/metrics`  |
| Triton            | native Triton Prometheus exporter          | `metrics:8002` `/metrics` |
| GPU (optional)    | NVIDIA DCGM exporter (install separately)  | `DCGM_FI_*`        |
| Hosted NIM        | not scrapeable (SaaS) — LLM latency shows up in the agent metrics |

The MCP servers speak SSE (no HTTP `/metrics`); their behavior is observed
through the agents that call them and through Triton.

## Install

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install kube-prometheus-stack \
  prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace \
  -f k8s/monitoring/values-kps.yaml
```

`values-kps.yaml` makes Prometheus pick up all ServiceMonitors and lets the
Grafana sidecar auto-import dashboards from any namespace.

## Apply the reg-agents monitors + dashboard + guardrail alerts

```bash
kubectl apply -f k8s/monitoring/servicemonitors.yaml
kubectl apply -f k8s/monitoring/grafana-dashboard.yaml
kubectl apply -f k8s/monitoring/prometheusrules.yaml   # guardrail alerts
```

### Guardrail alerts (`prometheusrules.yaml`)

| Alert | Guards against |
|-------|----------------|
| `HighFraudBlockRate` | BLOCK rate > 50% (10m) — drift / bad threshold / attack |
| `FraudGuardrailTriggered` | input clamps or prob-reset firing (data quality / model) |
| `FraudServingOnHeuristic` | Triton path down, scores fell back to the heuristic |
| `AgentHighErrorRate` / `AgentHighLatencyP95` | agent health |
| `TritonInferenceFailures` | Triton inference errors |
| `GPUTemperatureHigh` / `GPUMemoryNearFull` / `GPUSaturated` | GPU health (DCGM) |

They evaluate in Prometheus (see the **Alerts** tab); attach an Alertmanager
receiver to page. The same rules run locally via `monitoring/alerts.yml`.

## Open Grafana

```bash
kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
# http://localhost:3000  (user: admin, pass: reg-agents — from values-kps.yaml)
```

Open the **"reg-agents — agents & Triton"** dashboard. Generate traffic with the
demo (`scripts/demo_run.py`, `scripts/lifecycle_run.py`) and watch request rate,
p95 latency, error rate, and Triton inference/compute latency populate.

## Verify scrape targets

```bash
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090:9090
# http://localhost:9090/targets  → reg-agents-agents + reg-agents-triton should be UP
```

## Optional: GPU metrics (DCGM)

The GPU utilization panel needs the NVIDIA DCGM exporter. On GKE either enable
GKE system GPU metrics or install the exporter (it ships a ServiceMonitor that
kube-prometheus-stack will pick up):

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm install dcgm-exporter gpu-helm-charts/dcgm-exporter -n monitoring
```
