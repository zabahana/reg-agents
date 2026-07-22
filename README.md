# reg-agents

**Regulatory Intelligence & Model Governance — a multi-agent system on the NVIDIA AI stack.**

A production-shaped **mock demo** of agentic AI for banking model risk
management. It runs an end-to-end **SR 11-7 model governance review**: specialist
agents validate a model, analyze its fraud performance, retrieve the relevant
regulations, and compose an audit-ready report — all coordinated over **A2A**,
calling tools over **MCP**, and served by **NVIDIA NIM / NeMo Retriever /
Triton**.

> Built to be demoable on a laptop (OpenAI + lexical/heuristic fallbacks) and to
> run the *real* NVIDIA stack on a **GKE GPU** cluster with a one-line provider flip.

---

## What it demonstrates

| Capability | Where it shows up here |
|---|---|
| NVIDIA NIM / NeMo / TensorRT / Triton | `nim-llm`, `nemo-retriever`, `triton` (k8s GPU tier); OpenAI-compatible client |
| GNNs / XGBoost for fraud | Fraud model (mirrors NVIDIA fraud AI Blueprint), served via Triton |
| Generative & Agentic AI | 10 cooperating agents across three orchestrated flows |
| NLP / complaint intelligence | Complaint→regulation classifier (real CFPB data): binary gate + RAG/LLM labeling with citations |
| Model risk management lifecycle | Developer → Validator → Audit agents (three lines of defense) + real model bake-off |
| Banking/payments + regulation | SR 11-7, ECOA/Reg B, FCRA, UDAAP, BSA/AML corpus + fraud use case |
| Production deployment / MLOps | Docker, docker-compose, Kubernetes/GKE, GitHub Actions CI/CD |
| Python, distributed serving | FastAPI services, RAPIDS/cuVS/Milvus GPU path documented |

Core building blocks: **Kubernetes**, **MCP**, **A2A**, **Docker**, and **CI/CD**.

Three orchestrated flows share the same agent/MCP infrastructure:

1. **Governance review** — validate a registered model, analyze fraud
   performance, retrieve regulations, produce an audit-ready report.
2. **Model-development lifecycle** — a Developer agent trains candidate models
   and selects a champion, an independent Validator performs *effective
   challenge*, and Internal Audit reviews the process. This mirrors the banking
   **three lines of defense**. See committed artifacts in
   [`docs/lifecycle/`](docs/lifecycle/README.md).
3. **Complaint classification** — a two-stage model over **real CFPB complaint
   narratives**: a binary classifier gates *regulatory vs not*, then RAG over
   the regulation corpus + LLM reasoning (few-shot) assigns one of **24
   regulation categories** (UDAAP, sales practices, FCRA, Reg E, …) with a
   cited excerpt. Development + validation documentation with accuracy figures
   ships as markdown **and PDF** in
   [`docs/complaint_model/`](docs/complaint_model/README.md).

---

## Architecture

```
                         ┌──────────────────────────────┐
   user / UI  ─────────► │   Orchestrator (A2A server)  │
                         └───────────────┬──────────────┘
                       A2A message/send  │  (fan-out)
        ┌───────────────────┬────────────┼─────────────────┐
        ▼                   ▼             ▼                 ▼
  Retriever Agent    Validation Agent  Fraud Agent     Report Agent
        │                   │             │                 
        │ MCP               │ MCP         │ MCP             
        ▼                   ▼             ▼                 
  regulations-mcp    model-registry-mcp  fraud-mcp          
        │                   │             │                 
        ▼                   ▼             ▼                 
  NeMo Retriever      (model cards)     Triton (GNN+XGBoost)
   + vector store                        on GPU
        │
        ▼
   NVIDIA NIM (LLM reasoning, OpenAI-compatible) — used by every agent
```

A second orchestrator runs the **model-development lifecycle** over three more
agents (three lines of defense):

```
                         ┌──────────────────────────────┐
   user / UI  ─────────► │  Lifecycle Orchestrator (A2A)│
                         └───────────────┬──────────────┘
             A2A message/send (sequential, artifacts threaded)
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                        ▼
  Developer Agent  ──►   Validator Agent   ──►     Audit Agent
   (1st line)             (2nd line)                (3rd line)
        │ MCP
        ▼
   modeling-mcp  (scikit-learn bake-off → champion; GPU: RAPIDS cuML / XGBoost)
```

A third flow classifies consumer complaints (real CFPB narratives) against the
regulation taxonomy:

```
   user / UI ──► Complaint Agent (A2A :8110) ──► complaint-mcp (:9105)
                                                     │
                       stage 1: TF-IDF binary gate — regulatory?  (logistic/XGBoost)
                       stage 2: RAG (regulation corpus) + LLM few-shot reasoning
                                → 1 of 24 categories + citation + rationale
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for design detail and
request-flow diagrams.

---

## Quickstart (local, no GPU, no Docker)

```bash
cd /Users/zelalemabahana/reg-agents
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: add OPENAI_API_KEY for real LLM output

# Run tests (fully offline)
pytest -q

# Launch the whole stack (MCP servers + agents) in the background
bash scripts/run_local.sh

# Run an end-to-end governance review
python scripts/demo_run.py --model FRAUD-XGB-GNN-001

# Run the model-development lifecycle (develop -> validate -> audit)
python scripts/lifecycle_run.py --task fraud

# Or the UI
streamlit run reg_agents/ui/app.py

# Stop everything
bash scripts/stop_local.sh
```

Without any API key the system still runs end-to-end: retrieval uses a lexical
fallback, fraud scoring uses a transparent heuristic, and agents return the raw
tool output with a note that the LLM was unavailable. Add `OPENAI_API_KEY` (or
switch to NIM) to get fully synthesized, cited output.

### Sample outputs

- **Governance reports** — the review pipeline generalizing across model types
  (fraud, GenAI complaint classifier, AML): [`docs/reports/`](docs/reports/README.md).
  Regenerate with `python scripts/generate_report.py --all`.
- **Model-development lifecycle** — bake-off leaderboard, development document,
  independent validation report, and internal audit report (three lines of
  defense): [`docs/lifecycle/`](docs/lifecycle/README.md). Runs unchanged across
  model types — **fraud** (champion: a tree) and **credit PD** (champion:
  logistic regression) — to show the pipeline generalizes. Regenerate with
  `python scripts/generate_lifecycle.py --all`.
- **Complaint model documentation** — development document + independent
  validation report with accuracy figures and tables, as markdown **and PDF**:
  [`docs/complaint_model/`](docs/complaint_model/README.md). Data is real
  (CFPB Consumer Complaint Database; refresh with
  `python scripts/fetch_cfpb_complaints.py`). Regenerate docs with
  `python scripts/generate_complaint_model_docs.py`.
- **Publication-grade Model Development Document** — the full research
  protocol for the complaint model's stage-1 gate (EDA, stratified 80/10/10
  split, minority-balanced 4-model bake-off incl. fine-tuned DistilBERT,
  validation-based selection with a validation-optimized decision cut-off,
  OOV + sensitivity analyses) with every artifact
  (fitted models, split indices, environment manifest):
  [`docs/model_development/`](docs/model_development/README.md). Regenerate
  with `python scripts/generate_model_development_doc.py`.

## Run on real NVIDIA infra

- **Local Docker (CPU agents + hosted NIM):** set `LLM_PROVIDER=nim` and a
  `NIM_API_KEY` from [build.nvidia.com](https://build.nvidia.com), then
  `docker compose up`.
- **NVIDIA GPUs, no hyperscaler (Brev):** run Triton + the fraud model on a
  self-serve [NVIDIA Brev](https://brev.nvidia.com) GPU VM with hosted NIM for
  the LLM — the whole demo on NVIDIA infra, end to end. See
  [`brev/README.md`](brev/README.md) (`docker compose -f docker-compose.yml -f
  docker-compose.gpu.yml --profile monitoring up`).
- **GKE (hosted NIM for the LLM + self-hosted Triton for the fraud model):**
  follow [`k8s/README.md`](k8s/README.md). Triton is the only GPU workload, so
  the GPU pool is a single node. To self-host NIM on GPU instead, apply
  [`k8s/optional/nim-selfhosted.yaml`](k8s/optional/nim-selfhosted.yaml).

Switching OpenAI → NVIDIA NIM is a single env change (`LLM_PROVIDER`), because
NIM exposes an OpenAI-compatible API. That migration story is a core talking
point of the demo.

## Observability (metrics + traces)

Every A2A agent exposes Prometheus `/metrics` (request rate, p95 latency,
errors), Triton exports native inference metrics, and the DCGM exporter adds GPU
metrics (util, memory, temp, power). The fraud model has runtime **guardrails**
(input clamping + output-range reset + Triton→heuristic fallback) surfaced as
metrics, plus Prometheus **alerts** (block-rate spike, guardrail fired,
serving-on-heuristic, GPU hot/full, agent errors) routed through **Alertmanager**
(warning → Slack). The same Grafana dashboard, alerts, and routing run locally
and on GKE.

**Distributed tracing:** the agents are instrumented with **OpenTelemetry**, and
because trace context propagates over the A2A/MCP HTTP calls, one governance run
shows up as a single connected **trace** across agents (orchestrator → validation
/ fraud / retriever → report, plus the MCP tool spans) in **Jaeger**. This is the
same span model NVIDIA's NeMo Agent Toolkit (AIQ) emits, so it's the seam for
AIQ-based tracing/eval later. Tracing activates only when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set (the `monitoring` profile points it at
Jaeger); it is a silent no-op otherwise.

- **Local:** `docker compose --profile monitoring up` → Grafana at
  [localhost:3000](http://localhost:3000) (`admin` / `reg-agents`), dashboard
  **"reg-agents — agents, model & GPU"**; **Jaeger** UI at
  [localhost:16686](http://localhost:16686). Generate traffic with the demo
  scripts and watch metrics populate and traces appear.
- **GKE:** `kube-prometheus-stack` + ServiceMonitors + the dashboard —
  see [`k8s/monitoring/README.md`](k8s/monitoring/README.md).

---

## Repo layout

```
reg_agents/
  common/        llm, embeddings, vector store, A2A protocol, MCP client, corpus, modeling, complaints, telemetry (OTel)
  mcp_servers/   regulations / model-registry / fraud / modeling / complaint  (MCP tool servers, SSE)
  agents/        governance: orchestrator + retriever / validation / fraud / report
                 lifecycle:  lifecycle-orchestrator + developer / validator / audit (A2A)
                 complaints: complaint agent (two-stage classifier + citations)
  ui/            Streamlit demo
data/            regulations corpus, model cards, sample transactions + credit data + CFPB complaints (real)
docs/            architecture, sample reports, lifecycle artifacts (fraud + credit), complaint-model docs (MD + PDF)
triton/          Triton FIL model repository (config.pbtxt) + export script
brev/            NVIDIA Brev GPU runbook (Triton on NVIDIA infra, no hyperscaler)
k8s/             GKE manifests (Triton GPU tier + CPU agents), optional self-hosted NIM
k8s/monitoring/  kube-prometheus-stack values + ServiceMonitors + Grafana dashboard
monitoring/      local Prometheus + Grafana + guardrail alert rules (compose)
scripts/         run_local.sh / stop_local.sh / demo_run.py / lifecycle_run.py / export_triton_model.py / fetch_cfpb_complaints.py / generate_*.py
tests/           offline unit tests
.github/         CI/CD pipeline
```
