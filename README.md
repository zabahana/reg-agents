# reg-agents

**Regulatory Intelligence & Model Governance — a multi-agent system on the NVIDIA AI stack.**

A production-shaped demo for the NVIDIA *Senior Solutions Architect, Financial
Services Banking* role. It runs an end-to-end **SR 11-7 model governance
review**: specialist agents validate a model, analyze its fraud performance,
retrieve the relevant regulations, and compose an audit-ready report — all
coordinated over **A2A**, calling tools over **MCP**, and served by **NVIDIA
NIM / NeMo Retriever / Triton**.

> Built to be demoable on a laptop (OpenAI + lexical/heuristic fallbacks) and to
> run the *real* NVIDIA stack on a **GKE GPU** cluster with a one-line provider flip.

---

## What it demonstrates (maps to the job description)

| Job requirement | Where it shows up here |
|---|---|
| NVIDIA NIM / NeMo / TensorRT / Triton | `nim-llm`, `nemo-retriever`, `triton` (k8s GPU tier); OpenAI-compatible client |
| GNNs / XGBoost for fraud | Fraud model (mirrors NVIDIA fraud AI Blueprint), served via Triton |
| Generative & Agentic AI | 9 cooperating agents across two orchestrated flows |
| Model risk management lifecycle | Developer → Validator → Audit agents (three lines of defense) + real model bake-off |
| Banking/payments + regulation | SR 11-7, ECOA/Reg B, FCRA, UDAAP, BSA/AML corpus + fraud use case |
| Production deployment / MLOps | Docker, docker-compose, Kubernetes/GKE, GitHub Actions CI/CD |
| Python, distributed serving | FastAPI services, RAPIDS/cuVS/Milvus GPU path documented |

Plus the four **explicitly requested** components: **Kubernetes**, **MCP**,
**A2A**, **Docker**, and **CI/CD**.

Two orchestrated flows share the same agent/MCP infrastructure:

1. **Governance review** — validate a registered model, analyze fraud
   performance, retrieve regulations, produce an audit-ready report.
2. **Model-development lifecycle** — a Developer agent trains candidate models
   and selects a champion, an independent Validator performs *effective
   challenge*, and Internal Audit reviews the process. This mirrors the banking
   **three lines of defense**. See committed artifacts in
   [`docs/lifecycle/`](docs/lifecycle/README.md).

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

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detail and
[`docs/STUDY_GUIDE.md`](docs/STUDY_GUIDE.md) for the interview talking points.

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
  defense): [`docs/lifecycle/`](docs/lifecycle/README.md).
  Regenerate with `python scripts/generate_lifecycle.py --task fraud`.

## Run on real NVIDIA infra

- **Local Docker (CPU agents + hosted NIM):** set `LLM_PROVIDER=nim` and a
  `NIM_API_KEY` from [build.nvidia.com](https://build.nvidia.com), then
  `docker compose up`.
- **GKE with GPUs (self-hosted NIM + NeMo Retriever + Triton):** follow
  [`k8s/README.md`](k8s/README.md).

Switching OpenAI → NVIDIA NIM is a single env change (`LLM_PROVIDER`), because
NIM exposes an OpenAI-compatible API. That migration story is a core talking
point of the demo.

---

## Repo layout

```
reg_agents/
  common/        llm, embeddings, vector store, A2A protocol, MCP client, corpus, modeling
  mcp_servers/   regulations / model-registry / fraud / modeling  (MCP tool servers, SSE)
  agents/        governance: orchestrator + retriever / validation / fraud / report
                 lifecycle:  lifecycle-orchestrator + developer / validator / audit (A2A)
  ui/            Streamlit demo
data/            regulations corpus, model cards, sample transactions
docs/            architecture, study guide, sample reports + lifecycle artifacts
k8s/             GKE manifests (GPU NIM tier + CPU agents) + setup guide
scripts/         run_local.sh / stop_local.sh / demo_run.py / lifecycle_run.py / generate_*.py
tests/           offline unit tests
.github/         CI/CD pipeline
```
