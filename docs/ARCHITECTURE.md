# Architecture

## Design goals
1. **Showcase the NVIDIA AI stack** as it would be deployed for a bank: NIM for
   LLM inference, NeMo Retriever for embeddings, Triton for the fraud model, on
   Kubernetes GPUs.
2. **Be a faithful agentic system**: real MCP tool servers and real A2A
   agent-to-agent messaging, not a monolith pretending to be agents.
3. **Always demoable**: laptop-friendly fallbacks so nothing hard-blocks a demo.

## Component responsibilities

### MCP tool servers (Model Context Protocol)
Tools are decoupled from agents. Any MCP client (our agents, Cursor, Claude
Desktop) can use them.

- **regulations-mcp** — `search_regulations`, `list_regulation_sources`.
  Semantic search over the banking-regulation corpus (NeMo Retriever embeddings
  → vector store; lexical fallback).
- **model-registry-mcp** — `list_models`, `get_model_metadata`,
  `get_model_documentation`. Stands in for a bank's MRM inventory.
- **fraud-mcp** — `score_transaction`. Proxies **Triton** (GNN+XGBoost) in the
  demo; heuristic locally.
- **modeling-mcp** — `list_tasks`, `list_candidate_models`, `run_model_bakeoff`,
  `get_champion`. Trains candidate models and selects a champion. scikit-learn
  locally; **RAPIDS cuML / XGBoost** analog on GPU.

### A2A agents (Agent-to-Agent protocol)
Each agent publishes an **Agent Card** at `/.well-known/agent-card.json` and
accepts JSON-RPC `message/send`. Each is independently deployable/scalable.

Governance-review flow:

- **retriever-agent** — grounds answers in retrieved regulations.
- **validation-agent** — SR 11-7 findings from model docs + regulations.
- **fraud-agent** — fraud score + analyst explanation + consumer-protection lens.
- **report-agent** — audit-ready governance report.
- **orchestrator** — A2A *client* that fans out to the specialists and composes
  the report; also an A2A *server* so it is itself composable.

Model-development lifecycle flow (three lines of defense):

- **developer-agent** (1st line) — runs the modeling-mcp bake-off, selects a
  champion against a documented primary metric, writes the model development
  document.
- **validator-agent** (2nd line) — independent *effective challenge*: critiques
  the champion selection (e.g. ROC-AUC vs PR-AUC/recall under class imbalance),
  reviews conceptual soundness, data, outcomes, and fair lending; issues a
  disposition (Approve / Approve with Conditions / Reject).
- **audit-agent** (3rd line) — audits the *process* (validation independence,
  documentation, effective-challenge evidence, approvals) and issues an audit
  opinion. Does not re-do the math.
- **lifecycle-orchestrator** — sequences developer → validator → audit, threading
  each artifact to the next agent; also an A2A server.

### Inference tier (NVIDIA, on GPU)
- **NIM (LLM)** — OpenAI-compatible chat completions used by every agent.
- **NeMo Retriever** — embedding NIM for RAG.
- **Triton** — serves the fraud GNN+XGBoost model (FIL/ONNX), dynamic batching,
  low-latency inference.

## Request flow (governance review)

```mermaid
sequenceDiagram
    participant U as User / UI
    participant O as Orchestrator (A2A)
    participant V as Validation Agent
    participant F as Fraud Agent
    participant R as Retriever Agent
    participant Rep as Report Agent
    participant MCP as MCP servers
    participant NV as NIM / NeMo / Triton

    U->>O: review(model_id, transaction)
    O->>V: A2A message/send(model_id)
    V->>MCP: get_model_metadata / docs, search_regulations
    V->>NV: NIM chat (SR 11-7 findings)
    O->>F: A2A message/send(transaction)
    F->>MCP: score_transaction
    MCP->>NV: Triton infer (GNN+XGBoost)
    F->>NV: NIM chat (explanation)
    O->>R: A2A message/send(reg question)
    R->>MCP: search_regulations
    R->>NV: NIM chat (grounded answer)
    O->>Rep: A2A message/send(combined artifacts)
    Rep->>NV: NIM chat (final report)
    Rep-->>O: governance_report
    O-->>U: report + trace
```

## Request flow (model-development lifecycle)

```mermaid
sequenceDiagram
    participant U as User / UI
    participant L as Lifecycle Orchestrator (A2A)
    participant D as Developer Agent (1st line)
    participant V as Validator Agent (2nd line)
    participant A as Audit Agent (3rd line)
    participant M as modeling-mcp
    participant Reg as regulations-mcp
    participant NV as NIM / NeMo

    U->>L: run_lifecycle(task_id)
    L->>D: A2A message/send(task_id)
    D->>M: run_model_bakeoff (train candidates, pick champion)
    D->>NV: NIM chat (model development document)
    D-->>L: dev document + leaderboard
    L->>V: A2A message/send(dev doc + leaderboard)
    V->>Reg: search_regulations (validation / fair lending)
    V->>NV: NIM chat (effective challenge + disposition)
    V-->>L: validation report
    L->>A: A2A message/send(dev doc + validation report)
    A->>Reg: search_regulations (governance / audit)
    A->>NV: NIM chat (audit opinion)
    A-->>L: audit report
    L-->>U: 3 documents + trace
```

## Local vs GPU parity

| Concern | Local dev | GKE GPU demo |
|---|---|---|
| LLM | OpenAI (`gpt-4o-mini`) | NIM (`llama-3.1-8b-instruct`) |
| Embeddings | OpenAI embeddings | NeMo Retriever (`nv-embedqa-e5-v5`) |
| Vector search | FAISS (CPU) / numpy | Milvus + cuVS (GPU) |
| Fraud model | heuristic | Triton (GNN+XGBoost) |
| Model bake-off | scikit-learn (CPU) | RAPIDS cuML / XGBoost (GPU) |
| Orchestration | same code | same code |

The agents/MCP servers are **identical** across environments; only config
(env vars in `ConfigMap`) changes. That portability is the core value: develop
locally against OpenAI, deploy on the NVIDIA stack with no code changes.

## Scaling & production notes
- Each agent/MCP server scales independently (HPA on CPU/RPS).
- NIM and Triton scale on the GPU pool; use separate GPU nodes or MIG slices to
  co-locate the 8B LLM, the embedder, and Triton cost-effectively.
- Add observability (OpenTelemetry traces across A2A hops), auth on the A2A
  endpoints, and per-tool rate limiting before real production.
