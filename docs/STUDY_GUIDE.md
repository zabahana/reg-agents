# Study Guide — Mastering reg-agents for the NVIDIA SA Interview

Target role: **Senior Solutions Architect, Financial Services Banking (NVIDIA)**.
Likely hiring manager: **David Williams** (Financial Solutions Architect Manager;
payments, fraud, transformers/LLMs, inference serving). Likely peers: **Flora H.**
(agentic AI with NIM/NeMo/AIQ, GNNs for fraud/credit).

This project was built so you can speak, at a senior level, to **every core
NVIDIA component in the JD** using something you actually built.

---

## 1. The 60-second pitch (memorize this)

> "I built `reg-agents`, a multi-agent model-governance system for banking. Five
> agents coordinate over Google's **A2A** protocol; they call tools over
> **MCP** — regulation search, a model registry, and fraud scoring. The LLM
> reasoning runs on **NVIDIA NIM**, retrieval uses **NeMo Retriever**
> embeddings, and the fraud model — a **GNN-enhanced XGBoost**, like NVIDIA's
> fraud AI Blueprint — is served on **Triton**. It's containerized, runs on
> **GKE with a GPU node pool**, and ships via a **GitHub Actions CI/CD**
> pipeline. Because NIM is OpenAI-API-compatible, I develop locally against
> OpenAI and flip one env var to run the whole thing on NVIDIA inference."

---

## 2. Component-by-component talking points

### NVIDIA NIM (inference microservices)
- **What/why:** containerized, OpenAI-compatible inference microservices; fast
  path to production LLM serving with enterprise support.
- **In the project:** `reg_agents/common/llm.py` — one `OpenAI` client, provider
  chosen by `LLM_PROVIDER`. On GKE, `nim-llm` serves `llama-3.1-8b-instruct`.
- **SA angle:** "Migrating a customer from OpenAI to NIM is a `base_url` change —
  same SDK, data stays in their VPC/on-prem, predictable latency, no per-token
  egress." Know NIM health endpoint (`/v1/health/ready`) and NGC pull secrets.

### NeMo Retriever (embeddings) + NeMo (customization)
- **What/why:** NeMo Retriever = optimized embedding/reranking NIMs for RAG;
  NeMo = training/customization/eval (LoRA, distillation).
- **In the project:** `embeddings.py` uses `nv-embedqa-e5-v5` with `input_type`
  query/passage. RAG over the regulatory corpus.
- **SA angle:** for banking, tie to **data privacy** and **domain accuracy** —
  fine-tune/distill small models on the bank's data (cite the NVIDIA financial
  distillation blueprint: 1B model matching 70B at ~98% lower inference cost).

### Triton Inference Server (+ TensorRT)
- **What/why:** multi-framework model serving; dynamic batching, concurrent model
  execution, FIL backend for XGBoost, low latency. TensorRT/TensorRT-LLM =
  compilation + quantization for max throughput.
- **In the project:** `fraud-mcp` calls Triton's KServe v2 `/v2/models/.../infer`.
  Model card quotes P95 8 ms with FP16 + dynamic batching.
- **SA angle:** be ready to design **sub-50ms P95** serving: batching, FP16/INT8
  quantization, concurrent instances, GPU memory budgeting.

### GNN + XGBoost fraud (RAPIDS / cuGraph / cuVS)
- **What/why:** GraphSAGE embeddings over the cardholder–merchant–device graph
  concatenated with tabular features → XGBoost. RAPIDS `cuDF` for GPU ETL,
  `cuGraph` for GNNs, `cuVS` for GPU vector search (Milvus backend).
- **In the project:** modeled by `FRAUD-XGB-GNN-001` + the fraud MCP server;
  vector store abstraction swaps FAISS → Milvus/cuVS.
- **SA angle:** this **is** NVIDIA's fraud AI Blueprint — reduce false positives
  by adding graph features; American Express / bunq references.

### Agentic AI: A2A + MCP
- **A2A** (agent-to-agent): agent cards, `message/send`, tasks/artifacts —
  interoperable agents across teams/vendors. `reg_agents/common/a2a.py`.
- **MCP** (model context protocol): standard tool/resource interface; our three
  servers are reusable by any MCP client. `reg_agents/common/mcp_client.py`.
- **AIQ** (NVIDIA AgentIQ / NeMo Agent toolkit): know it exists — profiling,
  evaluation, and orchestration of agent systems. Say you'd add AIQ for **agent
  evaluation and tracing** in production. (Flora's stack.)

### Kubernetes / Docker / CI-CD
- One image, per-service command; `docker-compose.yml` locally; `k8s/` splits a
  **GPU inference tier** (NIM/NeMo/Triton) from **CPU agent/MCP tier**.
- GPU node pool with `nvidia-l4`, driver DaemonSet, NGC secrets, readiness
  probes, scale-to-zero for cost. CI: ruff + pytest + docker build + gated GKE
  rollout.

---

## 3. Likely interview questions → your answer hook

- *"Design real-time fraud detection for a top-5 issuer, sub-50ms P95, on-prem."*
  → RAPIDS ETL → cuGraph GNN features → XGBoost (FIL on Triton) → TensorRT/FP16,
  dynamic batching, concurrent instances; Milvus+cuVS for entity lookups; NIM for
  the analyst copilot. Walk your `fraud-mcp` + Triton design.
- *"How would you move a bank off OpenAI onto NVIDIA?"* → the `LLM_PROVIDER`
  story; data residency, cost, latency SLAs, NIM on their K8s.
- *"Explain a GNN to a non-technical exec."* (they ask this) → message passing /
  "who you transact with is predictive"; graph features cut false positives.
- *"How do you validate an AI model for a regulator?"* → SR 11-7: effective
  challenge, conceptual soundness, outcomes analysis, ongoing monitoring,
  documentation; adverse-action/ECOA explainability. This IS the validation agent.
- *"Quantization / inference optimization?"* (David's specialty) → INT8/FP8 with
  TensorRT-LLM, KV-cache, batching, throughput vs latency trade-offs, accuracy
  checks.
- *"Why agents / MCP / A2A vs one big prompt?"* → separation of concerns,
  independent scaling, tool governance, interoperability across teams/vendors.

---

## 4. Three-day mastery plan

**Day 1 — build & run it end to end.**
- `pip install -r requirements.txt`, `pytest`, `scripts/run_local.sh`,
  `demo_run.py`, Streamlit UI. Add your `OPENAI_API_KEY` and watch real output.
- Read every file in `reg_agents/`. You must be able to explain each in 2 lines.
- Get a free NIM key at build.nvidia.com; set `LLM_PROVIDER=nim` and re-run.

**Day 2 — go deep on the NVIDIA stack.**
- Study: NIM, NeMo Retriever, Triton (FIL backend), TensorRT-LLM quantization,
  RAPIDS/cuGraph/cuVS, the fraud AI Blueprint, AIQ. For each, write 3 sentences
  in your own words in this file.
- (Optional, if GCP ready) deploy `k8s/` to a small L4 GKE pool; get NIM + one
  Triton model live. Even partial (NIM only) is a strong story.
- Draw the architecture from memory on a whiteboard 3×.

**Day 3 — interview reps.**
- Do the sub-50ms fraud design out loud, timed. Do the OpenAI→NIM migration.
- Rehearse SR 11-7 validation using the validation agent's output as your script.
- Prepare 5 questions to ask David/Flora (below). Sleep.

---

## 5. Smart questions to ask them
- "Where does AIQ fit in your production agent evaluation today?"
- "For banking customers, how much is NIM self-hosted vs the hosted catalog?"
- "How do you split GPU nodes across NIM, retriever, and Triton for FSI POCs?"
- "What does 'design win' look like for the banking SA team this year?"
- "How does the team balance quant/HPC work vs GenAI/agentic work?"

---

## 6. Gaps to be honest about
- **C++:** JD wants strong C++. This project is Python. If pressed, be candid;
  emphasize performance work (Triton/TensorRT config, batching) and willingness.
- **Portfolio optimization (cuOpt/cuFOLIO):** intentionally out of scope — this
  is a *Banking/Consumer Finance* role, not Capital Markets.
- **Live GPU deploy:** if you don't finish the GKE deploy, say "the manifests are
  production-shaped; here's exactly how it runs on an L4 pool" and walk `k8s/`.
