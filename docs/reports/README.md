# Sample governance reports

Committed artifacts produced by the `reg-agents` pipeline (validation +
fraud/performance + retriever + report agents), captured in-process so they are
deterministic. Regenerate with:

```bash
python scripts/generate_report.py --all           # every model in the registry
python scripts/generate_report.py --model AML-TM-021
```

Each report shows the **same** agent logic generalizing across model types —
only the tailored regulatory query and the performance section differ per model.

| Model | Type | Report |
| --- | --- | --- |
| `FRAUD-XGB-GNN-001` | Fraud (XGBoost + GNN) — live transaction scoring | [FRAUD-XGB-GNN-001.md](FRAUD-XGB-GNN-001.md) |
| `GENAI-COMPLAINT-030` | GenAI complaint classifier (RAG + fine-tune) | [GENAI-COMPLAINT-030.md](GENAI-COMPLAINT-030.md) |
| `AML-TM-021` | AML transaction monitoring (rules + IsolationForest + GNN) | [AML-TM-021.md](AML-TM-021.md) |

The fraud detector gets a **live transaction-scoring** section (Triton proxy /
local heuristic); other models get a **performance & monitoring summary** derived
from their model card. The retriever pulls model-appropriate rules automatically
(e.g. BSA/AML for the AML model, ECOA + GLBA privacy for the GenAI classifier).
