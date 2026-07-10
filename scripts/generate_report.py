"""Generate sample governance report artifact(s) (robust, in-process).

Runs the SAME agent logic (identical prompts + tools) as the live A2A/MCP stack,
but in-process -- direct tool calls and LLM calls, no HTTP/SSE -- so it is
deterministic and cannot hang. The live distributed path (A2A over HTTP, tools
over MCP/SSE) runs in docker-compose / k8s and the Streamlit UI; this script
just captures clean artifacts.

The pipeline generalizes across model types: the fraud detector gets a live
transaction-scoring section; other models (credit, AML, GenAI, PPNR) get a
performance summary derived from their model card. The regulatory-context query
is tailored to each model's stated use.

    python scripts/generate_report.py --model FRAUD-XGB-GNN-001
    python scripts/generate_report.py --model GENAI-COMPLAINT-030 --model AML-TM-021
    python scripts/generate_report.py --all
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reg_agents.agents.fraud_agent import _SYS as FRAUD_SYS  # noqa: E402
from reg_agents.agents.report_agent import _SYS as REPORT_SYS  # noqa: E402
from reg_agents.agents.retriever_agent import _SYS as RETRIEVER_SYS  # noqa: E402
from reg_agents.agents.validation_agent import _SYS as VALIDATION_SYS  # noqa: E402
from reg_agents.common import llm  # noqa: E402
from reg_agents.common.corpus import RegulationRetriever  # noqa: E402
from reg_agents.config import get_settings  # noqa: E402
from reg_agents.mcp_servers.fraud_server import score_transaction  # noqa: E402
from reg_agents.mcp_servers.model_registry_server import (  # noqa: E402
    _INVENTORY,
    get_model_documentation,
    get_model_metadata,
)

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "reports"
)

_PERF_SYS = (
    "You are a model validator summarizing a model's performance and monitoring "
    "for a governance report. Using ONLY the model card, summarize reported "
    "performance metrics, monitoring approach, and the top risks/limitations in "
    "4-6 sentences. Do not invent numbers."
)

_retriever = RegulationRetriever()


def _search(query: str, k: int = 5) -> str:
    hits = _retriever.search(query, k)
    return json.dumps(
        [
            {
                "source": h.document.metadata.get("source", ""),
                "heading": h.document.metadata.get("heading", ""),
                "score": round(h.score, 4),
                "text": h.document.text,
            }
            for h in hits
        ],
        indent=2,
    )


def _reason(system: str, user: str, fallback: str, **kw) -> str:
    try:
        return llm.system_user(system, user, **kw)
    except Exception as exc:  # noqa: BLE001
        return f"[LLM unavailable: {exc}]\n\n{fallback}"


def _reg_query(use: str) -> str:
    return (
        f"SR 11-7 model risk management requirements and applicable fair-lending / "
        f"consumer-protection / BSA-AML rules (ECOA, FCRA, UDAAP, Reg E, FFIEC "
        f"BSA/AML, NIST AI RMF) for validating, explaining, and monitoring a model "
        f"used for: {use}"
    )


def build_report(model_id: str) -> dict:
    meta_raw = get_model_metadata(model_id)
    meta = json.loads(meta_raw)
    if "error" in meta:
        raise SystemExit(f"unknown model_id: {model_id}")
    docs = get_model_documentation(model_id)
    use = meta.get("use", "")

    validation = _reason(
        VALIDATION_SYS,
        f"MODEL METADATA:\n{meta_raw}\n\nMODEL DOCUMENTATION:\n{docs}\n\n"
        f"RELEVANT REGULATIONS:\n{_search('model validation and monitoring requirements: ' + use, 5)}",
        fallback=f"Metadata:\n{meta_raw}",
        max_tokens=1400,
    )

    is_fraud_detector = model_id == "FRAUD-XGB-GNN-001"
    if is_fraud_detector:
        txn = {"amount": 4200.0, "is_foreign": True, "merchant_risk": 0.6,
               "hour": 2, "velocity_24h": 9}
        score = score_transaction(**txn)
        perf = _reason(FRAUD_SYS, f"Fraud model output (JSON):\n{score}",
                       fallback=f"Fraud output:\n{score}")
        perf_section = (
            f"Transaction under review: `{txn}`\n\n```json\n{score}\n```\n\n{perf}"
        )
        perf_title = "Fraud Agent analysis (live transaction scoring)"
    else:
        perf = _reason(_PERF_SYS, f"MODEL CARD:\n{docs}",
                       fallback="Performance summary unavailable.")
        perf_section = perf
        perf_title = "Performance & Monitoring summary"

    reg_rules = _search(_reg_query(use), 6)
    reg_ctx = _reason(
        RETRIEVER_SYS,
        f"Question:\nWhat regulatory requirements apply to validating and "
        f"monitoring a model used for '{use}'?\n\nRegulation excerpts (JSON):\n{reg_rules}",
        fallback=f"Excerpts:\n{reg_rules}",
        max_tokens=1200,
    )

    combined = (
        f"MODEL ID: {model_id} ({meta.get('name')})\nUSE: {use}\n\n"
        f"## Validation Findings\n{validation}\n\n"
        f"## {perf_title}\n{perf_section}\n\n"
        f"## Regulatory Context\n{reg_ctx}\n"
    )
    report = _reason(REPORT_SYS, f"Source material:\n\n{combined}",
                     fallback=f"# Governance Report\n\n{combined}", max_tokens=1800)
    return {
        "meta": meta, "validation": validation, "perf_title": perf_title,
        "perf_section": perf_section, "regulatory_context": reg_ctx, "report": report,
    }


def write_report(model_id: str) -> str:
    s = get_settings()
    r = build_report(model_id)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    m = r["meta"]
    doc = f"""# Sample Governance Report — {model_id}

**{m.get('name')}** · type: {m.get('type')} · tier: {m.get('tier')} · owner: {m.get('owner')}

> Generated by the `reg-agents` pipeline on {ts}.
> LLM: **{s.llm_provider}** (`{s.active_model}`) · Embeddings: **{s.embedding_provider}** ·
> Vector backend: **{s.vector_backend}** · Fraud backend: **{s.triton_url or 'local heuristic'}**
>
> Same agent logic as the live A2A/MCP stack (validation, fraud/performance,
> retriever, report), captured as a committed artifact. See docs/ARCHITECTURE.md.

---

## Final Report (Report Agent)

{r['report']}

---

## Appendix A — Validation Agent findings

{r['validation']}

---

## Appendix B — {r['perf_title']}

{r['perf_section']}

---

## Appendix C — Retriever Agent regulatory context

{r['regulatory_context']}
"""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, f"{model_id}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="generate for every model")
    args = ap.parse_args()

    models = list(_INVENTORY) if args.all else (args.model or ["FRAUD-XGB-GNN-001"])
    for mid in models:
        out = write_report(mid)
        size = os.path.getsize(out)
        print(f"wrote {out} ({size} bytes)")


if __name__ == "__main__":
    main()
