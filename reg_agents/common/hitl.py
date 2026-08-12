"""Human-in-the-loop (HITL) dispositions for complaint classification.

Analysts approve, override the regulation label, or escalate. Decisions are
appended to a JSONL audit log under ``data/hitl/`` so every override is
reconstructible. Elevated systemic-risk signals auto-queue for review.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HITL_DIR = os.path.join(_ROOT, "data", "hitl")
HITL_LOG = os.path.join(HITL_DIR, "complaint_decisions.jsonl")

DISPOSITIONS = ("approve", "override", "escalate")


def _ensure_dir() -> None:
    os.makedirs(HITL_DIR, exist_ok=True)


def hitl_required(classification: Dict) -> bool:
    """Auto-queue when risk intelligence is elevated/moderate or ECOA/sales."""
    risk = classification.get("risk_intelligence") or {}
    signal = str(risk.get("systemic_signal", ""))
    if signal in {"elevated", "moderate"}:
        return True
    label = str((classification.get("stage2") or {}).get("label", ""))
    return label in {"ECOA_DISCRIMINATION", "SALES_PRACTICES", "BSA_AML"}


def attach_hitl_status(classification: Dict) -> Dict:
    """Annotate a classification with HITL routing metadata (no I/O)."""
    required = hitl_required(classification)
    classification["hitl"] = {
        "required": required,
        "status": "pending_review" if required else "auto_routed",
        "allowed_dispositions": list(DISPOSITIONS),
        "note": (
            "Analyst must approve, override the label, or escalate before "
            "final disposition."
            if required else
            "Auto-routed; analyst may still override or escalate."
        ),
    }
    return classification


def submit_decision(
    narrative: str,
    classification: Dict,
    disposition: str,
    analyst: str = "analyst",
    override_label: str = "",
    rationale: str = "",
) -> Dict:
    """Persist a HITL decision and return the audit record."""
    disposition = disposition.strip().lower()
    if disposition not in DISPOSITIONS:
        raise ValueError(f"disposition must be one of {DISPOSITIONS}")

    from reg_agents.common.complaints import REGULATIONS

    s2 = dict(classification.get("stage2") or {})
    final_label = s2.get("label", "")
    if disposition == "override":
        override_label = override_label.strip().upper()
        if override_label not in REGULATIONS:
            raise ValueError(
                f"override_label must be a taxonomy code; got {override_label!r}"
            )
        final_label = override_label

    record = {
        "decision_id": str(uuid.uuid4()),
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analyst": analyst or "analyst",
        "disposition": disposition,
        "model_label": s2.get("label"),
        "final_label": final_label,
        "override_label": override_label or None,
        "rationale": rationale.strip(),
        "systemic_signal": (classification.get("risk_intelligence") or {}).get(
            "systemic_signal"
        ),
        "stage1_probability": (classification.get("stage1") or {}).get("probability"),
        "narrative_excerpt": (narrative or "")[:400],
        "hitl_was_required": bool(
            (classification.get("hitl") or {}).get("required")
        ),
    }
    _ensure_dir()
    with open(HITL_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def list_decisions(limit: int = 50) -> List[Dict]:
    if not os.path.exists(HITL_LOG):
        return []
    rows: List[Dict] = []
    with open(HITL_LOG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-max(1, int(limit)):]


def decision_counts() -> Dict[str, int]:
    counts = {d: 0 for d in DISPOSITIONS}
    for row in list_decisions(limit=10_000):
        d = row.get("disposition")
        if d in counts:
            counts[d] += 1
    counts["total"] = sum(counts[d] for d in DISPOSITIONS)
    return counts
