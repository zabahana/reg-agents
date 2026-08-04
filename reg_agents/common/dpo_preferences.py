"""RLAIF preference pairs from the dual-judge panel (NIM + OpenAI).

Preference rule (Bradley-Terry / DPO):
  - When both judges return a verdict and agree, that consensus is *chosen*
    and the opposite regulatory label is *rejected*.
  - Rows where consensus also contradicts the logistic-regression gate are
    marked ``high_value`` — these are the disputed-set teaching signal.
  - Rows where judges disagree (or either abstains) are skipped for RLAIF;
    optional ``weak_contrastive`` pairs can bootstrap volume from weak labels.

OpenAI and NIM are first-class judges here — neither is a fallback.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

PROMPT_TEMPLATE = (
    "You are an independent compliance judge at a bank. Decide whether a "
    "consumer complaint is REGULATORY (it implicates a specific consumer-"
    "protection regulation such as FCRA, FDCPA, Reg E, Reg Z, RESPA, ECOA, "
    "TISA/Reg DD, Reg CC, GLBA, SCRA/MLA, BSA/AML, UDAAP) or NON-REGULATORY "
    "(a routine customer-service matter with no specific regulatory hook).\n"
    "Judge only from the complaint text. Reply with STRICT JSON only:\n"
    '{{"is_regulatory": true|false, "confidence": 0.0-1.0, '
    '"reason": "<one sentence>"}}\n\n'
    "COMPLAINT:\n{narrative}\n\nJSON VERDICT:"
)

_REASON = {
    True: "Complaint implicates a consumer-protection regulation.",
    False: "Routine customer-service matter with no specific regulatory hook.",
}


def verdict_json(is_regulatory: bool, reason: str = "",
                 confidence: float = 0.8) -> str:
    return json.dumps({
        "is_regulatory": bool(is_regulatory),
        "confidence": float(confidence),
        "reason": (reason or _REASON[bool(is_regulatory)]).strip(),
    }, ensure_ascii=False)


def load_panel_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_rlaif_pairs(panel_rows: Iterable[Dict]) -> List[Dict]:
    """Consensus-of-judges → chosen; opposite label → rejected."""
    pairs = []
    for row in panel_rows:
        nim, oa = row.get("nim"), row.get("openai")
        if not isinstance(nim, bool) or not isinstance(oa, bool):
            continue
        if nim != oa:
            continue
        chosen_label = bool(nim)
        rejected_label = not chosen_label
        reason = row.get("nim_reason") or row.get("openai_reason") or ""
        conf = row.get("nim_confidence") or row.get("openai_confidence") or 0.8
        gate = row.get("gate")
        high_value = isinstance(gate, bool) and gate != chosen_label
        narrative = row["narrative"]
        pairs.append({
            "narrative": narrative,
            "prompt": PROMPT_TEMPLATE.format(narrative=narrative[:1800]),
            "chosen_label": chosen_label,
            "rejected_label": rejected_label,
            "chosen": verdict_json(chosen_label, reason, float(conf)),
            "rejected": verdict_json(rejected_label),
            "source": "rlaif_disputed" if high_value else "rlaif_consensus",
            "high_value": high_value,
            "gate": gate,
            "weak": row.get("weak"),
            "nim": nim,
            "openai": oa,
        })
    return pairs


def recover_disputed_from_summary(summary_path: str,
                                  full_texts: List[str]) -> List[Dict]:
    """Recover full narratives for disputed rows stored truncated in the summary JSON.

    Used when ``judge_agreement_rows.jsonl`` is not yet available (pre-persist runs).
    """
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    disputed = summary.get("disputed", {}).get("rows", [])
    recovered = []
    for d in disputed:
        prefix = (d.get("narrative") or "")[:120]
        match = next((t for t in full_texts if t.startswith(prefix[:80])), None)
        if match is None:
            continue
        # Both judges disagree with gate and agree with each other by construction.
        chosen = bool(d["nim"])  # == d["openai"]
        recovered.append({
            "narrative": match,
            "gate": bool(d["gate"]),
            "weak": bool(d["weak"]),
            "nim": bool(d["nim"]),
            "openai": bool(d["openai"]),
            "nim_reason": "",
            "openai_reason": "",
            "nim_confidence": 0.8,
            "openai_confidence": 0.8,
        })
    return recovered


def build_weak_contrastive(texts: List[str], labels: List[bool],
                           limit: Optional[int] = None) -> List[Dict]:
    """Bootstrap pairs: weak label chosen, opposite rejected."""
    pairs = []
    for text, lab in zip(texts, labels):
        lab = bool(lab)
        pairs.append({
            "narrative": text,
            "prompt": PROMPT_TEMPLATE.format(narrative=text[:1800]),
            "chosen_label": lab,
            "rejected_label": not lab,
            "chosen": verdict_json(lab),
            "rejected": verdict_json(not lab),
            "source": "weak_contrastive",
            "high_value": False,
            "gate": None,
            "weak": lab,
            "nim": None,
            "openai": None,
        })
        if limit and len(pairs) >= limit:
            break
    return pairs


def write_pairs(pairs: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


def summarize_pairs(pairs: List[Dict]) -> Dict:
    by_src: Dict[str, int] = {}
    for p in pairs:
        by_src[p["source"]] = by_src.get(p["source"], 0) + 1
    n_reg = sum(1 for p in pairs if p["chosen_label"])
    return {
        "n_pairs": len(pairs),
        "by_source": by_src,
        "chosen_regulatory_rate": round(n_reg / max(len(pairs), 1), 4),
        "high_value": sum(1 for p in pairs if p.get("high_value")),
    }
