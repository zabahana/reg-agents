"""Risk intelligence layer — elevates complaint classification into control risk.

Classification answers *what* a complaint is (regulation taxonomy). Risk
intelligence asks whether the same narrative signals a **systemic control
failure** rather than an isolated service event.

Inputs are the stage-1/stage-2 classification dict from
``reg_agents.common.complaints``. Outputs are deterministic (rule + TF-IDF
similarity + local feature attribution) with an optional LLM hypothesis for
the control-failure narrative when an LLM is available.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# Regulation label → first-line control domain (bank risk taxonomy framing).
CONTROL_DOMAINS: Dict[str, Dict[str, str]] = {
    "FCRA_ACCURACY": {
        "domain": "Credit Reporting Controls",
        "failure_mode": "Inaccurate tradeline / identity data reaching consumer reports",
    },
    "FCRA_INVESTIGATION": {
        "domain": "Credit Reporting Controls",
        "failure_mode": "Dispute reinvestigation SLA or reasonableness failure",
    },
    "FCRA_PERMISSIBLE_PURPOSE": {
        "domain": "Credit Reporting Controls",
        "failure_mode": "Improper soft/hard inquiry or unauthorized report use",
    },
    "FDCPA_DEBT_VALIDATION": {
        "domain": "Collections Controls",
        "failure_mode": "Collection without validation or of a debt not owed",
    },
    "FDCPA_COMMUNICATION": {
        "domain": "Collections Controls",
        "failure_mode": "Harassing / improper collector contact pattern",
    },
    "FDCPA_THREATS": {
        "domain": "Collections Controls",
        "failure_mode": "False statements or prohibited collection threats",
    },
    "REG_E_UNAUTHORIZED": {
        "domain": "Payments / EFT Controls",
        "failure_mode": "Unauthorized electronic transfer intake or claim handling",
    },
    "REG_E_ERROR_RESOLUTION": {
        "domain": "Payments / EFT Controls",
        "failure_mode": "Error-resolution timeline or provisional-credit failure",
    },
    "REG_Z_BILLING": {
        "domain": "Consumer Credit Controls",
        "failure_mode": "Billing-error / chargeback handling control break",
    },
    "REG_Z_DISCLOSURE": {
        "domain": "Consumer Credit Controls",
        "failure_mode": "Fee / APR / promotional-rate disclosure control break",
    },
    "TILA_ORIGINATION": {
        "domain": "Consumer Credit Controls",
        "failure_mode": "Origination disclosure or underwriting notice failure",
    },
    "ECOA_DISCRIMINATION": {
        "domain": "Fair Lending Controls",
        "failure_mode": "Potential prohibited-basis discrimination in credit",
    },
    "ECOA_ADVERSE_ACTION": {
        "domain": "Fair Lending Controls",
        "failure_mode": "Adverse-action notice completeness / timing failure",
    },
    "RESPA_SERVICING": {
        "domain": "Mortgage Servicing Controls",
        "failure_mode": "Payment application, escrow, or QWR handling failure",
    },
    "RESPA_LOSS_MITIGATION": {
        "domain": "Mortgage Servicing Controls",
        "failure_mode": "Loss-mitigation / dual-tracking / foreclosure control break",
    },
    "TISA_REG_DD": {
        "domain": "Deposit Account Controls",
        "failure_mode": "Deposit fee / rate disclosure or overdraft practice failure",
    },
    "REG_CC_FUNDS": {
        "domain": "Deposit Account Controls",
        "failure_mode": "Funds-availability hold policy or disclosure failure",
    },
    "BSA_AML": {
        "domain": "Financial Crime / AML Controls",
        "failure_mode": "Account freeze/closure without adequate customer communication",
    },
    "GLBA_PRIVACY": {
        "domain": "Privacy & Information Security Controls",
        "failure_mode": "NPI sharing, safeguarding, or opt-out control failure",
    },
    "UDAAP": {
        "domain": "Conduct / UDAAP Controls",
        "failure_mode": "Unfair, deceptive, or abusive act or practice pattern",
    },
    "SALES_PRACTICES": {
        "domain": "Sales Practices Controls",
        "failure_mode": "Unauthorized account/product opening or enrollment",
    },
    "SCRA_MLA": {
        "domain": "Servicemember Protection Controls",
        "failure_mode": "SCRA/MLA rate-cap or protection handling failure",
    },
    "LOAN_SERVICING": {
        "domain": "Loan Servicing Controls",
        "failure_mode": "Non-mortgage payment, payoff, or hardship handling failure",
    },
}

# Higher base weight = more thematic / exam-sensitive when pattern language appears.
_SEVERITY: Dict[str, float] = {
    "ECOA_DISCRIMINATION": 0.35,
    "SALES_PRACTICES": 0.32,
    "BSA_AML": 0.28,
    "UDAAP": 0.28,
    "ECOA_ADVERSE_ACTION": 0.22,
    "FDCPA_THREATS": 0.22,
    "RESPA_LOSS_MITIGATION": 0.22,
    "FCRA_INVESTIGATION": 0.18,
    "REG_E_UNAUTHORIZED": 0.18,
    "GLBA_PRIVACY": 0.18,
    "SCRA_MLA": 0.20,
}

_PATTERN_RE = re.compile(
    r"\b(?:"
    r"always|repeatedly|every\s+(?:day|week|month|time)|multiple\s+times|"
    r"over\s+and\s+over|again\s+and\s+again|system(?:atic|ically)?|"
    r"automated|company[- ]wide|across\s+(?:all|the)|everyone|all\s+(?:my\s+)?"
    r"(?:accounts?|loans?|cards?)|policy|script|boilerplate|never\s+fix"
    r")\b",
    re.I,
)

_CONTROL_BREAK_RE = re.compile(
    r"\b(?:"
    r"never\s+investigat\w*|no\s+investigation|ignored|refused\s+to|"
    r"failed\s+to|within\s+30\s+days|no\s+response|never\s+respond\w*|"
    r"without\s+(?:my\s+)?consent|opened\s+without|did\s+not\s+notify|"
    r"no\s+notice|provisional\s+credit|dual\s+track|"
    r"verified\s+without|closed\s+(?:my\s+)?account\s+without"
    r")\b",
    re.I,
)

_RISK_SYS = (
    "You are a bank operational-risk analyst. Given a consumer complaint and "
    "its regulation classification, hypothesize whether this is an isolated "
    "service event or evidence of a systemic control failure. Reply STRICT "
    "JSON only: "
    '{"hypothesis": "<1-2 sentences>", '
    '"control_test": "<one concrete 1LOD/2LOD test to run>", '
    '"systemic_lean": "isolated"|"cluster"|"systemic"}'
)


def _family_of(label: str) -> str:
    from reg_agents.common.complaints import FAMILY, NON_REGULATORY

    if label == NON_REGULATORY:
        return NON_REGULATORY
    return FAMILY.get(label, label.split("_")[0])


def _signal_band(score: float, is_regulatory: bool) -> str:
    if not is_regulatory:
        return "none"
    if score >= 0.65:
        return "elevated"
    if score >= 0.40:
        return "moderate"
    return "isolated"


def _recommended_action(signal: str, domain: str) -> str:
    if signal == "none":
        return "Route to standard service recovery — no regulatory / control nexus."
    if signal == "elevated":
        return (
            f"Open a thematic review of {domain}: sample peer complaints, "
            "test the implicated control, and escalate to compliance / OR if "
            "the cluster confirms."
        )
    if signal == "moderate":
        return (
            f"Flag for 1LOD control owner of {domain}; pull a short lookback "
            "of same-family complaints before closing as one-off."
        )
    return (
        f"Treat as case-level remediation under {domain}; monitor for "
        "recurrence in the same control domain."
    )


def local_explanation(text: str, top_k: int = 8) -> Dict:
    """Top TF-IDF n-grams pushing the stage-1 champion toward regulatory.

    For logistic regression uses signed coefficient × tf-idf weight; for
    tree models uses |contribution| from feature × importance as a local
    surrogate. Returns empty lists when the stage-1 cache is unavailable.
    """
    from reg_agents.common.complaints import _stage1_cached

    try:
        s1 = _stage1_cached()
    except Exception:  # noqa: BLE001
        return {"top_positive": [], "top_negative": [], "model": None}

    vec, model, champion = s1["vectorizer"], s1["models"][s1["champion"]], s1["champion"]
    xt = vec.transform([text])
    feature_names = vec.get_feature_names_out()
    row = xt.tocsr()
    idx = row.indices
    data = row.data
    if len(idx) == 0:
        return {"top_positive": [], "top_negative": [], "model": champion}

    if hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        scored = [(feature_names[i], float(coef[i] * v)) for i, v in zip(idx, data)]
    else:
        # XGBoost / tree: weight by gain-style feature importance when present.
        try:
            importances = model.feature_importances_
            scored = [
                (feature_names[i], float(importances[i] * v))
                for i, v in zip(idx, data)
            ]
        except Exception:  # noqa: BLE001
            scored = [(feature_names[i], float(v)) for i, v in zip(idx, data)]

    scored.sort(key=lambda x: x[1], reverse=True)
    pos = [{"term": t, "contribution": round(s, 4)} for t, s in scored if s > 0][:top_k]
    neg = [{"term": t, "contribution": round(s, 4)}
           for t, s in sorted(scored, key=lambda x: x[1]) if s < 0][:top_k]
    return {"top_positive": pos, "top_negative": neg, "model": champion}


@lru_cache(maxsize=1)
def _prior_case_index() -> Tuple:
    """Cached TF-IDF matrix over the curated complaint corpus for similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    from reg_agents.common.complaints import load_complaints

    df = load_complaints()
    narratives = df["narrative"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    ids = (df["complaint_id"].astype(str).tolist()
           if "complaint_id" in df.columns
           else [str(i) for i in range(len(df))])
    products = (df["product"].astype(str).tolist()
                if "product" in df.columns
                else [""] * len(df))
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2)
    xt = normalize(vec.fit_transform(narratives))
    return vec, xt, narratives, labels, ids, products


def similar_prior_cases(text: str, label: str, k: int = 5,
                        min_sim: float = 0.15) -> List[Dict]:
    """Nearest prior CFPB cases, preferring the same regulation family."""
    import numpy as np
    from sklearn.preprocessing import normalize

    from reg_agents.common.complaints import NON_REGULATORY

    if not text or label == NON_REGULATORY:
        return []
    try:
        vec, xt, narratives, labels, ids, products = _prior_case_index()
    except Exception:  # noqa: BLE001
        return []

    q = normalize(vec.transform([text]))
    sims = np.asarray((q @ xt.T).todense()).ravel()
    family = _family_of(label)
    order = np.argsort(-sims)
    out: List[Dict] = []
    # Pass 1: same family; pass 2: fill from any label.
    for require_family in (True, False):
        for i in order:
            if sims[i] < min_sim:
                break
            if narratives[i].strip() == text.strip():
                continue
            if require_family and _family_of(labels[i]) != family:
                continue
            if any(c["complaint_id"] == ids[i] for c in out):
                continue
            out.append({
                "complaint_id": ids[i],
                "label": labels[i],
                "product": products[i],
                "similarity": round(float(sims[i]), 3),
                "excerpt": narratives[i][:220].rstrip() + ("…" if len(narratives[i]) > 220 else ""),
            })
            if len(out) >= k:
                return out
    return out


def _heuristic_score(text: str, classification: Dict) -> Tuple[float, List[str]]:
    """Deterministic systemic-risk score in [0, 1] plus human-readable drivers."""
    from reg_agents.common.complaints import NON_REGULATORY

    s1 = classification.get("stage1") or {}
    s2 = classification.get("stage2") or {}
    label = str(s2.get("label", NON_REGULATORY))
    is_reg = bool(s1.get("is_regulatory"))
    drivers: List[str] = []

    if not is_reg or label == NON_REGULATORY:
        return 0.0, ["Stage-1 gate: no regulatory nexus"]

    score = 0.18  # base for any regulatory complaint
    drivers.append(f"Regulatory classification: {label}")

    sev = _SEVERITY.get(label, 0.12)
    score += sev
    if sev >= 0.28:
        drivers.append(f"Exam-sensitive control domain ({label})")

    pattern_hits = _PATTERN_RE.findall(text)
    if pattern_hits:
        bump = min(0.08 * len(pattern_hits), 0.24)
        score += bump
        drivers.append(
            "Recurrence / pattern language: "
            + ", ".join(sorted({h.lower() for h in pattern_hits})[:4])
        )

    break_hits = _CONTROL_BREAK_RE.findall(text)
    if break_hits:
        bump = min(0.07 * len(break_hits), 0.21)
        score += bump
        drivers.append("Control-process failure cues in the narrative")

    conf = s2.get("confidence")
    if isinstance(conf, (int, float)) and conf >= 0.75:
        score += 0.06
        drivers.append(f"High stage-2 confidence ({conf:.0%})")

    prob = s1.get("probability")
    if isinstance(prob, (int, float)) and prob >= 0.85:
        score += 0.05
        drivers.append(f"Strong stage-1 regulatory probability (p={prob:.2f})")

    return min(score, 0.98), drivers


def _llm_hypothesis(text: str, classification: Dict, domain: str,
                    failure_mode: str) -> Optional[Dict]:
    try:
        from reg_agents.common import llm

        s2 = classification.get("stage2") or {}
        user = (
            f"CONTROL DOMAIN: {domain}\n"
            f"DEFAULT FAILURE MODE: {failure_mode}\n"
            f"LABEL: {s2.get('label')} — {s2.get('regulation_name')}\n"
            f"MODEL RATIONALE: {s2.get('rationale', '')}\n\n"
            f"COMPLAINT:\n{text[:1800]}\n\nJSON:"
        )
        raw = llm.system_user(_RISK_SYS, user, temperature=0.0, max_tokens=220)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        parsed = json.loads(m.group(0))
        lean = str(parsed.get("systemic_lean", "")).strip().lower()
        if lean not in {"isolated", "cluster", "systemic"}:
            lean = "cluster"
        return {
            "hypothesis": str(parsed.get("hypothesis", "")).strip(),
            "control_test": str(parsed.get("control_test", "")).strip(),
            "systemic_lean": lean,
            "mode": "llm",
        }
    except Exception as exc:  # noqa: BLE001
        return {"mode": "fallback", "error": str(exc)[:160]}


def assess_risk_intelligence(text: str, classification: Dict,
                             use_llm: bool = True) -> Dict:
    """Post-classification risk intelligence for one complaint narrative."""
    from reg_agents.common.complaints import NON_REGULATORY

    s1 = classification.get("stage1") or {}
    s2 = classification.get("stage2") or {}
    label = str(s2.get("label", NON_REGULATORY))
    is_reg = bool(s1.get("is_regulatory")) and label != NON_REGULATORY

    meta = CONTROL_DOMAINS.get(label, {
        "domain": "General Conduct Controls",
        "failure_mode": "Potential consumer-protection control weakness",
    })
    if not is_reg:
        meta = {"domain": "—", "failure_mode": "No regulatory control nexus"}

    score, drivers = _heuristic_score(text, classification)
    similars = similar_prior_cases(text, label) if is_reg else []
    if len(similars) >= 3 and similars[0]["similarity"] >= 0.35:
        score = min(score + 0.12, 0.98)
        drivers.append(
            f"{len(similars)} similar prior cases "
            f"(top similarity {similars[0]['similarity']:.0%})"
        )
    elif len(similars) >= 1 and similars[0]["similarity"] >= 0.45:
        score = min(score + 0.07, 0.98)
        drivers.append(
            f"Close prior-case match (similarity {similars[0]['similarity']:.0%})"
        )

    signal = _signal_band(score, is_reg)
    explanation = local_explanation(text) if is_reg else {
        "top_positive": [], "top_negative": [], "model": None,
    }

    hypothesis: Optional[Dict] = None
    if is_reg and use_llm:
        hypothesis = _llm_hypothesis(
            text, classification, meta["domain"], meta["failure_mode"],
        )
        lean = (hypothesis or {}).get("systemic_lean")
        if lean == "systemic" and signal != "elevated":
            score = min(score + 0.08, 0.98)
            signal = _signal_band(score, True)
            drivers.append("LLM lean: systemic")
        elif lean == "isolated" and signal == "elevated" and score < 0.75:
            score = max(score - 0.06, 0.0)
            signal = _signal_band(score, True)
            drivers.append("LLM lean: isolated (score tempered)")

    return {
        "systemic_signal": signal,
        "score": round(float(score), 3),
        "control_domain": meta["domain"],
        "failure_mode": meta["failure_mode"],
        "drivers": drivers,
        "similar_prior_cases": similars,
        "local_explanation": explanation,
        "recommended_action": _recommended_action(signal, meta["domain"]),
        "hypothesis": hypothesis,
        "anchor": {
            "stage1_model": s1.get("model"),
            "stage1_probability": s1.get("probability"),
            "stage1_threshold": s1.get("threshold"),
            "stage2_label": label,
            "stage2_confidence": s2.get("confidence"),
            "note": (
                "Classification is the model anchor (TF-IDF gate + regulation "
                "label). Risk intelligence interprets whether that label "
                "signals a systemic control failure."
            ),
        },
    }
