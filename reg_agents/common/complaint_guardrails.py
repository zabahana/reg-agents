"""Complaint-path guardrails (NeMo Guardrails when installed + native rails).

Native rails always run (taxonomy whitelist, citation grounding, input
hardening). When ``nemoguardrails`` is installed and
``COMPLAINT_NEMO_GUARDRAILS=1``, LLM user prompts are additionally checked
through a Colang rails config under ``reg_agents/guardrails/complaint/``.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions|system\s*prompt|"
    r"jailbreak|you\s+are\s+now\s+dan|"
    r"reveal\s+(your|the)\s+(system|hidden)\s+prompt)",
    re.I,
)

_MAX_NARRATIVE_CHARS = 8000


def _nemo_enabled() -> bool:
    return os.getenv("COMPLAINT_NEMO_GUARDRAILS", "0").strip() in {"1", "true", "yes"}


def check_input(narrative: str) -> Tuple[str, List[str]]:
    """Sanitize / gate the complaint narrative. Returns (text, triggered_rules)."""
    rules: List[str] = []
    text = (narrative or "").strip()
    if not text:
        rules.append("empty_narrative")
        return text, rules
    # Scan the full narrative before truncation so injection cues in the
    # tail are still detected.
    if _INJECTION_RE.search(text):
        rules.append("prompt_injection_heuristic")
        text = _INJECTION_RE.sub("[filtered]", text)
    if len(text) > _MAX_NARRATIVE_CHARS:
        text = text[:_MAX_NARRATIVE_CHARS]
        rules.append("narrative_truncated")
    if _nemo_enabled():
        ok, nemo_rules = _nemo_input_check(text)
        rules.extend(nemo_rules)
        if not ok:
            rules.append("nemo_guardrails_blocked_input")
    return text, rules


def check_stage2_output(result: Dict, allowed_sources: Optional[List[str]] = None) -> List[str]:
    """Validate stage-2 LLM/keyword output against taxonomy + citation grounding."""
    from reg_agents.common.complaints import REGULATIONS

    rules: List[str] = []
    label = str(result.get("label", "")).strip().upper()
    if label not in REGULATIONS:
        rules.append("label_not_in_taxonomy")
        result["label"] = "UDAAP"
        result["rationale"] = (
            (result.get("rationale") or "")
            + " [guardrail: label reset to UDAAP — outside whitelist]"
        ).strip()
        result["confidence"] = min(float(result.get("confidence") or 0.3), 0.3)

    conf = result.get("confidence")
    try:
        conf_f = float(conf)
        if not (0.0 <= conf_f <= 1.0):
            rules.append("confidence_out_of_range")
            result["confidence"] = max(0.0, min(conf_f, 1.0))
    except (TypeError, ValueError):
        rules.append("confidence_invalid")
        result["confidence"] = 0.4

    cite = result.get("citation")
    if cite and allowed_sources:
        src = str(cite.get("source", "")).lower()
        if src and not any(a.lower() in src or src in a.lower() for a in allowed_sources):
            rules.append("citation_not_in_retrieval")
            # Keep a retrieved excerpt rather than an invented source.
            result["citation_source"] = allowed_sources[0]

    return rules


def _nemo_input_check(text: str) -> Tuple[bool, List[str]]:
    """Best-effort NeMo Guardrails check; no-op if package/config missing."""
    try:
        from nemoguardrails import LLMRails, RailsConfig  # type: ignore
    except Exception:
        return True, ["nemo_guardrails_unavailable"]

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "guardrails", "complaint",
    )
    if not os.path.isdir(cfg_path):
        return True, ["nemo_guardrails_config_missing"]

    try:
        config = RailsConfig.from_path(cfg_path)
        rails = LLMRails(config)
        # generate() may call an LLM; for input-only we use check if available.
        if hasattr(rails, "check_rails"):
            # Older/newer APIs differ; treat exceptions as pass-through.
            pass
        resp = rails.generate(messages=[{"role": "user", "content": text[:1500]}])
        content = ""
        if isinstance(resp, dict):
            content = str(resp.get("content") or resp.get("response") or "")
        else:
            content = str(resp)
        if "not allowed" in content.lower() or "cannot assist" in content.lower():
            return False, ["nemo_guardrails_blocked"]
        return True, ["nemo_guardrails_ok"]
    except Exception as exc:  # noqa: BLE001
        return True, [f"nemo_guardrails_error:{str(exc)[:80]}"]
