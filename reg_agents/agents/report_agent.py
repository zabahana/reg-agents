"""Report Agent (A2A server on :8104).

Skill: assemble an audit-ready governance report from upstream artifacts
(validation findings, fraud analysis, regulatory context). Produces the kind of
SR 11-7 documentation an MRM function would file.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from reg_agents.agents.base import run
from reg_agents.common import llm
from reg_agents.common.a2a import (
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Task,
    TextPart,
    build_a2a_app,
)
from reg_agents.common.mrm_document_instructions import (
    GOVERNANCE_MDD_VAL_BRIDGE,
    MDD_SECTION_INSTRUCTIONS,
    VALIDATION_SECTION_INSTRUCTIONS,
)

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_UNAVAILABLE_RE = re.compile(r"\[[^\]]+\bunavailable:[^\]]+\]", re.IGNORECASE)

CARD = AgentCard(
    name="Report Agent",
    description="Generates audit-ready model governance reports.",
    url="http://localhost:8104",
    skills=[
        AgentSkill(
            id="governance-report",
            name="Governance report",
            description="Compose SR 11-7-style documentation from findings.",
            tags=["reporting", "governance", "audit"],
        )
    ],
)

_SYS = (
    "You are a senior model-governance lead (PhD econometrics, 20 years in "
    "model risk at large banks) assembling an audit-ready report an examiner "
    "could rely on.\n\n"
    f"{GOVERNANCE_MDD_VAL_BRIDGE}\n\n"
    "Compose these sections, in order, with those exact titles:\n"
    "1. Executive Summary\n"
    "2. Model Overview (summarize MDD coverage: purpose, data, split protocol, "
    "bake-off, champion, threshold — mirror the MDD section checklist)\n"
    "3. Validation Findings (mirror the independent validation section "
    "checklist; include disposition)\n"
    "4. Performance Analysis\n"
    "5. Regulatory Mapping\n"
    "6. Open Gaps & Remediation\n"
    "7. Risk Rating (Low/Medium/High) with a one-sentence justification\n\n"
    "MDD section checklist (use when summarizing development evidence):\n"
    f"{MDD_SECTION_INSTRUCTIONS}\n\n"
    "Validation section checklist (use when summarizing 2nd-line evidence):\n"
    f"{VALIDATION_SECTION_INSTRUCTIONS}\n\n"
    "Describe the model exactly as the source material characterizes it — "
    "never invent a model type, purpose, or numbers; quote reported metrics "
    "verbatim. Prefer markdown tables for findings "
    "(#: Severity | Finding | Remediation | Owner) and regulatory mapping "
    "(Regulation | Relevance | Evidence)."
)


def _split_artifacts(material: str) -> Dict[str, str]:
    """Split the orchestrator's labelled source material into named artifacts."""
    matches = list(_SECTION_RE.finditer(material))
    artifacts: Dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(material)
        artifacts[match.group(1).strip().lower()] = material[match.end():end].strip()
    return artifacts


def _json_values(text: str) -> list[Any]:
    """Return embedded JSON values without requiring a fixed tool payload shape."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(text):
        match = re.search(r"[\[{]", text[index:])
        if not match:
            break
        start = index + match.start()
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        values.append(value)
        index = start + consumed
    return values


def _non_json_text(text: str) -> str:
    """Retain narrative evidence while removing all serialized JSON payloads."""
    decoder = json.JSONDecoder()
    parts: list[str] = []
    index = 0
    while index < len(text):
        match = re.search(r"[\[{]", text[index:])
        if not match:
            parts.append(text[index:])
            break
        start = index + match.start()
        parts.append(text[index:start])
        try:
            _, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            parts.append(text[start:start + 1])
            index = start + 1
        else:
            index = start + consumed
    clean_lines = []
    for line in "".join(parts).splitlines():
        line = _UNAVAILABLE_RE.sub("", line).strip()
        if line and not line.lower().startswith(("model metadata:", "relevant rules:",
                                                 "retrieved excerpts:")):
            clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def _first_mapping(values: list[Any], required_key: str) -> Dict[str, Any]:
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, dict) and required_key in candidate:
                return candidate
    return {}


def _citation_rows(values: list[Any], narrative: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("source"):
                rows.append((
                    str(candidate["source"]),
                    str(candidate.get("heading") or "Relevant excerpt"),
                    str(candidate.get("text") or "No excerpt returned."),
                ))
    for source in re.findall(r"\[([^\]\n]+)\]", narrative):
        if not any(source == row[0] for row in rows):
            rows.append((source, "Citation supplied by upstream agent", "No excerpt returned."))
    return rows


def _risk_rating(validation: str, source_unavailable: bool) -> tuple[str, str]:
    evidence = validation.lower()
    if re.search(r"\b(high|critical)\b", evidence):
        return "High", "Upstream validation evidence identifies a high-severity matter."
    if source_unavailable or re.search(r"insufficient evidence|gap|unavailable", evidence):
        return "Medium", "Evidence is incomplete and requires independent review before reliance."
    return "Low", "No unresolved high-severity matter was supplied in the available artifacts."


def deterministic_report(material: str) -> str:
    """Build an audit-readable report from artifacts when LLM synthesis fails.

    This intentionally renders selected fields and prose, never serialized tool
    payloads. It remains useful when any upstream agent also used its fallback.
    """
    artifacts = _split_artifacts(material)
    model_id_match = re.search(r"^MODEL ID:\s*(.+?)\s*$", material, re.MULTILINE)
    model_id = model_id_match.group(1).strip() if model_id_match else "Not supplied"
    metadata_text = artifacts.get("model metadata (registry of record)", "")
    validation_text = artifacts.get("validation findings", "")
    regulatory_text = artifacts.get("regulatory context", "")
    metadata = _first_mapping(_json_values(metadata_text), "model_id")
    if metadata.get("model_id"):
        model_id = str(metadata["model_id"])

    validation_narrative = _non_json_text(validation_text)
    regulatory_narrative = _non_json_text(regulatory_text)
    citations = _citation_rows(_json_values(regulatory_text) + _json_values(validation_text),
                               f"{regulatory_narrative}\n{validation_narrative}")
    unavailable = bool(_UNAVAILABLE_RE.search(validation_text) or
                       _UNAVAILABLE_RE.search(regulatory_text) or not metadata)
    rating, rationale = _risk_rating(validation_narrative, unavailable)

    metadata_rows = [
        ("Model ID", model_id),
        ("Name", metadata.get("name", "Not available from registry")),
        ("Model type", metadata.get("type", "Not available from registry")),
        ("Owner", metadata.get("owner", "Not available from registry")),
        ("Tier", metadata.get("tier", "Not available from registry")),
        ("Intended use", metadata.get("use", "Not available from registry")),
        ("Protected-class feature treatment",
         metadata.get("protected_class_features", "Not available from registry")),
    ]
    metadata_table = "\n".join(f"| {label} | {value} |" for label, value in metadata_rows)
    findings = validation_narrative or (
        "No narrative validation finding was returned. The underlying validation "
        "artifact must be obtained and independently reviewed."
    )
    citation_table = (
        "\n".join(f"| {source} | {heading} | {text} |"
                  for source, heading, text in citations)
        if citations else
        "| No citation artifact returned | Evidence gap | Retrieve applicable regulatory passages before approval. |"
    )
    conditions = [
        "Obtain and retain the complete independent validation workpapers and issue disposition.",
        "Confirm model inventory metadata, ownership, tier, and intended-use controls are current.",
        "Document regulatory applicability and link each requirement to a testable control.",
    ]
    if unavailable:
        conditions.insert(0, "Do not treat this report as an LLM-authored approval opinion; "
                             "complete a human review of the source artifacts.")
    if rating != "Low":
        conditions.append("Close or formally accept identified evidence gaps through the model-risk issue process.")

    return "\n".join([
        "# Governance Report",
        "",
        "## Executive Summary",
        f"This deterministic governance report covers **{model_id}** using the artifacts "
        "available at report time. LLM synthesis was unavailable, so no interpretation "
        "beyond the supplied evidence has been introduced.",
        f"Current governance posture: **{rating} risk** — {rationale}",
        "",
        "## Model Metadata",
        "| Field | Recorded value |",
        "| --- | --- |",
        metadata_table,
        "",
        "## Validation Findings",
        findings,
        "",
        "## Regulatory Basis / Citations",
        "| Source | Heading | Evidence |",
        "| --- | --- | --- |",
        citation_table,
        "",
        "## Risk Assessment",
        f"**Rating: {rating}**",
        "",
        rationale,
        "",
        "## Recommendations / Conditions",
        *[f"{number}. {condition}" for number, condition in enumerate(conditions, 1)],
        "",
        "## Audit Trail",
        f"- Model identifier requested: `{model_id}`.",
        f"- Registry metadata: {'received' if metadata else 'not available'}.",
        f"- Validation artifact: {'received' if validation_text else 'not available'}.",
        f"- Regulatory artifact: {'received' if regulatory_text else 'not available'}.",
        "- LLM synthesis status: unavailable; deterministic artifact-based fallback generated.",
    ])


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    material = message.as_text()
    try:
        report = llm.system_user(
            _SYS,
            f"Source material to synthesize into the report:\n\n{material}",
            max_tokens=2400,
        )
    except Exception:  # noqa: BLE001 - deterministic fallback preserves report availability
        report = deterministic_report(material)
    return Task(artifacts=[Artifact(name="governance_report", parts=[TextPart(text=report)])])


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.report_agent:app", 8104)
