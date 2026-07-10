"""Lifecycle Orchestrator (A2A server on :8108).

Coordinates the full model-development lifecycle across the three lines of
defense, over A2A:

    Developer Agent  ─►  Validator Agent  ─►  Audit Agent
    (1st line)           (2nd line)           (3rd line)
    dev document         validation report    audit report

Exposes a "model-lifecycle" A2A skill and a `run_lifecycle()` helper used by the
CLI and Streamlit UI.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from reg_agents.agents.base import run
from reg_agents.common.a2a import (
    A2AClient,
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Task,
    TextPart,
    build_a2a_app,
)
from reg_agents.config import get_settings

CARD = AgentCard(
    name="Lifecycle Orchestrator",
    description="Runs develop → validate → audit across the three lines of defense.",
    url="http://localhost:8108",
    skills=[
        AgentSkill(
            id="model-lifecycle",
            name="Model development lifecycle",
            description="Develop a champion model, validate it, and audit the process.",
            tags=["orchestration", "a2a", "mrm", "three-lines-of-defense"],
        )
    ],
)


def _safe(fn, label: str) -> str:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return f"[{label} unavailable: {exc}]"


def run_lifecycle(task_id: str = "fraud") -> Dict[str, str]:
    """Fan out develop → validate → audit over A2A, threading artifacts along."""
    s = get_settings()
    developer = A2AClient(s.developer_agent_url)
    validator = A2AClient(s.validator_agent_url)
    auditor = A2AClient(s.audit_agent_url)

    steps: Dict[str, str] = {}

    try:
        dev_task = developer.send(task_id, {"task_id": task_id})
        dev_doc = _artifact(dev_task, "model_development_document")
        leaderboard = _artifact(dev_task, "bakeoff_leaderboard")
    except Exception as exc:  # noqa: BLE001
        dev_doc = f"[developer agent unavailable: {exc}]"
        leaderboard = ""
    steps["model_development_document"] = dev_doc
    steps["bakeoff_leaderboard"] = leaderboard

    steps["validation_report"] = _safe(
        lambda: validator.send_text(
            dev_doc,
            {"task_id": task_id, "development_document": dev_doc, "leaderboard": leaderboard},
        ),
        "validator agent",
    )

    steps["audit_report"] = _safe(
        lambda: auditor.send_text(
            steps["validation_report"],
            {
                "task_id": task_id,
                "development_document": dev_doc,
                "validation_report": steps["validation_report"],
            },
        ),
        "audit agent",
    )
    return steps


def _artifact(task: Task, name: str) -> str:
    """Pull a named artifact's text out of an A2A Task result."""
    for art in task.artifacts:
        if art.name == name:
            return art.as_text()
    return ""


def handle(message: Message, metadata: Dict[str, Any]) -> Task:
    task_id = (metadata.get("task_id") or message.as_text() or "fraud").strip()
    result = run_lifecycle(task_id)
    return Task(
        artifacts=[
            Artifact(name="model_development_document",
                     parts=[TextPart(text=result["model_development_document"])]),
            Artifact(name="validation_report",
                     parts=[TextPart(text=result["validation_report"])]),
            Artifact(name="audit_report",
                     parts=[TextPart(text=result["audit_report"])]),
            Artifact(name="trace", parts=[TextPart(text=json.dumps(result, indent=2)[:8000])]),
        ],
        metadata={"task_id": task_id},
    )


app = build_a2a_app(CARD, handle)

if __name__ == "__main__":
    run("reg_agents.agents.lifecycle_orchestrator:app", 8108)
