from reg_agents.agents import report_agent
from reg_agents.common.a2a import Message

_MATERIAL = """MODEL ID: FRAUD-XGB-GNN-001

## Model Metadata (registry of record)
{
  "model_id": "FRAUD-XGB-GNN-001",
  "name": "Card Transaction Fraud Detector",
  "type": "GNN-enhanced XGBoost",
  "owner": "Payments Risk",
  "tier": "1 (high risk)",
  "use": "Real-time card transaction fraud scoring",
  "protected_class_features": "none (age/zip excluded)"
}

## Validation Findings
[LLM unavailable: NIM request timed out]

Model metadata:
{"model_id": "FRAUD-XGB-GNN-001"}

Relevant rules:
[
  {
    "source": "SR 11-7",
    "heading": "Model validation",
    "text": "Validation should be independent and ongoing."
  }
]

## Regulatory Context
[LLM unavailable: NIM request timed out]

Retrieved excerpts:
[
  {
    "source": "ECOA",
    "heading": "Equal credit opportunity",
    "text": "Credit decisions must not discriminate on a prohibited basis."
  }
]
"""


def test_deterministic_report_formats_artifacts_without_raw_json():
    report = report_agent.deterministic_report(_MATERIAL)

    for heading in (
        "## Executive Summary",
        "## Model Metadata",
        "## Validation Findings",
        "## Regulatory Basis / Citations",
        "## Risk Assessment",
        "## Recommendations / Conditions",
        "## Audit Trail",
    ):
        assert heading in report
    assert "Card Transaction Fraud Detector" in report
    assert "SR 11-7" in report
    assert "ECOA" in report
    assert "Validation should be independent and ongoing." in report
    assert "**Rating: Medium**" in report
    assert '"model_id":' not in report
    assert '"source":' not in report
    assert "[LLM unavailable:" not in report


def test_handle_uses_deterministic_report_when_llm_times_out(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("NIM request timed out")

    monkeypatch.setattr(report_agent.llm, "system_user", timeout)

    task = report_agent.handle(Message.text(_MATERIAL), {})
    report = task.artifacts[0].as_text()

    assert report.startswith("# Governance Report")
    assert "## Audit Trail" in report
    assert '"model_id":' not in report


def test_handle_preserves_llm_enhanced_report(monkeypatch):
    monkeypatch.setattr(
        report_agent.llm,
        "system_user",
        lambda *args, **kwargs: "# Enhanced report\n\nLLM-authored synthesis.",
    )

    task = report_agent.handle(Message.text(_MATERIAL), {})

    assert task.artifacts[0].as_text() == "# Enhanced report\n\nLLM-authored synthesis."
