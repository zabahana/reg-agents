"""Tests for RLAIF preference-pair construction from the judge panel."""

from __future__ import annotations

import json

from reg_agents.common import dpo_preferences as P


def test_build_rlaif_pairs_consensus_and_disputed():
    panel = [
        # judges agree with each other AND with gate → consensus, not high-value
        {"narrative": "debt collector calls daily about a debt not mine",
         "nim": True, "openai": True, "gate": True, "weak": True,
         "nim_reason": "FDCPA", "openai_reason": "FDCPA",
         "nim_confidence": 0.9, "openai_confidence": 0.85},
        # judges agree, contradict gate → high-value disputed
        {"narrative": "branch moved my appointment twice",
         "nim": False, "openai": False, "gate": True, "weak": False,
         "nim_reason": "service only", "openai_reason": "",
         "nim_confidence": 0.7, "openai_confidence": 0.6},
        # judges disagree → skipped
        {"narrative": "unclear complaint text",
         "nim": True, "openai": False, "gate": True, "weak": True},
        # abstention → skipped
        {"narrative": "timeout row", "nim": None, "openai": True,
         "gate": True, "weak": True},
    ]
    pairs = P.build_rlaif_pairs(panel)
    assert len(pairs) == 2
    assert pairs[0]["source"] == "rlaif_consensus"
    assert pairs[0]["chosen_label"] is True
    assert pairs[0]["rejected_label"] is False
    assert pairs[0]["high_value"] is False
    assert json.loads(pairs[0]["chosen"])["is_regulatory"] is True

    assert pairs[1]["source"] == "rlaif_disputed"
    assert pairs[1]["high_value"] is True
    assert pairs[1]["chosen_label"] is False
    assert "NON-REGULATORY" in pairs[1]["prompt"] or "compliance judge" in pairs[1]["prompt"]


def test_weak_contrastive_and_summary():
    pairs = P.build_weak_contrastive(
        ["reg text", "service text"], [True, False], limit=2)
    assert len(pairs) == 2
    assert all(p["source"] == "weak_contrastive" for p in pairs)
    assert pairs[0]["chosen_label"] is True
    assert pairs[1]["chosen_label"] is False
    s = P.summarize_pairs(pairs)
    assert s["n_pairs"] == 2
    assert s["by_source"]["weak_contrastive"] == 2


def test_recover_disputed_from_summary(tmp_path):
    texts = [
        "I got an email saying that my prepaid card materials had been returned "
        "as undeliverable, and that I needed to update my physical address.",
        "unrelated other complaint about a mortgage payment",
    ]
    summary = {
        "disputed": {
            "n": 1,
            "rows": [{
                "gate": False, "nim": True, "openai": True, "weak": True,
                "judges_match_weak": True,
                "narrative": texts[0][:220],
            }],
        }
    }
    path = tmp_path / "judge_agreement.json"
    path.write_text(json.dumps(summary))
    recovered = P.recover_disputed_from_summary(str(path), texts)
    assert len(recovered) == 1
    assert recovered[0]["narrative"] == texts[0]
    pairs = P.build_rlaif_pairs(recovered)
    assert len(pairs) == 1
    assert pairs[0]["high_value"] is True
