"""Tests for the complaint → regulation classification model."""

from __future__ import annotations

import os

import pytest

from reg_agents.common import complaints as C

pytestmark = pytest.mark.skipif(
    not os.path.exists(C._DATA_CSV), reason="CFPB dataset not fetched"
)


def test_taxonomy_has_24_categories():
    assert len(C.REGULATIONS) == 24
    assert C.NON_REGULATORY in C.REGULATIONS
    # every category has a name, description and at least one keyword
    for reg in C.REGULATIONS.values():
        assert reg.name and reg.description and reg.keywords


def test_few_shots_reference_valid_labels():
    for _text, label in C.FEW_SHOTS:
        assert label in C.REGULATIONS


def test_family_map_covers_taxonomy():
    assert set(C.FAMILY) == set(C.REGULATIONS)


def test_weak_labeling_on_real_data():
    df = C.load_complaints()
    assert len(df) >= 1000
    assert set(df["label"]).issubset(set(C.REGULATIONS))
    # both classes present and mostly regulatory (CFPB skews regulatory)
    assert 0 < df["is_regulatory"].mean() < 1
    # keyword overrides fire
    assert C.weak_label("Managing an account",
                        "I was discriminated against because of my race") \
        == "ECOA_DISCRIMINATION"
    assert C.weak_label("Managing an account",
                        "someone made unauthorized charges with my stolen card") \
        == "REG_E_UNAUTHORIZED"


def test_stage1_beats_chance_and_classifies():
    s1 = C.train_stage1()
    champ = s1["leaderboard"][0]
    assert champ["roc_auc"] > 0.7
    assert champ["pr_auc"] > 0.9
    # champion is selected on validation PR-AUC over an 80/10/10 split
    # (after the 5% scoring holdout is reserved)
    assert champ["val_pr_auc"] > 0.9
    ds = s1["dataset"]
    assert ds["n_train"] + ds["n_val"] + ds["n_test"] + ds["n_holdout"] \
        == ds["n_rows"]
    assert ds["n_val"] == ds["n_test"]
    assert ds["n_holdout"] == round(C.HOLDOUT_FRAC * ds["n_rows"])
    # decision cut-off is optimized on validation, never assumed to be 0.5
    assert 0.0 < s1["threshold"] < 1.0
    assert champ["threshold"] == s1["threshold"]
    # regularization is validation-tuned and the generalization gap is
    # tracked in every leaderboard row (train memorization guard)
    assert "train_roc_auc" in champ and "train_test_gap" in champ
    if champ["model"] == "logistic_regression":
        assert champ["params"].startswith(("l1", "l2"))
        assert champ["train_test_gap"] < 0.2
    # Realistic-length narrative (training data is >= 120 chars; very short
    # texts carry few TF-IDF features and sit near the no-information zone).
    out = C.classify_binary(
        "A debt collector calls me ten times a day about a debt that is not "
        "mine. I sent a written dispute and asked for validation of the debt, "
        "but they continue calling my workplace and threatened to garnish my "
        "wages and report the account to the credit bureaus."
    )
    assert out["is_regulatory"] is True
    assert 0.0 <= out["probability"] <= 1.0


def test_keyword_fallback_returns_valid_label():
    label, conf = C.keyword_classify(
        "The collector threatened to sue me and garnish my wages."
    )
    assert label in C.REGULATIONS
    assert label.startswith("FDCPA")
    assert 0.0 <= conf <= 1.0


def test_full_pipeline_without_llm():
    res = C.classify_complaint(
        "There is a hard inquiry on my credit report I never authorized and "
        "the bureau refuses to remove it.",
        use_llm=False,
    )
    assert res["stage1"]["is_regulatory"] is True
    assert res["stage2"]["label"] in C.REGULATIONS
    assert res["stage2"]["citation"] is not None


def test_scoring_holdout_reserved_before_split():
    df = C.load_complaints()
    model_df, holdout_df = C.split_scoring_holdout(df)
    # 5% of the full dataset, stratified, and disjoint from the modeling pool
    assert len(holdout_df) == round(C.HOLDOUT_FRAC * len(df))
    assert len(model_df) + len(holdout_df) == len(df)
    assert set(model_df.index).isdisjoint(set(holdout_df.index))
    assert abs(holdout_df["is_regulatory"].mean()
               - df["is_regulatory"].mean()) < 0.02
    # the 80/10/10 split never touches holdout rows
    x_tr, x_va, x_te, *_ = C.split_stage1(df)
    split_idx = set(x_tr.index) | set(x_va.index) | set(x_te.index)
    assert split_idx.isdisjoint(set(holdout_df.index))
    assert len(split_idx) == len(model_df)


def test_score_batch_output_schema():
    batch = C.scoring_holdout().head(3)
    scored = C.score_batch(batch, use_llm=False)
    assert len(scored) == 3
    for col in ("complaint_id", "complaint", "score", "is_regulatory",
                "label", "llm_reasoning"):
        assert col in scored.columns
    assert scored["score"].between(0, 1).all()
    assert scored["label"].isin(C.REGULATIONS).all()


def test_score_batch_accepts_alias_text_column():
    import pandas as pd

    batch = pd.DataFrame({
        "complaint": ["A debt collector keeps calling me about a debt "
                      "that is not mine and threatens to sue me."],
    })
    scored = C.score_batch(batch, use_llm=False)
    assert scored.iloc[0]["complaint_id"] == 1
    assert scored.iloc[0]["label"] in C.REGULATIONS
