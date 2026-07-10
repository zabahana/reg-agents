"""Model bake-off tests (no LLM / no keys required)."""

from reg_agents.common import modeling


def test_bakeoff_returns_all_candidates():
    result = modeling.run_bakeoff("fraud")
    names = {r["model"] for r in result["leaderboard"]}
    assert names == set(modeling.CANDIDATE_DESCRIPTIONS)
    assert result["dataset"]["n_rows"] == 1000


def test_champion_is_top_of_leaderboard_by_primary_metric():
    result = modeling.run_bakeoff("fraud")
    board = result["leaderboard"]
    metric = result["primary_metric"]
    assert result["champion"] == board[0]
    assert board[0][metric] == max(r[metric] for r in board)


def test_ml_beats_naive_baseline():
    result = modeling.run_bakeoff("fraud")
    scores = {r["model"]: r["roc_auc"] for r in result["leaderboard"]}
    assert result["champion"]["roc_auc"] > scores["rules_baseline"]


def test_bakeoff_is_deterministic():
    a = modeling.run_bakeoff("fraud")
    b = modeling.run_bakeoff("fraud")
    assert a["champion"]["model"] == b["champion"]["model"]
    assert a["leaderboard"] == b["leaderboard"]


def test_unknown_task_raises():
    try:
        modeling.run_bakeoff("does-not-exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown task")
