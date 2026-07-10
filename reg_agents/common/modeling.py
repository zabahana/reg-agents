"""Model development bake-off: train candidate models and select a champion.

Mirrors a bank's *first line of defense* under SR 11-7 / OCC 2011-12: several
candidate algorithms are trained and compared on held-out data, a champion is
selected against a documented primary metric, and the challengers are retained
for benchmarking (a form of "effective challenge"). The result feeds the
Developer / Validator / Audit agents.

Everything is deterministic (fixed seed) so the committed artifacts are stable.

On GPU this maps cleanly to RAPIDS cuML / XGBoost; the scikit-learn estimators
here are intentionally swappable (same fit/predict_proba contract).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
)

SEED = 42
PRIMARY_METRIC = "roc_auc"


@dataclass
class Task:
    """A supervised model-development task backed by a CSV dataset."""

    task_id: str
    name: str
    objective: str
    csv_path: str
    target: str
    numeric_features: List[str]
    bool_features: List[str] = field(default_factory=list)


TASKS: Dict[str, Task] = {
    "fraud": Task(
        task_id="fraud",
        name="Card Transaction Fraud Detection",
        objective=(
            "Predict whether a card transaction is fraudulent (binary "
            "classification) for real-time authorization decisioning."
        ),
        csv_path=os.path.join(_DATA_DIR, "transactions", "sample_transactions.csv"),
        target="label",
        numeric_features=["amount", "merchant_risk", "hour", "velocity_24h"],
        bool_features=["is_foreign"],
    ),
}


def _candidates(seed: int) -> Dict[str, Callable[[], object]]:
    """Candidate model factories considered during development.

    A rules-style baseline is included so the leaderboard shows the uplift of
    the ML challengers over a naive benchmark (documented in the dev report).
    """
    return {
        "rules_baseline": lambda: DummyClassifier(strategy="stratified", random_state=seed),
        "logistic_regression": lambda: Pipeline(
            [("scale", StandardScaler()),
             ("clf", LogisticRegression(max_iter=1000, random_state=seed))]
        ),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=250, max_depth=7, n_jobs=-1, random_state=seed
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=seed),
    }


CANDIDATE_DESCRIPTIONS = {
    "rules_baseline": "Naive stratified baseline (benchmark / effective-challenge floor).",
    "logistic_regression": "Scaled logistic regression — transparent, easily explainable.",
    "decision_tree": "Shallow CART (depth 4) — interpretable rule splits.",
    "random_forest": "Random forest (250 trees, depth 7) — strong tabular baseline.",
    "gradient_boosting": "Gradient-boosted trees — production analog of XGBoost/cuML.",
}


def list_candidate_models(task_id: str = "fraud") -> List[Dict[str, str]]:
    if task_id not in TASKS:
        raise KeyError(f"unknown task_id: {task_id}")
    return [{"model": k, "description": v} for k, v in CANDIDATE_DESCRIPTIONS.items()]


def _load_xy(task: Task):
    df = pd.read_csv(task.csv_path)
    x = pd.DataFrame(index=df.index)
    for col in task.numeric_features:
        x[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in task.bool_features:
        x[col] = (
            df[col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
        )
    y = pd.to_numeric(df[task.target], errors="coerce").fillna(0).astype(int)
    return x, y, list(x.columns)


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation (standard credit/fraud discrimination metric)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def run_bakeoff(task_id: str = "fraud", test_size: float = 0.3, seed: int = SEED) -> Dict:
    """Train every candidate, evaluate on a held-out split, and pick a champion.

    Returns a JSON-serializable dict with the dataset summary, the full
    leaderboard (sorted by the primary metric), the champion, and a rationale.
    """
    if task_id not in TASKS:
        raise KeyError(f"unknown task_id: {task_id}")
    task = TASKS[task_id]
    x, y, features = _load_xy(task)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    leaderboard: List[Dict] = []
    for name, factory in _candidates(seed).items():
        model = factory()
        model.fit(x_tr, y_tr)
        proba = model.predict_proba(x_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        leaderboard.append(
            {
                "model": name,
                "description": CANDIDATE_DESCRIPTIONS[name],
                "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
                "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
                "ks": round(_ks_statistic(y_te.to_numpy(), proba), 4),
                "f1": round(float(f1_score(y_te, preds, zero_division=0)), 4),
                "precision": round(float(precision_score(y_te, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_te, preds, zero_division=0)), 4),
                "brier": round(float(brier_score_loss(y_te, proba)), 4),
            }
        )

    leaderboard.sort(key=lambda r: (r[PRIMARY_METRIC], r["pr_auc"]), reverse=True)
    champion = leaderboard[0]
    runner_up = leaderboard[1] if len(leaderboard) > 1 else None

    rationale = (
        f"'{champion['model']}' selected as champion: highest {PRIMARY_METRIC} "
        f"({champion[PRIMARY_METRIC]}) with KS {champion['ks']} and PR-AUC "
        f"{champion['pr_auc']} on a stratified {int(test_size * 100)}% hold-out"
    )
    if runner_up:
        rationale += (
            f", ahead of the next-best '{runner_up['model']}' "
            f"({PRIMARY_METRIC} {runner_up[PRIMARY_METRIC]}). Challengers are "
            f"retained for ongoing benchmarking (effective challenge)."
        )

    return {
        "task": {"task_id": task.task_id, "name": task.name, "objective": task.objective},
        "dataset": {
            "path": os.path.relpath(task.csv_path, os.path.dirname(_DATA_DIR)),
            "n_rows": int(len(x)),
            "n_train": int(len(x_tr)),
            "n_test": int(len(x_te)),
            "positive_rate": round(float(y.mean()), 4),
            "features": features,
        },
        "primary_metric": PRIMARY_METRIC,
        "leaderboard": leaderboard,
        "champion": champion,
        "selection_rationale": rationale,
    }
