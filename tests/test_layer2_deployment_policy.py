from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from src.modeling.layer2_situational.deployment_policy import (
    build_layer2_deployment_policy,
    load_layer2_deployment_policy,
    resolve_league_policy,
    tune_alpha,
)


def _workspace_tmp_dir(prefix: str) -> Path:
    base = Path(".pytest_tmp")
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"{prefix}{uuid4().hex}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def test_tune_alpha_prefers_lower_alpha_on_tie() -> None:
    y_true = pd.Series([0.0, 0.0, 0.0]).to_numpy()
    y_pred = pd.Series([0.0, 0.0, 0.0]).to_numpy()

    alpha, metrics = tune_alpha(y_true, y_true, y_pred, y_pred, alpha_grid=(0.0, 0.5, 1.0))

    assert alpha == 0.0
    assert metrics["home_rmse"] == 0.0
    assert metrics["away_rmse"] == 0.0


def test_build_policy_enables_league_when_gates_pass() -> None:
    n = 60
    holdout = pd.DataFrame(
        {
            "league_code": ["E0"] * n,
            "y_home_true": [0.2] * n,
            "y_away_true": [-0.1] * n,
            "y_home_pred": [0.2] * n,
            "y_away_pred": [-0.1] * n,
        }
    )
    cv_rows = [
        {"league_code": "E0", "mean_lift": 0.01, "test_n": 20},
        {"league_code": "E0", "mean_lift": 0.02, "test_n": 20},
        {"league_code": "E0", "mean_lift": 0.03, "test_n": 20},
    ]

    policy = build_layer2_deployment_policy(holdout, cv_rows, model_version="v2")
    league = policy["leagues"]["E0"]

    assert league["enabled"] is True
    assert league["reason"] == "enabled"
    assert 0.0 < league["alpha"] <= 0.5
    assert league["gate"]["cv_positive_folds"] >= 2
    assert league["gate"]["cv_total_folds"] >= 3
    assert league["gate"]["recent_n"] == 60


def test_build_policy_disables_league_when_gates_fail() -> None:
    holdout = pd.DataFrame(
        {
            "league_code": ["E1"] * 10,
            "y_home_true": [0.2] * 10,
            "y_away_true": [-0.1] * 10,
            "y_home_pred": [0.0] * 10,
            "y_away_pred": [0.0] * 10,
        }
    )
    cv_rows = [{"league_code": "E1", "mean_lift": -0.02, "test_n": 10}]

    policy = build_layer2_deployment_policy(holdout, cv_rows, model_version="v2")
    league = policy["leagues"]["E1"]

    assert league["enabled"] is False
    assert league["alpha"] == 0.0
    assert "insufficient_recent_sample" in league["gate"]["fail_reasons"]
    assert "insufficient_cv_folds" in league["gate"]["fail_reasons"]


def test_load_policy_missing_uses_safe_default() -> None:
    tmp_dir = _workspace_tmp_dir("policy_missing_")
    try:
        policy = load_layer2_deployment_policy(tmp_dir)
        assert policy["defaults"]["enabled"] is False
        assert policy["defaults"]["alpha"] == 0.0
        assert policy["leagues"] == {}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_resolve_league_policy_prefers_league_then_default() -> None:
    tmp_dir = _workspace_tmp_dir("policy_resolve_")
    payload = {
        "defaults": {"enabled": False, "alpha": 0.0, "reason": "default_disabled"},
        "leagues": {"E0": {"enabled": True, "alpha": 0.5, "reason": "enabled"}},
    }
    try:
        (tmp_dir / "layer2_deployment_policy.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        policy = load_layer2_deployment_policy(tmp_dir)

        e0 = resolve_league_policy(policy, "E0")
        e1 = resolve_league_policy(policy, "E1")

        assert e0["enabled"] is True
        assert e0["alpha"] == 0.5
        assert e1["enabled"] is False
        assert e1["alpha"] == 0.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
