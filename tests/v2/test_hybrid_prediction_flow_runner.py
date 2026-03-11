from __future__ import annotations

import argparse

from src.modeling.v2.run_hybrid_prediction_flow import build_run_plan


def _args(**overrides: object) -> argparse.Namespace:
    payload = {
        "python_bin": "python",
        "scope": "model_v2/market_scope.yaml",
        "scoreline_dir": "model_artifacts/v2/scoreline",
        "anytime_dir": "model_artifacts/v2/anytime",
        "out_dir": "artifacts/v2/predictions/hybrid",
        "league": None,
        "days": 3,
        "limit": None,
        "base_model_name": "market_outcome_gbm",
        "base_model_version": "fixtures_first_prematch_v1",
        "model_name": "market_outcome_v2",
        "model_version": "hybrid_v1",
        "run_risk": False,
        "risk_history_days": 180,
        "export_out": None,
        "dry_run": False,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_build_run_plan_seeds_from_legacy_then_overlays_v2_families() -> None:
    plan = build_run_plan(_args())
    names = [name for name, _ in plan]
    assert names == ["predict.base_legacy", "predict.scoreline", "predict.anytime"]
    commands = dict(plan)
    assert commands["predict.base_legacy"][1].endswith("predict_market_outcomes_fixtures_first.py")
    for name in ["predict.scoreline", "predict.anytime"]:
        assert "--model-name" in commands[name] and "market_outcome_v2" in commands[name]
        assert "--model-version" in commands[name] and "hybrid_v1" in commands[name]
        assert "--write-db" in commands[name]


def test_build_run_plan_adds_risk_and_export_for_hybrid_identity() -> None:
    plan = build_run_plan(
        _args(
            league="E0",
            limit=25,
            run_risk=True,
            export_out="storage/reports/hybrid.csv",
        )
    )
    names = [name for name, _ in plan]
    assert names == [
        "predict.base_legacy",
        "predict.scoreline",
        "predict.anytime",
        "risk.assess",
        "export.csv",
    ]
    commands = dict(plan)
    assert "--limit" in commands["predict.base_legacy"] and "25" in commands["predict.base_legacy"]
    assert "--model" in commands["risk.assess"] and "market_outcome_v2" in commands["risk.assess"]
    assert any(arg.endswith("hybrid.csv") for arg in commands["export.csv"])