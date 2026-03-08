from __future__ import annotations

import argparse

from src.modeling.v2.run_prediction_flow import build_run_plan


def _args(**overrides: object) -> argparse.Namespace:
    payload = {
        "python_bin": "python",
        "scope": "model_v2/market_scope.yaml",
        "scoreline_dir": "model_artifacts/v2/scoreline",
        "corners_dir": "model_artifacts/v2/corners",
        "anytime_dir": "model_artifacts/v2/anytime",
        "out_dir": "artifacts/v2/predictions",
        "league": None,
        "days": 3,
        "limit": None,
        "model_name": "market_outcome_v2",
        "model_version": "challenger_v1",
        "write_db": False,
        "run_risk": False,
        "export_out": None,
        "risk_history_days": 180,
        "dry_run": False,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_build_run_plan_bundles_family_predictions_under_shared_identity() -> None:
    plan = build_run_plan(_args())
    names = [name for name, _ in plan]
    assert names == ["predict.scoreline", "predict.corners", "predict.anytime"]
    for _, command in plan:
        assert "--model-name" in command and "market_outcome_v2" in command
        assert "--model-version" in command and "challenger_v1" in command
        assert "--write-db" not in command


def test_build_run_plan_adds_db_risk_and_export_steps_when_requested() -> None:
    plan = build_run_plan(
        _args(
            league="EPL",
            limit=25,
            write_db=True,
            run_risk=True,
            export_out="storage/reports/challenger.csv",
        )
    )
    names = [name for name, _ in plan]
    assert names == [
        "predict.scoreline",
        "predict.corners",
        "predict.anytime",
        "risk.assess",
        "export.csv",
    ]
    commands = dict(plan)
    for name in ["predict.scoreline", "predict.corners", "predict.anytime"]:
        assert "--write-db" in commands[name]
        assert "EPL" in commands[name]
        assert "25" in commands[name]
    assert "--model" in commands["risk.assess"] and "market_outcome_v2" in commands["risk.assess"]
    assert "--version" in commands["risk.assess"] and "challenger_v1" in commands["risk.assess"]
    assert any(arg.endswith("challenger.csv") for arg in commands["export.csv"])