from __future__ import annotations

from src.jobs.tick_due_fixtures_v1 import Options, build_hybrid_predict_command


def _options(**overrides: object) -> Options:
    payload = {
        "leagues_raw": None,
        "predict_days": 3,
        "settlement_delay_minutes": 180,
        "max_settle": 25,
        "max_predict": 50,
        "max_score": 500,
        "score_since_days": 30,
        "skip_settle": False,
        "skip_predict": False,
        "skip_score": False,
        "log_file": None,
        "predict_runtime": "hybrid_v2",
        "hybrid_model_name": "market_outcome_v2",
        "hybrid_model_version": "hybrid_v1",
        "hybrid_export_out": "storage/reports/market_predictions_hybrid_v1.csv",
        "dry_run": False,
    }
    payload.update(overrides)
    return Options(**payload)


def test_build_hybrid_predict_command_targets_hybrid_identity() -> None:
    command = build_hybrid_predict_command(_options(), "E0")
    assert command[1].endswith("run_hybrid_prediction_flow.py")
    assert "--league" in command and "E0" in command
    assert "--model-name" in command and "market_outcome_v2" in command
    assert "--model-version" in command and "hybrid_v1" in command
    assert "--run-risk" in command
    assert any(arg.endswith("market_predictions_hybrid_v1.csv") for arg in command)


def test_build_hybrid_predict_command_forwards_dry_run() -> None:
    command = build_hybrid_predict_command(_options(dry_run=True), "E1")
    assert "--dry-run" in command