from __future__ import annotations

from pathlib import Path

from src.jobs.tick_due_fixtures_v1 import (
    Options,
    build_hybrid_predict_command,
    build_postmatch_repair_commands,
    build_score_command,
)


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


def test_build_score_command_targets_hybrid_identity_for_hybrid_runtime() -> None:
    command = build_score_command(_options(), "E0")
    assert command[1].endswith("score_market_outcomes_fixtures_first.py")
    assert "--league" in command and "E0" in command
    assert "--model" in command and "market_outcome_v2" in command
    assert "--version" in command and "hybrid_v1" in command


def test_build_score_command_uses_legacy_defaults_for_legacy_runtime() -> None:
    command = build_score_command(_options(predict_runtime="legacy"), "E1")
    assert command[1].endswith("score_market_outcomes_fixtures_first.py")
    assert "--league" in command and "E1" in command
    assert "--model" not in command
    assert "--version" not in command


def test_build_postmatch_repair_commands_cover_stats_incidents_and_lead_states() -> None:
    fixture_ids_file = Path("artifacts/tmp/test_fixture_ids_runtime.csv")
    fixture_ids_file.parent.mkdir(parents=True, exist_ok=True)
    fixture_ids_file.write_text("fixture_id\n123\n456\n", encoding="utf-8")
    try:
        commands = build_postmatch_repair_commands(fixture_ids_file)
        assert len(commands) == 3
        assert commands[0][1].endswith("ingest_sofascore_stats.py")
        assert commands[1][1].endswith("ingest_sofascore_incidents.py")
        assert commands[2][1].endswith("build_incident_lead_state_features.py")
        for command in commands:
            assert "--fixture-ids-file" in command and str(fixture_ids_file) in command
    finally:
        if fixture_ids_file.exists():
            fixture_ids_file.unlink()
