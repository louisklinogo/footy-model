from __future__ import annotations

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

import json
from pathlib import Path

import importlib.util
import sys

import pytest


def _load_local_test_helpers() -> object:
    root_dir = Path(__file__).resolve().parents[1]
    conftest_path = root_dir / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_local_conftest", conftest_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load conftest at {conftest_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_helpers = _load_local_test_helpers()
days_from_now = _helpers.days_from_now
insert_fixture = _helpers.insert_fixture
insert_league = _helpers.insert_league
insert_team = _helpers.insert_team
run_script = _helpers.run_script


def test_calibration_check_smoke(db_case) -> None:
    league_code = f"TST_{db_case.suffix[:6]}"
    db_case.league_code = league_code

    model_name = "calib_test"
    model_version = "v1"
    market_code = "home_win"

    fixture_ids: list[int] = []
    try:
        with db_case.conn.cursor() as cur:
            insert_league(cur, league_code)
            home_team = insert_team(cur, league_code, f"Home {db_case.suffix}")
            away_team = insert_team(cur, league_code, f"Away {db_case.suffix}")

            for idx in range(100):
                fixture_id = insert_fixture(
                    cur,
                    league_code=league_code,
                    flashscore_id=f"tst_cal_{db_case.suffix}_{idx}",
                    home_team_id=home_team,
                    away_team_id=away_team,
                    kickoff=days_from_now(-(200 - idx)),
                    status="ft",
                )
                fixture_ids.append(fixture_id)

            for idx, fixture_id in enumerate(fixture_ids):
                p_model = 0.05 + (idx % 90) / 100.0
                actual = 1.0 if idx % 2 == 0 else 0.0
                cur.execute(
                    """
                    INSERT INTO predictions (fixture_id, market_code, model_name, model_version, p_model, metadata_json)
                    VALUES (%s, %s, %s, %s, %s, '{}'::jsonb)
                    RETURNING prediction_id
                    """,
                    (fixture_id, market_code, model_name, model_version, p_model),
                )
                prediction_id = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO prediction_scores (prediction_id, actual, brier, log_loss, hit)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (prediction_id, actual, 0.0, 0.0, bool(actual)),
                )

        db_case.conn.commit()

        output_dir = Path("tmp_calibration_out")
        result = run_script(
            [
                "src/modeling/evaluation/calibration_check.py",
                "--model",
                model_name,
                "--version",
                model_version,
                "--limit",
                "200",
                "--output-dir",
                str(output_dir),
            ]
        )
        assert result.returncode == 0, result.stderr

        results_path = output_dir / "calibration_results.json"
        assert results_path.exists()
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        assert market_code in payload
    finally:
        if output_dir.exists():
            for path in output_dir.glob("*"):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                output_dir.rmdir()
            except OSError:
                pass
        with db_case.conn.cursor() as cur:
            if fixture_ids:
                cur.execute(
                    "DELETE FROM prediction_scores WHERE prediction_id IN (SELECT prediction_id FROM predictions WHERE fixture_id = ANY(%s))",
                    (fixture_ids,),
                )
                cur.execute(
                    "DELETE FROM predictions WHERE fixture_id = ANY(%s)", (fixture_ids,)
                )
                cur.execute(
                    "DELETE FROM fixture_results WHERE fixture_id = ANY(%s)",
                    (fixture_ids,),
                )
                cur.execute("DELETE FROM fixtures WHERE fixture_id = ANY(%s)", (fixture_ids,))
        db_case.conn.commit()
