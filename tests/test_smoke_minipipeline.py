# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnusedParameter=false

import csv
from pathlib import Path

from tests.conftest import (
    days_from_now,
    insert_fixture,
    insert_league,
    insert_odds_snapshot,
    insert_premium_stats,
    insert_team,
    run_script,
)


def test_smoke_minipipeline_snapshot_predict_export(db_case):
    with db_case.conn.cursor() as cur:
        insert_league(cur, db_case.league_code)
        home_team = insert_team(cur, db_case.league_code, f"SmokeHome {db_case.suffix}")
        away_team = insert_team(cur, db_case.league_code, f"SmokeAway {db_case.suffix}")

        historical = insert_fixture(
            cur,
            league_code=db_case.league_code,
            flashscore_id=f"tst_smoke_hist_{db_case.suffix}",
            home_team_id=home_team,
            away_team_id=away_team,
            kickoff=days_from_now(-1),
            status="ft",
        )
        upcoming_flashscore_id = f"tst_smoke_future_{db_case.suffix}"
        upcoming = insert_fixture(
            cur,
            league_code=db_case.league_code,
            flashscore_id=upcoming_flashscore_id,
            home_team_id=home_team,
            away_team_id=away_team,
            kickoff=days_from_now(1),
            status="scheduled",
        )

        insert_premium_stats(cur, historical, h_xg=1.25, a_xg=0.75)
        insert_odds_snapshot(cur, upcoming, snapshot_time=days_from_now(0))

    db_case.conn.commit()

    snapshots = run_script(["src/features/build_team_premium_snapshots_v1.py", "--league", db_case.league_code])
    assert snapshots.returncode == 0, snapshots.stderr or snapshots.stdout

    predict = run_script(
        [
            "src/modeling/predict_v3_fixtures_first.py",
            "--league",
            db_case.league_code,
            "--days",
            "7",
            "--limit",
            "1",
        ]
    )
    assert predict.returncode == 0, predict.stderr or predict.stdout

    out_path = Path("data/v1/daily") / f"predictions_v3_fixtures_first_{db_case.suffix}.csv"
    db_case.created_files.append(Path(__file__).resolve().parents[1] / out_path)

    export = run_script(
        [
            "models/export_predictions_v3_fixtures_first.py",
            "--league",
            db_case.league_code,
            "--days",
            "7",
            "--out",
            str(out_path),
        ]
    )
    assert export.returncode == 0, export.stderr or export.stdout

    absolute_out = Path(__file__).resolve().parents[1] / out_path
    assert absolute_out.exists()

    with absolute_out.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert any(row.get("flashscore_id") == upcoming_flashscore_id for row in rows)
