# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnusedParameter=false

from scrapers.ingest_discovered_fixtures_v1 import upsert_fixture, upsert_league, upsert_team
from tests.conftest import days_from_now


def test_upsert_fixture_is_idempotent(db_case):
    flashscore_id = f"tst_fixture_{db_case.suffix}"
    kickoff = days_from_now(1).isoformat()

    with db_case.conn.cursor() as cur:
        upsert_league(cur, db_case.league_code, "Test League")
        home_team_id = upsert_team(cur, db_case.league_code, f"Home {db_case.suffix}")
        away_team_id = upsert_team(cur, db_case.league_code, f"Away {db_case.suffix}")

        upsert_fixture(
            cur,
            flashscore_id=flashscore_id,
            league_code=db_case.league_code,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            match_datetime_utc=kickoff,
            status="scheduled",
        )
        upsert_fixture(
            cur,
            flashscore_id=flashscore_id,
            league_code=db_case.league_code,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            match_datetime_utc=kickoff,
            status="scheduled",
        )

        cur.execute("SELECT COUNT(*) FROM fixtures WHERE flashscore_id = %s", (flashscore_id,))
        fixture_count = int(cur.fetchone()[0])

    db_case.conn.commit()
    assert fixture_count == 1
