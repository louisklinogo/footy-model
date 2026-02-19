# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from psycopg2.extras import Json

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_utils import connect_db
from pipeline_logging import create_pipeline_run, finalize_pipeline_run


PREMIUM_DIR_CANDIDATES = [
    Path("data/v1/premium"),
]

TERMINAL_STATUSES = {"ft", "postponed", "cancelled", "abandoned"}


class FixtureInfo(TypedDict):
    fixture_id: int
    status: str | None
    match_datetime_utc: datetime | None


def parse_val(val_str):
    if val_str is None:
        return None
    if isinstance(val_str, (int, float)):
        return float(val_str)

    s = str(val_str).strip()
    if not s:
        return None

    if "%" in s:
        s = s.split("%", 1)[0]
    if "(" in s:
        s = s.split("(", 1)[0].strip()

    try:
        return float(s)
    except Exception:
        return None


def parse_int(val):
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def normalize_result_status(status_raw) -> str | None:
    if status_raw is None:
        return None
    status = str(status_raw).strip().lower()
    if not status:
        return None
    if status in TERMINAL_STATUSES:
        return status
    if "finished" in status or status in {"full time", "full-time"}:
        return "ft"
    if "postponed" in status:
        return "postponed"
    if "cancelled" in status or "canceled" in status:
        return "cancelled"
    if "abandoned" in status:
        return "abandoned"
    return None


def extract_terminal_result(entry: dict[str, object]) -> dict[str, object] | None:
    result_raw = entry.get("result")
    result = result_raw if isinstance(result_raw, dict) else None
    if not result:
        return None

    status = normalize_result_status(
        result.get("result_status") or result.get("fixture_status") or result.get("status")
    )
    if status not in TERMINAL_STATUSES:
        return None

    return {
        "result_status": status,
        "home_goals": parse_int(result.get("home_goals")),
        "away_goals": parse_int(result.get("away_goals")),
        "scraped_at_utc": result.get("scraped_at_utc"),
    }


def parse_settled_at_utc(raw_value) -> str | None:
    if raw_value is None:
        return None
    raw = str(raw_value).strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def resolve_premium_dir() -> Path:
    for p in PREMIUM_DIR_CANDIDATES:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        "No premium directory found. Expected: "
        + ", ".join(str(p) for p in PREMIUM_DIR_CANDIDATES)
    )


def file_mtime_utc_iso(p: Path) -> str:
    ts = p.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def get_fixture_info(cur, flashscore_id: str) -> FixtureInfo | None:
    cur.execute(
        "SELECT fixture_id, status, match_datetime_utc FROM fixtures WHERE flashscore_id=%s",
        (flashscore_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    fixture_id, status_raw, match_dt_raw = row
    status = status_raw if isinstance(status_raw, str) else None
    match_dt = match_dt_raw if isinstance(match_dt_raw, datetime) else None
    return {
        "fixture_id": int(fixture_id),
        "status": status,
        "match_datetime_utc": match_dt,
    }


def choose_snapshot_type(fixture_status: str | None, match_datetime_utc) -> str:
    if fixture_status == "ft":
        return "closing"
    if match_datetime_utc is not None and match_datetime_utc < datetime.now(timezone.utc):
        return "closing"
    return "latest_pre_match"


def upsert_fixture_stats_premium(cur, fixture_id: int, entry: dict[str, object]):
    stats_raw = entry.get("stats")
    stats = stats_raw if isinstance(stats_raw, dict) else {}
    h_raw = stats.get("home")
    h = h_raw if isinstance(h_raw, dict) else {}
    a_raw = stats.get("away")
    a = a_raw if isinstance(a_raw, dict) else {}

    def g(obj, key):
        return parse_val(obj.get(key))

    payload = {
        "h_xg": g(h, "Expected goals (xG)"),
        "a_xg": g(a, "Expected goals (xG)"),
        "h_xgot": g(h, "xG on target (xGOT)"),
        "a_xgot": g(a, "xG on target (xGOT)"),
        "h_xa": g(h, "Expected assists (xA)"),
        "a_xa": g(a, "Expected assists (xA)"),
        "h_big_chances": g(h, "Big chances"),
        "a_big_chances": g(a, "Big chances"),
        "h_possession": g(h, "Ball possession"),
        "a_possession": g(a, "Ball possession"),
        "h_box_touches": g(h, "Touches in opposition box"),
        "a_box_touches": g(a, "Touches in opposition box"),
        "h_crosses": g(h, "Crosses"),
        "a_crosses": g(a, "Crosses"),
        "h_blocked_shots": g(h, "Blocked shots"),
        "a_blocked_shots": g(a, "Blocked shots"),
        "h_through_passes": g(h, "Accurate through passes"),
        "a_through_passes": g(a, "Accurate through passes"),
        "h_sot": g(h, "Shots on target"),
        "a_sot": g(a, "Shots on target"),
        "h_shots_inside_box": g(h, "Shots inside the box"),
        "a_shots_inside_box": g(a, "Shots inside the box"),
        "h_corners": g(h, "Corner kicks"),
        "a_corners": g(a, "Corner kicks"),
        "h_goals_prevented": g(h, "Goals prevented"),
        "a_goals_prevented": g(a, "Goals prevented"),
        "h_tackles_pct": g(h, "Tackles"),
        "a_tackles_pct": g(a, "Tackles"),
        "h_interceptions": g(h, "Interceptions"),
        "a_interceptions": g(a, "Interceptions"),
        "h_errors_lead_to_shot": g(h, "Errors leading to shot"),
        "a_errors_lead_to_shot": g(a, "Errors leading to shot"),
    }

    fidelity_inputs = [payload["h_xg"], payload["a_xg"], payload["h_xgot"], payload["a_xgot"]]
    fidelity_score = sum(1 for x in fidelity_inputs if x is not None) / 4.0

    cols = [
        "fixture_id",
        "h_xg",
        "a_xg",
        "h_xgot",
        "a_xgot",
        "h_xa",
        "a_xa",
        "h_big_chances",
        "a_big_chances",
        "h_possession",
        "a_possession",
        "h_box_touches",
        "a_box_touches",
        "h_crosses",
        "a_crosses",
        "h_blocked_shots",
        "a_blocked_shots",
        "h_through_passes",
        "a_through_passes",
        "h_sot",
        "a_sot",
        "h_shots_inside_box",
        "a_shots_inside_box",
        "h_corners",
        "a_corners",
        "h_goals_prevented",
        "a_goals_prevented",
        "h_tackles_pct",
        "a_tackles_pct",
        "h_interceptions",
        "a_interceptions",
        "h_errors_lead_to_shot",
        "a_errors_lead_to_shot",
        "fidelity_score",
        "raw_json",
    ]

    params = {"fixture_id": fixture_id, **payload, "fidelity_score": fidelity_score, "raw_json": Json(entry)}
    placeholders = ", ".join(f"%({c})s" for c in cols)
    insert_cols = ", ".join(cols)
    update_cols = ", ".join(
        [f"{c} = EXCLUDED.{c}" for c in cols if c != "fixture_id"] + ["ingested_at = NOW()"]
    )

    cur.execute(
        f"""
        INSERT INTO fixture_stats_premium ({insert_cols}, ingested_at)
        VALUES ({placeholders}, NOW())
        ON CONFLICT (fixture_id) DO UPDATE SET {update_cols}
        """,
        params,
    )


def upsert_fixture_odds_snapshot(
    cur,
    fixture_id: int,
    entry: dict[str, object],
    snapshot_time_iso: str,
    snapshot_type: str,
):
    odds_raw = entry.get("odds")
    odds = odds_raw if isinstance(odds_raw, dict) else {}
    ou_raw = odds.get("ou")
    ou = ou_raw if isinstance(ou_raw, dict) else {}
    ah_raw = odds.get("ah")
    ah = ah_raw if isinstance(ah_raw, dict) else {}

    cur.execute(
        """
        INSERT INTO fixture_odds_snapshots (
            fixture_id,
            snapshot_time_utc,
            snapshot_type,
            ou_json,
            ah_json
        ) VALUES (%s, %s::timestamptz, %s, %s, %s)
        ON CONFLICT (fixture_id, snapshot_time_utc, snapshot_type) DO UPDATE SET
            ou_json = EXCLUDED.ou_json,
            ah_json = EXCLUDED.ah_json
        """,
        (fixture_id, snapshot_time_iso, snapshot_type, Json(ou), Json(ah)),
    )


def upsert_fixture_result(
    cur,
    fixture_id: int,
    terminal_result: dict[str, object],
    settled_at_utc_iso: str | None,
):
    cur.execute(
        """
        INSERT INTO fixture_results (
            fixture_id,
            home_goals,
            away_goals,
            result_status,
            result_source,
            settled_at
        ) VALUES (%s, %s, %s, %s, 'flashscore', COALESCE(%s::timestamptz, NOW()))
        ON CONFLICT (fixture_id) DO UPDATE SET
            home_goals = EXCLUDED.home_goals,
            away_goals = EXCLUDED.away_goals,
            result_status = EXCLUDED.result_status,
            result_source = EXCLUDED.result_source,
            settled_at = EXCLUDED.settled_at
        """,
        (
            fixture_id,
            terminal_result.get("home_goals"),
            terminal_result.get("away_goals"),
            terminal_result["result_status"],
            settled_at_utc_iso,
        ),
    )


def update_fixture_status_if_terminal(cur, fixture_id: int, terminal_status: str) -> int:
    cur.execute(
        """
        UPDATE fixtures
        SET status = %s,
            updated_at = NOW()
        WHERE fixture_id = %s
          AND status IS DISTINCT FROM %s
        """,
        (terminal_status, fixture_id, terminal_status),
    )
    return int(cur.rowcount or 0)


def ingest_league(league_code: str, limit: int | None):
    premium_root = resolve_premium_dir()
    league_dir = premium_root / league_code
    if not league_dir.exists():
        raise FileNotFoundError(f"Premium league directory not found: {league_dir}")

    files = sorted([p for p in league_dir.glob("*.json") if p.is_file()])
    if limit is not None:
        files = files[:limit]

    stats_upserts = 0
    odds_upserts = 0
    results_upserts = 0
    fixture_status_updates = 0
    skipped_missing_fixture = 0
    skipped_bad_json = 0

    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                for p in files:
                    try:
                        entry = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        skipped_bad_json += 1
                        continue

                    fs_id = str(entry.get("id") or p.stem).strip()
                    if not fs_id:
                        skipped_bad_json += 1
                        continue

                    fixture = get_fixture_info(cur, fs_id)
                    if not fixture:
                        skipped_missing_fixture += 1
                        continue

                    fixture_id = fixture["fixture_id"]
                    fixture_status = fixture.get("status")
                    if not isinstance(fixture_status, str):
                        fixture_status = None
                    match_datetime_utc = fixture.get("match_datetime_utc")
                    if not isinstance(match_datetime_utc, datetime):
                        match_datetime_utc = None
                    snapshot_type = choose_snapshot_type(fixture_status, match_datetime_utc)
                    snapshot_time_iso = file_mtime_utc_iso(p)

                    upsert_fixture_stats_premium(cur, fixture_id, entry)
                    stats_upserts += 1

                    upsert_fixture_odds_snapshot(cur, fixture_id, entry, snapshot_time_iso, snapshot_type)
                    odds_upserts += 1

                    terminal_result = extract_terminal_result(entry)
                    if terminal_result:
                        fixture_status_updates += update_fixture_status_if_terminal(
                            cur,
                            fixture_id,
                            str(terminal_result["result_status"]),
                        )
                        if (
                            str(terminal_result["result_status"]) == "ft"
                            and terminal_result.get("home_goals") is not None
                            and terminal_result.get("away_goals") is not None
                        ):
                            settled_at_utc_iso = parse_settled_at_utc(
                                terminal_result.get("scraped_at_utc")
                            )
                            upsert_fixture_result(
                                cur,
                                fixture_id,
                                terminal_result,
                                settled_at_utc_iso,
                            )
                            results_upserts += 1

    finally:
        conn.close()

    return {
        "league_code": league_code,
        "premium_root": str(premium_root),
        "files": len(files),
        "stats_upserts": stats_upserts,
        "odds_upserts": odds_upserts,
        "results_upserts": results_upserts,
        "fixture_status_updates": fixture_status_updates,
        "skipped_missing_fixture": skipped_missing_fixture,
        "skipped_bad_json": skipped_bad_json,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest premium JSON into fixtures-first tables")
    parser.add_argument("--league", required=True, help="League code, e.g. E0")
    parser.add_argument("--limit", type=int, default=None, help="Optional max JSON files to process")
    args = parser.parse_args()

    run_id = create_pipeline_run(
        "ingest_premium_fixtures_v1",
        details_json={"league_code": args.league, "limit": args.limit},
    )
    try:
        result = ingest_league(args.league, args.limit)
        finalize_pipeline_run(
            run_id,
            "success",
            message=(
                f"Premium ingest for {args.league}: stats={result['stats_upserts']}, "
                f"odds={result['odds_upserts']}, "
                f"results_upserts={result['results_upserts']}, "
                f"status_updates={result['fixture_status_updates']}"
            ),
            details_json=result,
        )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        finalize_pipeline_run(
            run_id,
            "fail",
            message=str(exc),
            details_json={"league_code": args.league, "limit": args.limit},
        )
        raise


if __name__ == "__main__":
    if not (os.getenv("DATABASE_URL") or os.getenv("DEV_DATABASE_URL")):
        raise SystemExit(
            "Missing DATABASE_URL (or DEV_DATABASE_URL). Set DATABASE_URL before running this script."
        )
    main()
