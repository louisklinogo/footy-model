"""
Ingest player availability (missing players and lineups) from SofaScore.
Supports both scheduled fixtures (pre-match) and historical fixtures (backfill).

Usage:
    python src/ingest/ingest_sofascore_availability.py --limit 50 --status scheduled
    python src/ingest/ingest_sofascore_availability.py --limit 200 --status ft --league E0
    python src/ingest/ingest_sofascore_availability.py --dry-run --limit 5
"""

import sys
import asyncio
import argparse
from pathlib import Path
import datetime
import json
from typing import List, Dict, Any

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportArgumentType=false

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from src.db.db_utils import connect_db
from sofascore_wrapper.api import SofascoreAPI

DEFAULT_COOLDOWN_FILE = ROOT_DIR / "artifacts" / "state" / "availability_404_cooldown.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest player availability from SofaScore")
    parser.add_argument("--league", type=str, default=None, help="Filter by league code")
    parser.add_argument("--limit", type=int, default=50, help="Max fixtures to process")
    parser.add_argument("--status", type=str, default="scheduled",
                        choices=["scheduled", "ft", "all"], help="Fixture status to process")
    parser.add_argument(
        "--scheduled-start-hours",
        type=int,
        default=0,
        help="When status=scheduled, include fixtures starting from NOW + this many hours (default: 0).",
    )
    parser.add_argument(
        "--scheduled-end-hours",
        type=int,
        default=72,
        help="When status=scheduled, include fixtures up to NOW + this many hours (default: 72).",
    )
    parser.add_argument(
        "--scheduled-order",
        type=str,
        default="asc",
        choices=["asc", "desc"],
        help="When status=scheduled, kickoff ordering direction (default: asc / nearest first).",
    )
    parser.add_argument(
        "--retry-404-cooldown-hours",
        type=int,
        default=6,
        help="Skip scheduled fixtures that returned lineup 404 within this cooldown window (default: 6h).",
    )
    parser.add_argument(
        "--cooldown-file",
        type=Path,
        default=DEFAULT_COOLDOWN_FILE,
        help="Local JSON file used to persist lineup-404 cooldown state.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but don't write to DB")
    return parser.parse_args()


def _unix_to_datetime(raw: float) -> datetime.datetime | None:
    try:
        ts = float(raw)
    except (TypeError, ValueError):
        return None
    # Treat large values as milliseconds.
    if ts > 1e11:
        ts = ts / 1000.0
    try:
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def parse_expected_return(raw: object) -> datetime.datetime | None:
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        return _unix_to_datetime(raw)

    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        if value.isdigit():
            return _unix_to_datetime(int(value))
        try:
            dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt

    return None


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse_iso_utc(raw: object) -> datetime.datetime | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def load_cooldown_state(path: Path) -> dict[str, str]:
    try:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, str):
                out[key] = value
        return out
    except Exception:
        return {}


def save_cooldown_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def prune_cooldown_state(
    state: dict[str, str],
    keep_hours: int,
) -> dict[str, str]:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    threshold = now_utc - datetime.timedelta(hours=max(1, keep_hours))
    out: dict[str, str] = {}
    for fixture_id, ts in state.items():
        parsed = _parse_iso_utc(ts)
        if parsed is not None and parsed >= threshold:
            out[fixture_id] = ts
    return out


def should_skip_fixture_by_cooldown(
    fixture_id: int,
    cooldown_state: dict[str, str],
    cooldown_hours: int,
) -> bool:
    last_ts = cooldown_state.get(str(fixture_id))
    if not last_ts:
        return False
    last_dt = _parse_iso_utc(last_ts)
    if last_dt is None:
        return False
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    return last_dt >= now_utc - datetime.timedelta(hours=max(0, cooldown_hours))


def fetch_target_fixtures(
    league_code: str | None,
    status: str,
    limit: int,
    *,
    scheduled_start_hours: int = 0,
    scheduled_end_hours: int = 72,
    scheduled_order: str = "asc",
) -> List[Dict[str, Any]]:
    """Fetch fixtures that are missing availability data."""
    order = "ASC" if str(scheduled_order).lower() == "asc" else "DESC"
    query = """
    SELECT
        f.fixture_id,
        f.sofascore_id,
        f.home_team_id,
        f.away_team_id,
        f.league_code,
        f.status,
        f.match_datetime_utc
    FROM fixtures f
    LEFT JOIN (
        SELECT fixture_id, COUNT(*) AS avail_count
        FROM player_availability
        GROUP BY fixture_id
    ) pa ON pa.fixture_id = f.fixture_id
    WHERE f.sofascore_id IS NOT NULL
      AND (pa.avail_count IS NULL OR pa.avail_count < 10)
    """
    params: list = []
    if status != "all":
        query += " AND f.status = %s"
        params.append(status)

    if league_code:
        query += " AND f.league_code = %s"
        params.append(league_code)

    if status == "scheduled":
        if scheduled_end_hours <= scheduled_start_hours:
            raise ValueError("--scheduled-end-hours must be greater than --scheduled-start-hours")
        query += " AND f.match_datetime_utc > NOW() + (%s || ' hours')::interval"
        params.append(int(scheduled_start_hours))
        query += " AND f.match_datetime_utc <= NOW() + (%s || ' hours')::interval"
        params.append(int(scheduled_end_hours))
        query += f" ORDER BY f.match_datetime_utc {order}, f.fixture_id ASC LIMIT %s"
    else:
        # For FT/all backfills we keep newest-first behavior.
        query += " ORDER BY f.match_datetime_utc DESC, f.fixture_id DESC LIMIT %s"
    params.append(limit)

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def _has_availability_lineage_columns(cur) -> bool:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'player_availability'
          AND column_name IN ('first_recorded_at', 'last_refreshed_at', 'event_recorded_at')
        """
    )
    row = cur.fetchone()
    return bool(row and int(row[0]) == 3)


def _extract_lineup_event_recorded_at(lineups_raw: dict[str, Any]) -> datetime.datetime | None:
    if not isinstance(lineups_raw, dict):
        return None

    candidates: list[object] = []
    root_keys = (
        "updatedAtTimestamp",
        "updateTimestamp",
        "generatedAtTimestamp",
        "lastUpdatedTimestamp",
        "timestamp",
        "updatedAt",
        "generatedAt",
    )
    for key in root_keys:
        candidates.append(lineups_raw.get(key))

    for side in ("home", "away"):
        side_payload = lineups_raw.get(side)
        if not isinstance(side_payload, dict):
            continue
        for key in root_keys:
            candidates.append(side_payload.get(key))

    parsed: list[datetime.datetime] = []
    for raw in candidates:
        dt = parse_expected_return(raw)
        if dt is not None:
            parsed.append(dt)

    if not parsed:
        return None
    return max(parsed)


def _to_utc_datetime(raw: object) -> datetime.datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime.datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=datetime.timezone.utc)
        return raw.astimezone(datetime.timezone.utc)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    return None


def _resolve_event_recorded_at(
    fixture: Dict[str, Any],
    event_recorded_at: datetime.datetime | None,
    *,
    anchor_minutes: int = 60,
) -> datetime.datetime:
    """
    Build a leakage-safe event timestamp.

    For historical fixtures (status='ft'), any missing/post-kickoff timestamp is clamped
    to kickoff minus a fixed anchor window (default 60 minutes).
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    kickoff_utc = _to_utc_datetime(fixture.get("match_datetime_utc"))
    status = str(fixture.get("status") or "").lower()

    if event_recorded_at is None:
        if status == "ft" and kickoff_utc is not None:
            return kickoff_utc - datetime.timedelta(minutes=anchor_minutes)
        return now_utc

    if event_recorded_at.tzinfo is None:
        event_recorded_at = event_recorded_at.replace(tzinfo=datetime.timezone.utc)
    else:
        event_recorded_at = event_recorded_at.astimezone(datetime.timezone.utc)

    if kickoff_utc is not None and event_recorded_at > kickoff_utc:
        return kickoff_utc - datetime.timedelta(minutes=anchor_minutes)

    return event_recorded_at


async def ingest_availability(api: SofascoreAPI, conn, fixture: Dict[str, Any], dry_run: bool) -> str:
    ss_id = fixture["sofascore_id"]
    f_id = fixture["fixture_id"]

    print(f"  Processing Fixture {f_id} (SS: {ss_id}, Status: {fixture['status']})...")

    try:
        lineups_raw = await api._get(f"/event/{ss_id}/lineups")
    except Exception as e:
        print(f"    Error fetching lineups for {ss_id}: {e}")
        if "404" in str(e):
            return "not_found"
        return "error"

    if not lineups_raw:
        print(f"    No lineup data for {ss_id}")
        return "no_data"

    home_raw = lineups_raw.get("home", {})
    away_raw = lineups_raw.get("away", {})
    event_recorded_at = _extract_lineup_event_recorded_at(lineups_raw)
    event_recorded_at = _resolve_event_recorded_at(fixture, event_recorded_at)

    if dry_run:
        h_missing = len(home_raw.get("missingPlayers", []))
        a_missing = len(away_raw.get("missingPlayers", []))
        h_starters = sum(1 for p in home_raw.get("players", []) if not p.get("substitute"))
        a_starters = sum(1 for p in away_raw.get("players", []) if not p.get("substitute"))
        print(f"    [DRY RUN] Home: {h_starters} starters, {h_missing} missing | "
              f"Away: {a_starters} starters, {a_missing} missing")
        return "success"

    with conn.cursor() as cur:
        has_lineage_cols = _has_availability_lineage_columns(cur)
        _upsert_availability_rows(
            cur,
            f_id,
            fixture["home_team_id"],
            home_raw,
            has_lineage_cols=has_lineage_cols,
            event_recorded_at=event_recorded_at,
        )
        _upsert_availability_rows(
            cur,
            f_id,
            fixture["away_team_id"],
            away_raw,
            has_lineage_cols=has_lineage_cols,
            event_recorded_at=event_recorded_at,
        )
    conn.commit()
    print(f"    OK - committed availability for fixture {f_id}")
    return "success"


def _upsert_availability_rows(
    cur,
    fixture_id: int,
    team_id: int,
    data: dict,
    *,
    has_lineage_cols: bool,
    event_recorded_at: datetime.datetime | None,
):
    """Upsert starter/bench/missing/doubtful rows into player_availability."""

    # --- 1. Lineup players (Starter / Bench) ---
    for p_entry in data.get("players", []):
        p_info = p_entry.get("player", {})
        if not p_info or not p_info.get("id"):
            continue

        status = "bench" if p_entry.get("substitute") else "starter"
        db_p_id = _get_or_create_player(cur, p_info)

        if has_lineage_cols:
            cur.execute(
                """
                INSERT INTO player_availability (
                    fixture_id, player_id, team_id, status, source,
                    recorded_at, first_recorded_at, last_refreshed_at, event_recorded_at
                )
                VALUES (%s, %s, %s, %s, 'sofascore_lineup', NOW(), NOW(), NOW(), %s)
                ON CONFLICT (fixture_id, player_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    last_refreshed_at = NOW(),
                    event_recorded_at = COALESCE(EXCLUDED.event_recorded_at, player_availability.event_recorded_at)
                """,
                (fixture_id, db_p_id, team_id, status, event_recorded_at),
            )
        else:
            cur.execute(
                """
                INSERT INTO player_availability (fixture_id, player_id, team_id, status, source)
                VALUES (%s, %s, %s, %s, 'sofascore_lineup')
                ON CONFLICT (fixture_id, player_id) DO UPDATE SET
                    status = EXCLUDED.status
                """,
                (fixture_id, db_p_id, team_id, status),
            )

    # --- 2. Missing players (Missing / Doubtful) ---
    for m_entry in data.get("missingPlayers", []):
        p_info = m_entry.get("player", {})
        if not p_info or not p_info.get("id"):
            continue

        player_class = m_entry.get("playerClass", "")
        status = "doubtful" if "doubt" in player_class.lower() else "missing"

        description = m_entry.get("description") or m_entry.get("type") or "unknown"
        reason_code = m_entry.get("reason")
        expected_return = parse_expected_return(m_entry.get("expectedEndDate"))

        db_p_id = _get_or_create_player(cur, p_info)

        if has_lineage_cols:
            cur.execute(
                """
                INSERT INTO player_availability (
                    fixture_id, player_id, team_id, status, reason,
                    reason_code, expected_return, source,
                    recorded_at, first_recorded_at, last_refreshed_at, event_recorded_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'sofascore_missing', NOW(), NOW(), NOW(), %s)
                ON CONFLICT (fixture_id, player_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    reason = EXCLUDED.reason,
                    reason_code = EXCLUDED.reason_code,
                    expected_return = EXCLUDED.expected_return,
                    last_refreshed_at = NOW(),
                    event_recorded_at = COALESCE(EXCLUDED.event_recorded_at, player_availability.event_recorded_at)
                """,
                (
                    fixture_id,
                    db_p_id,
                    team_id,
                    status,
                    description,
                    reason_code,
                    expected_return,
                    event_recorded_at,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO player_availability (
                    fixture_id, player_id, team_id, status, reason,
                    reason_code, expected_return, source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'sofascore_missing')
                ON CONFLICT (fixture_id, player_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    reason = EXCLUDED.reason,
                    reason_code = EXCLUDED.reason_code,
                    expected_return = EXCLUDED.expected_return
                """,
                (fixture_id, db_p_id, team_id, status, description, reason_code, expected_return),
            )


def _get_or_create_player(cur, p_info: dict) -> int:
    """Atomically upsert player metadata and return internal player_id."""
    ss_id = str(p_info["id"])

    dob = None
    if p_info.get("dateOfBirthTimestamp"):
        dt = _unix_to_datetime(p_info["dateOfBirthTimestamp"])
        if dt:
            dob = dt.date()

    mv = (p_info.get("proposedMarketValueRaw") or {}).get("value")

    cur.execute("""
        INSERT INTO players (
            sofascore_id, name, slug, short_name, position,
            user_count, market_value_euro, nationality_code,
            country_name, date_of_birth, height, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (sofascore_id) DO UPDATE SET
            name = EXCLUDED.name,
            user_count = EXCLUDED.user_count,
            market_value_euro = EXCLUDED.market_value_euro,
            updated_at = NOW()
        RETURNING player_id
    """, (
        ss_id,
        p_info.get("name"),
        p_info.get("slug"),
        p_info.get("shortName"),
        p_info.get("position"),
        p_info.get("userCount"),
        mv,
        (p_info.get("country") or {}).get("alpha2"),
        (p_info.get("country") or {}).get("name"),
        dob,
        p_info.get("height"),
    ))
    return cur.fetchone()[0]


async def main():
    args = parse_args()
    overfetch_limit = args.limit
    if args.status == "scheduled":
        overfetch_limit = max(args.limit, args.limit * 4)

    cooldown_state = load_cooldown_state(args.cooldown_file)
    cooldown_state = prune_cooldown_state(
        cooldown_state, keep_hours=max(args.retry_404_cooldown_hours * 6, 24)
    )

    try:
        api = SofascoreAPI()
        try:
            fixtures = fetch_target_fixtures(
                args.league,
                args.status,
                overfetch_limit,
                scheduled_start_hours=args.scheduled_start_hours,
                scheduled_end_hours=args.scheduled_end_hours,
                scheduled_order=args.scheduled_order,
            )

            if args.status == "scheduled":
                filtered: list[Dict[str, Any]] = []
                skipped_on_cooldown = 0
                for fixture in fixtures:
                    fixture_id = int(fixture.get("fixture_id"))
                    if should_skip_fixture_by_cooldown(
                        fixture_id, cooldown_state, args.retry_404_cooldown_hours
                    ):
                        skipped_on_cooldown += 1
                        continue
                    filtered.append(fixture)
                    if len(filtered) >= args.limit:
                        break
                fixtures = filtered
                print(
                    "Scheduled window "
                    f"[+{args.scheduled_start_hours}h, +{args.scheduled_end_hours}h], "
                    f"order={args.scheduled_order}, "
                    f"skipped_on_404_cooldown={skipped_on_cooldown}"
                )

            if not fixtures:
                print(f"No fixtures found needing availability (Status: {args.status}).")
                return

            print(f"Found {len(fixtures)} fixtures to process.")
            cooldown_changed = False
            for f in fixtures:
                fixture_id = int(f.get("fixture_id"))
                conn = connect_db()
                try:
                    result = await ingest_availability(api, conn, f, args.dry_run)
                finally:
                    conn.close()
                if result == "not_found":
                    cooldown_state[str(fixture_id)] = _iso_now()
                    cooldown_changed = True
                elif result == "success":
                    if str(fixture_id) in cooldown_state:
                        del cooldown_state[str(fixture_id)]
                        cooldown_changed = True
                await asyncio.sleep(1)
            if cooldown_changed:
                save_cooldown_state(args.cooldown_file, cooldown_state)
        finally:
            await api.close()
    except Exception as e:
        print(f"Main Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
