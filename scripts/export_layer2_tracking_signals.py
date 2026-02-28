from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Layer 2 tracking signals for upcoming scheduled fixtures."
    )
    parser.add_argument("--days", type=int, default=4, help="Future horizon in days.")
    parser.add_argument("--league", type=str, default=None, help="Optional league filter.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional output markdown summary path.",
    )
    return parser.parse_args()


def fetch_rows(days: int, league: str | None) -> list[dict[str, object]]:
    query = """
    WITH latest AS (
        SELECT DISTINCT ON (p.fixture_id, p.market_code)
            p.fixture_id,
            p.market_code,
            p.metadata_json,
            p.created_at
        FROM predictions p
        WHERE p.model_name = 'situational_xgb'
          AND p.model_version = 'v2'
          AND p.market_code IN ('adj_lambda_home', 'adj_lambda_away')
        ORDER BY p.fixture_id, p.market_code, p.created_at DESC
    )
    SELECT
        f.fixture_id,
        f.match_datetime_utc,
        f.league_code,
        th.team_name AS home_team,
        ta.team_name AS away_team,
        ph.created_at AS home_pred_created_at,
        pa.created_at AS away_pred_created_at,
        ph.metadata_json AS home_meta,
        pa.metadata_json AS away_meta
    FROM fixtures f
    JOIN teams th ON th.team_id = f.home_team_id
    JOIN teams ta ON ta.team_id = f.away_team_id
    LEFT JOIN latest ph ON ph.fixture_id = f.fixture_id AND ph.market_code = 'adj_lambda_home'
    LEFT JOIN latest pa ON pa.fixture_id = f.fixture_id AND pa.market_code = 'adj_lambda_away'
    WHERE f.status = 'scheduled'
      AND f.match_datetime_utc >= NOW()
      AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
    """
    params: list[object] = [days]
    if league:
        query += " AND f.league_code = %s"
        params.append(league)
    query += " ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC"

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            cols = [d[0] for d in cur.description]
            out = []
            for row in cur.fetchall():
                out.append(dict(zip(cols, row)))
            return out
    finally:
        conn.close()


def _meta_value(meta: object, path: list[str], default: object = None) -> object:
    if not isinstance(meta, dict):
        return default
    cur: object = meta
    for part in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def flatten(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        home_meta = row.get("home_meta")
        away_meta = row.get("away_meta")

        flat = {
            "fixture_id": row.get("fixture_id"),
            "match_datetime_utc": row.get("match_datetime_utc"),
            "league_code": row.get("league_code"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "home_pred_created_at": row.get("home_pred_created_at"),
            "away_pred_created_at": row.get("away_pred_created_at"),
            "home_rule_layer_applied": _meta_value(home_meta, ["rule_layer_applied"], False),
            "away_rule_layer_applied": _meta_value(away_meta, ["rule_layer_applied"], False),
            "home_rule_layer_rules": _meta_value(home_meta, ["rule_layer_rules"], []),
            "away_rule_layer_rules": _meta_value(away_meta, ["rule_layer_rules"], []),
            "home_key_absent": _meta_value(home_meta, ["tracking_signals", "home_key_absent"]),
            "away_key_absent": _meta_value(away_meta, ["tracking_signals", "away_key_absent"]),
            "home_upcoming_tier": _meta_value(home_meta, ["tracking_signals", "home_upcoming_tier"]),
            "away_upcoming_tier": _meta_value(away_meta, ["tracking_signals", "away_upcoming_tier"]),
            "points_gap": _meta_value(home_meta, ["tracking_signals", "points_gap"]),
            "position_gap": _meta_value(home_meta, ["tracking_signals", "position_gap"]),
            "rest_delta": _meta_value(home_meta, ["tracking_signals", "rest_delta"]),
            "home_congestion_games_14d": _meta_value(home_meta, ["tracking_signals", "home_congestion_games_14d"]),
            "away_congestion_games_14d": _meta_value(away_meta, ["tracking_signals", "away_congestion_games_14d"]),
            "home_playing_top4": _meta_value(home_meta, ["tracking_signals", "home_playing_top4"]),
            "away_playing_top4": _meta_value(away_meta, ["tracking_signals", "away_playing_top4"]),
            "injury_impact": _meta_value(home_meta, ["tracking_signals", "injury_impact"]),
            "home_xg_lost": _meta_value(home_meta, ["tracking_signals", "home_xg_lost"]),
            "away_xg_lost": _meta_value(away_meta, ["tracking_signals", "away_xg_lost"]),
            "odds_model_gap_home": _meta_value(home_meta, ["tracking_signals", "odds_model_gap_home"]),
            "odds_model_gap_away": _meta_value(away_meta, ["tracking_signals", "odds_model_gap_away"]),
        }
        out.append(flat)
    return out


def write_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, object]], out_md: Path, out_csv: Path, days: int) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    home_key_absent = sum(1 for r in rows if int(r.get("home_key_absent") or 0) == 1)
    away_key_absent = sum(1 for r in rows if int(r.get("away_key_absent") or 0) == 1)
    home_big = sum(1 for r in rows if int(r.get("home_upcoming_tier") or 0) >= 2)
    away_big = sum(1 for r in rows if int(r.get("away_upcoming_tier") or 0) >= 2)
    home_rule = sum(1 for r in rows if bool(r.get("home_rule_layer_applied")))
    away_rule = sum(1 for r in rows if bool(r.get("away_rule_layer_applied")))

    lines = [
        "# Layer2 Tracking Signals Export",
        "",
        f"- generated_at_utc: {datetime.now(UTC).isoformat()}",
        f"- horizon_days: {days}",
        f"- fixtures: {total}",
        f"- csv: {out_csv}",
        "",
        "## Signal Counts",
        f"- home_key_absent=1: {home_key_absent}",
        f"- away_key_absent=1: {away_key_absent}",
        f"- home_upcoming_tier>=2: {home_big}",
        f"- away_upcoming_tier>=2: {away_big}",
        f"- home_rule_layer_applied=true: {home_rule}",
        f"- away_rule_layer_applied=true: {away_rule}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base_dir = Path("artifacts/reports/predictions")
    out_csv = args.out_csv or (base_dir / f"layer2_tracking_signals_{ts}.csv")
    out_md = args.out_md or (base_dir / f"layer2_tracking_signals_{ts}.md")

    rows = fetch_rows(days=args.days, league=args.league)
    flat = flatten(rows)
    write_csv(flat, out_csv)
    write_md(flat, out_md, out_csv, args.days)

    print(f"Exported {len(flat)} fixtures")
    print(f"CSV: {out_csv}")
    print(f"MD : {out_md}")


if __name__ == "__main__":
    main()
