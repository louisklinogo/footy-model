from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


MODEL_NAME = "market_outcome_gbm"
MODEL_VERSION = "fixtures_first_prematch_v1"
MARKETS = [
    "o15",
    "u15",
    "o25",
    "u25",
    "c85",
    "btts",
    "1x2_h",
    "1x2_d",
    "1x2_a",
    "dc_1x",
    "dc_x2",
    "dc_12",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export model probability vs bookmaker odds (with implied prob + edge)."
    )
    parser.add_argument("--days", type=int, default=4, help="Future horizon in days.")
    parser.add_argument("--league", type=str, default=None, help="Optional league_code filter.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Output markdown summary path.")
    parser.add_argument("--out-csv-wide", type=Path, default=None, help="Output wide CSV path (one row per fixture).")
    parser.add_argument("--out-md-wide", type=Path, default=None, help="Output wide markdown summary path.")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v


def _latest_price(snapshot: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(snapshot, dict):
        return None
    prices = snapshot.get("prices_latest")
    if not isinstance(prices, dict):
        return None
    return _to_float(prices.get(key))


def _odds_for_market(market_code: str, odds_map: dict[str, dict[str, Any]]) -> tuple[float | None, str | None]:
    if market_code == "1x2_h":
        return _latest_price(odds_map.get("1x2"), "home"), "1x2.home"
    if market_code == "1x2_d":
        return _latest_price(odds_map.get("1x2"), "draw"), "1x2.draw"
    if market_code == "1x2_a":
        return _latest_price(odds_map.get("1x2"), "away"), "1x2.away"

    if market_code == "o15":
        return _latest_price(odds_map.get("ou_1_5"), "over"), "ou_1_5.over"
    if market_code == "u15":
        return _latest_price(odds_map.get("ou_1_5"), "under"), "ou_1_5.under"
    if market_code == "o25":
        return _latest_price(odds_map.get("ou_2_5"), "over"), "ou_2_5.over"
    if market_code == "u25":
        return _latest_price(odds_map.get("ou_2_5"), "under"), "ou_2_5.under"

    if market_code == "btts":
        return _latest_price(odds_map.get("btts"), "yes"), "btts.yes"
    if market_code == "dc_1x":
        return _latest_price(odds_map.get("dc"), "home_draw"), "dc.home_draw"
    if market_code == "dc_x2":
        return _latest_price(odds_map.get("dc"), "draw_away"), "dc.draw_away"
    if market_code == "dc_12":
        return _latest_price(odds_map.get("dc"), "home_away"), "dc.home_away"
    if market_code == "c85":
        return _latest_price(odds_map.get("corners_ou_8_5"), "over"), "corners_ou_8_5.over"

    return None, None


def fetch_fixtures(days: int, league: str | None) -> list[dict[str, Any]]:
    query = """
    SELECT
        f.fixture_id,
        f.match_datetime_utc,
        f.league_code,
        th.team_name AS home_team,
        ta.team_name AS away_team
    FROM fixtures f
    JOIN teams th ON th.team_id = f.home_team_id
    JOIN teams ta ON ta.team_id = f.away_team_id
    WHERE f.status = 'scheduled'
      AND f.match_datetime_utc >= NOW()
      AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
    """
    params: list[Any] = [days]
    if league:
        query += " AND f.league_code = %s"
        params.append(league)
    query += " ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC"

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_predictions(fixture_ids: list[int]) -> dict[tuple[int, str], dict[str, Any]]:
    if not fixture_ids:
        return {}
    query = """
    SELECT DISTINCT ON (p.fixture_id, p.market_code)
        p.fixture_id,
        p.market_code,
        p.p_model,
        p.created_at
    FROM predictions p
    WHERE p.model_name = %s
      AND p.model_version = %s
      AND p.market_code = ANY(%s)
      AND p.fixture_id = ANY(%s)
    ORDER BY p.fixture_id, p.market_code, p.created_at DESC
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (MODEL_NAME, MODEL_VERSION, MARKETS, fixture_ids))
            out: dict[tuple[int, str], dict[str, Any]] = {}
            for fixture_id, market_code, p_model, created_at in cur.fetchall():
                out[(int(fixture_id), str(market_code))] = {
                    "p_model": _to_float(p_model),
                    "pred_created_at": created_at,
                }
            return out
    finally:
        conn.close()


def fetch_odds_snapshots(fixture_ids: list[int]) -> dict[int, dict[str, dict[str, Any]]]:
    if not fixture_ids:
        return {}
    query = """
    SELECT DISTINCT ON (fom.fixture_id, fom.market_code, COALESCE(fom.line_num, -1))
        fom.fixture_id,
        fom.market_code,
        fom.line_num,
        fom.snapshot_type,
        fom.snapshot_time_utc,
        fom.odds_json
    FROM fixture_odds_markets fom
    JOIN fixtures f ON f.fixture_id = fom.fixture_id
    WHERE fom.provider = 'sofascore'
      AND fom.fixture_id = ANY(%s)
      AND fom.market_code IN ('1x2', 'ou', 'btts', 'dc', 'corners_ou')
      AND fom.snapshot_type IN ('latest_pre_match', 'closing')
      AND fom.snapshot_time_utc <= f.match_datetime_utc
    ORDER BY
      fom.fixture_id,
      fom.market_code,
      COALESCE(fom.line_num, -1),
      (fom.snapshot_type = 'latest_pre_match') DESC,
      fom.snapshot_time_utc DESC
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (fixture_ids,))
            out: dict[int, dict[str, dict[str, Any]]] = {}
            for fixture_id, market_code, line_num, snapshot_type, snapshot_time_utc, odds_json in cur.fetchall():
                fid = int(fixture_id)
                out.setdefault(fid, {})
                code = str(market_code)
                line_key = None
                if line_num is not None:
                    try:
                        ln = float(line_num)
                    except (TypeError, ValueError):
                        ln = None
                    if ln is not None:
                        if code == "ou" and abs(ln - 1.5) < 1e-9:
                            line_key = "ou_1_5"
                        elif code == "ou" and abs(ln - 2.5) < 1e-9:
                            line_key = "ou_2_5"
                        elif code == "corners_ou" and abs(ln - 8.5) < 1e-9:
                            line_key = "corners_ou_8_5"
                key = line_key if line_key else code
                out[fid][key] = {
                    "snapshot_type": snapshot_type,
                    "snapshot_time_utc": snapshot_time_utc,
                    "line_num": float(line_num) if line_num is not None else None,
                    "odds_json": odds_json if isinstance(odds_json, dict) else {},
                }
            return out
    finally:
        conn.close()


def build_rows(
    fixtures: list[dict[str, Any]],
    predictions: dict[tuple[int, str], dict[str, Any]],
    odds: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        fid = int(fixture["fixture_id"])
        odds_map = odds.get(fid, {})
        for market in MARKETS:
            pred = predictions.get((fid, market))
            p_model = pred["p_model"] if pred else None
            pred_created_at = pred["pred_created_at"] if pred else None
            odds_value, odds_field = _odds_for_market(market, {k: v.get("odds_json", {}) for k, v in odds_map.items()})
            implied = (1.0 / odds_value) if odds_value and odds_value > 1.0 else None
            edge = (p_model - implied) if (p_model is not None and implied is not None) else None

            # pick snapshot info from mapped source key
            snapshot_source_key = None
            if odds_field:
                snapshot_source_key = odds_field.split(".")[0]
            snapshot = odds_map.get(snapshot_source_key) if snapshot_source_key else None

            rows.append(
                {
                    "fixture_id": fid,
                    "match_datetime_utc": fixture["match_datetime_utc"],
                    "league_code": fixture["league_code"],
                    "home_team": fixture["home_team"],
                    "away_team": fixture["away_team"],
                    "market_code": market,
                    "p_model": p_model,
                    "book_odds": odds_value,
                    "implied_prob": implied,
                    "edge": edge,
                    "odds_field": odds_field,
                    "odds_snapshot_type": snapshot.get("snapshot_type") if snapshot else None,
                    "odds_snapshot_time_utc": snapshot.get("snapshot_time_utc") if snapshot else None,
                    "prediction_created_at": pred_created_at,
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], out_md: Path, out_csv: Path, days: int) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(rows)
    n_with_pred = sum(1 for r in rows if r.get("p_model") is not None)
    n_with_odds = sum(1 for r in rows if r.get("book_odds") is not None)
    n_with_edge = sum(1 for r in rows if r.get("edge") is not None)
    positive_edges = [float(r["edge"]) for r in rows if isinstance(r.get("edge"), (int, float)) and float(r["edge"]) > 0.0]

    lines = [
        "# Market Odds vs Model Export",
        "",
        f"- generated_at_utc: {datetime.now(UTC).isoformat()}",
        f"- horizon_days: {days}",
        f"- rows: {n_total}",
        f"- rows_with_model_prob: {n_with_pred}",
        f"- rows_with_book_odds: {n_with_odds}",
        f"- rows_with_edge: {n_with_edge}",
        f"- rows_with_positive_edge: {len(positive_edges)}",
        f"- csv: {out_csv}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_wide_rows(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in long_rows:
        fixture_id = int(row["fixture_id"])
        market = str(row["market_code"])
        item = grouped.get(fixture_id)
        if item is None:
            item = {
                "fixture_id": fixture_id,
                "match_datetime_utc": row.get("match_datetime_utc"),
                "league_code": row.get("league_code"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
            }
            grouped[fixture_id] = item
        item[f"p_model_{market}"] = row.get("p_model")
        item[f"book_odds_{market}"] = row.get("book_odds")
        item[f"implied_prob_{market}"] = row.get("implied_prob")
        item[f"edge_{market}"] = row.get("edge")

    # Ensure stable column presence for all configured markets.
    for item in grouped.values():
        for market in MARKETS:
            item.setdefault(f"p_model_{market}", None)
            item.setdefault(f"book_odds_{market}", None)
            item.setdefault(f"implied_prob_{market}", None)
            item.setdefault(f"edge_{market}", None)

    return sorted(grouped.values(), key=lambda r: (r.get("match_datetime_utc"), r["fixture_id"]))


def write_wide_md(rows: list[dict[str, Any]], out_md: Path, out_csv: Path, days: int) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(rows)
    n_with_any_edge = 0
    for row in rows:
        has_edge = any(row.get(f"edge_{m}") is not None for m in MARKETS)
        if has_edge:
            n_with_any_edge += 1
    lines = [
        "# Market Odds vs Model Export (Wide)",
        "",
        f"- generated_at_utc: {datetime.now(UTC).isoformat()}",
        f"- horizon_days: {days}",
        f"- fixtures: {n_total}",
        f"- fixtures_with_any_edge: {n_with_any_edge}",
        f"- csv: {out_csv}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base_dir = Path("artifacts/reports/predictions")
    out_csv = args.out_csv or (base_dir / f"market_odds_vs_model_{ts}.csv")
    out_md = args.out_md or (base_dir / f"market_odds_vs_model_{ts}.md")
    out_csv_wide = args.out_csv_wide or (base_dir / f"market_odds_vs_model_wide_{ts}.csv")
    out_md_wide = args.out_md_wide or (base_dir / f"market_odds_vs_model_wide_{ts}.md")

    fixtures = fetch_fixtures(days=args.days, league=args.league)
    fixture_ids = [int(f["fixture_id"]) for f in fixtures]
    predictions = fetch_predictions(fixture_ids)
    odds = fetch_odds_snapshots(fixture_ids)
    rows = build_rows(fixtures, predictions, odds)
    wide_rows = build_wide_rows(rows)

    write_csv(rows, out_csv)
    write_md(rows, out_md, out_csv, args.days)
    write_csv(wide_rows, out_csv_wide)
    write_wide_md(wide_rows, out_md_wide, out_csv_wide, args.days)

    print(f"Exported {len(rows)} rows")
    print(f"CSV: {out_csv}")
    print(f"MD : {out_md}")
    print(f"WIDE CSV: {out_csv_wide}")
    print(f"WIDE MD : {out_md_wide}")


if __name__ == "__main__":
    main()
