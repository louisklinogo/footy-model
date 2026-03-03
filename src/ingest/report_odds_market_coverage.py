from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.market_contract import get_linchpin_market_codes
from src.betting.odds_resolver import resolve_odds_for_market
from src.db.db_utils import connect_db


DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "reports" / "odds_coverage"


@dataclass(frozen=True)
class FixtureRow:
    fixture_id: int
    league_code: str
    match_datetime_utc: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report odds coverage by prediction market for upcoming fixtures."
    )
    parser.add_argument("--days", type=int, default=7, help="Upcoming horizon in days.")
    parser.add_argument("--league", type=str, default=None, help="Optional league filter.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional fixture cap for faster dry runs."
    )
    parser.add_argument(
        "--markets",
        type=str,
        default=None,
        help="Optional comma-separated prediction market codes (default: linchpin contract).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for JSON/Markdown reports.",
    )
    return parser.parse_args()


def _bucket_for_kickoff(kickoff: datetime, now_utc: datetime) -> str:
    delta_h = (kickoff - now_utc).total_seconds() / 3600.0
    if delta_h <= 24.0:
        return "0_24h"
    if delta_h <= 72.0:
        return "24_72h"
    return "72h_plus"


def _parse_market_list(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(get_linchpin_market_codes())
    out: list[str] = []
    for item in raw.split(","):
        code = item.strip()
        if code:
            out.append(code)
    return out


def fetch_fixtures(days: int, league: str | None, limit: int | None) -> list[FixtureRow]:
    query = """
    SELECT fixture_id, league_code, match_datetime_utc
    FROM fixtures
    WHERE status = 'scheduled'
      AND match_datetime_utc IS NOT NULL
      AND match_datetime_utc > NOW()
      AND match_datetime_utc <= NOW() + (%s || ' days')::interval
    """
    params: list[Any] = [days]
    if league:
        query += " AND league_code = %s"
        params.append(league)
    query += " ORDER BY match_datetime_utc ASC, fixture_id ASC"
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
    finally:
        conn.close()

    out: list[FixtureRow] = []
    for fixture_id, league_code, match_dt in rows:
        if not isinstance(match_dt, datetime):
            continue
        kickoff = (
            match_dt.replace(tzinfo=timezone.utc)
            if match_dt.tzinfo is None
            else match_dt.astimezone(timezone.utc)
        )
        out.append(
            FixtureRow(
                fixture_id=int(fixture_id),
                league_code=str(league_code or ""),
                match_datetime_utc=kickoff,
            )
        )
    return out


def fetch_odds_rows(fixture_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    if not fixture_ids:
        return {}
    query = """
    SELECT
        fixture_id,
        provider,
        snapshot_type,
        snapshot_time_utc,
        market_code,
        line_num,
        odds_json
    FROM fixture_odds_markets
    WHERE fixture_id = ANY(%s)
      AND snapshot_type IN ('latest_pre_match', 'closing')
    ORDER BY fixture_id ASC, snapshot_time_utc DESC
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (fixture_ids,))
            rows = cur.fetchall()
    finally:
        conn.close()

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fixture_id, provider, snap_type, snap_time, market_code, line_num, odds_json in rows:
        grouped[int(fixture_id)].append(
            {
                "provider": provider,
                "snapshot_type": snap_type,
                "snapshot_time_utc": snap_time,
                "market_code": market_code,
                "line_num": line_num,
                "odds_json": odds_json,
            }
        )
    return dict(grouped)


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(part / total)


def build_coverage_report(
    fixtures: list[FixtureRow],
    odds_by_fixture: dict[int, list[dict[str, Any]]],
    markets: list[str],
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    overall_total = len(fixtures)

    overall_counts = {m: {"with_odds": 0, "total": overall_total} for m in markets}
    league_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"with_odds": 0, "total": 0}
    )
    bucket_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"with_odds": 0, "total": 0}
    )

    for fixture in fixtures:
        odds_rows = odds_by_fixture.get(fixture.fixture_id, [])
        bucket = _bucket_for_kickoff(fixture.match_datetime_utc, now_utc)
        for market in markets:
            overall_counts[market]["total"] += 0
            league_key = (fixture.league_code, market)
            league_counts[league_key]["total"] += 1
            bucket_key = (bucket, market)
            bucket_counts[bucket_key]["total"] += 1

            resolution = resolve_odds_for_market(
                fixture.match_datetime_utc, odds_rows, market
            )
            available = resolution.odds_used is not None and resolution.odds_used > 1.0
            if available:
                overall_counts[market]["with_odds"] += 1
                league_counts[league_key]["with_odds"] += 1
                bucket_counts[bucket_key]["with_odds"] += 1

    coverage_by_market = []
    for market in markets:
        counts = overall_counts[market]
        coverage_by_market.append(
            {
                "market_code": market,
                "fixtures_with_odds": int(counts["with_odds"]),
                "fixtures_total": int(counts["total"]),
                "coverage_pct": _pct(int(counts["with_odds"]), int(counts["total"])),
            }
        )
    coverage_by_market.sort(key=lambda item: item["market_code"])

    coverage_by_league_market = []
    for (league_code, market), counts in league_counts.items():
        coverage_by_league_market.append(
            {
                "league_code": league_code,
                "market_code": market,
                "fixtures_with_odds": int(counts["with_odds"]),
                "fixtures_total": int(counts["total"]),
                "coverage_pct": _pct(int(counts["with_odds"]), int(counts["total"])),
            }
        )
    coverage_by_league_market.sort(
        key=lambda item: (item["league_code"], item["market_code"])
    )

    coverage_by_bucket_market = []
    for (bucket, market), counts in bucket_counts.items():
        coverage_by_bucket_market.append(
            {
                "kickoff_bucket": bucket,
                "market_code": market,
                "fixtures_with_odds": int(counts["with_odds"]),
                "fixtures_total": int(counts["total"]),
                "coverage_pct": _pct(int(counts["with_odds"]), int(counts["total"])),
            }
        )
    coverage_by_bucket_market.sort(
        key=lambda item: (item["kickoff_bucket"], item["market_code"])
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_count": overall_total,
        "markets": markets,
        "coverage_by_market": coverage_by_market,
        "coverage_by_league_market": coverage_by_league_market,
        "coverage_by_bucket_market": coverage_by_bucket_market,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Odds Coverage Report")
    lines.append("")
    lines.append(f"- generated_at_utc: {report['generated_at_utc']}")
    lines.append(f"- fixture_count: {report['fixture_count']}")
    lines.append("")
    lines.append("## Coverage by Market")
    lines.append("")
    lines.append("| Market | Fixtures w/ Odds | Fixtures Total | Coverage |")
    lines.append("|---|---:|---:|---:|")
    for row in report["coverage_by_market"]:
        pct = float(row["coverage_pct"]) * 100.0
        lines.append(
            f"| {row['market_code']} | {row['fixtures_with_odds']} | {row['fixtures_total']} | {pct:.1f}% |"
        )
    lines.append("")
    lines.append("## Coverage by Kickoff Bucket")
    lines.append("")
    lines.append("| Bucket | Market | Fixtures w/ Odds | Fixtures Total | Coverage |")
    lines.append("|---|---|---:|---:|---:|")
    for row in report["coverage_by_bucket_market"]:
        pct = float(row["coverage_pct"]) * 100.0
        lines.append(
            f"| {row['kickoff_bucket']} | {row['market_code']} | {row['fixtures_with_odds']} | {row['fixtures_total']} | {pct:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    markets = _parse_market_list(args.markets)
    fixtures = fetch_fixtures(days=args.days, league=args.league, limit=args.limit)
    fixture_ids = [item.fixture_id for item in fixtures]
    odds_by_fixture = fetch_odds_rows(fixture_ids)
    report = build_coverage_report(fixtures, odds_by_fixture, markets)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    json_path = out_dir / f"{stamp}.json"
    md_path = out_dir / f"{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Saved coverage report: {json_path}")
    print(f"Saved coverage report: {md_path}")


if __name__ == "__main__":
    main()
