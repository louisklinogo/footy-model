from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.betting.odds_resolver import _MARKET_MAP, _SIDE_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLV Proxy (Opening vs Latest Pre-match)."
    )
    parser.add_argument(
        "--since-days", type=int, default=365, help="Lookback horizon in days."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/reports/backtests/clv_proxy.md"),
        help="Output markdown path.",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        v = float(value)
        if v <= 1.0:
            return None
        return v
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _line_matches(provider_line: Any, target_line: float | None) -> bool:
    if target_line is None:
        return True
    provider_line_num = _safe_float(provider_line)
    if provider_line_num is None:
        return False
    return abs(provider_line_num - target_line) <= 0.01


def _extract_price(odds_json: Any, side: str, price_key: str) -> float | None:
    payload = _as_dict(odds_json)
    prices = payload.get(price_key)
    if not isinstance(prices, dict):
        return None

    keys = _SIDE_KEYS.get(side, (side,))
    for k in keys:
        val = _to_float(prices.get(k))
        if val is not None:
            return val
    return None


def fetch_clv_data(since_days: int) -> list[dict[str, Any]]:
    query = """
    SELECT 
        p.prediction_id,
        p.fixture_id,
        p.market_code,
        p.kickoff,
        p.league_code,
        p.action,
        fom.market_code as provider_market_code,
        fom.line_num as provider_line_num,
        fom.snapshot_type,
        fom.snapshot_time_utc,
        fom.odds_json
    FROM (
        SELECT 
            p.prediction_id,
            p.fixture_id,
            p.market_code,
            f.match_datetime_utc as kickoff,
            f.league_code,
            pra.action
        FROM predictions p
        JOIN fixtures f ON f.fixture_id = p.fixture_id
        JOIN prediction_scores ps ON ps.prediction_id = p.prediction_id
        LEFT JOIN prediction_risk_assessments pra ON pra.prediction_id = p.prediction_id
        WHERE f.match_datetime_utc >= NOW() - (%s || ' days')::interval
          AND f.match_datetime_utc <= NOW()
    ) p
    JOIN fixture_odds_markets fom ON fom.fixture_id = p.fixture_id
    WHERE fom.provider = 'sofascore'
      AND fom.snapshot_time_utc <= p.kickoff
    ORDER BY p.prediction_id, fom.snapshot_time_utc DESC
    """

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (since_days,))
            if cur.description is None:
                return []
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _normalize_market_code(market_code: str | None) -> str:
    if not market_code:
        return ""
    return market_code[2:] if market_code.startswith("p_") else market_code


def _snapshot_priority(snapshot_type: str) -> int:
    if snapshot_type == "latest_pre_match":
        return 0
    if snapshot_type == "closing":
        return 1
    return 2


def _pick_odds_row(
    rows: list[dict[str, Any]],
    selections: tuple[Any, ...],
    kickoff: datetime | None,
) -> tuple[dict[str, Any] | None, Any | None]:
    if kickoff is None:
        return None, None

    eligible_rows: list[tuple[dict[str, Any], datetime]] = []
    for row in rows:
        snapshot_time = _parse_datetime(row.get("snapshot_time_utc"))
        if snapshot_time is None or snapshot_time > kickoff:
            continue
        eligible_rows.append((row, snapshot_time))

    if not eligible_rows:
        return None, None

    candidates: list[tuple[dict[str, Any], Any, datetime]] = []
    for selection in selections:
        target_market_codes = set(selection.market_codes)
        for row, snapshot_time in eligible_rows:
            if row.get("provider_market_code") not in target_market_codes:
                continue
            if not _line_matches(row.get("provider_line_num"), selection.line_num):
                continue
            candidates.append((row, selection, snapshot_time))

    if not candidates:
        return None, None

    def _sort_key(item: tuple[dict[str, Any], Any, datetime]) -> tuple[Any, ...]:
        row, selection, snapshot_time = item
        snapshot_type = str(row.get("snapshot_type") or "")
        return (
            _snapshot_priority(snapshot_type),
            -snapshot_time.timestamp(),
            snapshot_type,
            str(row.get("provider_market_code") or ""),
            _safe_float(row.get("provider_line_num")) or 0.0,
            str(selection.side),
        )

    candidates.sort(key=_sort_key)
    picked_row, picked_selection, _ = candidates[0]
    return picked_row, picked_selection


def _action_bucket(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower()
    return normalized or "unknown"


def main() -> int:
    args = parse_args()
    try:
        raw_rows = fetch_clv_data(args.since_days)
    except RuntimeError as exc:
        print(f"CLV unavailable: {exc}")
        return 0

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in raw_rows:
        prediction_id = row.get("prediction_id")
        if prediction_id is None:
            continue
        grouped.setdefault(int(prediction_id), []).append(row)

    results: list[dict[str, Any]] = []
    missing_counts = {
        "market_unmapped": 0,
        "selection_unavailable": 0,
        "opening_missing": 0,
        "latest_missing": 0,
        "both_missing": 0,
    }

    for odds_rows in grouped.values():
        base_row = odds_rows[0]
        market_code = str(base_row.get("market_code") or "")
        normalized_code = _normalize_market_code(market_code)
        config = _MARKET_MAP.get(normalized_code)
        if not config:
            missing_counts["market_unmapped"] += 1
            continue

        kickoff = _parse_datetime(base_row.get("kickoff"))
        primary_selections, fallback_selections = config
        odds_row, selection = _pick_odds_row(odds_rows, primary_selections, kickoff)
        if odds_row is None:
            odds_row, selection = _pick_odds_row(
                odds_rows, fallback_selections, kickoff
            )
        if odds_row is None or selection is None:
            missing_counts["selection_unavailable"] += 1
            continue

        opening_price = _extract_price(
            odds_row.get("odds_json"), selection.side, "prices_opening"
        )
        latest_price = _extract_price(
            odds_row.get("odds_json"), selection.side, "prices_latest"
        )
        if opening_price is None and latest_price is None:
            missing_counts["both_missing"] += 1
            continue
        if opening_price is None:
            missing_counts["opening_missing"] += 1
            continue
        if latest_price is None:
            missing_counts["latest_missing"] += 1
            continue

        implied_open = 1.0 / opening_price
        implied_latest = 1.0 / latest_price
        clv = implied_latest - implied_open

        action = _action_bucket(base_row.get("action"))
        results.append(
            {
                "market": market_code,
                "clv": clv,
                "league": base_row.get("league_code"),
                "action": action,
            }
        )

    missing_total = sum(missing_counts.values())
    total_samples = len(grouped)
    available_count = len(results)

    if not results:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(
                [
                    "# CLV Proxy Report",
                    "",
                    f"No CLV values available for the last {args.since_days} days.",
                    f"Samples evaluated: {total_samples}",
                    f"CLV unavailable: {missing_total}",
                    "",
                    "## CLV Unavailable Breakdown",
                    f"- market_unmapped: {missing_counts['market_unmapped']}",
                    f"- selection_unavailable: {missing_counts['selection_unavailable']}",
                    f"- opening_missing: {missing_counts['opening_missing']}",
                    f"- latest_missing: {missing_counts['latest_missing']}",
                    f"- both_missing: {missing_counts['both_missing']}",
                ]
            ),
            encoding="utf-8",
        )
        print(f"Report written to {args.out} (CLV unavailable for all samples)")
        return 0

    by_market: dict[str, list[float]] = {}
    by_action: dict[str, list[float]] = {}
    for r in results:
        m = r["market"]
        a = r["action"]
        by_market.setdefault(m, []).append(r["clv"])
        by_action.setdefault(a, []).append(r["clv"])

    summary_lines = [
        "# CLV Proxy Report",
        f"- Generated at: {datetime.now(UTC).isoformat()}",
        f"- Lookback: {args.since_days} days",
        f"- Samples evaluated: {total_samples}",
        f"- CLV available: {available_count}",
        f"- CLV unavailable: {missing_total}",
        f"- Missing opening prices: {missing_counts['opening_missing'] + missing_counts['both_missing']}",
        f"- Missing latest prices: {missing_counts['latest_missing'] + missing_counts['both_missing']}",
        "",
        "## Mean CLV by Action",
        "| Action | Count | Mean CLV |",
        "| :--- | :--- | :--- |",
    ]
    for a in sorted(by_action.keys()):
        vals = by_action[a]
        mean_clv = sum(vals) / len(vals)
        summary_lines.append(f"| {a} | {len(vals)} | {mean_clv:+.4f} |")

    summary_lines.extend(
        [
            "",
            "## Mean CLV by Market",
            "| Market | Count | Mean CLV |",
            "| :--- | :--- | :--- |",
        ]
    )

    for m in sorted(by_market.keys()):
        vals = by_market[m]
        mean_clv = sum(vals) / len(vals)
        summary_lines.append(f"| {m} | {len(vals)} | {mean_clv:+.4f} |")

    summary_lines.extend(
        [
            "",
            "## CLV Unavailable Breakdown",
            "| Reason | Count |",
            "| :--- | :--- |",
            f"| market_unmapped | {missing_counts['market_unmapped']} |",
            f"| selection_unavailable | {missing_counts['selection_unavailable']} |",
            f"| opening_missing | {missing_counts['opening_missing']} |",
            f"| latest_missing | {missing_counts['latest_missing']} |",
            f"| both_missing | {missing_counts['both_missing']} |",
        ]
    )

    overall_clv = sum(r["clv"] for r in results) / len(results)
    summary_lines.extend(
        [
            "",
            f"**Overall Mean CLV: {overall_clv:+.4f}**",
            "",
            "Note: CLV proxy = implied_prob_latest - implied_prob_opening using the same selected side/line from one leakage-safe pre-match snapshot row.",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
