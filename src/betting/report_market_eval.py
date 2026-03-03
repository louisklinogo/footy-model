from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

METRICS_FILE = (
    ROOT_DIR / "model_artifacts" / "market_models" / "metrics_walkforward.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-market evaluation report"
    )
    parser.add_argument(
        "--since-days", type=int, default=365, help="Days back for DB evaluation"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/reports/backtests/market_eval.md",
        help="Output report path",
    )
    return parser.parse_args()


def compute_bss(brier: float, base_rate: float) -> float:
    if base_rate <= 0 or base_rate >= 1:
        return 0.0
    return 1.0 - (brier / (base_rate * (1.0 - base_rate)))


def get_odds_band(odds: float | None) -> str:
    if odds is None:
        return "N/A"
    if odds < 1.5:
        return "<1.5"
    if odds < 2.0:
        return "1.5-2.0"
    if odds < 3.0:
        return "2.0-3.0"
    return "3.0+"


def fetch_market_stats_db(since_days: int) -> list[dict[str, object]]:
    query = """
    SELECT 
        p.market_code,
        COUNT(*) as n,
        AVG(ps.actual) as base_rate,
        AVG(ps.brier) as brier,
        AVG(ps.edge) as mean_edge,
        AVG(ps.roi_unit) as roi_unit,
        COUNT(ps.odds_used) as n_with_odds
    FROM prediction_scores ps
    JOIN predictions p ON p.prediction_id = ps.prediction_id
    JOIN fixtures f ON f.fixture_id = p.fixture_id
    WHERE f.match_datetime_utc >= NOW() - (%s || ' days')::interval
      AND ps.actual IS NOT NULL
    GROUP BY p.market_code
    ORDER BY p.market_code
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (since_days,))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        print(f"DB Error: {e}")
        return []
    finally:
        conn.close()


def fetch_odds_bands_db(since_days: int) -> dict[str, list[dict[str, object]]]:
    query = """
    SELECT 
        p.market_code,
        ps.odds_used,
        ps.actual,
        ps.roi_unit
    FROM prediction_scores ps
    JOIN predictions p ON p.prediction_id = ps.prediction_id
    JOIN fixtures f ON f.fixture_id = p.fixture_id
    WHERE f.match_datetime_utc >= NOW() - (%s || ' days')::interval
      AND ps.odds_used IS NOT NULL
      AND ps.actual IS NOT NULL
    """
    conn = connect_db()
    bands_data = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "hits": 0, "roi_sum": 0.0})
    )
    try:
        with conn.cursor() as cur:
            cur.execute(query, (since_days,))
            for row in cur.fetchall():
                m_code, odds, actual, roi = row
                band = get_odds_band(float(odds))
                stats = bands_data[m_code][band]
                stats["n"] += 1
                stats["hits"] += int(actual)
                stats["roi_sum"] += float(roi) if roi is not None else 0.0
    except Exception as e:
        print(f"DB Error (odds bands): {e}")
    finally:
        conn.close()
    return bands_data


def load_fallback_metrics() -> list[dict[str, object]]:
    if not METRICS_FILE.exists():
        return []
    with open(METRICS_FILE, "r") as f:
        data = json.load(f)

    results = []
    for item in data:
        results.append(
            {
                "market_code": item["market"],
                "base_rate": item["base_rate_test"],
                "auc": item.get("auc"),
                "brier": item["brier"],
                "n": item["test_n"],
                "mean_edge": None,
                "roi_unit": None,
                "n_with_odds": 0,
                "is_fallback": True,
            }
        )
    return results


def generate_report(stats: list[dict[str, object]], bands_data: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Market Evaluation Report",
        f"Generated at: {datetime.now().isoformat()}",
        "",
        "## Summary Table",
        "",
        "| Market | N | Base Rate | AUC | Brier | BSS | Mean Edge | ROI (Unit) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for s in stats:
        m = s["market_code"]
        n = s["n"]
        br = s["base_rate"]
        auc = s.get("auc", "N/A")
        brier = s["brier"]
        bss = (
            compute_bss(float(brier), float(br))
            if brier is not None and br is not None
            else "N/A"
        )
        edge = s["mean_edge"]
        roi = s["roi_unit"]

        auc_str = f"{auc:.4f}" if isinstance(auc, float) else "N/A"
        br_str = f"{br:.4f}" if br is not None else "N/A"
        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        bss_str = f"{bss:.4f}" if isinstance(bss, float) else "N/A"
        edge_str = f"{edge:.4f}" if edge is not None else "N/A"
        roi_str = f"{roi:.4f}" if roi is not None else "N/A"

        lines.append(
            f"| {m} | {n} | {br_str} | {auc_str} | {brier_str} | {bss_str} | {edge_str} | {roi_str} |"
        )

    if bands_data:
        lines.append("")
        lines.append("## Odds Band Breakdown")
        lines.append("")
        lines.append("| Market | Band | N | Hit Rate | ROI (Unit) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")

        for m_code in sorted(bands_data.keys()):
            for band in ["<1.5", "1.5-2.0", "2.0-3.0", "3.0+"]:
                if band in bands_data[m_code]:
                    s = bands_data[m_code][band]
                    hr = s["hits"] / s["n"] if s["n"] > 0 else 0
                    roi = s["roi_sum"] / s["n"] if s["n"] > 0 else 0
                    lines.append(
                        f"| {m_code} | {band} | {s['n']} | {hr:.4f} | {roi:.4f} |"
                    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {out_path}")


def main():
    args = parse_args()
    out_path = Path(args.out)

    stats = fetch_market_stats_db(args.since_days)
    bands_data = {}

    if not stats:
        print("No DB stats found or DB error. Falling back to metrics_walkforward.json")
        stats = load_fallback_metrics()
        if not stats:
            print("Error: No metrics available in DB or fallback file.")
            sys.exit(1)
    else:
        bands_data = fetch_odds_bands_db(args.since_days)
        # Try to enrich with AUC from fallback if missing in DB (DB doesn't store AUC in prediction_scores)
        fallback = {
            item["market_code"]: item["auc"] for item in load_fallback_metrics()
        }
        for s in stats:
            if s.get("auc") is None:
                s["auc"] = fallback.get(s["market_code"])

    generate_report(stats, bands_data, out_path)


if __name__ == "__main__":
    main()
