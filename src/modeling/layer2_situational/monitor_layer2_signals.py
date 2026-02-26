from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.layer2_situational.train_situational_residual import (
    add_odds_model_gap,
    load_feature_data,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def compute_odds_gap_drift(df: pd.DataFrame, days: int, min_rows: int = 30) -> dict:
    now = _utcnow()
    df = df.copy()
    df["match_datetime_utc"] = pd.to_datetime(
        df["match_datetime_utc"], utc=True, errors="coerce"
    )

    recent_start = now - timedelta(days=days)
    prev_start = now - timedelta(days=2 * days)

    recent = df[df["match_datetime_utc"] >= recent_start]
    prev = df[
        (df["match_datetime_utc"] >= prev_start)
        & (df["match_datetime_utc"] < recent_start)
    ]

    recent_vals = recent["odds_model_gap_home"].dropna()
    prev_vals = prev["odds_model_gap_home"].dropna()

    report = {
        "window_days": days,
        "recent_count": int(recent_vals.shape[0]),
        "prev_count": int(prev_vals.shape[0]),
        "recent_mean": float(recent_vals.mean()) if not recent_vals.empty else None,
        "prev_mean": float(prev_vals.mean()) if not prev_vals.empty else None,
        "recent_std": float(recent_vals.std()) if not recent_vals.empty else None,
        "prev_std": float(prev_vals.std()) if not prev_vals.empty else None,
    }

    if recent_vals.shape[0] >= min_rows and prev_vals.shape[0] >= min_rows:
        mean_shift = abs(report["recent_mean"] - report["prev_mean"])
        report["mean_shift"] = float(mean_shift)
        report["drift_flag"] = mean_shift > 0.05
    else:
        report["mean_shift"] = None
        report["drift_flag"] = False
        report["note"] = "insufficient_rows_for_drift_check"

    return report


def compute_availability_coverage(days: int) -> dict:
    conn = connect_db()
    now = _utcnow()
    horizon = now + timedelta(days=days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT fixture_id
            FROM fixtures
            WHERE status = 'scheduled'
              AND match_datetime_utc >= %s
              AND match_datetime_utc <= %s
            """,
            (now, horizon),
        )
        fixture_ids = [row[0] for row in cur.fetchall()]

        if not fixture_ids:
            conn.close()
            return {
                "window_days": days,
                "fixtures_total": 0,
                "fixtures_with_availability": 0,
                "coverage_pct": 0.0,
            }

        cur.execute(
            """
            SELECT COUNT(DISTINCT fixture_id)
            FROM player_availability
            WHERE fixture_id = ANY(%s)
            """,
            (fixture_ids,),
        )
        with_avail = int(cur.fetchone()[0])
    conn.close()

    total = len(fixture_ids)
    coverage = with_avail / total if total else 0.0
    return {
        "window_days": days,
        "fixtures_total": total,
        "fixtures_with_availability": with_avail,
        "coverage_pct": coverage,
        "coverage_flag": coverage < 0.5 if total else False,
    }


def compute_same_kickoff_groups(days: int) -> dict:
    conn = connect_db()
    now = _utcnow()
    horizon = now + timedelta(days=days)
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH base AS (
                SELECT fixture_id, match_datetime_utc
                FROM fixtures
                WHERE status = 'scheduled'
                  AND match_datetime_utc >= %s
                  AND match_datetime_utc <= %s
            ),
            grouped AS (
                SELECT match_datetime_utc, COUNT(*) AS fixture_count
                FROM base
                GROUP BY match_datetime_utc
                HAVING COUNT(*) > 1
            )
            SELECT COUNT(*) FROM grouped
            """,
            (now, horizon),
        )
        group_count = int(cur.fetchone()[0])
    conn.close()
    return {"window_days": days, "same_kickoff_groups": group_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Layer 2 signal health")
    parser.add_argument("--days", type=int, default=14, help="Window size in days")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/monitoring"),
        help="Output directory for monitoring report",
    )
    args = parser.parse_args()

    print("Loading feature data for odds gap drift...")
    df = load_feature_data()
    df = add_odds_model_gap(df)

    report = {
        "generated_at": _utcnow().isoformat(),
        "odds_gap_drift": compute_odds_gap_drift(df, args.days),
        "availability_coverage": compute_availability_coverage(args.days),
        "same_kickoff_groups": compute_same_kickoff_groups(args.days),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"layer2_monitor_{_utcnow().date().isoformat()}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
