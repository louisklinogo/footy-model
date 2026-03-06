"""
Validate pre-match backbone health before downstream market scoring/export.

This gate keeps Layer-1 lambda coverage/timing strict, while treating adjusted
lambda as an optional overlay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.layer2_situational.deployment_policy import (
    load_layer2_deployment_policy,
    resolve_league_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate lambda backbone coverage and timing integrity."
    )
    parser.add_argument("--days", type=int, default=3, help="Scheduled horizon in days")
    parser.add_argument("--league", type=str, default=None, help="Optional league filter")
    parser.add_argument(
        "--min-l1-coverage",
        type=float,
        default=0.98,
        help="Minimum required pair coverage for lambda_xgb.",
    )
    parser.add_argument(
        "--max-l1-late",
        type=int,
        default=0,
        help="Maximum allowed late Layer-1 pairs (as-of timestamp after kickoff).",
    )
    parser.add_argument(
        "--max-l1-unknown-asof",
        type=int,
        default=0,
        help="Maximum allowed Layer-1 pairs with unknown as-of timestamp.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--enforce-adj-enabled-league",
        action="store_true",
        help="When set, require adjusted-lambda coverage for leagues with Layer-2 enabled in deployment policy.",
    )
    parser.add_argument(
        "--min-adj-enabled-coverage",
        type=float,
        default=0.95,
        help="Minimum adjusted-lambda pair coverage for policy-enabled leagues when enforcement is active.",
    )
    parser.add_argument(
        "--layer2-model-dir",
        type=Path,
        default=ROOT_DIR / "model_artifacts" / "situational_model",
        help="Directory that contains layer2_deployment_policy.json.",
    )
    return parser.parse_args()


def _query_window_metrics(days: int, league: str | None) -> tuple[dict[str, object], list[dict[str, object]]]:
    params: list[object] = [days]
    league_filter = ""
    if league:
        league_filter = " AND f.league_code = %s"
        params.append(league)

    base_cte = f"""
    WITH target AS (
        SELECT f.fixture_id, f.league_code, f.match_datetime_utc
        FROM fixtures f
        WHERE f.status = 'scheduled'
          AND f.match_datetime_utc IS NOT NULL
          AND f.match_datetime_utc > NOW()
          AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
          {league_filter}
    ),
    l1h AS (
        SELECT DISTINCT ON (p.fixture_id)
            p.fixture_id,
            p.created_at,
            p.feature_asof_utc
        FROM predictions p
        JOIN target t ON t.fixture_id = p.fixture_id
        WHERE p.model_name = 'lambda_xgb'
          AND p.market_code = 'lambda_home'
        ORDER BY p.fixture_id, p.created_at DESC
    ),
    l1a AS (
        SELECT DISTINCT ON (p.fixture_id)
            p.fixture_id,
            p.created_at,
            p.feature_asof_utc
        FROM predictions p
        JOIN target t ON t.fixture_id = p.fixture_id
        WHERE p.model_name = 'lambda_xgb'
          AND p.market_code = 'lambda_away'
        ORDER BY p.fixture_id, p.created_at DESC
    ),
    l2h AS (
        SELECT DISTINCT ON (p.fixture_id)
            p.fixture_id,
            p.created_at,
            p.feature_asof_utc
        FROM predictions p
        JOIN target t ON t.fixture_id = p.fixture_id
        WHERE p.model_name = 'situational_xgb'
          AND p.market_code = 'adj_lambda_home'
        ORDER BY p.fixture_id, p.created_at DESC
    ),
    l2a AS (
        SELECT DISTINCT ON (p.fixture_id)
            p.fixture_id,
            p.created_at,
            p.feature_asof_utc
        FROM predictions p
        JOIN target t ON t.fixture_id = p.fixture_id
        WHERE p.model_name = 'situational_xgb'
          AND p.market_code = 'adj_lambda_away'
        ORDER BY p.fixture_id, p.created_at DESC
    ),
    wide AS (
        SELECT
            t.fixture_id,
            t.league_code,
            t.match_datetime_utc,
            l1h.created_at AS l1h_created_at,
            l1a.created_at AS l1a_created_at,
            l1h.feature_asof_utc AS l1h_asof_utc,
            l1a.feature_asof_utc AS l1a_asof_utc,
            l2h.created_at AS l2h_created_at,
            l2a.created_at AS l2a_created_at,
            l2h.feature_asof_utc AS l2h_asof_utc,
            l2a.feature_asof_utc AS l2a_asof_utc
        FROM target t
        LEFT JOIN l1h ON l1h.fixture_id = t.fixture_id
        LEFT JOIN l1a ON l1a.fixture_id = t.fixture_id
        LEFT JOIN l2h ON l2h.fixture_id = t.fixture_id
        LEFT JOIN l2a ON l2a.fixture_id = t.fixture_id
    )
    """

    summary_sql = (
        base_cte
        + """
    SELECT
        COUNT(*) AS fixtures_total,
        SUM((l1h_created_at IS NOT NULL AND l1a_created_at IS NOT NULL)::int) AS l1_pair_count,
        SUM((l2h_created_at IS NOT NULL AND l2a_created_at IS NOT NULL)::int) AS adj_pair_count,
        SUM(
            (
                (l1h_created_at IS NOT NULL AND l1a_created_at IS NOT NULL)
                AND (
                    COALESCE(l1h_asof_utc, l1h_created_at) > match_datetime_utc
                    OR COALESCE(l1a_asof_utc, l1a_created_at) > match_datetime_utc
                )
            )::int
        ) AS l1_late_count,
        SUM(
            (
                (l1h_created_at IS NOT NULL AND l1a_created_at IS NOT NULL)
                AND (l1h_asof_utc IS NULL OR l1a_asof_utc IS NULL)
            )::int
        ) AS l1_unknown_asof_count,
        SUM(
            (
                (l2h_created_at IS NOT NULL AND l2a_created_at IS NOT NULL)
                AND (
                    COALESCE(l2h_asof_utc, l2h_created_at) > match_datetime_utc
                    OR COALESCE(l2a_asof_utc, l2a_created_at) > match_datetime_utc
                )
            )::int
        ) AS adj_late_count,
        SUM(
            (
                (l2h_created_at IS NOT NULL AND l2a_created_at IS NOT NULL)
                AND (l2h_asof_utc IS NULL OR l2a_asof_utc IS NULL)
            )::int
        ) AS adj_unknown_asof_count
    FROM wide
    """
    )

    sample_sql = (
        base_cte
        + """
    SELECT fixture_id, league_code, match_datetime_utc
    FROM wide
    WHERE NOT (l1h_created_at IS NOT NULL AND l1a_created_at IS NOT NULL)
    ORDER BY match_datetime_utc ASC, fixture_id ASC
    LIMIT 25
    """
    )

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(summary_sql, tuple(params))
            row = cur.fetchone()
            assert row is not None
            summary = {
                "fixtures_total": int(row[0] or 0),
                "l1_pair_count": int(row[1] or 0),
                "adj_pair_count": int(row[2] or 0),
                "l1_late_count": int(row[3] or 0),
                "l1_unknown_asof_count": int(row[4] or 0),
                "adj_late_count": int(row[5] or 0),
                "adj_unknown_asof_count": int(row[6] or 0),
            }

            cur.execute(sample_sql, tuple(params))
            missing_rows = []
            for fixture_id, league_code, kickoff in cur.fetchall():
                missing_rows.append(
                    {
                        "fixture_id": int(fixture_id),
                        "league_code": league_code,
                        "match_datetime_utc": (
                            kickoff.isoformat() if kickoff is not None else None
                        ),
                    }
                )
    finally:
        conn.close()

    return summary, missing_rows


def main() -> None:
    args = parse_args()
    summary, missing_rows = _query_window_metrics(days=args.days, league=args.league)

    total = int(summary["fixtures_total"])
    l1_pairs = int(summary["l1_pair_count"])
    adj_pairs = int(summary["adj_pair_count"])
    l1_cov = (l1_pairs / total) if total else 1.0
    adj_cov = (adj_pairs / total) if total else 0.0

    failures: list[str] = []
    if total > 0 and l1_cov + 1e-12 < float(args.min_l1_coverage):
        failures.append(
            f"l1_coverage_below_threshold(actual={l1_cov:.4f}, threshold={args.min_l1_coverage:.4f})"
        )
    if int(summary["l1_late_count"]) > int(args.max_l1_late):
        failures.append(
            f"l1_late_count_exceeds_threshold(actual={summary['l1_late_count']}, threshold={args.max_l1_late})"
        )
    if int(summary["l1_unknown_asof_count"]) > int(args.max_l1_unknown_asof):
        failures.append(
            "l1_unknown_asof_exceeds_threshold"
            f"(actual={summary['l1_unknown_asof_count']}, threshold={args.max_l1_unknown_asof})"
        )

    layer2_enabled_for_league: bool | None = None
    layer2_reason_for_league: str | None = None
    if args.enforce_adj_enabled_league:
        if not args.league:
            failures.append("adj_enabled_gate_requires_league_filter")
        else:
            policy = load_layer2_deployment_policy(args.layer2_model_dir)
            league_policy = resolve_league_policy(policy, args.league)
            layer2_enabled_for_league = bool(league_policy.get("enabled", False))
            layer2_reason_for_league = str(league_policy.get("reason", "unknown"))
            if (
                layer2_enabled_for_league
                and total > 0
                and adj_cov + 1e-12 < float(args.min_adj_enabled_coverage)
            ):
                failures.append(
                    "adj_coverage_enabled_league_below_threshold"
                    f"(actual={adj_cov:.4f}, threshold={args.min_adj_enabled_coverage:.4f}, league={args.league})"
                )

    status = "pass" if not failures else "fail"
    report = {
        "status": status,
        "args": {
            "days": int(args.days),
            "league": args.league,
            "min_l1_coverage": float(args.min_l1_coverage),
            "max_l1_late": int(args.max_l1_late),
            "max_l1_unknown_asof": int(args.max_l1_unknown_asof),
            "enforce_adj_enabled_league": bool(args.enforce_adj_enabled_league),
            "min_adj_enabled_coverage": float(args.min_adj_enabled_coverage),
            "layer2_model_dir": str(args.layer2_model_dir),
        },
        "summary": {
            **summary,
            "l1_pair_coverage": l1_cov,
            "adj_pair_coverage": adj_cov,
            "layer2_enabled_for_league": layer2_enabled_for_league,
            "layer2_reason_for_league": layer2_reason_for_league,
        },
        "failures": failures,
        "examples_missing_l1_pair": missing_rows,
        "notes": [
            "Layer-1 coverage/timing is enforced.",
            "Adjusted-lambda coverage is gated only when enforcement is active and league policy is enabled.",
        ],
    }

    print(
        "Backbone health: "
        f"status={status} total={total} "
        f"l1_pairs={l1_pairs} ({l1_cov:.2%}) "
        f"adj_pairs={adj_pairs} ({adj_cov:.2%}) "
        f"l1_late={summary['l1_late_count']} "
        f"l1_unknown_asof={summary['l1_unknown_asof_count']} "
        f"layer2_enabled={layer2_enabled_for_league}"
    )
    if failures:
        print("Failure reasons:")
        for reason in failures:
            print(f"- {reason}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved backbone health report to {args.output}")

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
