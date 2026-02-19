"""Premium data readiness gate for fixtures-first tables."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_utils import connect_db
from pipeline_logging import create_data_quality_run, create_pipeline_run, finalize_pipeline_run


OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_OVERALL_COVERAGE = 0.85
MIN_HIGH_FIDELITY_RATE = 0.80
MIN_LEAGUE_COVERAGE = 0.50
HIGH_FIDELITY_CUTOFF = 0.9
JOB_NAME = "premium_readiness_gate_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check premium ingest readiness")
    parser.add_argument("--league", help="league_code scope")
    parser.add_argument("--since", help="scope by fixture date >= YYYY-MM-DD")
    return parser.parse_args()


def _build_scope_clause(league: str | None, since: str | None) -> tuple[str, list[object], str | None]:
    clauses = ["f.status = 'ft'"]
    params: list[object] = []
    parsed_since: str | None = None

    if league:
        clauses.append("f.league_code = %s")
        params.append(league)

    if since:
        parsed_since = date.fromisoformat(since).isoformat()
        clauses.append("f.match_datetime_utc::date >= %s")
        params.append(parsed_since)

    return " AND ".join(clauses), params, parsed_since


def check_readiness(league: str | None = None, since: str | None = None) -> dict[str, Any]:
    scope_clause, scope_params, parsed_since = _build_scope_clause(league, since)

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS total, COUNT(p.fixture_id) AS with_premium
                FROM fixtures f
                LEFT JOIN fixture_stats_premium p ON p.fixture_id = f.fixture_id
                WHERE {scope_clause}
                """,
                scope_params,
            )
            overall_row = cur.fetchone() or (0, 0)

            cur.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE p.fidelity_score >= %s) AS high_fidelity_count,
                    COUNT(*) FILTER (WHERE p.fidelity_score >= 0.5 AND p.fidelity_score < %s) AS mid_fidelity_count,
                    COUNT(*) FILTER (WHERE p.fidelity_score < 0.5 OR p.fidelity_score IS NULL) AS low_fidelity_count,
                    COUNT(*) AS premium_count,
                    AVG(p.fidelity_score) AS fidelity_avg
                FROM fixtures f
                JOIN fixture_stats_premium p ON p.fixture_id = f.fixture_id
                WHERE {scope_clause}
                """,
                [HIGH_FIDELITY_CUTOFF, HIGH_FIDELITY_CUTOFF, *scope_params],
            )
            fidelity_row = cur.fetchone() or (0, 0, 0, 0, None)

            cur.execute(
                f"""
                SELECT
                    COALESCE(f.league_code, 'unknown') AS league_code,
                    COUNT(*) AS total,
                    COUNT(p.fixture_id) AS with_premium
                FROM fixtures f
                LEFT JOIN fixture_stats_premium p ON p.fixture_id = f.fixture_id
                WHERE {scope_clause}
                GROUP BY COALESCE(f.league_code, 'unknown')
                ORDER BY league_code
                """,
                scope_params,
            )
            league_rows = cur.fetchall()
    finally:
        conn.close()

    total_fixtures = int(overall_row[0] or 0)
    with_premium = int(overall_row[1] or 0)
    high_fidelity_count = int(fidelity_row[0] or 0)
    mid_fidelity_count = int(fidelity_row[1] or 0)
    low_fidelity_count = int(fidelity_row[2] or 0)
    premium_count = int(fidelity_row[3] or 0)
    fidelity_avg = float(fidelity_row[4]) if fidelity_row[4] is not None else None

    coverage_rate = (with_premium / total_fixtures) if total_fixtures else 0.0
    high_fidelity_rate = (high_fidelity_count / premium_count) if premium_count else 0.0

    league_coverage: list[dict[str, object]] = []
    min_league_coverage = 1.0
    for league_code, league_total, league_with_premium in league_rows:
        league_total_i = int(league_total or 0)
        league_with_premium_i = int(league_with_premium or 0)
        league_cov = (league_with_premium_i / league_total_i) if league_total_i else 0.0
        min_league_coverage = min(min_league_coverage, league_cov)
        league_coverage.append(
            {
                "league_code": league_code,
                "total": league_total_i,
                "with_premium": league_with_premium_i,
                "coverage_rate": league_cov,
            }
        )

    if not league_coverage:
        min_league_coverage = 0.0

    ready = coverage_rate >= MIN_OVERALL_COVERAGE and high_fidelity_rate >= MIN_HIGH_FIDELITY_RATE

    metrics = {
        "scope": {"league": league, "since": parsed_since},
        "thresholds": {
            "min_overall_coverage": MIN_OVERALL_COVERAGE,
            "min_high_fidelity_rate": MIN_HIGH_FIDELITY_RATE,
            "min_league_coverage": MIN_LEAGUE_COVERAGE,
            "high_fidelity_cutoff": HIGH_FIDELITY_CUTOFF,
        },
        "overall": {
            "finished_fixtures": total_fixtures,
            "with_premium": with_premium,
            "coverage_rate": coverage_rate,
        },
        "fidelity": {
            "premium_count": premium_count,
            "high_fidelity_count": high_fidelity_count,
            "mid_fidelity_count": mid_fidelity_count,
            "low_fidelity_count": low_fidelity_count,
            "high_fidelity_rate": high_fidelity_rate,
            "fidelity_avg": fidelity_avg,
        },
        "league_coverage": league_coverage,
        "min_league_coverage": min_league_coverage,
        "ready": ready,
    }
    return metrics


def _print_report(metrics: Mapping[str, Any]) -> None:
    overall = metrics["overall"]
    fidelity = metrics["fidelity"]
    scope = metrics["scope"]
    leagues = metrics["league_coverage"]

    print("=" * 60)
    print("PREMIUM DATA READINESS AUDIT")
    print("=" * 60)
    print("\nSCOPE:")
    print(f"  League: {scope['league'] or 'ALL'}")
    print(f"  Since: {scope['since'] or 'NONE'}")
    print("\nOVERALL COVERAGE:")
    print(f"  Finished fixtures: {overall['finished_fixtures']:,}")
    print(f"  Premium rows: {overall['with_premium']:,}")
    print(
        f"  Coverage: {overall['coverage_rate']:.1%} "
        f"{'[OK]' if overall['coverage_rate'] >= MIN_OVERALL_COVERAGE else '[FAIL]'}"
    )
    print("\nFIDELITY:")
    print(f"  Premium fixtures in scope: {fidelity['premium_count']:,}")
    print(
        f"  Bands: high={fidelity['high_fidelity_count']:,} "
        f"mid={fidelity['mid_fidelity_count']:,} low={fidelity['low_fidelity_count']:,}"
    )
    print(
        f"  High fidelity (>={HIGH_FIDELITY_CUTOFF}): {fidelity['high_fidelity_rate']:.1%} "
        f"{'[OK]' if fidelity['high_fidelity_rate'] >= MIN_HIGH_FIDELITY_RATE else '[FAIL]'}"
    )

    print("\nLEAGUE COVERAGE:")
    if leagues:
        for row in sorted(leagues, key=lambda entry: entry["coverage_rate"]):
            tag = "[OK]" if row["coverage_rate"] >= MIN_LEAGUE_COVERAGE else "[WARN]"
            print(f"  {row['league_code']}: {row['coverage_rate']:.1%} {tag}")
    else:
        print("  No finished fixtures found for scope")

    print("\n" + "=" * 60)
    if metrics["ready"]:
        print("VERDICT: DATA READY FOR PRODUCTION TRAINING [PASS]")
    else:
        print("VERDICT: DATA NOT READY - GAPS DETECTED [FAIL]")
    print("=" * 60)


def main() -> int:
    args = _parse_args()
    run_metadata = {
        "league": args.league,
        "since": args.since,
        "thresholds": {
            "min_overall_coverage": MIN_OVERALL_COVERAGE,
            "min_high_fidelity_rate": MIN_HIGH_FIDELITY_RATE,
            "min_league_coverage": MIN_LEAGUE_COVERAGE,
            "high_fidelity_cutoff": HIGH_FIDELITY_CUTOFF,
        },
    }
    run_id = create_pipeline_run(
        job_name=JOB_NAME,
        message="Premium readiness gate started",
        details_json=run_metadata,
    )

    try:
        metrics = check_readiness(league=args.league, since=args.since)
        _print_report(metrics)

        dq_id = create_data_quality_run(
            passed=bool(metrics["ready"]),
            coverage_pct=float(metrics["overall"]["coverage_rate"]),
            fidelity_avg=metrics["fidelity"]["fidelity_avg"],
            null_rate=None,
            details_json=metrics,
        )
        report = {"job_name": JOB_NAME, "dq_id": dq_id, **metrics}
        with (OUT_DIR / "premium_readiness_v1.json").open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        finalize_pipeline_run(
            run_id=run_id,
            status="success" if metrics["ready"] else "fail",
            message="Premium readiness gate passed" if metrics["ready"] else "Premium readiness gate failed",
            details_json=report,
        )
        return 0 if metrics["ready"] else 1
    except Exception as exc:
        finalize_pipeline_run(
            run_id=run_id,
            status="fail",
            message=f"Premium readiness gate failed: {exc}",
            details_json=run_metadata,
        )
        if isinstance(exc, ValueError):
            print(f"Invalid input: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
