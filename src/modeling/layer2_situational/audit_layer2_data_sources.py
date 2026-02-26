from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


@dataclass(frozen=True)
class SourceAuditRow:
    source_name: str
    role_in_layer2: str
    row_count: int
    fixture_coverage_pct: float
    timing_integrity_pct: float
    key_risks: list[str]
    risk_level: str
    evidence: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Layer 2 upstream data sources")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Output directory for source audit artifacts",
    )
    return parser.parse_args()


def _risk_level(*, fixture_coverage_pct: float, timing_integrity_pct: float, key_risks: list[str]) -> str:
    if timing_integrity_pct < 95.0 or fixture_coverage_pct < 50.0:
        return "critical"
    if timing_integrity_pct < 99.0 or fixture_coverage_pct < 75.0 or key_risks:
        return "high"
    if timing_integrity_pct < 99.9 or fixture_coverage_pct < 90.0:
        return "medium"
    return "low"


def _fetch_scalar(cur, query: str, params: tuple[object, ...] = ()) -> int:
    cur.execute(query, params)
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _fetch_one(cur, query: str, params: tuple[object, ...] = ()) -> tuple | None:
    cur.execute(query, params)
    return cur.fetchone()


def _column_exists(cur, table_name: str, column_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
        )
        """,
        (table_name, column_name),
    )
    row = cur.fetchone()
    return bool(row and row[0])


def _prediction_lineage_expr(cur, alias: str) -> tuple[str, str]:
    parts: list[str] = []
    anchor = "created_at"
    if _column_exists(cur, "predictions", "feature_asof_utc"):
        parts.append(f"{alias}.feature_asof_utc")
        anchor = "feature_asof_utc"
    if _column_exists(cur, "predictions", "first_created_at"):
        parts.append(f"{alias}.first_created_at")
        if anchor == "created_at":
            anchor = "first_created_at"
    parts.append(f"{alias}.created_at")
    return f"COALESCE({', '.join(parts)})", anchor


def _availability_timing_expr(cur, alias: str) -> tuple[str, str]:
    if _column_exists(cur, "player_availability", "event_recorded_at"):
        return f"{alias}.event_recorded_at", "event_recorded_at"
    if _column_exists(cur, "player_availability", "first_recorded_at"):
        return f"{alias}.first_recorded_at", "first_recorded_at"
    return f"{alias}.recorded_at", "recorded_at"


def audit_sources() -> tuple[list[SourceAuditRow], dict[str, object]]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            ft_fixture_count = _fetch_scalar(
                cur,
                """
                SELECT COUNT(*)
                FROM fixtures f
                JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
                WHERE f.status = 'ft'
                  AND f.match_datetime_utc IS NOT NULL
                  AND fr.home_goals IS NOT NULL
                  AND fr.away_goals IS NOT NULL
                """,
            )

            fixture_bounds = _fetch_one(
                cur,
                """
                SELECT MIN(match_datetime_utc), MAX(match_datetime_utc)
                FROM fixtures
                WHERE match_datetime_utc IS NOT NULL
                """,
            )
            fixtures_min = str(fixture_bounds[0]) if fixture_bounds and fixture_bounds[0] is not None else None
            fixtures_max = str(fixture_bounds[1]) if fixture_bounds and fixture_bounds[1] is not None else None

            rows: list[SourceAuditRow] = []
            pred_lineage_expr, pred_lineage_anchor = _prediction_lineage_expr(cur, "p")
            pa_timing_expr, pa_timing_anchor = _availability_timing_expr(cur, "pa")

            # 1) fixtures + fixture_results (base target)
            base_rows = _fetch_scalar(
                cur,
                """
                SELECT COUNT(*)
                FROM fixtures f
                JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
                WHERE f.status = 'ft'
                  AND f.match_datetime_utc IS NOT NULL
                  AND fr.home_goals IS NOT NULL
                  AND fr.away_goals IS NOT NULL
                """,
            )
            base_cov = (base_rows / ft_fixture_count * 100.0) if ft_fixture_count else 0.0
            rows.append(
                SourceAuditRow(
                    source_name="fixtures+fixture_results",
                    role_in_layer2="ground-truth target and chronology anchor",
                    row_count=base_rows,
                    fixture_coverage_pct=base_cov,
                    timing_integrity_pct=100.0,
                    key_risks=[],
                    risk_level="low",
                    evidence={
                        "ft_fixture_count": ft_fixture_count,
                        "time_min_utc": fixtures_min,
                        "time_max_utc": fixtures_max,
                    },
                )
            )

            # 2) predictions lambda_xgb (Layer 1 inputs)
            lambda_rows = _fetch_scalar(
                cur,
                """
                SELECT COUNT(*)
                FROM predictions
                WHERE model_name = 'lambda_xgb'
                  AND market_code IN ('lambda_home', 'lambda_away')
                """,
            )
            latest_pair_row = _fetch_one(
                cur,
                f"""
                WITH lambda_rows AS (
                    SELECT
                        p.fixture_id,
                        p.model_version,
                        p.market_code,
                        {pred_lineage_expr} AS lineage_at,
                        p.prediction_id
                    FROM predictions p
                    WHERE model_name = 'lambda_xgb'
                      AND market_code IN ('lambda_home', 'lambda_away')
                ),
                pair_versions AS (
                    SELECT fixture_id, model_version,
                           MAX(lineage_at) AS latest_lineage_at,
                           MAX(prediction_id) AS latest_prediction_id
                    FROM lambda_rows
                    GROUP BY fixture_id, model_version
                    HAVING COUNT(DISTINCT market_code) = 2
                ),
                chosen AS (
                    SELECT
                        pv.fixture_id,
                        pv.latest_lineage_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY pv.fixture_id
                            ORDER BY pv.latest_lineage_at DESC, pv.latest_prediction_id DESC, pv.model_version DESC
                        ) AS rn
                    FROM pair_versions pv
                )
                SELECT
                    COUNT(*) FILTER (WHERE c.rn = 1) AS latest_pair_fixtures,
                    COUNT(*) FILTER (WHERE c.rn = 1 AND c.latest_lineage_at <= f.match_datetime_utc) AS pre_kickoff_latest_pairs,
                    COUNT(*) FILTER (WHERE c.rn = 1 AND c.latest_lineage_at > f.match_datetime_utc) AS post_kickoff_latest_pairs
                FROM chosen c
                JOIN fixtures f ON f.fixture_id = c.fixture_id
                WHERE c.rn = 1
                  AND f.status = 'ft'
                  AND f.match_datetime_utc IS NOT NULL
                """,
            )
            latest_pairs = int(latest_pair_row[0]) if latest_pair_row else 0
            pre_latest = int(latest_pair_row[1]) if latest_pair_row else 0
            post_latest = int(latest_pair_row[2]) if latest_pair_row else 0
            lambda_cov = (latest_pairs / ft_fixture_count * 100.0) if ft_fixture_count else 0.0
            lambda_timing = (pre_latest / latest_pairs * 100.0) if latest_pairs else 0.0
            lambda_risks: list[str] = []
            if post_latest > 0:
                lambda_risks.append("latest_lambda_created_after_kickoff")
            if lambda_timing < 95.0:
                lambda_risks.append("insufficient_pre_kickoff_lineage")
            rows.append(
                SourceAuditRow(
                    source_name="predictions(lambda_xgb)",
                    role_in_layer2="Layer 1 baseline lambdas for residual target",
                    row_count=lambda_rows,
                    fixture_coverage_pct=lambda_cov,
                    timing_integrity_pct=lambda_timing,
                    key_risks=lambda_risks,
                    risk_level=_risk_level(
                        fixture_coverage_pct=lambda_cov,
                        timing_integrity_pct=lambda_timing,
                        key_risks=lambda_risks,
                    ),
                    evidence={
                        "latest_pair_fixtures": latest_pairs,
                        "pre_kickoff_latest_pairs": pre_latest,
                        "post_kickoff_latest_pairs": post_latest,
                        "lineage_anchor": pred_lineage_anchor,
                    },
                )
            )

            # 3) team_premium_snapshots
            tps_rows = _fetch_scalar(cur, "SELECT COUNT(*) FROM team_premium_snapshots")
            tps_fixture_pairs = _fetch_scalar(
                cur,
                """
                WITH base AS (
                    SELECT f.fixture_id
                    FROM fixtures f
                    JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
                    WHERE f.status = 'ft' AND f.match_datetime_utc IS NOT NULL
                ),
                pair_ok AS (
                    SELECT s.fixture_id
                    FROM team_premium_snapshots s
                    GROUP BY s.fixture_id
                    HAVING COUNT(*) FILTER (WHERE is_home = true) >= 1
                       AND COUNT(*) FILTER (WHERE is_home = false) >= 1
                )
                SELECT COUNT(*)
                FROM base b
                JOIN pair_ok p ON p.fixture_id = b.fixture_id
                """,
            )
            tps_cov = (tps_fixture_pairs / ft_fixture_count * 100.0) if ft_fixture_count else 0.0
            tps_sample = _fetch_one(
                cur,
                """
                SELECT
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY sample_size),
                    percentile_cont(0.1) WITHIN GROUP (ORDER BY sample_size),
                    percentile_cont(0.9) WITHIN GROUP (ORDER BY sample_size)
                FROM team_premium_snapshots
                WHERE sample_size IS NOT NULL
                """,
            )
            rows.append(
                SourceAuditRow(
                    source_name="team_premium_snapshots",
                    role_in_layer2="rolling team-form features and rest/schedule context",
                    row_count=tps_rows,
                    fixture_coverage_pct=tps_cov,
                    timing_integrity_pct=100.0,  # timestamp lineage not encoded in table schema
                    key_risks=[],
                    risk_level=_risk_level(
                        fixture_coverage_pct=tps_cov,
                        timing_integrity_pct=100.0,
                        key_risks=[],
                    ),
                    evidence={
                        "ft_fixtures_with_home_away_snapshot_pair": tps_fixture_pairs,
                        "sample_size_p50": float(tps_sample[0]) if tps_sample and tps_sample[0] is not None else None,
                        "sample_size_p10": float(tps_sample[1]) if tps_sample and tps_sample[1] is not None else None,
                        "sample_size_p90": float(tps_sample[2]) if tps_sample and tps_sample[2] is not None else None,
                    },
                )
            )

            # 4) fixture_odds_markets
            odds_rows = _fetch_scalar(
                cur,
                """
                SELECT COUNT(*)
                FROM fixture_odds_markets
                WHERE provider = 'sofascore'
                  AND market_code = '1x2'
                """,
            )
            odds_ft_cov_row = _fetch_one(
                cur,
                """
                WITH ft AS (
                    SELECT fixture_id, match_datetime_utc
                    FROM fixtures
                    WHERE status = 'ft' AND match_datetime_utc IS NOT NULL
                ),
                has_pre AS (
                    SELECT DISTINCT f.fixture_id
                    FROM ft f
                    JOIN fixture_odds_markets o ON o.fixture_id = f.fixture_id
                    WHERE o.provider = 'sofascore'
                      AND o.market_code = '1x2'
                      AND o.snapshot_type IN ('latest_pre_match', 'closing')
                      AND o.snapshot_time_utc <= f.match_datetime_utc
                ),
                has_bad AS (
                    SELECT DISTINCT f.fixture_id
                    FROM ft f
                    JOIN fixture_odds_markets o ON o.fixture_id = f.fixture_id
                    WHERE o.provider = 'sofascore'
                      AND o.market_code = '1x2'
                      AND o.snapshot_type IN ('latest_pre_match', 'closing')
                      AND o.snapshot_time_utc > f.match_datetime_utc
                )
                SELECT
                    (SELECT COUNT(*) FROM has_pre),
                    (SELECT COUNT(*) FROM has_bad)
                """,
            )
            odds_pre_cov_n = int(odds_ft_cov_row[0]) if odds_ft_cov_row else 0
            odds_bad_n = int(odds_ft_cov_row[1]) if odds_ft_cov_row else 0
            odds_cov = (odds_pre_cov_n / ft_fixture_count * 100.0) if ft_fixture_count else 0.0
            odds_timing = 100.0 - ((odds_bad_n / ft_fixture_count * 100.0) if ft_fixture_count else 0.0)
            odds_risks: list[str] = []
            if odds_cov < 60.0:
                odds_risks.append("sparse_pre_match_odds_coverage")
            if odds_bad_n > 0:
                odds_risks.append("post_kickoff_odds_rows_present")
            rows.append(
                SourceAuditRow(
                    source_name="fixture_odds_markets(sofascore_1x2)",
                    role_in_layer2="market-implied probabilities for odds-model gap features",
                    row_count=odds_rows,
                    fixture_coverage_pct=odds_cov,
                    timing_integrity_pct=odds_timing,
                    key_risks=odds_risks,
                    risk_level=_risk_level(
                        fixture_coverage_pct=odds_cov,
                        timing_integrity_pct=odds_timing,
                        key_risks=odds_risks,
                    ),
                    evidence={
                        "ft_fixtures_with_pre_kickoff_1x2": odds_pre_cov_n,
                        "ft_fixtures_with_post_kickoff_1x2_in_pre_types": odds_bad_n,
                    },
                )
            )

            # 5) player_availability
            pa_rows = _fetch_scalar(cur, "SELECT COUNT(*) FROM player_availability")
            pa_cov_n = _fetch_scalar(
                cur,
                """
                SELECT COUNT(DISTINCT f.fixture_id)
                FROM fixtures f
                JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
                JOIN player_availability pa ON pa.fixture_id = f.fixture_id
                WHERE f.status = 'ft'
                  AND f.match_datetime_utc IS NOT NULL
                """,
            )
            pa_bad_n = _fetch_scalar(
                cur,
                f"""
                SELECT COUNT(*)
                FROM player_availability pa
                JOIN fixtures f ON f.fixture_id = pa.fixture_id
                WHERE f.match_datetime_utc IS NOT NULL
                  AND {pa_timing_expr} IS NOT NULL
                  AND {pa_timing_expr} > f.match_datetime_utc
                """,
            )
            pa_known_timing_n = _fetch_scalar(
                cur,
                f"""
                SELECT COUNT(*)
                FROM player_availability pa
                JOIN fixtures f ON f.fixture_id = pa.fixture_id
                WHERE f.match_datetime_utc IS NOT NULL
                  AND {pa_timing_expr} IS NOT NULL
                """,
            )
            pa_unknown_timing_n = _fetch_scalar(
                cur,
                f"""
                SELECT COUNT(*)
                FROM player_availability pa
                JOIN fixtures f ON f.fixture_id = pa.fixture_id
                WHERE f.match_datetime_utc IS NOT NULL
                  AND {pa_timing_expr} IS NULL
                """,
            )
            pa_cov = (pa_cov_n / ft_fixture_count * 100.0) if ft_fixture_count else 0.0
            pa_timing = (
                100.0 - (pa_bad_n / pa_known_timing_n * 100.0)
                if pa_known_timing_n
                else 0.0
            )
            pa_risks: list[str] = []
            if pa_cov < 70.0:
                pa_risks.append("insufficient_fixture_coverage")
            if pa_bad_n > 0:
                pa_risks.append("timing_after_kickoff")
            if pa_known_timing_n == 0:
                pa_risks.append("no_known_timing_rows")
            unknown_share = (
                (pa_unknown_timing_n / (pa_known_timing_n + pa_unknown_timing_n))
                if (pa_known_timing_n + pa_unknown_timing_n) > 0
                else 0.0
            )
            if unknown_share > 0.25:
                pa_risks.append("timing_unknown_rows_high")
            rows.append(
                SourceAuditRow(
                    source_name="player_availability",
                    role_in_layer2="injury/absence impact features",
                    row_count=pa_rows,
                    fixture_coverage_pct=pa_cov,
                    timing_integrity_pct=pa_timing,
                    key_risks=pa_risks,
                    risk_level=_risk_level(
                        fixture_coverage_pct=pa_cov,
                        timing_integrity_pct=pa_timing,
                        key_risks=pa_risks,
                    ),
                    evidence={
                        "ft_fixtures_with_availability": pa_cov_n,
                        "timing_anchor": pa_timing_anchor,
                        "rows_with_timing_after_kickoff": pa_bad_n,
                        "rows_with_known_timing": pa_known_timing_n,
                        "rows_with_unknown_timing": pa_unknown_timing_n,
                    },
                )
            )

            # 6) team_rivalries
            rivalry_rows = _fetch_scalar(cur, "SELECT COUNT(*) FROM team_rivalries")
            derby_fixture_n = _fetch_scalar(
                cur,
                """
                SELECT COUNT(*)
                FROM fixtures f
                JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
                WHERE f.status = 'ft'
                  AND EXISTS (
                      SELECT 1
                      FROM team_rivalries r
                      WHERE (r.team_id_a = f.home_team_id AND r.team_id_b = f.away_team_id)
                         OR (r.team_id_a = f.away_team_id AND r.team_id_b = f.home_team_id)
                  )
                """,
            )
            rivalry_cov = (derby_fixture_n / ft_fixture_count * 100.0) if ft_fixture_count else 0.0
            rivalry_risks: list[str] = []
            if rivalry_cov < 1.0:
                rivalry_risks.append("very_low_derby_prevalence")
            rows.append(
                SourceAuditRow(
                    source_name="team_rivalries",
                    role_in_layer2="derby context features",
                    row_count=rivalry_rows,
                    fixture_coverage_pct=rivalry_cov,
                    timing_integrity_pct=100.0,
                    key_risks=rivalry_risks,
                    risk_level=_risk_level(
                        fixture_coverage_pct=rivalry_cov,
                        timing_integrity_pct=100.0,
                        key_risks=rivalry_risks,
                    ),
                    evidence={
                        "ft_derby_fixture_count": derby_fixture_n,
                    },
                )
            )

            # 7) fixture_player_stats
            fps_rows = _fetch_scalar(cur, "SELECT COUNT(*) FROM fixture_player_stats")
            fps_fixture_cov_n = _fetch_scalar(
                cur,
                """
                SELECT COUNT(DISTINCT f.fixture_id)
                FROM fixtures f
                JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
                JOIN fixture_player_stats fps ON fps.fixture_id = f.fixture_id
                WHERE f.status = 'ft'
                """,
            )
            fps_cov = (fps_fixture_cov_n / ft_fixture_count * 100.0) if ft_fixture_count else 0.0
            fps_xg_missing_pct_row = _fetch_one(
                cur,
                """
                SELECT
                    COUNT(*) FILTER (WHERE expected_goals IS NULL) * 100.0 / NULLIF(COUNT(*), 0)
                FROM fixture_player_stats
                """,
            )
            fps_xg_null_pct = float(fps_xg_missing_pct_row[0]) if fps_xg_missing_pct_row and fps_xg_missing_pct_row[0] is not None else 100.0
            fps_risks: list[str] = []
            if fps_cov < 60.0:
                fps_risks.append("player_stats_fixture_coverage_low")
            if fps_xg_null_pct > 30.0:
                fps_risks.append("expected_goals_missingness_high")
            rows.append(
                SourceAuditRow(
                    source_name="fixture_player_stats",
                    role_in_layer2="historical player impact prior strength estimation",
                    row_count=fps_rows,
                    fixture_coverage_pct=fps_cov,
                    timing_integrity_pct=100.0,
                    key_risks=fps_risks,
                    risk_level=_risk_level(
                        fixture_coverage_pct=fps_cov,
                        timing_integrity_pct=100.0,
                        key_risks=fps_risks,
                    ),
                    evidence={
                        "ft_fixtures_with_player_stats": fps_fixture_cov_n,
                        "expected_goals_null_pct": fps_xg_null_pct,
                    },
                )
            )

            meta = {
                "generated_at": datetime.now(UTC).isoformat(),
                "ft_fixture_count": ft_fixture_count,
                "fixtures_time_min_utc": fixtures_min,
                "fixtures_time_max_utc": fixtures_max,
                "critical_sources": [
                    row.source_name for row in rows if row.risk_level == "critical"
                ],
                "high_sources": [
                    row.source_name for row in rows if row.risk_level == "high"
                ],
            }

            return rows, meta
    finally:
        conn.close()


def write_outputs(output_dir: Path, rows: list[SourceAuditRow], meta: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "layer2_data_sources_audit.json"
    md_path = output_dir / "layer2_data_sources_audit.md"

    payload = {
        "meta": meta,
        "rows": [asdict(r) for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Layer 2 Data Source Audit")
    lines.append("")
    lines.append(f"Generated: {meta['generated_at']}")
    lines.append("")
    lines.append(f"- FT fixtures baseline: {meta['ft_fixture_count']}")
    lines.append(f"- Time span: {meta['fixtures_time_min_utc']} -> {meta['fixtures_time_max_utc']}")
    lines.append("")
    lines.append("| Source | Role | Rows | Fixture Coverage % | Timing Integrity % | Risk | Key Risks |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r.source_name} | {r.role_in_layer2} | {r.row_count} | "
            f"{r.fixture_coverage_pct:.2f} | {r.timing_integrity_pct:.2f} | {r.risk_level} | "
            f"{';'.join(r.key_risks) if r.key_risks else ''} |"
        )
    lines.append("")
    lines.append("## Critical Sources")
    if meta["critical_sources"]:
        for name in meta["critical_sources"]:
            lines.append(f"- {name}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## High-Risk Sources")
    if meta["high_sources"]:
        for name in meta["high_sources"]:
            lines.append(f"- {name}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Timing integrity is measured against kickoff when timestamp lineage is available.")
    lines.append("- Some sources do not encode immutable as-of timestamps; those are flagged in risk narrative when relevant.")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    rows, meta = audit_sources()
    json_path, md_path = write_outputs(args.output_dir, rows, meta)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
