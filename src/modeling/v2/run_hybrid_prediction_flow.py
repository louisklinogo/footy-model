from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.db.db_utils import connect_db


DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_SCORELINE_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline_external_context_v1_candidate_20260316"
DEFAULT_ANYTIME_DIR = ROOT_DIR / "model_artifacts" / "v2" / "anytime_direct_monotone_v1_candidate_20260308"
DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "v2" / "predictions" / "hybrid"
DEFAULT_EXPORT_PATH = ROOT_DIR / "storage" / "reports" / "market_predictions_hybrid_v1.csv"
DEFAULT_BASE_MODEL_NAME = "market_outcome_gbm"
DEFAULT_BASE_MODEL_VERSION = "fixtures_first_prematch_v1"
DEFAULT_MODEL_NAME = "market_outcome_v2"
DEFAULT_MODEL_VERSION = "hybrid_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid v2 prediction flow.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--scoreline-dir", type=Path, default=DEFAULT_SCORELINE_DIR)
    parser.add_argument("--anytime-dir", type=Path, default=DEFAULT_ANYTIME_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--league", type=str, default=None)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-model-name", type=str, default=DEFAULT_BASE_MODEL_NAME)
    parser.add_argument("--base-model-version", type=str, default=DEFAULT_BASE_MODEL_VERSION)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", type=str, default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--run-risk", action="store_true")
    parser.add_argument("--risk-history-days", type=int, default=180)
    parser.add_argument("--export-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_run_plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python_bin = str(args.python_bin)
    model_name = str(args.model_name)
    model_version = str(args.model_version)
    plan: list[tuple[str, list[str]]] = []

    base_cmd = [
        python_bin,
        str(ROOT_DIR / "src" / "modeling" / "evaluation" / "predict_market_outcomes_fixtures_first.py"),
        "--days",
        str(int(args.days)),
    ]
    if args.league:
        base_cmd.extend(["--league", str(args.league)])
    if args.limit is not None and int(args.limit) > 0:
        base_cmd.extend(["--limit", str(int(args.limit))])
    plan.append(("predict.base_legacy", base_cmd))

    family_specs = [
        (
            "predict.scoreline",
            ROOT_DIR / "src" / "modeling" / "v2" / "families" / "scoreline" / "predict_scoreline.py",
            Path(args.scoreline_dir),
        ),
        (
            "predict.anytime",
            ROOT_DIR / "src" / "modeling" / "v2" / "families" / "anytime" / "predict_anytime.py",
            Path(args.anytime_dir),
        ),
    ]
    for name, script_path, artifact_dir in family_specs:
        cmd = [
            python_bin,
            str(script_path),
            "--scope",
            str(args.scope),
            "--artifact-dir",
            str(artifact_dir),
            "--out-dir",
            str(Path(args.out_dir)),
            "--days",
            str(int(args.days)),
            "--model-name",
            model_name,
            "--model-version",
            model_version,
            "--write-db",
        ]
        if args.league:
            cmd.extend(["--league", str(args.league)])
        if args.limit is not None and int(args.limit) > 0:
            cmd.extend(["--limit", str(int(args.limit))])
        plan.append((name, cmd))

    if bool(args.run_risk):
        risk_cmd = [
            python_bin,
            str(ROOT_DIR / "src" / "modeling" / "evaluation" / "assess_prediction_risk.py"),
            "--days",
            str(int(args.days)),
            "--history-days",
            str(int(args.risk_history_days)),
            "--model",
            model_name,
            "--version",
            model_version,
        ]
        if args.league:
            risk_cmd.extend(["--league", str(args.league)])
        if args.limit is not None and int(args.limit) > 0:
            risk_cmd.extend(["--limit", str(int(args.limit))])
        plan.append(("risk.assess", risk_cmd))

    if args.export_out is not None:
        export_cmd = [
            python_bin,
            str(ROOT_DIR / "src" / "modeling" / "export" / "export_market_outcomes_fixtures_first.py"),
            "--days",
            str(int(args.days)),
            "--model",
            model_name,
            "--version",
            model_version,
            "--out",
            str(Path(args.export_out)),
        ]
        if args.league:
            export_cmd.extend(["--league", str(args.league)])
        plan.append(("export.csv", export_cmd))

    return plan


def _fixture_scope_sql(league: str | None, days: int, limit: int | None) -> tuple[str, list[object]]:
    clauses = [
        "f.status = 'scheduled'",
        "f.match_datetime_utc IS NOT NULL",
        "f.match_datetime_utc > NOW()",
        "f.match_datetime_utc <= NOW() + (%s || ' days')::interval",
    ]
    params: list[object] = [int(days)]
    if league:
        clauses.append("f.league_code = %s")
        params.append(league)
    sql = (
        "SELECT f.fixture_id FROM fixtures f WHERE "
        + " AND ".join(clauses)
        + " ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC"
    )
    if limit is not None and int(limit) > 0:
        sql += " LIMIT %s"
        params.append(int(limit))
    return sql, params


def seed_hybrid_predictions(args: argparse.Namespace) -> dict[str, Any]:
    fixture_scope_sql, fixture_scope_params = _fixture_scope_sql(args.league, int(args.days), args.limit)
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"WITH target_fixtures AS ({fixture_scope_sql}) SELECT COUNT(*) FROM target_fixtures", fixture_scope_params)
                fixture_count = int(cur.fetchone()[0])

                delete_sql = f"""
                WITH target_fixtures AS ({fixture_scope_sql})
                DELETE FROM predictions p
                USING target_fixtures tf
                WHERE p.fixture_id = tf.fixture_id
                  AND p.model_name = %s
                  AND p.model_version = %s
                """
                cur.execute(delete_sql, [*fixture_scope_params, str(args.model_name), str(args.model_version)])
                deleted_rows = int(cur.rowcount)

                insert_sql = f"""
                WITH target_fixtures AS ({fixture_scope_sql})
                INSERT INTO predictions (
                    fixture_id,
                    market_code,
                    model_name,
                    model_version,
                    p_model,
                    metadata_json,
                    created_at
                )
                SELECT
                    p.fixture_id,
                    p.market_code,
                    %s,
                    %s,
                    p.p_model,
                    COALESCE(p.metadata_json, '{{}}'::jsonb)
                        || jsonb_build_object(
                            'hybrid_source', 'live_seed_copy',
                            'hybrid_seeded_from_model_name', %s,
                            'hybrid_seeded_from_model_version', %s,
                            'hybrid_seeded_at_utc', to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
                        ),
                    NOW()
                FROM predictions p
                JOIN target_fixtures tf ON tf.fixture_id = p.fixture_id
                WHERE p.model_name = %s
                  AND p.model_version = %s
                ON CONFLICT (fixture_id, market_code, model_name, model_version)
                DO UPDATE SET
                    p_model = EXCLUDED.p_model,
                    metadata_json = EXCLUDED.metadata_json,
                    created_at = NOW()
                """
                cur.execute(
                    insert_sql,
                    [
                        *fixture_scope_params,
                        str(args.model_name),
                        str(args.model_version),
                        str(args.base_model_name),
                        str(args.base_model_version),
                        str(args.base_model_name),
                        str(args.base_model_version),
                    ],
                )
                seeded_rows = int(cur.rowcount)
        return {
            "name": "seed.hybrid_base",
            "fixture_count": fixture_count,
            "deleted_rows": deleted_rows,
            "seeded_rows": seeded_rows,
            "model_name": str(args.model_name),
            "model_version": str(args.model_version),
            "base_model_name": str(args.base_model_name),
            "base_model_version": str(args.base_model_version),
            "dry_run": False,
        }
    finally:
        conn.close()


def _run_step(name: str, command: list[str], *, dry_run: bool) -> dict[str, Any]:
    print(f"COMMAND[{name}]: {' '.join(command)}")
    if dry_run:
        return {
            "name": name,
            "command": command,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dry_run": True,
        }
    completed = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True)
    result = {
        "name": name,
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "dry_run": False,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {name}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return result


def main() -> None:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plan = build_run_plan(args)
    results: list[dict[str, Any]] = []
    for idx, (name, command) in enumerate(plan):
        results.append(_run_step(name, command, dry_run=bool(args.dry_run)))
        if idx == 0:
            if bool(args.dry_run):
                results.append(
                    {
                        "name": "seed.hybrid_base",
                        "fixture_count": 0,
                        "deleted_rows": 0,
                        "seeded_rows": 0,
                        "dry_run": True,
                    }
                )
            else:
                results.append(seed_hybrid_predictions(args))
    report = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dry_run": bool(args.dry_run),
        "base_model_name": str(args.base_model_name),
        "base_model_version": str(args.base_model_version),
        "model_name": str(args.model_name),
        "model_version": str(args.model_version),
        "scope": str(args.scope),
        "scoreline_dir": str(args.scoreline_dir),
        "anytime_dir": str(args.anytime_dir),
        "out_dir": str(args.out_dir),
        "export_out": str(args.export_out) if args.export_out is not None else None,
        "steps": results,
    }
    report_path = Path(args.out_dir) / "hybrid_prediction_flow_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved hybrid prediction flow report to {report_path}")


if __name__ == "__main__":
    main()
