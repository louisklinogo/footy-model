from __future__ import annotations

from collections.abc import Mapping

from psycopg2.extras import Json

from src.db.db_utils import connect_db


MAX_MESSAGE_LEN = 500


def _truncate_message(message: str | None) -> str | None:
    if message is None:
        return None
    text = str(message).strip()
    if not text:
        return None
    if len(text) <= MAX_MESSAGE_LEN:
        return text
    return text[: MAX_MESSAGE_LEN - 3] + "..."


def create_pipeline_run(
    job_name: str,
    message: str | None = None,
    details_json: Mapping[str, object] | None = None,
) -> int:
    details = dict(details_json or {})
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pipeline_runs (job_name, started_at, status, message, details_json)
                    VALUES (%s, NOW(), 'running', %s, %s)
                    RETURNING run_id
                    """,
                    (job_name, _truncate_message(message), Json(details)),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("Failed to create pipeline run")
                return int(row[0])
    finally:
        conn.close()


def finalize_pipeline_run(
    run_id: int,
    status: str,
    message: str | None = None,
    details_json: Mapping[str, object] | None = None,
) -> None:
    if status not in {"success", "fail"}:
        raise ValueError("status must be 'success' or 'fail'")

    details = dict(details_json or {})
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                    SET ended_at = NOW(),
                        status = %s,
                        message = %s,
                        details_json = %s
                    WHERE run_id = %s
                    """,
                    (status, _truncate_message(message), Json(details), run_id),
                )
    finally:
        conn.close()


def create_data_quality_run(
    passed: bool,
    coverage_pct: float | None = None,
    fidelity_avg: float | None = None,
    null_rate: float | None = None,
    details_json: Mapping[str, object] | None = None,
) -> int:
    details = dict(details_json or {})
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_quality_runs (
                        run_time,
                        coverage_pct,
                        fidelity_avg,
                        null_rate,
                        passed,
                        details_json
                    )
                    VALUES (NOW(), %s, %s, %s, %s, %s)
                    RETURNING dq_id
                    """,
                    (coverage_pct, fidelity_avg, null_rate, passed, Json(details)),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("Failed to create data quality run")
                return int(row[0])
    finally:
        conn.close()
