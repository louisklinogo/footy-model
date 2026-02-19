"""Frequent fixtures-first tick job: settle -> predict -> score."""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_utils import connect_db
from pipeline_logging import create_pipeline_run, finalize_pipeline_run


REGISTRY_PATH = ROOT / "scrapers" / "config" / "league_registry.json"
PREMIUM_ROOT = ROOT / "data" / "v1" / "premium"
TICK_IDS_BASE = ROOT / "data" / "v1" / "ids" / "tick"

OVERALL_JOB = "tick_due_fixtures_v1"
SETTLE_JOB = "tick_due_fixtures_v1.settle"
PREDICT_JOB = "tick_due_fixtures_v1.predict"
SCORE_JOB = "tick_due_fixtures_v1.score"


@dataclass(frozen=True)
class Options:
    leagues_raw: str | None
    predict_days: int
    settlement_delay_minutes: int
    max_settle: int
    max_predict: int
    max_score: int
    score_since_days: int
    dry_run: bool


@dataclass(frozen=True)
class SettleTarget:
    fixture_id: int
    flashscore_id: str
    league_code: str


@dataclass(frozen=True)
class PredictTarget:
    fixture_id: int
    league_code: str


def parse_args() -> Options:
    parser = argparse.ArgumentParser(description="Run fixtures-first settle/predict/score tick")
    _ = parser.add_argument("--leagues", default=None)
    _ = parser.add_argument("--predict-days", type=int, default=3)
    _ = parser.add_argument("--settlement-delay-minutes", type=int, default=180)
    _ = parser.add_argument("--max-settle", type=int, default=25)
    _ = parser.add_argument("--max-predict", type=int, default=50)
    _ = parser.add_argument("--max-score", type=int, default=500)
    _ = parser.add_argument("--score-since-days", type=int, default=30)
    _ = parser.add_argument("--dry-run", action="store_true")
    ns = parser.parse_args()
    return Options(
        leagues_raw=ns.leagues if isinstance(ns.leagues, str) else None,
        predict_days=int(ns.predict_days),
        settlement_delay_minutes=int(ns.settlement_delay_minutes),
        max_settle=int(ns.max_settle),
        max_predict=int(ns.max_predict),
        max_score=int(ns.max_score),
        score_since_days=int(ns.score_since_days),
        dry_run=bool(ns.dry_run),
    )


def parse_leagues(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    leagues = [part.strip() for part in raw.split(",") if part.strip()]
    if not leagues:
        raise ValueError("--leagues was provided but no valid league codes were found")
    return leagues


def load_enabled_leagues() -> list[str]:
    rows = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    output: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("enabled") and isinstance(row.get("league_code"), str):
            output.append(str(row["league_code"]).strip())
    return [code for code in output if code]


def _start_run(job_name: str, details: Mapping[str, object], dry_run: bool) -> int | None:
    try:
        return create_pipeline_run(job_name=job_name, message="tick phase started", details_json=dict(details))
    except Exception as exc:
        if dry_run:
            print(f"WARN: logging unavailable for {job_name}: {exc}")
            return None
        raise


def _finish_run(
    run_id: int | None,
    status: str,
    message: str,
    details: Mapping[str, object],
    dry_run: bool,
) -> None:
    _ = dry_run
    if run_id is None:
        return
    finalize_pipeline_run(run_id=run_id, status=status, message=message, details_json=dict(details))


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("COMMAND:", " ".join(cmd))
    if dry_run:
        return
    _ = subprocess.run(cmd, check=True)


def ensure_fixture_job_state_table() -> None:
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fixture_job_state (
                        fixture_id BIGINT PRIMARY KEY,
                        last_status_check_at TIMESTAMPTZ,
                        settle_attempts INTEGER NOT NULL DEFAULT 0,
                        last_settle_attempt_at TIMESTAMPTZ,
                        last_predict_at TIMESTAMPTZ,
                        last_score_at TIMESTAMPTZ,
                        CONSTRAINT fixture_job_state_fixture_fk
                            FOREIGN KEY (fixture_id)
                            REFERENCES fixtures (fixture_id)
                            ON DELETE CASCADE
                    )
                    """
                )
    finally:
        conn.close()


def select_settle_targets(
    leagues: Sequence[str], settlement_delay_minutes: int, max_settle: int
) -> list[SettleTarget]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.fixture_id, f.flashscore_id, f.league_code
                FROM fixtures f
                WHERE f.match_datetime_utc IS NOT NULL
                  AND f.match_datetime_utc + (%s || ' minutes')::interval < NOW()
                  AND COALESCE(f.status, '') NOT IN ('ft', 'cancelled', 'abandoned', 'postponed')
                  AND f.league_code = ANY(%s)
                ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC
                LIMIT %s
                """,
                (settlement_delay_minutes, list(leagues), max_settle),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    targets: list[SettleTarget] = []
    for row in rows:
        if isinstance(row[0], int) and isinstance(row[1], str) and isinstance(row[2], str):
            fs_id = row[1].strip()
            league = row[2].strip()
            if fs_id and league:
                targets.append(SettleTarget(fixture_id=row[0], flashscore_id=fs_id, league_code=league))
    return targets


def mark_settle_selection(fixture_ids: Sequence[int]) -> None:
    if not fixture_ids:
        return
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO fixture_job_state (fixture_id, last_status_check_at, settle_attempts, last_settle_attempt_at)
                    VALUES (%s, NOW(), 1, NOW())
                    ON CONFLICT (fixture_id)
                    DO UPDATE SET
                        last_status_check_at = NOW(),
                        settle_attempts = fixture_job_state.settle_attempts + 1,
                        last_settle_attempt_at = NOW()
                    """,
                    [(fixture_id,) for fixture_id in fixture_ids],
                )
    finally:
        conn.close()


def write_ids_file(ids_root: Path, league: str, flashscore_ids: Sequence[str]) -> None:
    ids_root.mkdir(parents=True, exist_ok=True)
    ids_path = ids_root / f"match_ids_{league}.json"
    ids_path.write_text(json.dumps([{"id": fs_id} for fs_id in flashscore_ids], indent=2), encoding="utf-8")


def delete_existing_premium_json(league: str, flashscore_ids: Sequence[str], dry_run: bool) -> None:
    for fs_id in flashscore_ids:
        p = PREMIUM_ROOT / league / f"{fs_id}.json"
        if not p.exists():
            continue
        if dry_run:
            print(f"DRY RUN DELETE: {p}")
            continue
        p.unlink()


def run_settle_phase(options: Options, leagues: Sequence[str]) -> None:
    details = {"leagues": list(leagues), "dry_run": options.dry_run}
    run_id = _start_run(SETTLE_JOB, details, options.dry_run)
    try:
        if options.dry_run:
            print("DRY RUN: settle DB selection and updates skipped")
            for league in leagues:
                fake_root = TICK_IDS_BASE / "<run-ts>"
                print(f"DRY RUN WRITE: {fake_root / f'match_ids_{league}.json'}")
                run_command(["node", str(ROOT / "scrapers" / "premium_enricher_v4.js"), league, "--ids-root", str(fake_root), "--out-root", str(PREMIUM_ROOT)], True)
                run_command([sys.executable, str(ROOT / "scrapers" / "ingest_premium_fixtures_v1.py"), "--league", league], True)
            _finish_run(run_id, "success", "settle dry-run complete", details, options.dry_run)
            return

        targets = select_settle_targets(leagues, options.settlement_delay_minutes, options.max_settle)
        mark_settle_selection([t.fixture_id for t in targets])
        grouped: dict[str, list[SettleTarget]] = defaultdict(list)
        for target in targets:
            grouped[target.league_code].append(target)

        ids_root = TICK_IDS_BASE / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        for league, league_targets in grouped.items():
            fs_ids = [t.flashscore_id for t in league_targets]
            write_ids_file(ids_root, league, fs_ids)
            delete_existing_premium_json(league, fs_ids, False)
            run_command(["node", str(ROOT / "scrapers" / "premium_enricher_v4.js"), league, "--ids-root", str(ids_root), "--out-root", str(PREMIUM_ROOT)], False)
            run_command([sys.executable, str(ROOT / "scrapers" / "ingest_premium_fixtures_v1.py"), "--league", league], False)

        _finish_run(run_id, "success", f"settle complete targets={len(targets)}", details, options.dry_run)
    except Exception as exc:
        _finish_run(run_id, "fail", f"settle failed: {exc}", details, options.dry_run)
        raise


def select_predict_targets(leagues: Sequence[str], predict_days: int, max_predict: int) -> list[PredictTarget]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.fixture_id, f.league_code
                FROM fixtures f
                WHERE f.status = 'scheduled'
                  AND f.match_datetime_utc IS NOT NULL
                  AND f.match_datetime_utc > NOW()
                  AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
                  AND f.league_code = ANY(%s)
                ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC
                LIMIT %s
                """,
                (predict_days, list(leagues), max_predict),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    targets: list[PredictTarget] = []
    for row in rows:
        if isinstance(row[0], int) and isinstance(row[1], str) and row[1].strip():
            targets.append(PredictTarget(fixture_id=row[0], league_code=row[1].strip()))
    return targets


def mark_predict_selection(fixture_ids: Sequence[int]) -> None:
    if not fixture_ids:
        return
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO fixture_job_state (fixture_id, last_predict_at)
                    VALUES (%s, NOW())
                    ON CONFLICT (fixture_id)
                    DO UPDATE SET last_predict_at = NOW()
                    """,
                    [(fixture_id,) for fixture_id in fixture_ids],
                )
    finally:
        conn.close()


def run_predict_phase(options: Options, leagues: Sequence[str]) -> None:
    details = {"leagues": list(leagues), "dry_run": options.dry_run}
    run_id = _start_run(PREDICT_JOB, details, options.dry_run)
    try:
        if options.dry_run:
            for league in leagues:
                run_command([sys.executable, str(ROOT / "scrapers" / "build_team_premium_snapshots_v1.py"), "--league", league], True)
                run_command([sys.executable, str(ROOT / "models" / "predict_v3_fixtures_first.py"), "--league", league, "--days", str(options.predict_days)], True)
                run_command([sys.executable, str(ROOT / "models" / "export_predictions_v3_fixtures_first.py"), "--league", league, "--days", str(options.predict_days)], True)
            _finish_run(run_id, "success", "predict dry-run complete", details, options.dry_run)
            return

        targets = select_predict_targets(leagues, options.predict_days, options.max_predict)
        mark_predict_selection([target.fixture_id for target in targets])

        grouped: dict[str, list[PredictTarget]] = defaultdict(list)
        for target in targets:
            grouped[target.league_code].append(target)

        for league in grouped:
            run_command([sys.executable, str(ROOT / "scrapers" / "build_team_premium_snapshots_v1.py"), "--league", league], False)
            run_command([sys.executable, str(ROOT / "models" / "predict_v3_fixtures_first.py"), "--league", league, "--days", str(options.predict_days)], False)
            run_command([sys.executable, str(ROOT / "models" / "export_predictions_v3_fixtures_first.py"), "--league", league, "--days", str(options.predict_days)], False)

        _finish_run(run_id, "success", f"predict complete targets={len(targets)}", details, options.dry_run)
    except Exception as exc:
        _finish_run(run_id, "fail", f"predict failed: {exc}", details, options.dry_run)
        raise


def select_score_targets(league: str, since_days: int, max_score: int) -> list[int]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.fixture_id
                FROM fixtures f
                WHERE f.league_code = %s
                  AND f.status = 'ft'
                  AND f.match_datetime_utc IS NOT NULL
                  AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
                ORDER BY f.match_datetime_utc DESC, f.fixture_id DESC
                LIMIT %s
                """,
                (league, since_days, max_score),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [row[0] for row in rows if isinstance(row[0], int)]


def mark_score_selection(fixture_ids: Sequence[int]) -> None:
    if not fixture_ids:
        return
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO fixture_job_state (fixture_id, last_score_at)
                    VALUES (%s, NOW())
                    ON CONFLICT (fixture_id)
                    DO UPDATE SET last_score_at = NOW()
                    """,
                    [(fixture_id,) for fixture_id in fixture_ids],
                )
    finally:
        conn.close()


def run_score_phase(options: Options, leagues: Sequence[str]) -> None:
    details = {"leagues": list(leagues), "dry_run": options.dry_run}
    run_id = _start_run(SCORE_JOB, details, options.dry_run)
    try:
        failures: list[str] = []
        for league in leagues:
            score_targets: list[int] = []
            if not options.dry_run:
                score_targets = select_score_targets(league, options.score_since_days, options.max_score)
            try:
                run_command([sys.executable, str(ROOT / "models" / "score_predictions_v3_fixtures_first.py"), "--league", league, "--since-days", str(options.score_since_days), "--limit", str(options.max_score)], options.dry_run)
            except Exception as exc:
                failures.append(f"{league}: {exc}")
            finally:
                if not options.dry_run and score_targets:
                    mark_score_selection(score_targets)

        if failures:
            raise RuntimeError("; ".join(failures))

        _finish_run(run_id, "success", "score complete", details, options.dry_run)
    except Exception as exc:
        _finish_run(run_id, "fail", f"score failed: {exc}", details, options.dry_run)
        raise


def main() -> int:
    options = parse_args()
    leagues = parse_leagues(options.leagues_raw) or load_enabled_leagues()
    if not leagues:
        raise RuntimeError("No leagues available for tick job")

    details = {
        "leagues": leagues,
        "predict_days": options.predict_days,
        "settlement_delay_minutes": options.settlement_delay_minutes,
        "max_settle": options.max_settle,
        "max_predict": options.max_predict,
        "max_score": options.max_score,
        "dry_run": options.dry_run,
    }
    run_id = _start_run(OVERALL_JOB, details, options.dry_run)
    try:
        if options.dry_run:
            print("DRY RUN: skipping fixture_job_state DDL and DB mutations")
        else:
            ensure_fixture_job_state_table()

        run_settle_phase(options, leagues)
        run_predict_phase(options, leagues)
        run_score_phase(options, leagues)

        _finish_run(run_id, "success", "tick complete", details, options.dry_run)
        return 0
    except Exception as exc:
        _finish_run(run_id, "fail", f"tick failed: {exc}", details, options.dry_run)
        print(f"Tick failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
