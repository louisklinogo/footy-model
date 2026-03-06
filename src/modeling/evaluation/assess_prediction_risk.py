"""
Assess pre-match prediction risk and write recommendation rows.

This script does not change `predictions`; it writes a separate risk layer into
`prediction_risk_assessments` keyed by prediction_id.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from psycopg2.extras import Json

from src.betting.odds_resolver import resolve_odds_for_market
from src.db.db_utils import connect_db


MODEL_NAME = "market_outcome_gbm"
MODEL_VERSION = "fixtures_first_prematch_v1"
DEFAULT_POLICY_PATH = Path("model_artifacts/market_models/risk_policy.json")
DEFAULT_GATING_PATH = Path("model_artifacts/market_models/market_gating.json")
DEFAULT_COVERAGE_DIR = Path("artifacts/reports/odds_coverage")
DEFAULT_EDGE_BUCKET_REPORT_DIR = Path("artifacts/reports/risk_edge_buckets")
EPS = 1e-6
EDGE_BUCKET_SPECS: tuple[tuple[str, float, float | None], ...] = (
    ("2-4%", 0.02, 0.04),
    ("4-6%", 0.04, 0.06),
    ("6%+", 0.06, None),
)
FOCUS_MARKETS = (
    "o15",
    "u35",
    "c75",
    "c85",
    "c95",
    "c105",
    "hc25",
    "hc35",
    "hc45",
    "hc55",
    "ac25",
    "ac35",
    "ac45",
    "ac55",
    "1x2_h",
    "1x2_d",
    "1x2_a",
    "dc_1x",
    "dc_x2",
    "dc_12",
    "ho15",
    "ao15",
    "h_1up",
    "a_1up",
    "h_2up",
    "a_2up",
)


DEFAULT_POLICY: dict[str, Any] = {
    "min_edge_watch": 0.01,
    "min_edge_bet_small": 0.02,
    "min_edge_bet": 0.04,
    "min_precision_rolling": 0.58,
    "precision_min_samples": 80,
    "precision_window": 60,
    "loss_streak_pause": 2,
    "hard_precision_gate": True,
    "hard_loss_streak_pause": True,
    "tradable_markets": [
        "o15",
        "u35",
        "1x2_h",
        "1x2_d",
        "1x2_a",
        "dc_1x",
        "dc_x2",
        "dc_12",
        "c85",
        "c95",
        "c105",
    ],
    "required_markets": [
        "o15",
        "u35",
        "1x2_h",
        "1x2_d",
        "1x2_a",
        "dc_1x",
        "dc_x2",
        "dc_12",
        "c85",
        "c95",
        "c105",
    ],
    "market_gating_path": str(DEFAULT_GATING_PATH),
    "odds_coverage_dir": str(DEFAULT_COVERAGE_DIR),
    "odds_coverage_report_path": None,
    "odds_coverage_min_pct": 0.6,
    "kelly_scale": 0.25,
    "max_stake_fraction": 0.02,
    "max_small_stake_fraction": 0.01,
    "calibration_min_samples": 80,
    "calibration_poor_brier": 0.22,
    "calibration_very_poor_brier": 0.24,
    "edge_bucket_min_samples": 60,
    "edge_bucket_min_precision": 0.54,
    "edge_bucket_loss_streak_pause": 3,
    "hard_edge_bucket_precision_gate": True,
    "hard_edge_bucket_loss_streak_pause": True,
    "edge_bucket_report_dir": str(DEFAULT_EDGE_BUCKET_REPORT_DIR),
    "penalties": {
        "missing_odds": 12.0,
        "fallback_used": 18.0,
        "feature_missing_unit": 0.75,
        "stale_odds_6h": 2.0,
        "stale_odds_24h": 4.0,
        "low_sample_lt8": 10.0,
        "low_sample_lt5": 18.0,
        "calibration_low_sample": 8.0,
        "calibration_poor": 10.0,
        "calibration_very_poor": 16.0,
        "precision_gate_fail": 25.0,
        "loss_streak_pause": 30.0,
        "edge_bucket_precision_gate_fail": 20.0,
        "edge_bucket_loss_streak_pause": 24.0,
    },
}


@dataclass(frozen=True)
class CalibrationStats:
    n: int
    mean_brier: float | None


@dataclass(frozen=True)
class RollingPerformance:
    n: int
    precision: float | None
    loss_streak: int


@dataclass(frozen=True)
class EdgeBucketPerformance:
    n: int
    precision: float | None
    loss_streak: int
    mean_roi_unit: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess pre-match risk for prediction rows"
    )
    parser.add_argument(
        "--league", type=str, default=None, help="Optional league_code filter"
    )
    parser.add_argument("--days", type=int, default=3, help="Future horizon in days")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional max prediction rows to assess"
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=180,
        help="Rolling history window used for calibration penalties",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument(
        "--version", type=str, default=MODEL_VERSION, help="Model version"
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Optional policy JSON path",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute but do not write to DB"
    )
    return parser.parse_args()


def ensure_risk_schema() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS prediction_risk_assessments (
        risk_assessment_id BIGSERIAL PRIMARY KEY,
        prediction_id BIGINT NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
        fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
        market_code TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        p_model DOUBLE PRECISION NOT NULL CHECK (p_model >= 0.0 AND p_model <= 1.0),
        p_conservative DOUBLE PRECISION NOT NULL CHECK (p_conservative >= 0.0 AND p_conservative <= 1.0),
        implied_probability DOUBLE PRECISION CHECK (implied_probability >= 0.0 AND implied_probability <= 1.0),
        odds_used DOUBLE PRECISION CHECK (odds_used IS NULL OR odds_used > 0.0),
        edge_raw DOUBLE PRECISION,
        edge_adjusted DOUBLE PRECISION,
        risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 100.0),
        action TEXT NOT NULL CHECK (action IN ('pass', 'watch', 'bet_small', 'bet')),
        kelly_fraction DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (kelly_fraction >= 0.0 AND kelly_fraction <= 1.0),
        stake_fraction DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (stake_fraction >= 0.0 AND stake_fraction <= 1.0),
        risk_flags_json JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(risk_flags_json) = 'array'),
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata_json) = 'object'),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (prediction_id),
        UNIQUE (fixture_id, market_code, model_name, model_version)
    );

    CREATE INDEX IF NOT EXISTS idx_prediction_risk_assessments_fixture_action
        ON prediction_risk_assessments (fixture_id, action);
    CREATE INDEX IF NOT EXISTS idx_prediction_risk_assessments_model_fixture
        ON prediction_risk_assessments (model_name, model_version, fixture_id);
    CREATE INDEX IF NOT EXISTS idx_prediction_risk_assessments_updated_at
        ON prediction_risk_assessments (updated_at DESC);
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_policy(path: Path) -> dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    if not path.exists():
        return policy
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return policy
    if not isinstance(payload, dict):
        return policy
    merged = _merge_dict(policy, payload)
    return merged


def _coerce_market_list(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    out: set[str] = set()
    for item in value:
        code = str(item or "").strip()
        if code:
            out.add(code)
    return out


def _load_gating_eligible(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict):
        return set()
    rows = payload.get("markets")
    if not isinstance(rows, list):
        return set()
    eligible: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("market_code") or "").strip()
        if not code:
            continue
        if bool(row.get("eligible")):
            eligible.add(code)
    return eligible


def _latest_coverage_report_path(root_dir: Path) -> Path | None:
    if not root_dir.exists() or not root_dir.is_dir():
        return None
    candidates = sorted(root_dir.glob("*.json"))
    if not candidates:
        return None
    return candidates[-1]


def _load_coverage_rates(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("coverage_by_market")
    if not isinstance(rows, list):
        return {}
    rates: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("market_code") or "").strip()
        if not code:
            continue
        pct_raw = row.get("coverage_pct")
        try:
            pct = float(pct_raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(pct):
            continue
        rates[code] = pct
    return rates


def resolve_tradable_markets(policy: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    required = _coerce_market_list(policy.get("required_markets"))
    configured = _coerce_market_list(policy.get("tradable_markets"))
    if not required:
        required = set(configured) if configured else set(FOCUS_MARKETS)

    gating_path = Path(str(policy.get("market_gating_path") or DEFAULT_GATING_PATH))
    gating_eligible = _load_gating_eligible(gating_path)
    if not gating_eligible:
        gating_eligible = set(required)

    coverage_rates: dict[str, float] = {}
    coverage_path_raw = str(policy.get("odds_coverage_report_path") or "").strip()
    if coverage_path_raw:
        coverage_path = Path(coverage_path_raw)
    else:
        coverage_dir_raw = str(policy.get("odds_coverage_dir") or "").strip()
        coverage_dir = Path(coverage_dir_raw) if coverage_dir_raw else DEFAULT_COVERAGE_DIR
        coverage_path = _latest_coverage_report_path(coverage_dir)

    coverage_path_str = None
    if coverage_path is not None:
        coverage_path_str = str(coverage_path)
        coverage_rates = _load_coverage_rates(coverage_path)

    min_cov = float(policy.get("odds_coverage_min_pct", 0.6))
    coverage_eligible = {
        code for code, pct in coverage_rates.items() if pct >= min_cov - EPS
    }

    tradable = set(required) & set(gating_eligible)
    coverage_applied = bool(coverage_rates)
    if coverage_applied:
        tradable &= coverage_eligible

    summary = {
        "required_count": len(required),
        "gating_eligible_count": len(gating_eligible),
        "coverage_applied": coverage_applied,
        "coverage_min_pct": min_cov,
        "coverage_report_path": coverage_path_str,
        "coverage_eligible_count": len(coverage_eligible),
        "resolved_tradable_count": len(tradable),
    }
    return tradable, summary


def fetch_candidates(
    *,
    model: str,
    version: str,
    league: str | None,
    days: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    query = """
    WITH latest_predictions AS (
        SELECT DISTINCT ON (p.fixture_id, p.market_code, p.model_name, p.model_version)
            p.prediction_id,
            p.fixture_id,
            p.market_code,
            p.model_name,
            p.model_version,
            COALESCE(p.p_final, p.p_model) AS p_model,
            p.metadata_json
        FROM predictions p
        JOIN fixtures f
          ON f.fixture_id = p.fixture_id
        WHERE p.model_name = %s
          AND p.model_version = %s
          AND p.market_code = ANY(%s)
          AND f.status = 'scheduled'
          AND f.match_datetime_utc IS NOT NULL
          AND f.match_datetime_utc > NOW()
          AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
    """
    params: list[object] = [model, version, list(FOCUS_MARKETS), days]
    if league:
        query += " AND f.league_code = %s"
        params.append(league)
    query += """
        ORDER BY
            p.fixture_id,
            p.market_code,
            p.model_name,
            p.model_version,
            p.created_at DESC,
            p.prediction_id DESC
    )
    SELECT
        lp.prediction_id,
        lp.fixture_id,
        lp.market_code,
        lp.model_name,
        lp.model_version,
        lp.p_model,
        lp.metadata_json,
        f.league_code,
        f.match_datetime_utc,
        fom.provider AS odds_provider,
        fom.snapshot_type AS odds_snapshot_type,
        fom.snapshot_time_utc AS odds_snapshot_time_utc,
        fom.line_num AS odds_line_num,
        fom.market_code AS odds_market_code,
        fom.odds_json
    FROM latest_predictions lp
    JOIN fixtures f
      ON f.fixture_id = lp.fixture_id
    LEFT JOIN LATERAL (
        -- Get the latest pre-match snapshot for ANY market matching this fixture
        -- We will filter for specific markets in Python code to keep query simple
        SELECT provider, snapshot_type, snapshot_time_utc, odds_json, line_num, market_code
        FROM fixture_odds_markets
        WHERE fixture_id = lp.fixture_id
          AND snapshot_time_utc <= f.match_datetime_utc
          AND snapshot_type IN ('latest_pre_match', 'closing')
        ORDER BY (snapshot_type = 'latest_pre_match') DESC, snapshot_time_utc DESC
        LIMIT 200 -- Fetch enough rows for market mapping + snapshot fallback.
    ) fom ON true
    ORDER BY f.match_datetime_utc ASC, lp.fixture_id ASC, lp.market_code ASC
    """
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            cols = [desc[0] for desc in (cur.description or [])]
    finally:
        conn.close()

    # Group by prediction_id to handle multiple markets from the lateral join
    grouped: dict[int, dict[str, Any]] = {}
    for raw_row in rows:
        d = dict(zip(cols, raw_row))
        p_id = d["prediction_id"]
        if p_id not in grouped:
            grouped[p_id] = {**d, "odds_rows": []}

        if d.get("odds_json"):
            blob = _as_dict(d["odds_json"])
            grouped[p_id]["odds_rows"].append(
                {
                    "provider": d.get("odds_provider"),
                    "snapshot_type": d.get("odds_snapshot_type"),
                    "snapshot_time_utc": d.get("odds_snapshot_time_utc"),
                    "market_code": blob.get("market_code") or d.get("odds_market_code"),
                    "line_num": d.get("odds_line_num"),
                    "odds_json": blob,
                    "prices_latest": blob.get("prices_latest"),
                }
            )

    return list(grouped.values())


def fetch_calibration_stats(
    *,
    model: str,
    version: str,
    history_days: int,
) -> tuple[dict[tuple[str, str], CalibrationStats], dict[str, CalibrationStats]]:
    query = """
    SELECT
        f.league_code,
        p.market_code,
        COUNT(*) AS n,
        AVG(ps.brier) AS mean_brier
    FROM predictions p
    JOIN prediction_scores ps
      ON ps.prediction_id = p.prediction_id
    JOIN fixtures f
      ON f.fixture_id = p.fixture_id
    WHERE p.model_name = %s
      AND p.model_version = %s
      AND p.market_code = ANY(%s)
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
    GROUP BY f.league_code, p.market_code
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (model, version, list(FOCUS_MARKETS), history_days))
            rows = cur.fetchall()
    finally:
        conn.close()

    by_league_market: dict[tuple[str, str], CalibrationStats] = {}
    by_market_accum: dict[str, list[tuple[int, float]]] = {}
    for league_code, market_code, n, mean_brier in rows:
        league = str(league_code or "")
        market = str(market_code or "")
        count = int(n or 0)
        brier = float(mean_brier) if mean_brier is not None else None
        by_league_market[(league, market)] = CalibrationStats(n=count, mean_brier=brier)
        if brier is not None and count > 0:
            by_market_accum.setdefault(market, []).append((count, brier))

    by_market: dict[str, CalibrationStats] = {}
    for market, entries in by_market_accum.items():
        total_n = sum(count for count, _ in entries)
        if total_n <= 0:
            by_market[market] = CalibrationStats(n=0, mean_brier=None)
            continue
        weighted = sum(count * brier for count, brier in entries) / total_n
        by_market[market] = CalibrationStats(n=total_n, mean_brier=float(weighted))

    return by_league_market, by_market


def _compute_loss_streak(hits_desc: list[bool]) -> int:
    streak = 0
    for hit in hits_desc:
        if hit:
            break
        streak += 1
    return streak


def fetch_recent_performance_stats(
    *,
    model: str,
    version: str,
    history_days: int,
    window: int,
) -> tuple[dict[tuple[str, str], RollingPerformance], dict[str, RollingPerformance]]:
    query = """
    SELECT
        f.league_code,
        p.market_code,
        ps.hit,
        f.match_datetime_utc
    FROM predictions p
    JOIN prediction_scores ps
      ON ps.prediction_id = p.prediction_id
    JOIN fixtures f
      ON f.fixture_id = p.fixture_id
    WHERE p.model_name = %s
      AND p.model_version = %s
      AND p.market_code = ANY(%s)
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
    ORDER BY p.market_code ASC, f.match_datetime_utc DESC
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (model, version, list(FOCUS_MARKETS), history_days))
            rows = cur.fetchall()
    finally:
        conn.close()

    per_league_market_hits: dict[tuple[str, str], list[bool]] = {}
    per_market_hits: dict[str, list[bool]] = {}
    for league_code, market_code, hit, _match_dt in rows:
        league = str(league_code or "")
        market = str(market_code or "")
        hit_bool = bool(hit)

        lm_key = (league, market)
        lm_hits = per_league_market_hits.setdefault(lm_key, [])
        if len(lm_hits) < window:
            lm_hits.append(hit_bool)

        market_hits = per_market_hits.setdefault(market, [])
        if len(market_hits) < window:
            market_hits.append(hit_bool)

    by_league_market: dict[tuple[str, str], RollingPerformance] = {}
    for key, hits in per_league_market_hits.items():
        n = len(hits)
        precision = float(sum(1 for h in hits if h) / n) if n > 0 else None
        by_league_market[key] = RollingPerformance(
            n=n,
            precision=precision,
            loss_streak=_compute_loss_streak(hits),
        )

    by_market: dict[str, RollingPerformance] = {}
    for market, hits in per_market_hits.items():
        n = len(hits)
        precision = float(sum(1 for h in hits if h) / n) if n > 0 else None
        by_market[market] = RollingPerformance(
            n=n,
            precision=precision,
            loss_streak=_compute_loss_streak(hits),
        )

    return by_league_market, by_market


def _edge_bucket_label(edge_value: float | None) -> str | None:
    if edge_value is None:
        return None
    edge = float(edge_value)
    if math.isnan(edge):
        return None
    for label, lo, hi in EDGE_BUCKET_SPECS:
        if edge < lo:
            continue
        if hi is None or edge < hi:
            return label
    return None


def _resolve_edge_bucket_gate(
    *,
    policy: dict[str, Any],
    market_code: str,
    edge_bucket: str | None,
) -> dict[str, Any]:
    resolved = {
        "min_samples": int(policy.get("edge_bucket_min_samples", 60)),
        "min_precision": float(policy.get("edge_bucket_min_precision", 0.54)),
        "loss_streak_pause": int(policy.get("edge_bucket_loss_streak_pause", 3)),
        "hard_precision_gate": bool(
            policy.get("hard_edge_bucket_precision_gate", True)
        ),
        "hard_loss_streak_pause": bool(
            policy.get("hard_edge_bucket_loss_streak_pause", True)
        ),
    }
    if not edge_bucket:
        return resolved

    overrides = policy.get("edge_bucket_overrides")
    if not isinstance(overrides, dict):
        return resolved
    market_overrides = overrides.get(market_code)
    if not isinstance(market_overrides, dict):
        return resolved
    bucket_override = market_overrides.get(edge_bucket)
    if not isinstance(bucket_override, dict):
        return resolved

    if "min_samples" in bucket_override:
        try:
            resolved["min_samples"] = max(1, int(bucket_override.get("min_samples")))
        except (TypeError, ValueError):
            pass
    if "min_precision" in bucket_override:
        try:
            resolved["min_precision"] = float(bucket_override.get("min_precision"))
        except (TypeError, ValueError):
            pass
    if "loss_streak_pause" in bucket_override:
        try:
            resolved["loss_streak_pause"] = int(bucket_override.get("loss_streak_pause"))
        except (TypeError, ValueError):
            pass
    if "hard_precision_gate" in bucket_override:
        resolved["hard_precision_gate"] = bool(bucket_override.get("hard_precision_gate"))
    if "hard_loss_streak_pause" in bucket_override:
        resolved["hard_loss_streak_pause"] = bool(
            bucket_override.get("hard_loss_streak_pause")
        )
    return resolved


def fetch_edge_bucket_performance_stats(
    *,
    model: str,
    version: str,
    history_days: int,
    window: int,
) -> tuple[
    dict[tuple[str, str], EdgeBucketPerformance],
    dict[str, EdgeBucketPerformance],
]:
    query = """
    SELECT
        p.market_code,
        ps.hit,
        ps.roi_unit,
        COALESCE(pra.edge_adjusted, ps.edge) AS edge_value,
        f.match_datetime_utc
    FROM predictions p
    JOIN prediction_scores ps
      ON ps.prediction_id = p.prediction_id
    JOIN fixtures f
      ON f.fixture_id = p.fixture_id
    LEFT JOIN prediction_risk_assessments pra
      ON pra.prediction_id = p.prediction_id
    WHERE p.model_name = %s
      AND p.model_version = %s
      AND p.market_code = ANY(%s)
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
    ORDER BY p.market_code ASC, f.match_datetime_utc DESC
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (model, version, list(FOCUS_MARKETS), history_days))
            rows = cur.fetchall()
    finally:
        conn.close()

    per_market_bucket_hits: dict[tuple[str, str], list[bool]] = {}
    per_market_bucket_roi: dict[tuple[str, str], list[float]] = {}
    per_bucket_hits: dict[str, list[bool]] = {}
    per_bucket_roi: dict[str, list[float]] = {}

    for market_code, hit, roi_unit, edge_value, _match_dt in rows:
        market = str(market_code or "")
        bucket = _edge_bucket_label(_safe_float(edge_value))
        if not market or not bucket:
            continue

        key = (market, bucket)
        hits = per_market_bucket_hits.setdefault(key, [])
        if len(hits) < window:
            hits.append(bool(hit))
        roi_rows = per_market_bucket_roi.setdefault(key, [])
        if len(roi_rows) < window:
            roi = _safe_float(roi_unit)
            if roi is not None:
                roi_rows.append(float(roi))

        bucket_hits = per_bucket_hits.setdefault(bucket, [])
        if len(bucket_hits) < window:
            bucket_hits.append(bool(hit))
        bucket_roi_rows = per_bucket_roi.setdefault(bucket, [])
        if len(bucket_roi_rows) < window:
            roi = _safe_float(roi_unit)
            if roi is not None:
                bucket_roi_rows.append(float(roi))

    by_market_bucket: dict[tuple[str, str], EdgeBucketPerformance] = {}
    for key, hits in per_market_bucket_hits.items():
        n = len(hits)
        precision = float(sum(1 for h in hits if h) / n) if n > 0 else None
        roi_rows = per_market_bucket_roi.get(key, [])
        mean_roi = float(sum(roi_rows) / len(roi_rows)) if roi_rows else None
        by_market_bucket[key] = EdgeBucketPerformance(
            n=n,
            precision=precision,
            loss_streak=_compute_loss_streak(hits),
            mean_roi_unit=mean_roi,
        )

    by_bucket: dict[str, EdgeBucketPerformance] = {}
    for bucket, hits in per_bucket_hits.items():
        n = len(hits)
        precision = float(sum(1 for h in hits if h) / n) if n > 0 else None
        roi_rows = per_bucket_roi.get(bucket, [])
        mean_roi = float(sum(roi_rows) / len(roi_rows)) if roi_rows else None
        by_bucket[bucket] = EdgeBucketPerformance(
            n=n,
            precision=precision,
            loss_streak=_compute_loss_streak(hits),
            mean_roi_unit=mean_roi,
        )

    return by_market_bucket, by_bucket


def write_edge_bucket_report(
    *,
    output_dir: Path,
    model: str,
    version: str,
    history_days: int,
    window: int,
    by_market_bucket: dict[tuple[str, str], EdgeBucketPerformance],
    by_bucket: dict[str, EdgeBucketPerformance],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"edge_bucket_report_{stamp}.json"
    rows_market = [
        {
            "market_code": market,
            "edge_bucket": bucket,
            "n": stats.n,
            "precision": stats.precision,
            "loss_streak": stats.loss_streak,
            "mean_roi_unit": stats.mean_roi_unit,
        }
        for (market, bucket), stats in sorted(by_market_bucket.items())
    ]
    rows_bucket = [
        {
            "edge_bucket": bucket,
            "n": stats.n,
            "precision": stats.precision,
            "loss_streak": stats.loss_streak,
            "mean_roi_unit": stats.mean_roi_unit,
        }
        for bucket, stats in sorted(by_bucket.items())
    ]
    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "model_name": model,
        "model_version": version,
        "history_days": int(history_days),
        "window": int(window),
        "edge_bucket_by_market": rows_market,
        "edge_bucket_global": rows_bucket,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _extract_numeric(value: Any) -> float | None:
    out = _safe_float(value)
    if out is not None:
        return out
    if isinstance(value, dict):
        for key in ("odds", "price", "value", "decimal"):
            if key in value:
                parsed = _safe_float(value.get(key))
                if parsed is not None:
                    return parsed
    return None


def _get_nested_odds(payload: dict[str, Any], line: str, side: str) -> float | None:
    line_payload = payload.get(line)
    if not isinstance(line_payload, dict):
        return None
    return _extract_numeric(line_payload.get(side))


def _extract_1x2_odds(
    payload: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    home = None
    draw = None
    away = None
    for key in ("1", "home", "h", "Home"):
        if key in payload:
            home = _extract_numeric(payload.get(key))
            if home is not None:
                break
    for key in ("X", "x", "draw", "d", "Draw"):
        if key in payload:
            draw = _extract_numeric(payload.get(key))
            if draw is not None:
                break
    for key in ("2", "away", "a", "Away"):
        if key in payload:
            away = _extract_numeric(payload.get(key))
            if away is not None:
                break
    return home, draw, away


def _clip_probability(prob: float) -> float:
    return float(min(max(prob, 0.0), 1.0))


def _is_tradable_market(policy: dict[str, Any], market_code: str) -> bool:
    configured = policy.get("tradable_markets")
    if not isinstance(configured, list) or not configured:
        return True
    allowed = {str(item) for item in configured}
    return market_code in allowed


def _risk_haircut_from_score(risk_score: float) -> float:
    # Convert a 0..100 score into a conservative probability haircut capped at 0.25.
    return float(min(0.25, max(0.0, risk_score) / 400.0))


def _kelly_fraction(prob: float, odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b
    return float(max(0.0, raw))


def _parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _bool_from_json(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _fallback_trace_used(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return False
        return normalized not in {"0", "false", "f", "no", "n", "none"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def assess_row(
    row: dict[str, Any],
    *,
    policy: dict[str, Any],
    tradable_summary: dict[str, Any] | None,
    calibration_by_league_market: dict[tuple[str, str], CalibrationStats],
    calibration_by_market: dict[str, CalibrationStats],
    performance_by_league_market: dict[tuple[str, str], RollingPerformance],
    performance_by_market: dict[str, RollingPerformance],
    edge_bucket_by_market: dict[tuple[str, str], EdgeBucketPerformance],
    edge_bucket_global: dict[str, EdgeBucketPerformance],
) -> dict[str, Any]:
    penalties = policy["penalties"]
    flags: list[str] = []
    breakdown: dict[str, float] = {}

    market_code = str(row["market_code"])
    league_code = str(row.get("league_code") or "")
    p_model = _clip_probability(float(row["p_model"]))
    metadata = _as_dict(row.get("metadata_json"))
    is_tradable = _is_tradable_market(policy, market_code)

    odds_rows = row.get("odds_rows", [])
    odds_resolution = resolve_odds_for_market(
        row.get("match_datetime_utc"), odds_rows, market_code
    )
    odds_used = odds_resolution.odds_used
    implied_probability = None
    if odds_used is not None and odds_used > 1.0:
        implied_probability = _clip_probability(1.0 / odds_used)
    else:
        odds_used = None

    risk_score = 0.0

    if is_tradable and odds_used is None:
        risk_score += float(penalties["missing_odds"])
        breakdown["missing_odds"] = float(penalties["missing_odds"])
        flags.append("missing_odds")
    elif not is_tradable:
        flags.append("non_tradable_market")

    fallback_trace = odds_resolution.fallback_used
    resolver_used_fallback = _fallback_trace_used(fallback_trace)
    metadata_used_fallback = _bool_from_json(metadata.get("fallback_used"))
    if resolver_used_fallback or metadata_used_fallback:
        risk_score += float(penalties["fallback_used"])
        breakdown["fallback_used"] = float(penalties["fallback_used"])
        flags.append("fallback_used")

    missing_count = int(metadata.get("features_missing_count", 0) or 0)
    if missing_count > 0:
        val = float(penalties["feature_missing_unit"]) * float(missing_count)
        risk_score += val
        breakdown["feature_missing_penalty"] = val
        flags.append("feature_missingness")

    home_sample = _safe_float(metadata.get("home_sample_size"))
    away_sample = _safe_float(metadata.get("away_sample_size"))
    min_sample = None
    if home_sample is not None and away_sample is not None:
        min_sample = min(home_sample, away_sample)
        if min_sample < 5:
            risk_score += float(penalties["low_sample_lt5"])
            breakdown["low_sample_lt5"] = float(penalties["low_sample_lt5"])
            flags.append("low_sample_lt5")
        elif min_sample < 8:
            risk_score += float(penalties["low_sample_lt8"])
            breakdown["low_sample_lt8"] = float(penalties["low_sample_lt8"])
            flags.append("low_sample_lt8")

    odds_snapshot = odds_resolution.snapshot_time_utc
    match_time = _parse_iso_datetime(row.get("match_datetime_utc"))
    if is_tradable and odds_snapshot is not None and match_time is not None:
        stale_hours = (match_time - odds_snapshot).total_seconds() / 3600.0
        if stale_hours >= 24.0:
            risk_score += float(penalties["stale_odds_24h"])
            breakdown["stale_odds_24h"] = float(penalties["stale_odds_24h"])
            flags.append("stale_odds_24h")
        elif stale_hours >= 6.0:
            risk_score += float(penalties["stale_odds_6h"])
            breakdown["stale_odds_6h"] = float(penalties["stale_odds_6h"])
            flags.append("stale_odds_6h")

    calib = calibration_by_league_market.get((league_code, market_code))
    if calib is None:
        calib = calibration_by_market.get(
            market_code, CalibrationStats(n=0, mean_brier=None)
        )
        if calib.n > 0:
            flags.append("calibration_global_fallback")
    min_calib_n = int(policy["calibration_min_samples"])
    if calib.n < min_calib_n:
        risk_score += float(penalties["calibration_low_sample"])
        breakdown["calibration_low_sample"] = float(penalties["calibration_low_sample"])
        flags.append("calibration_low_sample")
    elif calib.mean_brier is not None:
        very_poor = float(policy["calibration_very_poor_brier"])
        poor = float(policy["calibration_poor_brier"])
        if calib.mean_brier >= very_poor:
            risk_score += float(penalties["calibration_very_poor"])
            breakdown["calibration_very_poor"] = float(
                penalties["calibration_very_poor"]
            )
            flags.append("calibration_very_poor")
        elif calib.mean_brier >= poor:
            risk_score += float(penalties["calibration_poor"])
            breakdown["calibration_poor"] = float(penalties["calibration_poor"])
            flags.append("calibration_poor")

    perf = performance_by_league_market.get((league_code, market_code))
    if perf is None:
        perf = performance_by_market.get(
            market_code, RollingPerformance(n=0, precision=None, loss_streak=0)
        )
        if perf.n > 0:
            flags.append("performance_global_fallback")

    min_precision = float(policy.get("min_precision_rolling", 0.58))
    precision_min_samples = int(policy.get("precision_min_samples", 30))
    loss_streak_pause = int(policy.get("loss_streak_pause", 2))

    force_pass = False
    hard_precision_gate = bool(policy.get("hard_precision_gate", False))
    if (
        perf.n >= precision_min_samples
        and perf.precision is not None
        and perf.precision < min_precision
    ):
        risk_score += float(penalties.get("precision_gate_fail", 25.0))
        breakdown["precision_gate_fail"] = float(
            penalties.get("precision_gate_fail", 25.0)
        )
        flags.append("precision_gate_fail")
        if hard_precision_gate:
            force_pass = True

    hard_loss_streak_pause = bool(policy.get("hard_loss_streak_pause", True))
    if perf.loss_streak >= loss_streak_pause and loss_streak_pause > 0:
        risk_score += float(penalties.get("loss_streak_pause", 30.0))
        breakdown["loss_streak_pause"] = float(penalties.get("loss_streak_pause", 30.0))
        flags.append("loss_streak_pause")
        if hard_loss_streak_pause:
            force_pass = True

    edge_raw = None
    if implied_probability is not None:
        edge_raw = p_model - implied_probability

    edge_bucket = _edge_bucket_label(edge_raw)
    edge_bucket_stats: EdgeBucketPerformance | None = None
    edge_bucket_scope: str | None = None
    if edge_bucket:
        edge_bucket_stats = edge_bucket_by_market.get((market_code, edge_bucket))
        edge_bucket_scope = "market"
        if edge_bucket_stats is None:
            edge_bucket_stats = edge_bucket_global.get(edge_bucket)
            edge_bucket_scope = "global"
            if edge_bucket_stats and edge_bucket_stats.n > 0:
                flags.append("edge_bucket_global_fallback")

    edge_bucket_gate = _resolve_edge_bucket_gate(
        policy=policy, market_code=market_code, edge_bucket=edge_bucket
    )
    edge_bucket_min_samples = int(edge_bucket_gate["min_samples"])
    edge_bucket_min_precision = float(edge_bucket_gate["min_precision"])
    edge_bucket_loss_pause = int(edge_bucket_gate["loss_streak_pause"])
    if edge_bucket_stats is not None and edge_bucket_stats.n >= edge_bucket_min_samples:
        if (
            edge_bucket_stats.precision is not None
            and edge_bucket_stats.precision < edge_bucket_min_precision
        ):
            penalty = float(penalties.get("edge_bucket_precision_gate_fail", 20.0))
            risk_score += penalty
            breakdown["edge_bucket_precision_gate_fail"] = penalty
            flags.append("edge_bucket_precision_gate_fail")
            if bool(edge_bucket_gate["hard_precision_gate"]):
                force_pass = True
        if edge_bucket_loss_pause > 0 and edge_bucket_stats.loss_streak >= edge_bucket_loss_pause:
            penalty = float(penalties.get("edge_bucket_loss_streak_pause", 24.0))
            risk_score += penalty
            breakdown["edge_bucket_loss_streak_pause"] = penalty
            flags.append("edge_bucket_loss_streak_pause")
            if bool(edge_bucket_gate["hard_loss_streak_pause"]):
                force_pass = True

    risk_score = float(min(max(risk_score, 0.0), 100.0))
    haircut = _risk_haircut_from_score(risk_score)
    p_conservative = _clip_probability(p_model - haircut)
    edge_adjusted = None
    if implied_probability is not None:
        edge_adjusted = p_conservative - implied_probability

    min_watch = float(policy["min_edge_watch"])
    min_small = float(policy["min_edge_bet_small"])
    min_bet = float(policy["min_edge_bet"])

    if not is_tradable:
        action = "pass"
    elif force_pass:
        action = "pass"
    elif odds_used is None or edge_adjusted is None:
        action = "pass"
    elif risk_score >= 75.0:
        action = "pass"
    elif edge_adjusted < min_watch:
        action = "pass"
    elif edge_adjusted < min_small:
        action = "watch"
    elif edge_adjusted < min_bet or risk_score >= 50.0:
        action = "bet_small"
    elif risk_score <= 35.0:
        action = "bet"
    else:
        action = "bet_small"

    kelly = 0.0
    stake_fraction = 0.0
    if odds_used is not None and action in {"bet_small", "bet"}:
        kelly = _kelly_fraction(p_conservative, odds_used)
        scaled = kelly * float(policy["kelly_scale"]) * (1.0 - risk_score / 100.0)
        if action == "bet_small":
            scaled = min(scaled, float(policy["max_small_stake_fraction"]))
        else:
            scaled = min(scaled, float(policy["max_stake_fraction"]))
        stake_fraction = float(max(0.0, scaled))

    market_mode = "tradable" if is_tradable else "predict_only"

    return {
        "prediction_id": int(row["prediction_id"]),
        "fixture_id": int(row["fixture_id"]),
        "market_code": market_code,
        "model_name": str(row["model_name"]),
        "model_version": str(row["model_version"]),
        "p_model": p_model,
        "p_conservative": p_conservative,
        "implied_probability": implied_probability,
        "odds_used": odds_used,
        "edge_raw": edge_raw,
        "edge_adjusted": edge_adjusted,
        "risk_score": risk_score,
        "action": action,
        "kelly_fraction": kelly,
        "stake_fraction": stake_fraction,
        "risk_flags_json": flags,
        "metadata_json": {
            "penalty_breakdown": breakdown,
            "calibration_samples": calib.n,
            "calibration_mean_brier": calib.mean_brier,
            "rolling_precision_n": perf.n,
            "rolling_precision": perf.precision,
            "rolling_loss_streak": perf.loss_streak,
            "rolling_min_precision_gate": min_precision,
            "edge_bucket": edge_bucket,
            "edge_bucket_scope": edge_bucket_scope,
            "edge_bucket_n": edge_bucket_stats.n if edge_bucket_stats else 0,
            "edge_bucket_precision": (
                edge_bucket_stats.precision if edge_bucket_stats else None
            ),
            "edge_bucket_loss_streak": (
                edge_bucket_stats.loss_streak if edge_bucket_stats else None
            ),
            "edge_bucket_mean_roi_unit": (
                edge_bucket_stats.mean_roi_unit if edge_bucket_stats else None
            ),
            "edge_bucket_gate": edge_bucket_gate,
            "is_tradable_market": is_tradable,
            "market_mode": market_mode,
            "risk_haircut": haircut,
            "fallback_used": bool(resolver_used_fallback or metadata_used_fallback),
            "fallback_trace": fallback_trace,
            "odds_provider": odds_resolution.provider,
            "odds_snapshot_type": odds_resolution.snapshot_type,
            "odds_line_num": odds_resolution.line_num,
            "odds_field": odds_resolution.odds_field,
            "odds_snapshot_time_utc": odds_snapshot.isoformat()
            if odds_snapshot
            else None,
            "match_datetime_utc": match_time.isoformat() if match_time else None,
            "features_missing_count": missing_count,
            "min_team_sample_size": min_sample,
            "tradable_resolution": tradable_summary or {},
        },
    }


def upsert_assessments(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    query = """
    INSERT INTO prediction_risk_assessments (
        prediction_id,
        fixture_id,
        market_code,
        model_name,
        model_version,
        p_model,
        p_conservative,
        implied_probability,
        odds_used,
        edge_raw,
        edge_adjusted,
        risk_score,
        action,
        kelly_fraction,
        stake_fraction,
        risk_flags_json,
        metadata_json,
        created_at,
        updated_at
    )
    VALUES (
        %(prediction_id)s,
        %(fixture_id)s,
        %(market_code)s,
        %(model_name)s,
        %(model_version)s,
        %(p_model)s,
        %(p_conservative)s,
        %(implied_probability)s,
        %(odds_used)s,
        %(edge_raw)s,
        %(edge_adjusted)s,
        %(risk_score)s,
        %(action)s,
        %(kelly_fraction)s,
        %(stake_fraction)s,
        %(risk_flags_json)s,
        %(metadata_json)s,
        NOW(),
        NOW()
    )
    ON CONFLICT (prediction_id)
    DO UPDATE SET
        fixture_id = EXCLUDED.fixture_id,
        market_code = EXCLUDED.market_code,
        model_name = EXCLUDED.model_name,
        model_version = EXCLUDED.model_version,
        p_model = EXCLUDED.p_model,
        p_conservative = EXCLUDED.p_conservative,
        implied_probability = EXCLUDED.implied_probability,
        odds_used = EXCLUDED.odds_used,
        edge_raw = EXCLUDED.edge_raw,
        edge_adjusted = EXCLUDED.edge_adjusted,
        risk_score = EXCLUDED.risk_score,
        action = EXCLUDED.action,
        kelly_fraction = EXCLUDED.kelly_fraction,
        stake_fraction = EXCLUDED.stake_fraction,
        risk_flags_json = EXCLUDED.risk_flags_json,
        metadata_json = EXCLUDED.metadata_json,
        updated_at = NOW();
    """
    prepared: list[dict[str, Any]] = []
    for row in rows:
        prepared.append(
            {
                **row,
                "risk_flags_json": Json(row["risk_flags_json"]),
                "metadata_json": Json(row["metadata_json"]),
            }
        )

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.executemany(query, prepared)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def summarize_actions(rows: list[dict[str, Any]]) -> str:
    counts = {"pass": 0, "watch": 0, "bet_small": 0, "bet": 0}
    for row in rows:
        action = str(row["action"])
        if action in counts:
            counts[action] += 1
    return (
        f"pass={counts['pass']} "
        f"watch={counts['watch']} "
        f"bet_small={counts['bet_small']} "
        f"bet={counts['bet']}"
    )


def main() -> None:
    args = parse_args()
    ensure_risk_schema()
    policy = load_policy(args.policy_path)
    resolved_tradable, tradable_summary = resolve_tradable_markets(policy)
    policy["tradable_markets"] = sorted(resolved_tradable)
    print(
        "Resolved tradable markets: "
        f"{tradable_summary['resolved_tradable_count']} "
        f"(required={tradable_summary['required_count']}, "
        f"gating={tradable_summary['gating_eligible_count']}, "
        f"coverage_applied={tradable_summary['coverage_applied']})"
    )
    candidates = fetch_candidates(
        model=args.model,
        version=args.version,
        league=args.league,
        days=args.days,
        limit=args.limit,
    )
    if not candidates:
        print("No eligible scheduled predictions found for risk assessment.")
        return

    calibration_by_league_market, calibration_by_market = fetch_calibration_stats(
        model=args.model,
        version=args.version,
        history_days=args.history_days,
    )
    performance_by_league_market, performance_by_market = (
        fetch_recent_performance_stats(
            model=args.model,
            version=args.version,
            history_days=args.history_days,
            window=int(policy.get("precision_window", 60)),
        )
    )
    edge_bucket_by_market, edge_bucket_global = fetch_edge_bucket_performance_stats(
        model=args.model,
        version=args.version,
        history_days=args.history_days,
        window=int(policy.get("precision_window", 60)),
    )
    edge_bucket_report_dir_raw = str(
        policy.get("edge_bucket_report_dir") or DEFAULT_EDGE_BUCKET_REPORT_DIR
    ).strip()
    edge_bucket_report_path: str | None = None
    if edge_bucket_report_dir_raw:
        report_path = write_edge_bucket_report(
            output_dir=Path(edge_bucket_report_dir_raw),
            model=args.model,
            version=args.version,
            history_days=args.history_days,
            window=int(policy.get("precision_window", 60)),
            by_market_bucket=edge_bucket_by_market,
            by_bucket=edge_bucket_global,
        )
        edge_bucket_report_path = str(report_path)

    assessed = [
        assess_row(
            row,
            policy=policy,
            tradable_summary=tradable_summary,
            calibration_by_league_market=calibration_by_league_market,
            calibration_by_market=calibration_by_market,
            performance_by_league_market=performance_by_league_market,
            performance_by_market=performance_by_market,
            edge_bucket_by_market=edge_bucket_by_market,
            edge_bucket_global=edge_bucket_global,
        )
        for row in candidates
    ]

    if args.dry_run:
        print(f"[DRY RUN] Assessed {len(assessed)} rows: {summarize_actions(assessed)}")
        return

    written = upsert_assessments(assessed)
    if edge_bucket_report_path:
        print(f"Edge bucket report: {edge_bucket_report_path}")
    print(
        f"Assessed {len(assessed)} rows, upserted {written} risk rows: {summarize_actions(assessed)}"
    )


if __name__ == "__main__":
    main()
