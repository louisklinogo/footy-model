from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.modeling.evaluation.score_market_outcomes_fixtures_first import compute_actual
from src.modeling.v2.calibration.methods import apply_binary_calibrator
from src.modeling.v2.eval.metrics import binary_metric_values
from src.modeling.v2.io.artifact_identity import load_artifact_metadata
from src.modeling.v2.market_family_registry import (
    DEFAULT_FAMILY_ARTIFACT_SOURCES,
    DEFAULT_FAMILY_MARKETS,
    SCORELINE_SUBFAMILY_MARKETS,
)

ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts" / "v2" / "family_replacement"
DEFAULT_SCORELINE_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline"
DEFAULT_CORNERS_DIR = ROOT_DIR / "model_artifacts" / "v2" / "corners"
DEFAULT_ANYTIME_DIR = ROOT_DIR / "model_artifacts" / "v2" / "anytime"

FAMILY_MARKETS: dict[str, set[str]] = {
    family: set(markets) for family, markets in DEFAULT_FAMILY_MARKETS.items()
}
FAMILY_ARTIFACT_SOURCES: dict[str, str] = dict(DEFAULT_FAMILY_ARTIFACT_SOURCES)
OVERLAP_MARKETS = set().union(*FAMILY_MARKETS.values())


def resolve_holdout_artifact_dir(
    family: str,
    requested_dir: Path,
    *,
    search_root: Path | None = None,
) -> Path:
    holdout_path = requested_dir / "holdout_predictions.csv"
    if holdout_path.exists():
        return requested_dir

    root = search_root or (ROOT_DIR / "model_artifacts" / "v2")
    if not root.exists():
        raise FileNotFoundError(f"No model artifact root found at {root}")

    live_prefix = f"live_verification_{family}_"
    family_prefix = f"{family}_"
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "holdout_predictions.csv").exists()
        and (path.name.startswith(live_prefix) or path.name.startswith(family_prefix))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find holdout_predictions.csv for family '{family}' under {requested_dir} or {root}"
        )

    def _rank(path: Path) -> tuple[int, str]:
        return (0 if path.name.startswith(live_prefix) else 1, path.name)

    return sorted(candidates, key=_rank)[0]


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _python_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _is_missing_metric(value: Any) -> bool:
    if value is None:
        return True
    return bool(pd.isna(value))


def _resolve_datetime_series(
    frame: pd.DataFrame,
    *candidate_columns: str,
) -> pd.Series | None:
    for column in candidate_columns:
        if column in frame.columns:
            return pd.to_datetime(frame[column], utc=True, errors="coerce", format="mixed")
    return None


def _datetime_bounds(series: pd.Series | None) -> tuple[str | None, str | None]:
    if series is None:
        return None, None
    valid = series.dropna()
    if valid.empty:
        return None, None
    return str(valid.min()), str(valid.max())


def metric_row(frame: pd.DataFrame, p_col: str) -> dict[str, Any]:
    out = binary_metric_values(frame["actual_int"].to_numpy(), frame[p_col].to_numpy())
    return {key: _python_value(value) for key, value in out.items()}


def _delta_metrics(live_metrics: dict[str, Any], challenger_metrics: dict[str, Any]) -> dict[str, float | None]:
    live_auc = live_metrics.get("auc")
    challenger_auc = challenger_metrics.get("auc")

    def _delta(metric_name: str) -> float | None:
        live_value = live_metrics.get(metric_name)
        challenger_value = challenger_metrics.get(metric_name)
        if _is_missing_metric(live_value) or _is_missing_metric(challenger_value):
            return None
        return float(challenger_value) - float(live_value)

    return {
        "auc": None if _is_missing_metric(live_auc) or _is_missing_metric(challenger_auc) else float(challenger_auc) - float(live_auc),
        "brier": _delta("brier"),
        "log_loss": _delta("log_loss"),
        "ece": _delta("ece"),
    }


def _normalize_challenger_frame(challenger: pd.DataFrame) -> pd.DataFrame:
    required = {"family", "market_code", "fixture_id", "p_challenger"}
    missing = required - set(challenger.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"challenger frame missing required columns: {missing_csv}")
    out = challenger.copy()
    if "holdout_y_true" not in out.columns and "y_true" in out.columns:
        out["holdout_y_true"] = out["y_true"]
    out["fixture_id"] = out["fixture_id"].astype(int)
    out["p_challenger"] = pd.to_numeric(out["p_challenger"], errors="coerce")
    if "p_challenger_raw" not in out.columns:
        out["p_challenger_raw"] = out["p_challenger"]
    else:
        out["p_challenger_raw"] = pd.to_numeric(out["p_challenger_raw"], errors="coerce")
    if "challenger_calibrated" not in out.columns:
        out["challenger_calibrated"] = False
    else:
        out["challenger_calibrated"] = out["challenger_calibrated"].astype(bool)
    if "challenger_calibration_method" not in out.columns:
        out["challenger_calibration_method"] = "identity"
    out["challenger_calibration_method"] = (
        out["challenger_calibration_method"].fillna("identity").astype(str)
    )
    if "match_datetime_utc" in out.columns:
        out["match_datetime_utc"] = pd.to_datetime(out["match_datetime_utc"], utc=True, errors="coerce", format="mixed")
    return out


def _normalize_live_frame(live: pd.DataFrame) -> pd.DataFrame:
    required = {"fixture_id", "market_code", "p_live"}
    missing = required - set(live.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"live frame missing required columns: {missing_csv}")
    out = live.copy()
    out["fixture_id"] = out["fixture_id"].astype(int)
    metadata = out.get("metadata_json", pd.Series([{}] * len(out))).apply(_json_dict)
    if "fallback_used" not in out.columns:
        out["fallback_used"] = metadata.apply(lambda row: bool(row.get("fallback_used", False)))
    else:
        out["fallback_used"] = out["fallback_used"].astype(bool)
    if "fallback_reason" not in out.columns:
        out["fallback_reason"] = metadata.apply(lambda row: row.get("fallback_reason"))
    if "prediction_model_family" not in out.columns:
        out["prediction_model_family"] = metadata.apply(lambda row: row.get("prediction_model_family"))
    return out


def _compute_actual_series(frame: pd.DataFrame) -> pd.Series:
    if "actual" in frame.columns:
        return pd.to_numeric(frame["actual"], errors="coerce")
    return pd.to_numeric(frame.apply(lambda row: compute_actual(row.to_dict()), axis=1), errors="coerce")


def _build_per_market_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, market_code), group in frame.groupby(["family", "market_code"], sort=True):
        live_metrics = metric_row(group, "p_live")
        challenger_metrics = metric_row(group, "p_challenger")
        deltas = _delta_metrics(live_metrics, challenger_metrics)
        rows.append(
            {
                "family": family,
                "market_code": market_code,
                "n": int(len(group)),
                "fixtures": int(group["fixture_id"].nunique()),
                "fallback_rows": int(group["fallback_used"].sum()),
                "fallback_rate": float(group["fallback_used"].mean()),
                "live_auc": live_metrics["auc"],
                "challenger_auc": challenger_metrics["auc"],
                "delta_auc": deltas["auc"],
                "live_brier": live_metrics["brier"],
                "challenger_brier": challenger_metrics["brier"],
                "delta_brier": deltas["brier"],
                "live_log_loss": live_metrics["log_loss"],
                "challenger_log_loss": challenger_metrics["log_loss"],
                "delta_log_loss": deltas["log_loss"],
                "live_ece": live_metrics["ece"],
                "challenger_ece": challenger_metrics["ece"],
                "delta_ece": deltas["ece"],
                "challenger_better_auc": None if deltas["auc"] is None else deltas["auc"] > 0.0,
                "challenger_better_brier": deltas["brier"] < 0.0,
                "challenger_better_log_loss": deltas["log_loss"] < 0.0,
            }
        )
    if rows:
        return pd.DataFrame(rows).sort_values(["family", "market_code"]).reset_index(drop=True)
    return pd.DataFrame(
        columns=[
            "family",
            "market_code",
            "n",
            "fixtures",
            "fallback_rows",
            "fallback_rate",
            "live_auc",
            "challenger_auc",
            "delta_auc",
            "live_brier",
            "challenger_brier",
            "delta_brier",
            "live_log_loss",
            "challenger_log_loss",
            "delta_log_loss",
            "live_ece",
            "challenger_ece",
            "delta_ece",
            "challenger_better_auc",
            "challenger_better_brier",
            "challenger_better_log_loss",
        ]
    )


def _segment_summary(name: str, frame: pd.DataFrame, per_market: pd.DataFrame, markets: set[str]) -> dict[str, Any]:
    live_metrics = metric_row(frame, "p_live")
    challenger_metrics = metric_row(frame, "p_challenger")
    market_slice = per_market[per_market["market_code"].isin(markets)]
    return {
        "segment": name,
        "rows": int(len(frame)),
        "fixtures": int(frame["fixture_id"].nunique()),
        "markets": sorted(markets),
        "live": live_metrics,
        "challenger": challenger_metrics,
        "delta": _delta_metrics(live_metrics, challenger_metrics),
        "fallback_rows": int(frame["fallback_used"].sum()),
        "live_model_rows": int((~frame["fallback_used"]).sum()),
        "fallback_rate": float(frame["fallback_used"].mean()) if len(frame) else 0.0,
        "market_win_counts": {
            "auc": int((market_slice["challenger_better_auc"] == True).sum()),
            "brier": int((market_slice["challenger_better_brier"] == True).sum()),
            "log_loss": int((market_slice["challenger_better_log_loss"] == True).sum()),
            "markets_total": int(len(market_slice)),
        },
    }


def _append_segment(
    segment_rows: list[dict[str, Any]],
    *,
    name: str,
    frame: pd.DataFrame,
    markets: set[str],
) -> None:
    segments = [
        (name, frame),
        (f"{name}_nonfallback", frame[~frame["fallback_used"]].copy()),
        (f"{name}_fallback_only", frame[frame["fallback_used"]].copy()),
    ]
    for segment_name, segment_frame in segments:
        segment_rows.append(
            _segment_summary(
                segment_name,
                segment_frame,
                _build_per_market_comparison(segment_frame),
                markets,
            )
        )


def load_family_calibrators(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    calibrators_path = Path(artifact_dir) / "calibrators.joblib"
    if not calibrators_path.exists():
        return {}
    try:
        payload = joblib.load(calibrators_path)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def apply_holdout_calibrators(
    holdout_frame: pd.DataFrame,
    calibrators: Mapping[str, dict[str, Any]],
) -> pd.DataFrame:
    out = holdout_frame.copy()
    out["p_challenger"] = pd.to_numeric(out["p_challenger"], errors="coerce")
    out["p_challenger_raw"] = out["p_challenger"]
    out["challenger_calibrated"] = False
    out["challenger_calibration_method"] = "identity"
    for market_code, calibrator in calibrators.items():
        mask = out["market_code"] == str(market_code)
        if not bool(mask.any()):
            continue
        method = str(calibrator.get("method") or "identity")
        try:
            out.loc[mask, "p_challenger"] = apply_binary_calibrator(
                calibrator,
                out.loc[mask, "p_challenger_raw"].to_numpy(dtype=float),
            )
        except Exception:
            continue
        out.loc[mask, "challenger_calibration_method"] = method
        out.loc[mask, "challenger_calibrated"] = method != "identity"
    return out


def build_live_replacement_report(
    *,
    challenger: pd.DataFrame,
    live: pd.DataFrame,
    actuals: pd.DataFrame,
    family_markets: Mapping[str, set[str]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = {key: set(value) for key, value in (family_markets or FAMILY_MARKETS).items()}
    overlap_markets = set().union(*families.values())
    artifact_summary = dict(challenger.attrs.get("artifact_summary", {}))
    challenger_norm = _normalize_challenger_frame(challenger)
    live_norm = _normalize_live_frame(live)

    challenger_norm = challenger_norm[challenger_norm["market_code"].isin(overlap_markets)].copy()
    live_norm = live_norm[live_norm["market_code"].isin(overlap_markets)].copy()
    actuals_norm = actuals.copy()
    if "fixture_id" not in actuals_norm.columns:
        raise ValueError("actuals frame missing required column: fixture_id")
    actuals_norm["fixture_id"] = actuals_norm["fixture_id"].astype(int)

    merged = challenger_norm.merge(
        live_norm[["fixture_id", "market_code", "p_live", "fallback_used", "fallback_reason", "prediction_model_family"]],
        on=["fixture_id", "market_code"],
        how="inner",
    )
    merged = merged.merge(actuals_norm, on="fixture_id", how="left")
    match_time = _resolve_datetime_series(
        merged,
        "match_datetime_utc",
        "match_datetime_utc_x",
        "match_datetime_utc_y",
    )
    if match_time is not None:
        merged["match_datetime_utc"] = match_time
    merged["actual"] = _compute_actual_series(merged)
    merged["actual_available"] = merged["actual"].notna()
    merged["live_row_source"] = np.where(merged["fallback_used"], "fallback", "live_model")
    if "holdout_y_true" in merged.columns:
        merged["holdout_matches_actual"] = np.where(
            merged["actual_available"],
            np.isclose(pd.to_numeric(merged["holdout_y_true"], errors="coerce"), merged["actual"]),
            False,
        )
    else:
        merged["holdout_matches_actual"] = False

    comparison = merged[merged["actual_available"]].copy()
    comparison["actual_int"] = comparison["actual"].astype(int)

    per_market = _build_per_market_comparison(comparison)

    segment_rows: list[dict[str, Any]] = []
    _append_segment(
        segment_rows,
        name="full_overlap",
        frame=comparison,
        markets=overlap_markets,
    )
    for family, markets in sorted(families.items()):
        family_frame = comparison[comparison["market_code"].isin(markets)].copy()
        _append_segment(
            segment_rows,
            name=f"{family}_overlap",
            frame=family_frame,
            markets=markets,
        )

    league_rows: list[dict[str, Any]] = []
    group_keys = ["family", "league_code"] if "league_code" in comparison.columns else ["family"]
    for keys, group in comparison.groupby(group_keys, sort=True):
        family = keys[0] if isinstance(keys, tuple) else keys
        league_code = keys[1] if isinstance(keys, tuple) else None
        live_metrics = metric_row(group, "p_live")
        challenger_metrics = metric_row(group, "p_challenger")
        deltas = _delta_metrics(live_metrics, challenger_metrics)
        league_rows.append(
            {
                "family": family,
                "league_code": league_code,
                "rows": int(len(group)),
                "fixtures": int(group["fixture_id"].nunique()),
                "fallback_rows": int(group["fallback_used"].sum()),
                "fallback_rate": float(group["fallback_used"].mean()),
                "live_auc": live_metrics["auc"],
                "challenger_auc": challenger_metrics["auc"],
                "delta_auc": deltas["auc"],
                "live_brier": live_metrics["brier"],
                "challenger_brier": challenger_metrics["brier"],
                "delta_brier": deltas["brier"],
                "live_log_loss": live_metrics["log_loss"],
                "challenger_log_loss": challenger_metrics["log_loss"],
                "delta_log_loss": deltas["log_loss"],
                "live_ece": live_metrics["ece"],
                "challenger_ece": challenger_metrics["ece"],
                "delta_ece": deltas["ece"],
            }
        )
    if league_rows:
        league_family = pd.DataFrame(league_rows).sort_values(["family", "league_code"], na_position="last").reset_index(drop=True)
    else:
        league_family = pd.DataFrame(
            columns=[
                "family",
                "league_code",
                "rows",
                "fixtures",
                "fallback_rows",
                "fallback_rate",
                "live_auc",
                "challenger_auc",
                "delta_auc",
                "live_brier",
                "challenger_brier",
                "delta_brier",
                "live_log_loss",
                "challenger_log_loss",
                "delta_log_loss",
                "live_ece",
                "challenger_ece",
                "delta_ece",
            ]
        )

    challenger_time = _resolve_datetime_series(challenger_norm, "match_datetime_utc")
    matched_time = _resolve_datetime_series(comparison, "match_datetime_utc")
    actuals_time = _resolve_datetime_series(actuals_norm, "match_datetime_utc")
    challenger_date_min, challenger_date_max = _datetime_bounds(challenger_time)
    matched_date_min, matched_date_max = _datetime_bounds(matched_time)
    actuals_date_min, actuals_date_max = _datetime_bounds(actuals_time)
    report = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "artifact_summary": {
            "live_runtime": {
                "model_dir": str(ROOT_DIR / "model_artifacts" / "market_models"),
                "prediction_script": "src/modeling/evaluation/predict_market_outcomes_fixtures_first.py",
            },
            "challenger_apply_calibrators": bool(
                artifact_summary.get("challenger_apply_calibrators", False)
            ),
            "challenger_families": artifact_summary.get("families", {}),
        },
        "cohort_summary": {
            "challenger_rows_loaded": int(len(challenger_norm)),
            "challenger_fixtures_loaded": int(challenger_norm["fixture_id"].nunique()),
            "live_prediction_rows_loaded": int(len(live_norm)),
            "live_prediction_fixtures_loaded": int(live_norm["fixture_id"].nunique()),
            "matched_rows_before_actual_filter": int(len(merged)),
            "matched_rows_scored": int(len(comparison)),
            "matched_fixtures_scored": int(comparison["fixture_id"].nunique()),
            "fallback_rows_matched": int(comparison["fallback_used"].sum()),
            "nonfallback_rows_matched": int((~comparison["fallback_used"]).sum()),
            "holdout_vs_actual": {
                "rows_with_actual": int(merged["actual_available"].sum()),
                "rows_matching_actual": int((merged["actual_available"] & merged["holdout_matches_actual"]).sum()),
                "rows_mismatching_actual": int((merged["actual_available"] & ~merged["holdout_matches_actual"]).sum()),
                "rows_without_actual": int((~merged["actual_available"]).sum()),
            },
            "challenger_calibration": {
                "rows_calibrated": int(comparison.get("challenger_calibrated", pd.Series(dtype=bool)).sum()),
                "rows_identity": int(
                    len(comparison)
                    - int(comparison.get("challenger_calibrated", pd.Series(dtype=bool)).sum())
                ),
                "markets_calibrated": sorted(
                    comparison.loc[
                        comparison.get("challenger_calibrated", pd.Series(False, index=comparison.index)).astype(bool),
                        "market_code",
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
            },
            "challenger_date_min": challenger_date_min,
            "challenger_date_max": challenger_date_max,
            "matched_date_min": matched_date_min,
            "matched_date_max": matched_date_max,
            "actuals_date_min": actuals_date_min,
            "actuals_date_max": actuals_date_max,
            "date_min": matched_date_min,
            "date_max": matched_date_max,
        },
        "segment_definitions": {
            "scoreline_subfamily_markets": {
                family: sorted(families.get(family, set()))
                for family in SCORELINE_SUBFAMILY_MARKETS
                if family in families
            },
        },
        "segment_summaries": segment_rows,
    }
    return report, per_market, comparison, league_family


def write_live_replacement_artifacts(
    *,
    output_dir: Path,
    report: dict[str, Any],
    per_market: pd.DataFrame,
    matched_rows: pd.DataFrame,
    league_family: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    per_market_path = output_dir / "per_market_comparison.csv"
    matched_path = output_dir / "matched_prediction_rows.csv"
    league_family_path = output_dir / "league_family_comparison.csv"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    per_market.to_csv(per_market_path, index=False)
    matched_cols = [
        col
        for col in [
            "family",
            "fixture_id",
            "market_code",
            "league_code",
            "match_datetime_utc",
            "actual_int",
            "p_live",
            "p_challenger_raw",
            "p_challenger",
            "challenger_calibrated",
            "challenger_calibration_method",
            "fallback_used",
            "fallback_reason",
            "live_row_source",
            "prediction_model_family",
        ]
        if col in matched_rows.columns
    ]
    matched_rows[matched_cols].to_csv(matched_path, index=False)
    league_family.to_csv(league_family_path, index=False)
    return {
        "summary_path": str(summary_path),
        "per_market_path": str(per_market_path),
        "matched_rows_path": str(matched_path),
        "league_family_path": str(league_family_path),
    }


def load_family_holdout_predictions(
    family_dirs: Mapping[str, Path],
    family_markets: Mapping[str, set[str]] | None = None,
    *,
    apply_calibrators: bool = False,
    artifact_sources: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    families = family_markets or FAMILY_MARKETS
    sources = artifact_sources or FAMILY_ARTIFACT_SOURCES
    frames: list[pd.DataFrame] = []
    resolved_dirs: dict[str, Path] = {}
    calibrators_by_source: dict[str, dict[str, Any]] = {}
    artifact_summary: dict[str, Any] = {
        "challenger_apply_calibrators": bool(apply_calibrators),
        "families": {},
    }
    for family, markets in families.items():
        source_family = str(sources.get(family, family))
        requested_dir = family_dirs.get(source_family)
        if requested_dir is None:
            raise KeyError(
                f"Missing artifact directory for family '{family}' (source family '{source_family}')"
            )
        artifact_dir = resolved_dirs.get(source_family)
        if artifact_dir is None:
            artifact_dir = resolve_holdout_artifact_dir(source_family, Path(requested_dir))
            resolved_dirs[source_family] = artifact_dir
        holdout_path = artifact_dir / "holdout_predictions.csv"
        df = pd.read_csv(holdout_path).rename(columns={"market": "market_code", "p_model": "p_challenger", "y_true": "holdout_y_true"})
        df = df[df["market_code"].isin(markets)].copy()
        calibrator_market_count: int | None = None
        if apply_calibrators:
            calibrators = calibrators_by_source.get(source_family)
            if calibrators is None:
                calibrators = load_family_calibrators(artifact_dir)
                calibrators_by_source[source_family] = calibrators
            calibrator_market_count = int(len(calibrators))
            df = apply_holdout_calibrators(df, calibrators)
        family_time = _resolve_datetime_series(df, "match_datetime_utc")
        family_date_min, family_date_max = _datetime_bounds(family_time)
        artifact_summary["families"][family] = {
            "source_family": source_family,
            "requested_dir": str(requested_dir),
            "resolved_artifact_dir": str(artifact_dir),
            "rows_loaded": int(len(df)),
            "fixtures_loaded": int(df["fixture_id"].nunique()) if "fixture_id" in df.columns else 0,
            "markets_loaded": sorted(df["market_code"].astype(str).unique().tolist()),
            "date_min": family_date_min,
            "date_max": family_date_max,
            "calibrators_applied": bool(apply_calibrators),
            "calibrator_market_count": calibrator_market_count,
            "artifact_metadata": load_artifact_metadata(artifact_dir),
        }
        df["family"] = family
        frames.append(
            df[
                [
                    col
                    for col in [
                        "family",
                        "market_code",
                        "fixture_id",
                        "holdout_y_true",
                        "p_challenger_raw",
                        "p_challenger",
                        "challenger_calibrated",
                        "challenger_calibration_method",
                        "league_code",
                        "match_datetime_utc",
                    ]
                    if col in df.columns
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out.attrs["artifact_summary"] = artifact_summary
    return out


def load_live_predictions_for_fixture_ids(*, fixture_ids: list[int], days: int, backfill_days: int, league: str | None, limit: int | None) -> pd.DataFrame:
    from src.modeling.evaluation.predict_market_outcomes_fixtures_first import add_derived_features, apply_imputation, build_prediction_rows, fetch_candidate_fixtures, load_artifacts as load_live_artifacts

    fixtures = fetch_candidate_fixtures(days=days, league=league, limit=limit, backfill_days=backfill_days)
    fixtures = fixtures[fixtures["fixture_id"].isin(fixture_ids)].copy()
    fixtures["match_datetime_utc"] = pd.to_datetime(fixtures["match_datetime_utc"], utc=True, errors="coerce")
    if "odds_snapshot_time_utc" in fixtures.columns:
        fixtures["odds_snapshot_time_utc"] = pd.to_datetime(fixtures["odds_snapshot_time_utc"], utc=True, errors="coerce")
    features, imputation, models, model_failures, multiclass_model, multiclass_markets = load_live_artifacts()
    featured = add_derived_features(fixtures)
    featured["features_missing_count"] = featured.reindex(columns=features).isna().sum(axis=1)
    scored = apply_imputation(featured, features, imputation)
    rows, _ = build_prediction_rows(scored, features, models, model_failures, multiclass_model, multiclass_markets)
    return pd.DataFrame(rows, columns=["fixture_id", "market_code", "model_name", "model_version", "p_live", "metadata_json"])


def load_actuals_for_fixture_ids(fixture_ids: list[int]) -> pd.DataFrame:
    from src.db.db_utils import connect_db

    conn = connect_db()
    try:
        actuals = pd.read_sql(
            """
            SELECT f.fixture_id, f.match_datetime_utc, fr.home_goals, fr.away_goals,
                   fs.h_corners, fs.a_corners,
                   ils.home_led_by_1_any, ils.away_led_by_1_any,
                   ils.home_led_by_2_any, ils.away_led_by_2_any
            FROM fixtures f
            JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
            LEFT JOIN fixture_stats_premium fs ON fs.fixture_id = f.fixture_id
            LEFT JOIN fixture_incident_lead_states ils ON ils.fixture_id = f.fixture_id
            WHERE f.fixture_id = ANY(%s)
            """,
            conn,
            params=(fixture_ids,),
        )
    finally:
        conn.close()
    return actuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v2 family holdout predictions against the current live market-outcome stack.")
    parser.add_argument("--scoreline-dir", type=Path, default=DEFAULT_SCORELINE_DIR)
    parser.add_argument("--corners-dir", type=Path, default=DEFAULT_CORNERS_DIR)
    parser.add_argument("--anytime-dir", type=Path, default=DEFAULT_ANYTIME_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--backfill-days", type=int, default=60)
    parser.add_argument("--league", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--apply-calibrators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-family calibrators.joblib to challenger holdout probabilities before comparison. Use --no-apply-calibrators to compare raw holdout probabilities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family_dirs = {"scoreline": Path(args.scoreline_dir), "corners": Path(args.corners_dir), "anytime": Path(args.anytime_dir)}
    challenger = load_family_holdout_predictions(
        family_dirs,
        apply_calibrators=bool(args.apply_calibrators),
    )
    fixture_ids = sorted(challenger["fixture_id"].astype(int).unique().tolist())
    live = load_live_predictions_for_fixture_ids(
        fixture_ids=fixture_ids,
        days=int(args.days),
        backfill_days=int(args.backfill_days),
        league=args.league,
        limit=args.limit,
    )
    actuals = load_actuals_for_fixture_ids(fixture_ids)
    report, per_market, matched_rows, league_family = build_live_replacement_report(
        challenger=challenger,
        live=live,
        actuals=actuals,
    )
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.output_root) / (args.run_name or f"live_replacement_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}")
    artifacts = write_live_replacement_artifacts(
        output_dir=output_dir,
        report=report,
        per_market=per_market,
        matched_rows=matched_rows,
        league_family=league_family,
    )
    print(json.dumps({"output_dir": str(output_dir), **artifacts, "cohort_summary": report["cohort_summary"]}, indent=2))


if __name__ == "__main__":
    main()