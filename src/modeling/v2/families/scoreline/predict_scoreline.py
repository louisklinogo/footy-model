from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from math import factorial
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.evaluation.predict_market_outcomes_fixtures_first import (
    add_derived_features,
    fetch_candidate_fixtures,
)
from src.modeling.v2.calibration.methods import apply_binary_calibrator
from src.modeling.v2.families.scoreline.derive_markets import derive_and_validate
from src.modeling.v2.families.scoreline.residual_utils import (
    apply_residual_bundle,
    build_residual_source_frame,
    prepare_residual_feature_frame,
)
from src.modeling.v2.families.scoreline.total_intensity import (
    apply_total_intensity_correction,
    build_identity_total_intensity_correction,
)
from src.modeling.v2.io.artifact_identity import resolve_model_identity
from src.modeling.v2.io.baseline_registry import load_scope_markets


DEFAULT_ARTIFACT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline"
DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "v2" / "predictions"
MODEL_NAME = "scoreline_v2"
MODEL_VERSION = "poisson_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict v2 scoreline family markets.")
    parser.add_argument("--league", type=str, default=None, help="Optional league filter.")
    parser.add_argument("--days", type=int, default=3, help="Horizon in days.")
    parser.add_argument("--limit", type=int, default=None, help="Max fixtures.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing trained scoreline artifacts.",
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=DEFAULT_SCOPE,
        help="Market scope file path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Prediction artifact output directory.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=10,
        help="Scoreline matrix truncation cap.",
    )
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Upsert predictions into predictions table.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional override for model_name. Use to bundle multiple v2 families under one runtime identity.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Optional override for model_version. Defaults to artifact metadata when present.",
    )
    parser.add_argument(
        "--apply-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply scoreline family calibrators.joblib when present.",
    )
    return parser.parse_args()


def _poisson_pmf(lmbda: float, k: int) -> float:
    lmbda = max(1e-6, float(lmbda))
    return float(np.exp(-lmbda) * (lmbda**k) / float(factorial(k)))


def _score_matrix_independent_poisson(
    lambda_home: float, lambda_away: float, max_goals: int
) -> np.ndarray:
    home = np.array([_poisson_pmf(lambda_home, g) for g in range(max_goals + 1)], dtype=float)
    away = np.array([_poisson_pmf(lambda_away, g) for g in range(max_goals + 1)], dtype=float)
    mat = np.outer(home, away)
    total = float(mat.sum())
    if total <= 0.0:
        return np.full((max_goals + 1, max_goals + 1), 1.0 / ((max_goals + 1) ** 2))
    return mat / total


def _apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for feat, med in medians.items():
        out[feat] = out[feat].fillna(float(med))
    return out


def load_artifacts(
    artifact_dir: Path,
) -> tuple[
    object,
    object,
    list[str],
    dict[str, float],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any],
]:
    home_model = joblib.load(artifact_dir / "home_goals_model.pkl")
    away_model = joblib.load(artifact_dir / "away_goals_model.pkl")
    features = json.loads((artifact_dir / "features.json").read_text(encoding="utf-8"))
    imputation_payload = json.loads(
        (artifact_dir / "imputation.json").read_text(encoding="utf-8")
    )
    medians = {
        str(k): float(v)
        for k, v in (imputation_payload.get("global_medians") or {}).items()
        if isinstance(v, (int, float))
    }
    total_intensity_correction = build_identity_total_intensity_correction()
    correction_path = artifact_dir / "total_intensity_correction.json"
    if correction_path.exists():
        payload = json.loads(correction_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            total_intensity_correction.update(payload)
    residual_bundle: dict[str, Any] | None = None
    residual_path = artifact_dir / "residual_models.joblib"
    if residual_path.exists():
        payload = joblib.load(residual_path)
        residual_bundle = payload if isinstance(payload, dict) else None
    calibrators: dict[str, Any] = {}
    calibrator_path = artifact_dir / "calibrators.joblib"
    if calibrator_path.exists():
        payload = joblib.load(calibrator_path)
        calibrators = payload if isinstance(payload, dict) else {}
    return (
        home_model,
        away_model,
        list(features),
        medians,
        total_intensity_correction,
        residual_bundle,
        calibrators,
    )


def _apply_market_calibrators(
    market_frame: pd.DataFrame,
    calibrators: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str]]:
    out = market_frame.copy()
    applied: dict[str, str] = {}
    for market, calibrator in calibrators.items():
        if market not in out.columns or not isinstance(calibrator, dict):
            continue
        out[market] = apply_binary_calibrator(calibrator, out[market].to_numpy(dtype=float))
        applied[str(market)] = str(calibrator.get("method") or "identity")
    return out, applied


def _presence_mask(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return np.zeros(len(frame), dtype=bool)
    mask = np.ones(len(frame), dtype=bool)
    for column in existing:
        mask &= frame[column].notna().to_numpy(dtype=bool)
    return mask


def _scoreline_feature_tiers(frame: pd.DataFrame) -> np.ndarray:
    direct_odds = _presence_mask(
        frame,
        ("odds_over_15", "odds_under_15", "odds_over_35", "odds_under_35"),
    )
    refinement = _presence_mask(frame, ("adj_lambda_home_final", "adj_lambda_away_final"))
    availability = _presence_mask(
        frame,
        ("home_availability_known", "away_availability_known", "home_lineup_known", "away_lineup_known"),
    )
    structural = _presence_mask(
        frame,
        ("lambda_home_l1", "lambda_away_l1", "home_rolling_xg", "away_rolling_xg"),
    )
    tiers = np.full(len(frame), "D", dtype=object)
    tiers[structural] = "C"
    tiers[direct_odds] = "B"
    tiers[direct_odds & refinement & availability] = "A"
    return tiers


def upsert_predictions(rows: list[tuple[int, str, str, str, float, str]]) -> int:
    if not rows:
        return 0
    query = """
    INSERT INTO predictions (
        fixture_id,
        market_code,
        model_name,
        model_version,
        p_model,
        metadata_json,
        created_at
    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, NOW())
    ON CONFLICT (fixture_id, market_code, model_name, model_version)
    DO UPDATE SET
        p_model = EXCLUDED.p_model,
        metadata_json = EXCLUDED.metadata_json,
        created_at = NOW();
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.executemany(query, rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def main() -> None:
    args = parse_args()
    (
        home_model,
        away_model,
        features,
        medians,
        total_intensity_correction,
        residual_bundle,
        calibrators,
    ) = load_artifacts(args.artifact_dir)
    model_name, model_version = resolve_model_identity(
        args.artifact_dir,
        default_model_name=MODEL_NAME,
        default_model_version=MODEL_VERSION,
        override_model_name=args.model_name,
        override_model_version=args.model_version,
    )
    scope_markets = set(load_scope_markets(args.scope))

    fixtures = fetch_candidate_fixtures(
        days=args.days,
        league=args.league,
        limit=args.limit,
        backfill_days=None,
    )
    if fixtures.empty:
        print("No eligible fixtures for scoreline v2 prediction.")
        return

    fixtures["match_datetime_utc"] = pd.to_datetime(
        fixtures["match_datetime_utc"], utc=True, errors="coerce"
    )
    featured = add_derived_features(fixtures, include_external_team_match_context=True)
    for feat in features:
        if feat not in featured.columns:
            featured[feat] = np.nan
    feature_tiers = _scoreline_feature_tiers(featured)
    scored = _apply_imputation(featured, medians)

    x = scored[features]
    lambda_home_raw = np.clip(home_model.predict(x), 0.05, 8.0)
    lambda_away_raw = np.clip(away_model.predict(x), 0.05, 8.0)
    lambda_home, lambda_away = apply_total_intensity_correction(
        lambda_home=lambda_home_raw,
        lambda_away=lambda_away_raw,
        correction=total_intensity_correction,
    )

    base_market_rows: list[dict[str, float]] = []
    for lh, la in zip(lambda_home, lambda_away, strict=True):
        matrix = _score_matrix_independent_poisson(
            lambda_home=float(lh), lambda_away=float(la), max_goals=max(1, int(args.max_goals))
        )
        base_market_rows.append(derive_and_validate(matrix))
    base_market_frame = pd.DataFrame(base_market_rows)
    final_market_frame = base_market_frame.copy()
    publish_residual_overlay = bool(residual_bundle and residual_bundle.get("publish_to_canonical_surface"))
    if publish_residual_overlay:
        residual_source = build_residual_source_frame(
            raw_frame=featured,
            base_market_frame=base_market_frame,
            lambda_home=np.asarray(lambda_home, dtype=float),
            lambda_away=np.asarray(lambda_away, dtype=float),
        )
        residual_features, _, _ = prepare_residual_feature_frame(
            residual_source,
            feature_columns=list(residual_bundle.get("feature_columns") or []),
            imputation=dict(residual_bundle.get("imputation") or {}),
        )
        final_market_frame = apply_residual_bundle(
            base_market_frame=base_market_frame,
            residual_features=residual_features,
            residual_bundle=residual_bundle,
            direct_market_overrides=True,
        )
    calibration_methods: dict[str, str] = {}
    if bool(args.apply_calibration) and calibrators:
        final_market_frame, calibration_methods = _apply_market_calibrators(
            final_market_frame,
            calibrators,
        )

    rows_csv: list[dict[str, Any]] = []
    rows_db: list[tuple[int, str, str, str, float, str]] = []
    for row_idx, (fixture, lh, la) in enumerate(zip(
        scored.itertuples(index=False),
        lambda_home,
        lambda_away,
        strict=True,
    )):
        fixture_id = int(fixture.fixture_id)
        lh = float(lh)
        la = float(la)
        markets = final_market_frame.iloc[row_idx].to_dict()
        base_markets = base_market_frame.iloc[row_idx].to_dict()
        for market_code, prob in markets.items():
            if market_code not in scope_markets:
                continue
            p_model = float(np.clip(prob, 0.001, 0.999))
            metadata = {
                "model_family": "scoreline_v2",
                "lambda_home_pred": lh,
                "lambda_away_pred": la,
                "lambda_home_raw": float(lambda_home_raw[row_idx]),
                "lambda_away_raw": float(lambda_away_raw[row_idx]),
                "total_lambda_raw": float(lambda_home_raw[row_idx] + lambda_away_raw[row_idx]),
                "total_lambda_corrected": float(lh + la),
                "total_intensity_multiplier": float(total_intensity_correction.get("multiplier") or 1.0),
                "max_goals": int(args.max_goals),
                "residual_overlay_applied": publish_residual_overlay,
                "calibration_applied": market_code in calibration_methods,
                "calibration_method": calibration_methods.get(market_code),
                "feature_tier": str(feature_tiers[row_idx]),
                "p_model_base": float(np.clip(base_markets.get(market_code, p_model), 0.001, 0.999)),
                "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            }
            rows_csv.append(
                {
                    "fixture_id": fixture_id,
                    "market_code": market_code,
                    "model_name": model_name,
                    "model_version": model_version,
                    "p_model": p_model,
                    "p_model_base": float(np.clip(base_markets.get(market_code, p_model), 0.001, 0.999)),
                    "lambda_home_pred": lh,
                    "lambda_away_pred": la,
                    "lambda_home_raw": float(lambda_home_raw[row_idx]),
                    "lambda_away_raw": float(lambda_away_raw[row_idx]),
                    "total_lambda_raw": float(lambda_home_raw[row_idx] + lambda_away_raw[row_idx]),
                    "total_lambda_corrected": float(lh + la),
                    "total_intensity_multiplier": float(total_intensity_correction.get("multiplier") or 1.0),
                    "feature_tier": str(feature_tiers[row_idx]),
                    "residual_overlay_applied": publish_residual_overlay,
                    "calibration_method": calibration_methods.get(market_code),
                }
            )
            rows_db.append(
                (
                    fixture_id,
                    market_code,
                    model_name,
                    model_version,
                    p_model,
                    json.dumps(metadata),
                )
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"scoreline_v2_predictions_{stamp}.csv"
    pd.DataFrame(rows_csv).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote v2 scoreline predictions: {out_path}")

    if args.write_db:
        written = upsert_predictions(rows_db)
        print(f"Upserted {written} predictions into DB.")


if __name__ == "__main__":
    main()
