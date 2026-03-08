from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
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
from src.modeling.v2.families.anytime.derive_markets import (
    derive_and_validate_anytime,
    derive_and_validate_anytime_phase_split,
)
from src.modeling.v2.families.anytime.features import build_anytime_features
from src.modeling.v2.io.artifact_identity import resolve_model_identity
from src.modeling.v2.io.baseline_registry import load_scope_markets


DEFAULT_ARTIFACT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "anytime"
DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "v2" / "predictions"
MODEL_NAME = "anytime_v2"
MODEL_VERSION = "markov_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict v2 anytime family markets.")
    parser.add_argument("--league", type=str, default=None, help="Optional league filter.")
    parser.add_argument("--days", type=int, default=3, help="Horizon in days.")
    parser.add_argument("--limit", type=int, default=None, help="Max fixtures.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing trained anytime artifacts.",
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
    return parser.parse_args()


def load_artifacts(
    artifact_dir: Path,
) -> tuple[dict[str, object], list[str], dict[str, float], int, str]:
    features = json.loads((artifact_dir / "features.json").read_text(encoding="utf-8"))
    imputation_payload = json.loads(
        (artifact_dir / "imputation.json").read_text(encoding="utf-8")
    )
    medians = {
        str(k): float(v)
        for k, v in (imputation_payload.get("global_medians") or {}).items()
        if isinstance(v, (int, float))
    }
    config_path = artifact_dir / "model_config.json"
    max_goals = 8
    path_version = "constant"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            max_goals = int(payload.get("max_goals", 8))
            path_version = str(payload.get("path_version", "constant"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            max_goals = 8
            path_version = "constant"

    models: dict[str, object]
    if path_version == "phase_split":
        models = {
            "home_p1": joblib.load(artifact_dir / "home_goals_p1_model.pkl"),
            "away_p1": joblib.load(artifact_dir / "away_goals_p1_model.pkl"),
            "home_p2": joblib.load(artifact_dir / "home_goals_p2_model.pkl"),
            "away_p2": joblib.load(artifact_dir / "away_goals_p2_model.pkl"),
        }
    else:
        models = {
            "home": joblib.load(artifact_dir / "home_goals_model.pkl"),
            "away": joblib.load(artifact_dir / "away_goals_model.pkl"),
        }

    return models, list(features), medians, max(4, int(max_goals)), path_version


def _apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for feat, med in medians.items():
        out[feat] = out[feat].fillna(float(med))
    return out


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
    models, features, medians, max_goals, path_version = load_artifacts(args.artifact_dir)
    model_name, model_version = resolve_model_identity(
        args.artifact_dir,
        default_model_name=MODEL_NAME,
        default_model_version=MODEL_VERSION,
        override_model_name=args.model_name,
        override_model_version=args.model_version,
    )
    scope_markets = set(load_scope_markets(args.scope))
    anytime_markets = {"h_1up", "a_1up", "h_2up", "a_2up"} & scope_markets

    fixtures = fetch_candidate_fixtures(
        days=args.days,
        league=args.league,
        limit=args.limit,
        backfill_days=None,
    )
    if fixtures.empty:
        print("No eligible fixtures for anytime v2 prediction.")
        return

    featured = add_derived_features(fixtures)
    featured = build_anytime_features(featured)
    for feat in features:
        if feat not in featured.columns:
            featured[feat] = np.nan
    scored = _apply_imputation(featured, medians)

    x = scored[features]
    if path_version == "phase_split":
        home_lambda_p1 = np.clip(models["home_p1"].predict(x), 0.01, 6.0)
        away_lambda_p1 = np.clip(models["away_p1"].predict(x), 0.01, 6.0)
        home_lambda_p2 = np.clip(models["home_p2"].predict(x), 0.01, 6.0)
        away_lambda_p2 = np.clip(models["away_p2"].predict(x), 0.01, 6.0)
    else:
        home_lambda = np.clip(models["home"].predict(x), 0.05, 8.0)
        away_lambda = np.clip(models["away"].predict(x), 0.05, 8.0)

    rows_csv: list[dict[str, Any]] = []
    rows_db: list[tuple[int, str, str, str, float, str]] = []
    if path_version == "phase_split":
        iterator = zip(
            scored.itertuples(index=False),
            home_lambda_p1,
            away_lambda_p1,
            home_lambda_p2,
            away_lambda_p2,
            strict=True,
        )
    else:
        iterator = zip(
            scored.itertuples(index=False),
            home_lambda,
            away_lambda,
            strict=True,
        )

    for item in iterator:
        if path_version == "phase_split":
            fixture, lh_p1, la_p1, lh_p2, la_p2 = item
        else:
            fixture, lh, la = item
        fixture_id = int(fixture.fixture_id)
        if path_version == "phase_split":
            lambda_home_p1 = float(lh_p1)
            lambda_away_p1 = float(la_p1)
            lambda_home_p2 = float(lh_p2)
            lambda_away_p2 = float(la_p2)
            markets = derive_and_validate_anytime_phase_split(
                lambda_home_p1=lambda_home_p1,
                lambda_away_p1=lambda_away_p1,
                lambda_home_p2=lambda_home_p2,
                lambda_away_p2=lambda_away_p2,
                max_goals=max_goals,
            )
        else:
            lambda_home = float(lh)
            lambda_away = float(la)
            markets = derive_and_validate_anytime(
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                max_goals=max_goals,
            )
        for market_code, prob in markets.items():
            if market_code not in anytime_markets:
                continue
            p_model = float(np.clip(prob, 0.001, 0.999))
            metadata: dict[str, Any] = {
                "model_family": "anytime_v2",
                "markov_max_goals": int(max_goals),
                "path_version": path_version,
                "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            }
            if path_version == "phase_split":
                metadata.update(
                    {
                        "derivation": "markov_ctmc_phase_split",
                        "lambda_home_p1_pred": lambda_home_p1,
                        "lambda_away_p1_pred": lambda_away_p1,
                        "lambda_home_p2_pred": lambda_home_p2,
                        "lambda_away_p2_pred": lambda_away_p2,
                    }
                )
            else:
                metadata.update(
                    {
                        "derivation": "markov_ctmc",
                        "lambda_home_pred": lambda_home,
                        "lambda_away_pred": lambda_away,
                    }
                )
            rows_csv.append(
                {
                    "fixture_id": fixture_id,
                    "market_code": market_code,
                    "model_name": model_name,
                    "model_version": model_version,
                    "p_model": p_model,
                    "path_version": path_version,
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
    out_path = args.out_dir / f"anytime_v2_predictions_{stamp}.csv"
    pd.DataFrame(rows_csv).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote v2 anytime predictions: {out_path}")

    if args.write_db:
        written = upsert_predictions(rows_db)
        print(f"Upserted {written} predictions into DB.")


if __name__ == "__main__":
    main()
