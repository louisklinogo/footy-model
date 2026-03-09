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
from src.modeling.v2.calibration.methods import apply_binary_calibrator
from src.modeling.v2.families.anytime.derive_markets import (
    derive_and_validate_anytime,
    derive_and_validate_anytime_direct_monotone,
    derive_and_validate_anytime_phase_split,
    derive_and_validate_anytime_state_ladder,
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
) -> tuple[dict[str, object], list[str], dict[str, float], int, str, dict[str, Any]]:
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
    extra_config: dict[str, Any] = {}
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            max_goals = int(payload.get("max_goals", 8))
            path_version = str(payload.get("path_version", "constant"))
            extra_config = payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            max_goals = 8
            path_version = "constant"
            extra_config = {}

    models: dict[str, object]
    if path_version == "direct_monotone":
        models = {
            "home": joblib.load(artifact_dir / "home_goals_model.pkl"),
            "away": joblib.load(artifact_dir / "away_goals_model.pkl"),
            "home_1up_head": joblib.load(artifact_dir / "home_1up_head.pkl"),
            "away_1up_head": joblib.load(artifact_dir / "away_1up_head.pkl"),
            "home_2up_head": joblib.load(artifact_dir / "home_2up_head.pkl"),
            "away_2up_head": joblib.load(artifact_dir / "away_2up_head.pkl"),
            "direct_monotone_head_calibrators": joblib.load(artifact_dir / "direct_monotone_head_calibrators.joblib")
            if (artifact_dir / "direct_monotone_head_calibrators.joblib").exists()
            else {},
        }
    elif path_version == "state_ladder":
        models = {
            "home_p1": joblib.load(artifact_dir / "home_goals_p1_model.pkl"),
            "away_p1": joblib.load(artifact_dir / "away_goals_p1_model.pkl"),
            "home_p2": joblib.load(artifact_dir / "home_goals_p2_model.pkl"),
            "away_p2": joblib.load(artifact_dir / "away_goals_p2_model.pkl"),
            "home_1up_head": joblib.load(artifact_dir / "home_1up_head.pkl"),
            "away_1up_head": joblib.load(artifact_dir / "away_1up_head.pkl"),
            "home_2up_cond_head": joblib.load(artifact_dir / "home_2up_cond_head.pkl"),
            "away_2up_cond_head": joblib.load(artifact_dir / "away_2up_cond_head.pkl"),
            "state_ladder_head_calibrators": joblib.load(artifact_dir / "state_ladder_head_calibrators.joblib")
            if (artifact_dir / "state_ladder_head_calibrators.joblib").exists()
            else {},
        }
        if str(extra_config.get("state_ladder_prior_version", "phase_split")) == "constant_model":
            models["home_prior"] = joblib.load(artifact_dir / "home_prior_model.pkl")
            models["away_prior"] = joblib.load(artifact_dir / "away_prior_model.pkl")
    elif path_version == "phase_split":
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

    return models, list(features), medians, max(4, int(max_goals)), path_version, extra_config


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
    models, features, medians, max_goals, path_version, extra_config = load_artifacts(args.artifact_dir)
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
    if path_version == "direct_monotone":
        home_lambda = np.clip(models["home"].predict(x), 0.05, 8.0)
        away_lambda = np.clip(models["away"].predict(x), 0.05, 8.0)
    elif path_version in {"phase_split", "state_ladder"}:
        home_lambda_p1 = np.clip(models["home_p1"].predict(x), 0.01, 6.0)
        away_lambda_p1 = np.clip(models["away_p1"].predict(x), 0.01, 6.0)
        home_lambda_p2 = np.clip(models["home_p2"].predict(x), 0.01, 6.0)
        away_lambda_p2 = np.clip(models["away_p2"].predict(x), 0.01, 6.0)
    else:
        home_lambda = np.clip(models["home"].predict(x), 0.05, 8.0)
        away_lambda = np.clip(models["away"].predict(x), 0.05, 8.0)

    rows_csv: list[dict[str, Any]] = []
    rows_db: list[tuple[int, str, str, str, float, str]] = []
    if path_version == "direct_monotone":
        prior_rows = [
            derive_and_validate_anytime(
                lambda_home=float(lh),
                lambda_away=float(la),
                max_goals=max_goals,
            )
            for lh, la in zip(home_lambda, away_lambda, strict=True)
        ]
        prior_frame = pd.DataFrame(prior_rows).rename(
            columns={market: f"prior_{market}" for market in ["h_1up", "a_1up", "h_2up", "a_2up"]}
        )
        head_x = pd.concat([x.reset_index(drop=True), prior_frame.reset_index(drop=True)], axis=1)
        direct_monotone_blend = extra_config.get("direct_monotone_blend") or {}
        head_calibrators = models.get("direct_monotone_head_calibrators") or {}

        def _head_probs(name: str) -> np.ndarray:
            probs = models[name].predict_proba(head_x)
            classes = [int(c) for c in getattr(models[name], "classes_", [0, 1])]
            if probs.shape[1] == 1:
                return np.full(len(head_x), 1.0 if classes and classes[0] == 1 else 0.0)
            idx = classes.index(1) if 1 in classes else probs.shape[1] - 1
            return np.clip(probs[:, idx], 0.001, 0.999)

        raw_home_1up = _head_probs("home_1up_head")
        raw_away_1up = _head_probs("away_1up_head")
        raw_home_2up = _head_probs("home_2up_head")
        raw_away_2up = _head_probs("away_2up_head")
        cal_home_1up = apply_binary_calibrator(head_calibrators.get("home_1up", {"method": "identity"}), raw_home_1up)
        cal_away_1up = apply_binary_calibrator(head_calibrators.get("away_1up", {"method": "identity"}), raw_away_1up)
        cal_home_2up = apply_binary_calibrator(head_calibrators.get("home_2up", {"method": "identity"}), raw_home_2up)
        cal_away_2up = apply_binary_calibrator(head_calibrators.get("away_2up", {"method": "identity"}), raw_away_2up)
        prior_home_1up = prior_frame["prior_h_1up"].to_numpy(dtype=float)
        prior_away_1up = prior_frame["prior_a_1up"].to_numpy(dtype=float)
        prior_home_2up = prior_frame["prior_h_2up"].to_numpy(dtype=float)
        prior_away_2up = prior_frame["prior_a_2up"].to_numpy(dtype=float)
        home_1up_prob = np.clip(prior_home_1up + float(direct_monotone_blend.get("home_1up", 0.35)) * (cal_home_1up - prior_home_1up), 0.001, 0.999)
        away_1up_prob = np.clip(prior_away_1up + float(direct_monotone_blend.get("away_1up", 0.35)) * (cal_away_1up - prior_away_1up), 0.001, 0.999)
        home_2up_prob = np.clip(prior_home_2up + float(direct_monotone_blend.get("home_2up", 0.35)) * (cal_home_2up - prior_home_2up), 0.001, 0.999)
        away_2up_prob = np.clip(prior_away_2up + float(direct_monotone_blend.get("away_2up", 0.35)) * (cal_away_2up - prior_away_2up), 0.001, 0.999)
        iterator = zip(
            scored.itertuples(index=False),
            home_lambda,
            away_lambda,
            home_1up_prob,
            away_1up_prob,
            home_2up_prob,
            away_2up_prob,
            strict=True,
        )
    elif path_version == "phase_split":
        iterator = zip(
            scored.itertuples(index=False),
            home_lambda_p1,
            away_lambda_p1,
            home_lambda_p2,
            away_lambda_p2,
            strict=True,
        )
    elif path_version == "state_ladder":
        prior_version = str(extra_config.get("state_ladder_prior_version", "phase_split"))
        prior_home_total = None
        prior_away_total = None
        if prior_version == "constant_model":
            prior_home_total = np.clip(models["home_prior"].predict(x), 0.01, 6.0)
            prior_away_total = np.clip(models["away_prior"].predict(x), 0.01, 6.0)
        prior_rows = [
            (
                derive_and_validate_anytime(
                    lambda_home=(float(ph) if prior_home_total is not None else float(lh_p1) + float(lh_p2)),
                    lambda_away=(float(pa) if prior_away_total is not None else float(la_p1) + float(la_p2)),
                    max_goals=max_goals,
                )
                if prior_version in {"constant", "constant_model"}
                else derive_and_validate_anytime_phase_split(
                    lambda_home_p1=float(lh_p1),
                    lambda_away_p1=float(la_p1),
                    lambda_home_p2=float(lh_p2),
                    lambda_away_p2=float(la_p2),
                    max_goals=max_goals,
                )
            )
            for lh_p1, la_p1, lh_p2, la_p2, ph, pa in zip(
                home_lambda_p1,
                away_lambda_p1,
                home_lambda_p2,
                away_lambda_p2,
                prior_home_total if prior_home_total is not None else home_lambda_p1,
                prior_away_total if prior_away_total is not None else away_lambda_p1,
                strict=True,
            )
        ]
        prior_frame = pd.DataFrame(prior_rows).rename(
            columns={market: f"prior_{market}" for market in ["h_1up", "a_1up", "h_2up", "a_2up"]}
        )
        head_x = pd.concat([x.reset_index(drop=True), prior_frame.reset_index(drop=True)], axis=1)
        state_ladder_blend = extra_config.get("state_ladder_blend") or {}
        head_calibrators = models.get("state_ladder_head_calibrators") or {}

        def _conditional_from_joint(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
            cond = np.divide(numer, np.maximum(denom, 1e-6))
            return np.clip(cond, 0.001, 0.999)

        def _head_probs(name: str) -> np.ndarray:
            probs = models[name].predict_proba(head_x)
            classes = [int(c) for c in getattr(models[name], "classes_", [0, 1])]
            if probs.shape[1] == 1:
                return np.full(len(head_x), 1.0 if classes and classes[0] == 1 else 0.0)
            idx = classes.index(1) if 1 in classes else probs.shape[1] - 1
            return np.clip(probs[:, idx], 0.001, 0.999)

        raw_home_1up = _head_probs("home_1up_head")
        raw_away_1up = _head_probs("away_1up_head")
        raw_home_2up_cond = _head_probs("home_2up_cond_head")
        raw_away_2up_cond = _head_probs("away_2up_cond_head")
        cal_home_1up = apply_binary_calibrator(head_calibrators.get("home_1up", {"method": "identity"}), raw_home_1up)
        cal_away_1up = apply_binary_calibrator(head_calibrators.get("away_1up", {"method": "identity"}), raw_away_1up)
        cal_home_2up_cond = apply_binary_calibrator(head_calibrators.get("home_2up_cond", {"method": "identity"}), raw_home_2up_cond)
        cal_away_2up_cond = apply_binary_calibrator(head_calibrators.get("away_2up_cond", {"method": "identity"}), raw_away_2up_cond)
        prior_home_1up = prior_frame["prior_h_1up"].to_numpy(dtype=float)
        prior_away_1up = prior_frame["prior_a_1up"].to_numpy(dtype=float)
        prior_home_2up_cond = _conditional_from_joint(
            prior_frame["prior_h_2up"].to_numpy(dtype=float),
            prior_home_1up,
        )
        prior_away_2up_cond = _conditional_from_joint(
            prior_frame["prior_a_2up"].to_numpy(dtype=float),
            prior_away_1up,
        )
        home_1up_prob = np.clip(
            prior_home_1up + float(state_ladder_blend.get("home_1up", 0.35)) * (cal_home_1up - prior_home_1up),
            0.001,
            0.999,
        )
        away_1up_prob = np.clip(
            prior_away_1up + float(state_ladder_blend.get("away_1up", 0.35)) * (cal_away_1up - prior_away_1up),
            0.001,
            0.999,
        )
        home_2up_cond_prob = np.clip(
            prior_home_2up_cond
            + float(state_ladder_blend.get("home_2up_cond", 0.35)) * (cal_home_2up_cond - prior_home_2up_cond),
            0.001,
            0.999,
        )
        away_2up_cond_prob = np.clip(
            prior_away_2up_cond
            + float(state_ladder_blend.get("away_2up_cond", 0.35)) * (cal_away_2up_cond - prior_away_2up_cond),
            0.001,
            0.999,
        )
        iterator = zip(
            scored.itertuples(index=False),
            home_lambda_p1,
            away_lambda_p1,
            home_lambda_p2,
            away_lambda_p2,
            home_1up_prob,
            away_1up_prob,
            home_2up_cond_prob,
            away_2up_cond_prob,
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
        if path_version == "direct_monotone":
            fixture, lh, la, ph1, pa1, ph2, pa2 = item
        elif path_version == "phase_split":
            fixture, lh_p1, la_p1, lh_p2, la_p2 = item
        elif path_version == "state_ladder":
            fixture, lh_p1, la_p1, lh_p2, la_p2, ph1, pa1, ph2c, pa2c = item
        else:
            fixture, lh, la = item
        fixture_id = int(fixture.fixture_id)
        if path_version == "direct_monotone":
            lambda_home = float(lh)
            lambda_away = float(la)
            p_home_1up = float(ph1)
            p_away_1up = float(pa1)
            p_home_2up = float(ph2)
            p_away_2up = float(pa2)
            markets = derive_and_validate_anytime_direct_monotone(
                p_home_1up=p_home_1up,
                p_away_1up=p_away_1up,
                p_home_2up=p_home_2up,
                p_away_2up=p_away_2up,
            )
        elif path_version == "phase_split":
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
        elif path_version == "state_ladder":
            lambda_home_p1 = float(lh_p1)
            lambda_away_p1 = float(la_p1)
            lambda_home_p2 = float(lh_p2)
            lambda_away_p2 = float(la_p2)
            p_home_1up = float(ph1)
            p_away_1up = float(pa1)
            p_home_2up_cond = float(ph2c)
            p_away_2up_cond = float(pa2c)
            markets = derive_and_validate_anytime_state_ladder(
                p_home_1up=p_home_1up,
                p_away_1up=p_away_1up,
                p_home_2up_given_1up=p_home_2up_cond,
                p_away_2up_given_1up=p_away_2up_cond,
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
            if path_version == "direct_monotone":
                metadata.update(
                    {
                        "derivation": "direct_monotone_v1",
                        "lambda_home_pred": lambda_home,
                        "lambda_away_pred": lambda_away,
                        "p_home_1up_pred": p_home_1up,
                        "p_away_1up_pred": p_away_1up,
                        "p_home_2up_pred": p_home_2up,
                        "p_away_2up_pred": p_away_2up,
                        "direct_monotone_blend": extra_config.get("direct_monotone_blend", {}),
                    }
                )
            elif path_version == "phase_split":
                metadata.update(
                    {
                        "derivation": "markov_ctmc_phase_split",
                        "lambda_home_p1_pred": lambda_home_p1,
                        "lambda_away_p1_pred": lambda_away_p1,
                        "lambda_home_p2_pred": lambda_home_p2,
                        "lambda_away_p2_pred": lambda_away_p2,
                    }
                )
            elif path_version == "state_ladder":
                metadata.update(
                    {
                        "derivation": "state_ladder_v1",
                        "lambda_home_p1_pred": lambda_home_p1,
                        "lambda_away_p1_pred": lambda_away_p1,
                        "lambda_home_p2_pred": lambda_home_p2,
                        "lambda_away_p2_pred": lambda_away_p2,
                        "p_home_1up_pred": p_home_1up,
                        "p_away_1up_pred": p_away_1up,
                        "p_home_2up_given_1up_pred": p_home_2up_cond,
                        "p_away_2up_given_1up_pred": p_away_2up_cond,
                        "state_ladder_prior_version": extra_config.get("state_ladder_prior_version", "phase_split"),
                        "state_ladder_blend": extra_config.get("state_ladder_blend", {}),
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
