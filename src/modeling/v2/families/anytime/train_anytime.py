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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.layer2_markets import market_outcome_calibrator as legacy_calibrator
from src.modeling.v2.calibration.methods import apply_binary_calibrator, fit_binary_calibrator
from src.modeling.v2.eval.metrics import (
    aggregate_market_summary,
    binary_classification_row,
    build_prediction_frame,
    group_binary_classification_rows,
    summarize_binary_metric_rows,
)
from src.modeling.v2.families.anytime.derive_markets import (
    derive_and_validate_anytime,
    derive_and_validate_anytime_direct_monotone,
    derive_and_validate_anytime_phase_split,
    derive_and_validate_anytime_state_ladder,
)
from src.modeling.v2.families.anytime.features import build_anytime_features
from src.modeling.v2.families.anytime.labels import prepare_state_ladder_labels
from src.modeling.v2.db_reuse_features import build_feature_coverage_snapshot
from src.modeling.v2.io.artifact_identity import build_artifact_metadata, write_artifact_metadata
from src.modeling.v2.io.baseline_registry import load_scope_markets
from src.modeling.v2.io.contracts import load_feature_contract


DEFAULT_CONTRACT = ROOT_DIR / "model_v2" / "feature_contracts" / "anytime.yaml"
DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "anytime"
MODEL_NAME = "anytime_v2"
MODEL_VERSION = "markov_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v2 anytime family head.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional PIT dataset artifact (.parquet or .csv). Falls back to legacy fetch_dataset() when omitted.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="Path to anytime feature contract.",
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=DEFAULT_SCOPE,
        help="Path to v2 market scope.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for anytime artifacts.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=MODEL_VERSION,
        help="Persisted model_version to attach to this artifact set.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("histgb_poisson", "poisson_glm", "auto"),
        default="auto",
        help="Regressor family for home/away goal intensities. Use auto to select by walk-forward quality.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=8,
        help="Markov state truncation cap.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on training rows (for smoke runs).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=6,
        help="Number of expanding walk-forward folds.",
    )
    parser.add_argument(
        "--min-fold-test-n",
        type=int,
        default=60,
        help="Minimum valid test rows per fold/market to score walk-forward metrics.",
    )
    parser.add_argument(
        "--path-version",
        type=str,
        choices=("constant", "phase_split", "state_ladder", "direct_monotone"),
        default="constant",
        help="Anytime path derivation version.",
    )
    parser.add_argument(
        "--state-ladder-prior-version",
        type=str,
        choices=("constant", "constant_model", "phase_split"),
        default="phase_split",
        help="Backbone prior used inside state_ladder mode.",
    )
    parser.add_argument(
        "--home-poisson-alpha",
        type=float,
        default=1.0,
        help="Poisson GLM regularization for home-goal heads.",
    )
    parser.add_argument(
        "--away-poisson-alpha",
        type=float,
        default=1.0,
        help="Poisson GLM regularization for away-goal heads.",
    )
    return parser.parse_args()


def _build_regressor(model_type: str, *, poisson_alpha: float = 1.0) -> Any:
    if model_type == "poisson_glm":
        return PoissonRegressor(alpha=max(0.0, float(poisson_alpha)), max_iter=1000)
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=40,
        random_state=42,
    )


def _fit_binary_classifier(x: pd.DataFrame, y: pd.Series | np.ndarray) -> Any:
    target = pd.Series(y).astype(float)
    valid = target.notna()
    target = target.loc[valid].astype(int)
    x_valid = x.loc[valid].reset_index(drop=True)
    if target.empty:
        model = DummyClassifier(strategy="constant", constant=0)
        model.fit(pd.DataFrame({"_dummy": [0.0, 1.0]}), [0, 0])
        return model
    unique = sorted(target.unique().tolist())
    if len(unique) < 2:
        model = DummyClassifier(strategy="constant", constant=int(unique[0]))
        model.fit(x_valid, target)
        return model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    model.fit(x_valid, target)
    return model


def _positive_class_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(x)
    classes = [int(c) for c in getattr(model, "classes_", [0, 1])]
    if probs.ndim != 2:
        return np.full(len(x), 0.5, dtype=float)
    if probs.shape[1] == 1:
        value = 1.0 if classes and classes[0] == 1 else 0.0
        return np.full(len(x), value, dtype=float)
    idx = classes.index(1) if 1 in classes else probs.shape[1] - 1
    return np.clip(probs[:, idx], 0.001, 0.999)


def _build_state_ladder_head_frame(
    x: pd.DataFrame,
    base_probs: pd.DataFrame,
) -> pd.DataFrame:
    base = base_probs.rename(columns={market: f"prior_{market}" for market in base_probs.columns})
    return pd.concat([x.reset_index(drop=True), base.reset_index(drop=True)], axis=1)


def _derive_phase_split_prior_frame(
    *,
    preds: dict[str, np.ndarray],
    max_goals: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for home_p1, away_p1, home_p2, away_p2 in zip(
        preds["home_p1"],
        preds["away_p1"],
        preds["home_p2"],
        preds["away_p2"],
        strict=True,
    ):
        rows.append(
            derive_and_validate_anytime_phase_split(
                lambda_home_p1=float(home_p1),
                lambda_away_p1=float(away_p1),
                lambda_home_p2=float(home_p2),
                lambda_away_p2=float(away_p2),
                max_goals=max(4, int(max_goals)),
            )
        )
    return pd.DataFrame(rows)


def _derive_constant_prior_frame(
    *,
    preds: dict[str, np.ndarray] | None = None,
    total_preds: dict[str, np.ndarray] | None = None,
    max_goals: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    if total_preds is not None:
        iterator = zip(total_preds["home_total"], total_preds["away_total"], strict=True)
        for home_total, away_total in iterator:
            rows.append(
                derive_and_validate_anytime(
                    lambda_home=float(home_total),
                    lambda_away=float(away_total),
                    max_goals=max(4, int(max_goals)),
                )
            )
        return pd.DataFrame(rows)
    assert preds is not None
    for home_p1, away_p1, home_p2, away_p2 in zip(
        preds["home_p1"],
        preds["away_p1"],
        preds["home_p2"],
        preds["away_p2"],
        strict=True,
    ):
        rows.append(
            derive_and_validate_anytime(
                lambda_home=float(home_p1) + float(home_p2),
                lambda_away=float(away_p1) + float(away_p2),
                max_goals=max(4, int(max_goals)),
            )
        )
    return pd.DataFrame(rows)


def _derive_state_ladder_prior_frame(
    *,
    preds: dict[str, np.ndarray],
    max_goals: int,
    prior_version: str,
    total_preds: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    if str(prior_version) == "constant_model":
        return _derive_constant_prior_frame(total_preds=total_preds, max_goals=max_goals)
    if str(prior_version) == "constant":
        return _derive_constant_prior_frame(preds=preds, max_goals=max_goals)
    return _derive_phase_split_prior_frame(preds=preds, max_goals=max_goals)


def _build_state_ladder_oof_prior_frame(
    *,
    frame: pd.DataFrame,
    features: list[str],
    model_type: str,
    home_poisson_alpha: float,
    away_poisson_alpha: float,
    folds: int,
    prior_version: str,
) -> pd.DataFrame:
    sort_cols = ["fixture_id"]
    if "match_datetime_utc" in frame.columns:
        sort_cols = ["match_datetime_utc", "fixture_id"]
    ordered = frame.sort_values(sort_cols).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for fold_idx, (train_end, test_end) in enumerate(_walkforward_ranges(len(ordered), folds), start=1):
        train_df = ordered.iloc[:train_end].copy()
        test_df = ordered.iloc[train_end:test_end].copy()
        if train_df.empty or test_df.empty:
            continue
        imputation = _fit_imputation(train_df, features)
        train_imp = _apply_imputation(train_df, imputation)
        test_imp = _apply_imputation(test_df, imputation)
        phase_models = {
            "home_p1": _build_regressor(model_type, poisson_alpha=home_poisson_alpha),
            "away_p1": _build_regressor(model_type, poisson_alpha=away_poisson_alpha),
            "home_p2": _build_regressor(model_type, poisson_alpha=home_poisson_alpha),
            "away_p2": _build_regressor(model_type, poisson_alpha=away_poisson_alpha),
        }
        phase_models["home_p1"].fit(train_imp[features], train_imp["home_goals_p1"].astype(float))
        phase_models["away_p1"].fit(train_imp[features], train_imp["away_goals_p1"].astype(float))
        phase_models["home_p2"].fit(train_imp[features], train_imp["home_goals_p2"].astype(float))
        phase_models["away_p2"].fit(train_imp[features], train_imp["away_goals_p2"].astype(float))
        phase_preds = {
            "home_p1": np.clip(phase_models["home_p1"].predict(test_imp[features]), 0.01, 6.0),
            "away_p1": np.clip(phase_models["away_p1"].predict(test_imp[features]), 0.01, 6.0),
            "home_p2": np.clip(phase_models["home_p2"].predict(test_imp[features]), 0.01, 6.0),
            "away_p2": np.clip(phase_models["away_p2"].predict(test_imp[features]), 0.01, 6.0),
        }
        total_preds = None
        if prior_version == "constant_model":
            home_total_model = _build_regressor(model_type, poisson_alpha=home_poisson_alpha)
            away_total_model = _build_regressor(model_type, poisson_alpha=away_poisson_alpha)
            home_total_model.fit(train_imp[features], train_imp["home_goals"].astype(float))
            away_total_model.fit(train_imp[features], train_imp["away_goals"].astype(float))
            total_preds = {
                "home_total": np.clip(home_total_model.predict(test_imp[features]), 0.01, 6.0),
                "away_total": np.clip(away_total_model.predict(test_imp[features]), 0.01, 6.0),
            }
        prior_frame = _derive_state_ladder_prior_frame(
            preds=phase_preds,
            max_goals=8,
            prior_version=prior_version,
            total_preds=total_preds,
        )
        meta_frame = (
            test_df[["fixture_id", "match_datetime_utc"]].reset_index(drop=True)
            if "match_datetime_utc" in test_df.columns
            else pd.DataFrame(
                {
                    "fixture_id": test_df["fixture_id"].to_numpy(),
                    "match_datetime_utc": np.full(len(test_df), np.nan),
                }
            )
        )
        pieces.append(
            pd.concat(
                [
                    meta_frame,
                    pd.Series(np.full(len(test_df), fold_idx, dtype=int), name="state_ladder_oof_fold"),
                    prior_frame.reset_index(drop=True),
                ],
                axis=1,
            )
        )
    if not pieces:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "match_datetime_utc",
                "state_ladder_oof_fold",
                "h_1up",
                "a_1up",
                "h_2up",
                "a_2up",
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def _conditional_from_joint(numer: np.ndarray | pd.Series, denom: np.ndarray | pd.Series) -> np.ndarray:
    numer_arr = np.asarray(numer, dtype=float)
    denom_arr = np.asarray(denom, dtype=float)
    cond = np.divide(numer_arr, np.maximum(denom_arr, 1e-6))
    return np.clip(cond, 0.001, 0.999)


def _select_binary_calibrator(*, probs: np.ndarray, y_true: pd.Series | np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    valid = np.isfinite(y)
    if int(valid.sum()) < 80:
        return {"method": "identity"}
    y_valid = y[valid].astype(int)
    if np.unique(y_valid).size < 2:
        return {"method": "identity"}
    ordered = pd.DataFrame({"p_model": np.asarray(probs, dtype=float)[valid], "y_true": y_valid})
    fit_n = max(60, int(len(ordered) * 0.6))
    fit_n = min(fit_n, len(ordered) - 20)
    if fit_n < 60 or (len(ordered) - fit_n) < 20:
        return {"method": "identity"}
    fit_df = ordered.iloc[:fit_n].copy()
    eval_df = ordered.iloc[fit_n:].copy()
    raw = binary_classification_row(
        market="tmp",
        y_true=eval_df["y_true"].to_numpy(dtype=int),
        p_true=eval_df["p_model"].to_numpy(dtype=float),
    )
    best = {"method": "identity"}
    best_rank = (
        float(raw.get("brier") or np.inf),
        float(raw.get("log_loss") or np.inf),
        -float(raw.get("auc") or -np.inf),
    )
    raw_auc = raw.get("auc")
    for method in ("sigmoid", "isotonic"):
        try:
            candidate = fit_binary_calibrator(method, p_model=fit_df["p_model"], y_true=fit_df["y_true"])
            cand_probs = apply_binary_calibrator(candidate, eval_df["p_model"].to_numpy(dtype=float))
            metrics = binary_classification_row(
                market="tmp",
                y_true=eval_df["y_true"].to_numpy(dtype=int),
                p_true=cand_probs,
            )
            cand_auc = metrics.get("auc")
            if raw_auc is not None and cand_auc is not None and float(cand_auc) < float(raw_auc) - 0.01:
                continue
            rank = (
                float(metrics.get("brier") or np.inf),
                float(metrics.get("log_loss") or np.inf),
                -float(metrics.get("auc") or -np.inf),
            )
            if rank < best_rank:
                best_rank = rank
                best = candidate
        except Exception:
            continue
    return best


def _fit_blend_weight(*, prior_probs: np.ndarray | list[float], head_probs: np.ndarray | list[float], y_true: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    valid = np.isfinite(y)
    if int(valid.sum()) < 40:
        return 0.35
    y_valid = y[valid].astype(int)
    prior_valid = np.asarray(prior_probs, dtype=float)[valid]
    head_valid = np.asarray(head_probs, dtype=float)[valid]
    best_alpha = 0.35
    best_brier = float("inf")
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0):
        blended = np.clip(prior_valid + float(alpha) * (head_valid - prior_valid), 0.001, 0.999)
        brier = float(np.mean((blended - y_valid) ** 2))
        if brier < best_brier - 1e-12:
            best_brier = brier
            best_alpha = float(alpha)
    return best_alpha


def _state_ladder_head_probabilities(*, models: dict[str, Any], head_x: pd.DataFrame, prior_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    raw_home_1 = _positive_class_probability(models["home_1up_head"], head_x)
    raw_away_1 = _positive_class_probability(models["away_1up_head"], head_x)
    raw_home_2c = _positive_class_probability(models["home_2up_cond_head"], head_x)
    raw_away_2c = _positive_class_probability(models["away_2up_cond_head"], head_x)
    calibrators = models.get("state_ladder_head_calibrators") or {}
    cal_home_1 = apply_binary_calibrator(calibrators.get("home_1up", {"method": "identity"}), raw_home_1)
    cal_away_1 = apply_binary_calibrator(calibrators.get("away_1up", {"method": "identity"}), raw_away_1)
    cal_home_2c = apply_binary_calibrator(calibrators.get("home_2up_cond", {"method": "identity"}), raw_home_2c)
    cal_away_2c = apply_binary_calibrator(calibrators.get("away_2up_cond", {"method": "identity"}), raw_away_2c)
    prior_home_1 = prior_frame["h_1up"].to_numpy(dtype=float)
    prior_away_1 = prior_frame["a_1up"].to_numpy(dtype=float)
    prior_home_2c = _conditional_from_joint(prior_frame["h_2up"], prior_frame["h_1up"])
    prior_away_2c = _conditional_from_joint(prior_frame["a_2up"], prior_frame["a_1up"])
    blend = models.get("state_ladder_blend") or {}
    return {
        "p_home_1up_raw": raw_home_1,
        "p_away_1up_raw": raw_away_1,
        "p_home_2up_cond_raw": raw_home_2c,
        "p_away_2up_cond_raw": raw_away_2c,
        "p_home_1up": np.clip(prior_home_1 + float(blend.get("home_1up", 0.35)) * (cal_home_1 - prior_home_1), 0.001, 0.999),
        "p_away_1up": np.clip(prior_away_1 + float(blend.get("away_1up", 0.35)) * (cal_away_1 - prior_away_1), 0.001, 0.999),
        "p_home_2up_cond": np.clip(prior_home_2c + float(blend.get("home_2up_cond", 0.35)) * (cal_home_2c - prior_home_2c), 0.001, 0.999),
        "p_away_2up_cond": np.clip(prior_away_2c + float(blend.get("away_2up_cond", 0.35)) * (cal_away_2c - prior_away_2c), 0.001, 0.999),
    }


def _build_state_ladder_oof_head_probs(
    *,
    frame: pd.DataFrame,
    features: list[str],
    prior_frame: pd.DataFrame,
    folds: int,
) -> pd.DataFrame:
    ordered = frame.copy().reset_index(drop=True)
    ordered["__row_id"] = np.arange(len(ordered), dtype=int)
    ordered_prior = prior_frame.reset_index(drop=True).copy()
    if "match_datetime_utc" in ordered.columns:
        ordered = ordered.sort_values(["match_datetime_utc", "fixture_id", "__row_id"]).reset_index(drop=True)
    ordered_prior = ordered_prior.iloc[ordered["__row_id"].to_numpy(dtype=int)].reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for fold_idx, (train_end, test_end) in enumerate(_walkforward_ranges(len(ordered), folds), start=1):
        train_rows = ordered.iloc[:train_end].copy()
        test_rows = ordered.iloc[train_end:test_end].copy()
        if train_rows.empty or test_rows.empty:
            continue
        train_prior = ordered_prior.iloc[:train_end].reset_index(drop=True)
        test_prior = ordered_prior.iloc[train_end:test_end].reset_index(drop=True)
        head_x_train = _build_state_ladder_head_frame(train_rows[features], train_prior)
        head_x_test = _build_state_ladder_head_frame(test_rows[features], test_prior)
        home_1up_head = _fit_binary_classifier(head_x_train, train_rows["target_h_1up"])
        away_1up_head = _fit_binary_classifier(head_x_train, train_rows["target_a_1up"])
        home_2up_cond_head = _fit_binary_classifier(head_x_train, train_rows["target_h_2up_given_h_1up"])
        away_2up_cond_head = _fit_binary_classifier(head_x_train, train_rows["target_a_2up_given_a_1up"])
        pieces.append(
            pd.DataFrame(
                {
                    "__row_id": test_rows["__row_id"].to_numpy(dtype=int),
                    "state_ladder_head_oof_fold": np.full(len(test_rows), fold_idx, dtype=int),
                    "home_1up_oof": _positive_class_probability(home_1up_head, head_x_test),
                    "away_1up_oof": _positive_class_probability(away_1up_head, head_x_test),
                    "home_2up_cond_oof": _positive_class_probability(home_2up_cond_head, head_x_test),
                    "away_2up_cond_oof": _positive_class_probability(away_2up_cond_head, head_x_test),
                }
            )
        )
    if not pieces:
        return pd.DataFrame(
            columns=[
                "__row_id",
                "state_ladder_head_oof_fold",
                "home_1up_oof",
                "away_1up_oof",
                "home_2up_cond_oof",
                "away_2up_cond_oof",
            ]
        )
    return pd.concat(pieces, ignore_index=True).sort_values("__row_id").reset_index(drop=True)


def _build_direct_anytime_oof_prior_frame(
    *,
    frame: pd.DataFrame,
    features: list[str],
    model_type: str,
    home_poisson_alpha: float,
    away_poisson_alpha: float,
    folds: int,
) -> pd.DataFrame:
    ordered = frame.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for fold_idx, (train_end, test_end) in enumerate(_walkforward_ranges(len(ordered), folds), start=1):
        train_df = ordered.iloc[:train_end].copy()
        test_df = ordered.iloc[train_end:test_end].copy()
        if train_df.empty or test_df.empty:
            continue
        imputation = _fit_imputation(train_df, features)
        train_imp = _apply_imputation(train_df, imputation)
        test_imp = _apply_imputation(test_df, imputation)
        home_model = _build_regressor(model_type, poisson_alpha=home_poisson_alpha)
        away_model = _build_regressor(model_type, poisson_alpha=away_poisson_alpha)
        home_model.fit(train_imp[features], train_imp["home_goals"].astype(float))
        away_model.fit(train_imp[features], train_imp["away_goals"].astype(float))
        prior_frame = _derive_constant_prior_frame(
            total_preds={
                "home_total": np.clip(home_model.predict(test_imp[features]), 0.01, 6.0),
                "away_total": np.clip(away_model.predict(test_imp[features]), 0.01, 6.0),
            },
            max_goals=8,
        )
        pieces.append(
            pd.concat(
                [
                    test_df[["fixture_id", "match_datetime_utc"]].reset_index(drop=True),
                    pd.Series(np.full(len(test_df), fold_idx, dtype=int), name="direct_anytime_oof_fold"),
                    prior_frame.reset_index(drop=True),
                ],
                axis=1,
            )
        )
    if not pieces:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "match_datetime_utc",
                "direct_anytime_oof_fold",
                "h_1up",
                "a_1up",
                "h_2up",
                "a_2up",
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def _build_direct_anytime_oof_head_probs(
    *,
    frame: pd.DataFrame,
    features: list[str],
    prior_frame: pd.DataFrame,
    folds: int,
) -> pd.DataFrame:
    ordered = frame.copy().reset_index(drop=True)
    ordered["__row_id"] = np.arange(len(ordered), dtype=int)
    ordered_prior = prior_frame.reset_index(drop=True).copy()
    if "match_datetime_utc" in ordered.columns:
        ordered = ordered.sort_values(["match_datetime_utc", "fixture_id", "__row_id"]).reset_index(drop=True)
    ordered_prior = ordered_prior.iloc[ordered["__row_id"].to_numpy(dtype=int)].reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for fold_idx, (train_end, test_end) in enumerate(_walkforward_ranges(len(ordered), folds), start=1):
        train_rows = ordered.iloc[:train_end].copy()
        test_rows = ordered.iloc[train_end:test_end].copy()
        if train_rows.empty or test_rows.empty:
            continue
        train_prior = ordered_prior.iloc[:train_end].reset_index(drop=True)
        test_prior = ordered_prior.iloc[train_end:test_end].reset_index(drop=True)
        head_x_train = _build_state_ladder_head_frame(train_rows[features], train_prior)
        head_x_test = _build_state_ladder_head_frame(test_rows[features], test_prior)
        h1_head = _fit_binary_classifier(head_x_train, train_rows["target_h_1up"])
        a1_head = _fit_binary_classifier(head_x_train, train_rows["target_a_1up"])
        h2_head = _fit_binary_classifier(head_x_train, train_rows["target_h_2up"])
        a2_head = _fit_binary_classifier(head_x_train, train_rows["target_a_2up"])
        pieces.append(
            pd.DataFrame(
                {
                    "__row_id": test_rows["__row_id"].to_numpy(dtype=int),
                    "direct_anytime_head_oof_fold": np.full(len(test_rows), fold_idx, dtype=int),
                    "home_1up_oof": _positive_class_probability(h1_head, head_x_test),
                    "away_1up_oof": _positive_class_probability(a1_head, head_x_test),
                    "home_2up_oof": _positive_class_probability(h2_head, head_x_test),
                    "away_2up_oof": _positive_class_probability(a2_head, head_x_test),
                }
            )
        )
    if not pieces:
        return pd.DataFrame(
            columns=[
                "__row_id",
                "direct_anytime_head_oof_fold",
                "home_1up_oof",
                "away_1up_oof",
                "home_2up_oof",
                "away_2up_oof",
            ]
        )
    return pd.concat(pieces, ignore_index=True).sort_values("__row_id").reset_index(drop=True)


def _direct_monotone_head_probabilities(*, models: dict[str, Any], head_x: pd.DataFrame, prior_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    raw_home_1 = _positive_class_probability(models["home_1up_head"], head_x)
    raw_away_1 = _positive_class_probability(models["away_1up_head"], head_x)
    raw_home_2 = _positive_class_probability(models["home_2up_head"], head_x)
    raw_away_2 = _positive_class_probability(models["away_2up_head"], head_x)
    calibrators = models.get("direct_monotone_head_calibrators") or {}
    cal_home_1 = apply_binary_calibrator(calibrators.get("home_1up", {"method": "identity"}), raw_home_1)
    cal_away_1 = apply_binary_calibrator(calibrators.get("away_1up", {"method": "identity"}), raw_away_1)
    cal_home_2 = apply_binary_calibrator(calibrators.get("home_2up", {"method": "identity"}), raw_home_2)
    cal_away_2 = apply_binary_calibrator(calibrators.get("away_2up", {"method": "identity"}), raw_away_2)
    blend = models.get("direct_monotone_blend") or {}
    prior_home_1 = prior_frame["h_1up"].to_numpy(dtype=float)
    prior_away_1 = prior_frame["a_1up"].to_numpy(dtype=float)
    prior_home_2 = prior_frame["h_2up"].to_numpy(dtype=float)
    prior_away_2 = prior_frame["a_2up"].to_numpy(dtype=float)
    return {
        "p_home_1up": np.clip(prior_home_1 + float(blend.get("home_1up", 0.35)) * (cal_home_1 - prior_home_1), 0.001, 0.999),
        "p_away_1up": np.clip(prior_away_1 + float(blend.get("away_1up", 0.35)) * (cal_away_1 - prior_away_1), 0.001, 0.999),
        "p_home_2up": np.clip(prior_home_2 + float(blend.get("home_2up", 0.35)) * (cal_home_2 - prior_home_2), 0.001, 0.999),
        "p_away_2up": np.clip(prior_away_2 + float(blend.get("away_2up", 0.35)) * (cal_away_2 - prior_away_2), 0.001, 0.999),
    }


def _load_training_frame(dataset_path: Path | None) -> tuple[pd.DataFrame, str]:
    if dataset_path is None:
        return legacy_calibrator.fetch_dataset(), "legacy_fetch_dataset"

    if not dataset_path.exists():
        raise RuntimeError(f"Dataset path does not exist: {dataset_path}")
    validation_path = dataset_path.parent / "pit_validation_report.json"
    if validation_path.exists():
        try:
            validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read PIT validation report: {validation_path}") from exc
        if validation_report.get("status") == "failed":
            raise RuntimeError(f"PIT dataset failed validation: {validation_path}")
    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(dataset_path)
    elif suffix == ".csv":
        frame = pd.read_csv(dataset_path, low_memory=False)
    else:
        raise RuntimeError(f"Unsupported dataset path format: {dataset_path}")
    return frame, str(dataset_path)


def _select_features(df: pd.DataFrame, contract_path: Path) -> list[str]:
    contract = load_feature_contract(contract_path)
    disabled = set(contract.disabled_features)
    required = [feat for feat in contract.required_features if feat not in disabled]
    optional = [feat for feat in contract.optional_features if feat not in disabled]
    missingness = [feat for feat in contract.missingness_indicators if feat not in disabled]
    missing_required = [feat for feat in required if feat not in df.columns]
    if missing_required:
        raise RuntimeError(
            "Missing required anytime contract features: " + ", ".join(sorted(missing_required))
        )
    selected = list(required)
    selected.extend(
        feat for feat in optional if feat in df.columns and feat not in selected
    )
    selected.extend(
        feat
        for feat in missingness
        if feat in df.columns and feat not in selected
    )
    return selected


def _fit_imputation(train_df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for feat in features:
        med = train_df[feat].median()
        medians[feat] = float(0.0 if pd.isna(med) else med)
    return medians


def _apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for feat, med in medians.items():
        out[feat] = out[feat].fillna(med)
    return out


def _prepare_phase_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["home_goals_p1"] = pd.to_numeric(out.get("home_goals_p1"), errors="coerce")
    out["away_goals_p1"] = pd.to_numeric(out.get("away_goals_p1"), errors="coerce")
    home_goals = pd.to_numeric(out["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(out["away_goals"], errors="coerce")
    out["home_goals_p2"] = (home_goals - out["home_goals_p1"]).where(
        out["home_goals_p1"].notna(), np.nan
    )
    out["away_goals_p2"] = (away_goals - out["away_goals_p1"]).where(
        out["away_goals_p1"].notna(), np.nan
    )
    out["home_goals_p2"] = pd.to_numeric(out["home_goals_p2"], errors="coerce").clip(lower=0.0)
    out["away_goals_p2"] = pd.to_numeric(out["away_goals_p2"], errors="coerce").clip(lower=0.0)
    return out


def _fit_goal_models(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    path_version: str,
    home_poisson_alpha: float,
    away_poisson_alpha: float,
    state_ladder_prior_version: str = "phase_split",
) -> dict[str, Any]:
    x_train = train_df[features]
    if path_version == "direct_monotone":
        models = {
            "home": _build_regressor(model_type, poisson_alpha=home_poisson_alpha),
            "away": _build_regressor(model_type, poisson_alpha=away_poisson_alpha),
        }
        models["home"].fit(x_train, train_df["home_goals"].astype(float))
        models["away"].fit(x_train, train_df["away_goals"].astype(float))
        oof_prior = _build_direct_anytime_oof_prior_frame(
            frame=train_df,
            features=features,
            model_type=model_type,
            home_poisson_alpha=home_poisson_alpha,
            away_poisson_alpha=away_poisson_alpha,
            folds=6,
        )
        prior_frame = train_df[["fixture_id"]].merge(
            oof_prior.drop(columns=["match_datetime_utc"], errors="ignore"),
            on="fixture_id",
            how="left",
        )[["h_1up", "a_1up", "h_2up", "a_2up"]]
        missing_prior = prior_frame.isna().any(axis=1)
        if bool(missing_prior.any()):
            fallback_prior = _derive_constant_prior_frame(
                total_preds={
                    "home_total": np.clip(models["home"].predict(x_train.loc[missing_prior]), 0.01, 6.0),
                    "away_total": np.clip(models["away"].predict(x_train.loc[missing_prior]), 0.01, 6.0),
                },
                max_goals=8,
            )
            prior_frame.loc[missing_prior, ["h_1up", "a_1up", "h_2up", "a_2up"]] = fallback_prior.to_numpy()
        head_x = _build_state_ladder_head_frame(x_train, prior_frame)
        models["home_1up_head"] = _fit_binary_classifier(head_x, train_df["target_h_1up"])
        models["away_1up_head"] = _fit_binary_classifier(head_x, train_df["target_a_1up"])
        models["home_2up_head"] = _fit_binary_classifier(head_x, train_df["target_h_2up"])
        models["away_2up_head"] = _fit_binary_classifier(head_x, train_df["target_a_2up"])
        raw_head_probs = {
            "home_1up": _positive_class_probability(models["home_1up_head"], head_x),
            "away_1up": _positive_class_probability(models["away_1up_head"], head_x),
            "home_2up": _positive_class_probability(models["home_2up_head"], head_x),
            "away_2up": _positive_class_probability(models["away_2up_head"], head_x),
        }
        oof_head_probs = _build_direct_anytime_oof_head_probs(
            frame=train_df,
            features=features,
            prior_frame=prior_frame,
            folds=6,
        )
        oof_aligned = pd.DataFrame({"__row_id": np.arange(len(train_df), dtype=int)}).merge(
            oof_head_probs,
            on="__row_id",
            how="left",
        )
        models["direct_monotone_head_calibrators"] = {
            "home_1up": _select_binary_calibrator(probs=oof_aligned["home_1up_oof"].to_numpy(dtype=float), y_true=train_df["target_h_1up"]),
            "away_1up": _select_binary_calibrator(probs=oof_aligned["away_1up_oof"].to_numpy(dtype=float), y_true=train_df["target_a_1up"]),
            "home_2up": _select_binary_calibrator(probs=oof_aligned["home_2up_oof"].to_numpy(dtype=float), y_true=train_df["target_h_2up"]),
            "away_2up": _select_binary_calibrator(probs=oof_aligned["away_2up_oof"].to_numpy(dtype=float), y_true=train_df["target_a_2up"]),
        }
        calibrated_home_1 = apply_binary_calibrator(
            models["direct_monotone_head_calibrators"]["home_1up"],
            oof_aligned["home_1up_oof"].fillna(pd.Series(raw_head_probs["home_1up"])).to_numpy(dtype=float),
        )
        calibrated_away_1 = apply_binary_calibrator(
            models["direct_monotone_head_calibrators"]["away_1up"],
            oof_aligned["away_1up_oof"].fillna(pd.Series(raw_head_probs["away_1up"])).to_numpy(dtype=float),
        )
        calibrated_home_2 = apply_binary_calibrator(
            models["direct_monotone_head_calibrators"]["home_2up"],
            oof_aligned["home_2up_oof"].fillna(pd.Series(raw_head_probs["home_2up"])).to_numpy(dtype=float),
        )
        calibrated_away_2 = apply_binary_calibrator(
            models["direct_monotone_head_calibrators"]["away_2up"],
            oof_aligned["away_2up_oof"].fillna(pd.Series(raw_head_probs["away_2up"])).to_numpy(dtype=float),
        )
        models["direct_monotone_blend"] = {
            "home_1up": _fit_blend_weight(prior_probs=prior_frame["h_1up"].to_numpy(dtype=float), head_probs=calibrated_home_1, y_true=train_df["target_h_1up"]),
            "away_1up": _fit_blend_weight(prior_probs=prior_frame["a_1up"].to_numpy(dtype=float), head_probs=calibrated_away_1, y_true=train_df["target_a_1up"]),
            "home_2up": _fit_blend_weight(prior_probs=prior_frame["h_2up"].to_numpy(dtype=float), head_probs=calibrated_home_2, y_true=train_df["target_h_2up"]),
            "away_2up": _fit_blend_weight(prior_probs=prior_frame["a_2up"].to_numpy(dtype=float), head_probs=calibrated_away_2, y_true=train_df["target_a_2up"]),
        }
        return models

    if path_version in {"phase_split", "state_ladder"}:
        models = {
            "home_p1": _build_regressor(model_type, poisson_alpha=home_poisson_alpha),
            "away_p1": _build_regressor(model_type, poisson_alpha=away_poisson_alpha),
            "home_p2": _build_regressor(model_type, poisson_alpha=home_poisson_alpha),
            "away_p2": _build_regressor(model_type, poisson_alpha=away_poisson_alpha),
        }
        if path_version == "state_ladder" and state_ladder_prior_version == "constant_model":
            models["home_prior"] = _build_regressor(model_type, poisson_alpha=home_poisson_alpha)
            models["away_prior"] = _build_regressor(model_type, poisson_alpha=away_poisson_alpha)
        models["home_p1"].fit(x_train, train_df["home_goals_p1"].astype(float))
        models["away_p1"].fit(x_train, train_df["away_goals_p1"].astype(float))
        models["home_p2"].fit(x_train, train_df["home_goals_p2"].astype(float))
        models["away_p2"].fit(x_train, train_df["away_goals_p2"].astype(float))
        if path_version == "state_ladder":
            if state_ladder_prior_version == "constant_model":
                models["home_prior"].fit(x_train, train_df["home_goals"].astype(float))
                models["away_prior"].fit(x_train, train_df["away_goals"].astype(float))
            oof_prior = _build_state_ladder_oof_prior_frame(
                frame=train_df,
                features=features,
                model_type=model_type,
                home_poisson_alpha=home_poisson_alpha,
                away_poisson_alpha=away_poisson_alpha,
                folds=6,
                prior_version=state_ladder_prior_version,
            )
            prior_frame = train_df[["fixture_id"]].merge(
                oof_prior.drop(columns=["match_datetime_utc"], errors="ignore"),
                on="fixture_id",
                how="left",
            )[["h_1up", "a_1up", "h_2up", "a_2up"]]
            missing_prior = prior_frame.isna().any(axis=1)
            if bool(missing_prior.any()):
                fallback_preds = {
                    "home_p1": np.clip(models["home_p1"].predict(x_train.loc[missing_prior]), 0.01, 6.0),
                    "away_p1": np.clip(models["away_p1"].predict(x_train.loc[missing_prior]), 0.01, 6.0),
                    "home_p2": np.clip(models["home_p2"].predict(x_train.loc[missing_prior]), 0.01, 6.0),
                    "away_p2": np.clip(models["away_p2"].predict(x_train.loc[missing_prior]), 0.01, 6.0),
                }
                fallback_totals = None
                if state_ladder_prior_version == "constant_model":
                    fallback_totals = {
                        "home_total": np.clip(models["home_prior"].predict(x_train.loc[missing_prior]), 0.01, 6.0),
                        "away_total": np.clip(models["away_prior"].predict(x_train.loc[missing_prior]), 0.01, 6.0),
                    }
                fallback_prior = _derive_state_ladder_prior_frame(
                    preds=fallback_preds,
                    max_goals=8,
                    prior_version=state_ladder_prior_version,
                    total_preds=fallback_totals,
                )
                prior_frame.loc[missing_prior, ["h_1up", "a_1up", "h_2up", "a_2up"]] = fallback_prior.to_numpy()
            head_x = _build_state_ladder_head_frame(x_train, prior_frame)
            models["home_1up_head"] = _fit_binary_classifier(head_x, train_df["target_h_1up"])
            models["away_1up_head"] = _fit_binary_classifier(head_x, train_df["target_a_1up"])
            models["home_2up_cond_head"] = _fit_binary_classifier(
                head_x,
                train_df["target_h_2up_given_h_1up"],
            )
            models["away_2up_cond_head"] = _fit_binary_classifier(
                head_x,
                train_df["target_a_2up_given_a_1up"],
            )
            raw_head_probs = {
                "home_1up": _positive_class_probability(models["home_1up_head"], head_x),
                "away_1up": _positive_class_probability(models["away_1up_head"], head_x),
                "home_2up_cond": _positive_class_probability(models["home_2up_cond_head"], head_x),
                "away_2up_cond": _positive_class_probability(models["away_2up_cond_head"], head_x),
            }
            oof_head_probs = _build_state_ladder_oof_head_probs(
                frame=train_df,
                features=features,
                prior_frame=prior_frame,
                folds=6,
            )
            oof_aligned = pd.DataFrame({"__row_id": np.arange(len(train_df), dtype=int)}).merge(
                oof_head_probs,
                on="__row_id",
                how="left",
            )
            models["state_ladder_head_calibrators"] = {
                "home_1up": _select_binary_calibrator(probs=oof_aligned["home_1up_oof"].to_numpy(dtype=float), y_true=train_df["target_h_1up"]),
                "away_1up": _select_binary_calibrator(probs=oof_aligned["away_1up_oof"].to_numpy(dtype=float), y_true=train_df["target_a_1up"]),
                "home_2up_cond": _select_binary_calibrator(
                    probs=oof_aligned["home_2up_cond_oof"].to_numpy(dtype=float),
                    y_true=train_df["target_h_2up_given_h_1up"],
                ),
                "away_2up_cond": _select_binary_calibrator(
                    probs=oof_aligned["away_2up_cond_oof"].to_numpy(dtype=float),
                    y_true=train_df["target_a_2up_given_a_1up"],
                ),
            }
            calibrated_home_1 = apply_binary_calibrator(
                models["state_ladder_head_calibrators"]["home_1up"],
                oof_aligned["home_1up_oof"].fillna(pd.Series(raw_head_probs["home_1up"])).to_numpy(dtype=float),
            )
            calibrated_away_1 = apply_binary_calibrator(
                models["state_ladder_head_calibrators"]["away_1up"],
                oof_aligned["away_1up_oof"].fillna(pd.Series(raw_head_probs["away_1up"])).to_numpy(dtype=float),
            )
            calibrated_home_2c = apply_binary_calibrator(
                models["state_ladder_head_calibrators"]["home_2up_cond"],
                oof_aligned["home_2up_cond_oof"].fillna(pd.Series(raw_head_probs["home_2up_cond"])).to_numpy(dtype=float),
            )
            calibrated_away_2c = apply_binary_calibrator(
                models["state_ladder_head_calibrators"]["away_2up_cond"],
                oof_aligned["away_2up_cond_oof"].fillna(pd.Series(raw_head_probs["away_2up_cond"])).to_numpy(dtype=float),
            )
            models["state_ladder_blend"] = {
                "home_1up": _fit_blend_weight(
                    prior_probs=prior_frame["h_1up"].to_numpy(dtype=float),
                    head_probs=calibrated_home_1,
                    y_true=train_df["target_h_1up"],
                ),
                "away_1up": _fit_blend_weight(
                    prior_probs=prior_frame["a_1up"].to_numpy(dtype=float),
                    head_probs=calibrated_away_1,
                    y_true=train_df["target_a_1up"],
                ),
                "home_2up_cond": _fit_blend_weight(
                    prior_probs=_conditional_from_joint(prior_frame["h_2up"], prior_frame["h_1up"]),
                    head_probs=calibrated_home_2c,
                    y_true=train_df["target_h_2up_given_h_1up"],
                ),
                "away_2up_cond": _fit_blend_weight(
                    prior_probs=_conditional_from_joint(prior_frame["a_2up"], prior_frame["a_1up"]),
                    head_probs=calibrated_away_2c,
                    y_true=train_df["target_a_2up_given_a_1up"],
                ),
            }
        return models

    models = {
        "home": _build_regressor(model_type, poisson_alpha=home_poisson_alpha),
        "away": _build_regressor(model_type, poisson_alpha=away_poisson_alpha),
    }
    models["home"].fit(x_train, train_df["home_goals"].astype(float))
    models["away"].fit(x_train, train_df["away_goals"].astype(float))
    return models


def _predict_goal_rates(
    *,
    models: dict[str, Any],
    x: pd.DataFrame,
    path_version: str,
    state_ladder_prior_version: str = "phase_split",
) -> dict[str, np.ndarray]:
    if path_version == "direct_monotone":
        total_preds = {
            "home": np.clip(models["home"].predict(x), 0.05, 8.0),
            "away": np.clip(models["away"].predict(x), 0.05, 8.0),
        }
        prior_frame = _derive_constant_prior_frame(
            total_preds={
                "home_total": total_preds["home"],
                "away_total": total_preds["away"],
            },
            max_goals=8,
        )
        head_x = _build_state_ladder_head_frame(x, prior_frame)
        head_probs = _direct_monotone_head_probabilities(models=models, head_x=head_x, prior_frame=prior_frame)
        total_preds.update(
            {
                "prior_h_1up": prior_frame["h_1up"].to_numpy(dtype=float),
                "prior_a_1up": prior_frame["a_1up"].to_numpy(dtype=float),
                "prior_h_2up": prior_frame["h_2up"].to_numpy(dtype=float),
                "prior_a_2up": prior_frame["a_2up"].to_numpy(dtype=float),
                **head_probs,
            }
        )
        return total_preds

    if path_version in {"phase_split", "state_ladder"}:
        phase_preds = {
            "home_p1": np.clip(models["home_p1"].predict(x), 0.01, 6.0),
            "away_p1": np.clip(models["away_p1"].predict(x), 0.01, 6.0),
            "home_p2": np.clip(models["home_p2"].predict(x), 0.01, 6.0),
            "away_p2": np.clip(models["away_p2"].predict(x), 0.01, 6.0),
        }
        if path_version == "phase_split":
            return phase_preds
        total_preds = None
        if state_ladder_prior_version == "constant_model":
            total_preds = {
                "home_total": np.clip(models["home_prior"].predict(x), 0.01, 6.0),
                "away_total": np.clip(models["away_prior"].predict(x), 0.01, 6.0),
            }
        prior_frame = _derive_state_ladder_prior_frame(
            preds=phase_preds,
            max_goals=8,
            prior_version=state_ladder_prior_version,
            total_preds=total_preds,
        )
        head_x = _build_state_ladder_head_frame(x, prior_frame)
        head_probs = _state_ladder_head_probabilities(models=models, head_x=head_x, prior_frame=prior_frame)
        phase_preds.update(
            {
                "prior_h_1up": prior_frame["h_1up"].to_numpy(dtype=float),
                "prior_a_1up": prior_frame["a_1up"].to_numpy(dtype=float),
                "prior_h_2up": prior_frame["h_2up"].to_numpy(dtype=float),
                "prior_a_2up": prior_frame["a_2up"].to_numpy(dtype=float),
                **head_probs,
            }
        )
        return phase_preds

    return {
        "home": np.clip(models["home"].predict(x), 0.05, 8.0),
        "away": np.clip(models["away"].predict(x), 0.05, 8.0),
    }


def _derive_anytime_frame(
    *,
    preds: dict[str, np.ndarray],
    path_version: str,
    max_goals: int,
) -> pd.DataFrame:
    derived_rows: list[dict[str, float]] = []
    if path_version == "direct_monotone":
        for p_home_1up, p_away_1up, p_home_2up, p_away_2up in zip(
            preds["p_home_1up"],
            preds["p_away_1up"],
            preds["p_home_2up"],
            preds["p_away_2up"],
            strict=True,
        ):
            derived_rows.append(
                derive_and_validate_anytime_direct_monotone(
                    p_home_1up=float(p_home_1up),
                    p_away_1up=float(p_away_1up),
                    p_home_2up=float(p_home_2up),
                    p_away_2up=float(p_away_2up),
                )
            )
        return pd.DataFrame(derived_rows)

    if path_version == "phase_split":
        for home_p1, away_p1, home_p2, away_p2 in zip(
            preds["home_p1"],
            preds["away_p1"],
            preds["home_p2"],
            preds["away_p2"],
            strict=True,
        ):
            derived_rows.append(
                derive_and_validate_anytime_phase_split(
                    lambda_home_p1=float(home_p1),
                    lambda_away_p1=float(away_p1),
                    lambda_home_p2=float(home_p2),
                    lambda_away_p2=float(away_p2),
                    max_goals=max(4, int(max_goals)),
                )
            )
        return pd.DataFrame(derived_rows)

    if path_version == "state_ladder":
        for p_home_1up, p_away_1up, p_home_2up_cond, p_away_2up_cond in zip(
            preds["p_home_1up"],
            preds["p_away_1up"],
            preds["p_home_2up_cond"],
            preds["p_away_2up_cond"],
            strict=True,
        ):
            derived_rows.append(
                derive_and_validate_anytime_state_ladder(
                    p_home_1up=float(p_home_1up),
                    p_away_1up=float(p_away_1up),
                    p_home_2up_given_1up=float(p_home_2up_cond),
                    p_away_2up_given_1up=float(p_away_2up_cond),
                )
            )
        return pd.DataFrame(derived_rows)

    for home_mu, away_mu in zip(preds["home"], preds["away"], strict=True):
        derived_rows.append(
            derive_and_validate_anytime(
                lambda_home=float(home_mu),
                lambda_away=float(away_mu),
                max_goals=max(4, int(max_goals)),
            )
        )
    return pd.DataFrame(derived_rows)


def _market_metrics(market: str, y_true: np.ndarray, p_true: np.ndarray) -> dict[str, Any]:
    return binary_classification_row(market=market, y_true=y_true, p_true=p_true)


def _walkforward_ranges(total_rows: int, folds: int) -> list[tuple[int, int]]:
    if total_rows <= 0:
        return []
    steps = max(1, int(folds))
    points = np.linspace(0, total_rows, steps + 2, dtype=int)
    ranges: list[tuple[int, int]] = []
    for idx in range(1, len(points) - 1):
        train_end = int(points[idx])
        test_end = int(points[idx + 1])
        if train_end <= 0 or test_end <= train_end:
            continue
        ranges.append((train_end, test_end))
    return ranges


def _evaluate_walkforward(
    *,
    frame: pd.DataFrame,
    features: list[str],
    anytime_markets: set[str],
    model_type: str,
    max_goals: int,
    path_version: str,
    state_ladder_prior_version: str,
    home_poisson_alpha: float,
    away_poisson_alpha: float,
    folds: int,
    min_fold_test_n: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float | int | None]],
    list[dict[str, Any]],
]:
    ordered = frame.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    fold_rows: list[dict[str, Any]] = []
    fold_league_rows: list[dict[str, Any]] = []
    for fold_idx, (train_end, test_end) in enumerate(
        _walkforward_ranges(len(ordered), folds), start=1
    ):
        train_df = ordered.iloc[:train_end].copy()
        test_df = ordered.iloc[train_end:test_end].copy()
        if train_df.empty or test_df.empty:
            continue

        imputation = _fit_imputation(train_df, features)
        train_df = _apply_imputation(train_df, imputation)
        test_df = _apply_imputation(test_df, imputation)

        x_test = test_df[features]
        models = _fit_goal_models(
            train_df=train_df,
            features=features,
            model_type=model_type,
            path_version=path_version,
            home_poisson_alpha=home_poisson_alpha,
            away_poisson_alpha=away_poisson_alpha,
            state_ladder_prior_version=state_ladder_prior_version,
        )
        preds = _predict_goal_rates(
            models=models,
            x=x_test,
            path_version=path_version,
            state_ladder_prior_version=state_ladder_prior_version,
        )
        pred_frame = _derive_anytime_frame(
            preds=preds,
            path_version=path_version,
            max_goals=max_goals,
        )

        for market in sorted(anytime_markets):
            if market not in pred_frame.columns:
                continue
            target_col = f"target_{market}"
            if target_col not in test_df.columns:
                continue
            valid = test_df[target_col].notna().to_numpy(dtype=bool)
            valid_n = int(valid.sum())
            if valid_n < int(min_fold_test_n):
                continue
            y_true = test_df[target_col].to_numpy(dtype=float)[valid].astype(int)
            p_true = np.clip(pred_frame[market].to_numpy(dtype=float)[valid], 0.001, 0.999)
            base_fields = {
                "fold": int(fold_idx),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
            }
            fold_rows.append(
                binary_classification_row(
                    market=market,
                    y_true=y_true,
                    p_true=p_true,
                    extra_fields=base_fields,
                )
            )
            if "league_code" in test_df.columns:
                fold_league_rows.extend(
                    group_binary_classification_rows(
                        market=market,
                        group_values=test_df.loc[valid, "league_code"],
                        y_true=y_true,
                        p_true=p_true,
                        extra_fields=base_fields,
                    )
                )

    if not fold_rows:
        return fold_rows, {}, fold_league_rows
    summary = summarize_binary_metric_rows(fold_rows)
    return fold_rows, summary, fold_league_rows


def _aggregate_walkforward_quality(
    summary: dict[str, dict[str, float | int | None]],
) -> dict[str, float | int | None]:
    return aggregate_market_summary(summary)


def _select_model_type(
    *,
    frame: pd.DataFrame,
    features: list[str],
    anytime_markets: set[str],
    max_goals: int,
    path_version: str,
    state_ladder_prior_version: str,
    home_poisson_alpha: float,
    away_poisson_alpha: float,
    folds: int,
    min_fold_test_n: int,
) -> tuple[
    str,
    list[dict[str, Any]],
    dict[str, dict[str, float | int | None]],
    dict[str, dict[str, float | int | None]],
    list[dict[str, Any]],
]:
    candidates = ("histgb_poisson", "poisson_glm")
    collected: dict[
        str,
        tuple[
            list[dict[str, Any]],
            dict[str, dict[str, float | int | None]],
            dict[str, float | int | None],
            list[dict[str, Any]],
        ],
    ] = {}
    for model_type in candidates:
        folds_rows, summary, league_rows = _evaluate_walkforward(
            frame=frame,
            features=features,
            anytime_markets=anytime_markets,
            model_type=model_type,
            max_goals=max_goals,
            path_version=path_version,
            state_ladder_prior_version=state_ladder_prior_version,
            home_poisson_alpha=home_poisson_alpha,
            away_poisson_alpha=away_poisson_alpha,
            folds=folds,
            min_fold_test_n=min_fold_test_n,
        )
        collected[model_type] = (
            folds_rows,
            summary,
            _aggregate_walkforward_quality(summary),
            league_rows,
        )

    def _rank(model_type: str) -> tuple[float, float, int]:
        agg = collected[model_type][2]
        auc = agg.get("auc_mean")
        brier = agg.get("brier_mean")
        markets_used = int(agg.get("markets_used") or 0)
        auc_rank = float(auc) if isinstance(auc, (int, float)) else -1e9
        brier_rank = -float(brier) if isinstance(brier, (int, float)) else -1e9
        return auc_rank, brier_rank, markets_used

    selected = max(candidates, key=_rank)
    selected_rows, selected_summary, _selected_agg, selected_league_rows = collected[selected]
    model_selection_summary = {
        model_type: collected[model_type][2] for model_type in candidates
    }
    return selected, selected_rows, selected_summary, model_selection_summary, selected_league_rows


def main() -> None:
    args = parse_args()
    df, data_source = _load_training_frame(args.dataset_path)
    if df.empty:
        raise RuntimeError("No rows available for anytime training.")
    if args.max_rows is not None and args.max_rows > 0 and len(df) > args.max_rows:
        df = (
            df.sort_values(["match_datetime_utc", "fixture_id"])
            .tail(int(args.max_rows))
            .reset_index(drop=True)
        )

    if "prediction_time_utc" not in df.columns:
        df = legacy_calibrator.add_targets_and_derived(df)
    df["match_datetime_utc"] = pd.to_datetime(
        df["match_datetime_utc"], utc=True, errors="coerce"
    )
    if "home_goal_share" not in df.columns:
        df = build_anytime_features(df)
    path_version = str(args.path_version)
    if path_version in {"phase_split", "state_ladder"}:
        df = _prepare_phase_targets(df)
        df = df[df["home_goals_p1"].notna() & df["away_goals_p1"].notna()].copy()
        if df.empty:
            raise RuntimeError("No rows with phase-split goal labels available for anytime training.")
    if path_version == "state_ladder":
        df = prepare_state_ladder_labels(df)

    features = _select_features(df, args.contract)
    scope_markets = set(load_scope_markets(args.scope))
    anytime_markets = {"h_1up", "a_1up", "h_2up", "a_2up"} & scope_markets
    model_type_requested = str(args.model_type)
    model_selection_summary: dict[str, dict[str, float | int | None]]
    walkforward_league_rows: list[dict[str, Any]]
    if model_type_requested == "auto":
        model_type_selected, walkforward_rows, walkforward_summary, model_selection_summary, walkforward_league_rows = (
            _select_model_type(
                frame=df,
                features=features,
                anytime_markets=anytime_markets,
                max_goals=int(args.max_goals),
                path_version=path_version,
                state_ladder_prior_version=str(args.state_ladder_prior_version),
                home_poisson_alpha=float(args.home_poisson_alpha),
                away_poisson_alpha=float(args.away_poisson_alpha),
                folds=int(args.folds),
                min_fold_test_n=int(args.min_fold_test_n),
            )
        )
    else:
        model_type_selected = model_type_requested
        walkforward_rows, walkforward_summary, walkforward_league_rows = _evaluate_walkforward(
            frame=df,
            features=features,
            anytime_markets=anytime_markets,
            model_type=model_type_selected,
            max_goals=int(args.max_goals),
            path_version=path_version,
            state_ladder_prior_version=str(args.state_ladder_prior_version),
            home_poisson_alpha=float(args.home_poisson_alpha),
            away_poisson_alpha=float(args.away_poisson_alpha),
            folds=int(args.folds),
            min_fold_test_n=int(args.min_fold_test_n),
        )
        model_selection_summary = {
            model_type_selected: _aggregate_walkforward_quality(walkforward_summary)
        }

    train_df, test_df = legacy_calibrator.split_time_respecting(df)
    imputation = _fit_imputation(train_df, features)
    train_df = _apply_imputation(train_df, imputation)
    test_df = _apply_imputation(test_df, imputation)

    x_test = test_df[features]
    models = _fit_goal_models(
        train_df=train_df,
        features=features,
        model_type=model_type_selected,
        path_version=path_version,
        home_poisson_alpha=float(args.home_poisson_alpha),
        away_poisson_alpha=float(args.away_poisson_alpha),
        state_ladder_prior_version=str(args.state_ladder_prior_version),
    )
    preds = _predict_goal_rates(
        models=models,
        x=x_test,
        path_version=path_version,
        state_ladder_prior_version=str(args.state_ladder_prior_version),
    )
    pred_frame = _derive_anytime_frame(
        preds=preds,
        path_version=path_version,
        max_goals=int(args.max_goals),
    )

    market_metrics: list[dict[str, Any]] = []
    holdout_league_rows: list[dict[str, Any]] = []
    holdout_prediction_frames: list[pd.DataFrame] = []
    for market in sorted(anytime_markets):
        if market not in pred_frame.columns:
            continue
        target_col = f"target_{market}"
        if target_col not in test_df.columns:
            continue
        valid = test_df[target_col].notna().to_numpy(dtype=bool)
        if int(valid.sum()) < 50:
            continue
        y_true = test_df[target_col].to_numpy(dtype=float)[valid].astype(int)
        p_true = np.clip(pred_frame[market].to_numpy(dtype=float)[valid], 0.001, 0.999)
        extra_columns: dict[str, Any] = {}
        if "league_code" in test_df.columns:
            extra_columns["league_code"] = (
                test_df.loc[valid, "league_code"].fillna("__missing__").astype(str).to_numpy()
            )
        if "match_datetime_utc" in test_df.columns:
            extra_columns["match_datetime_utc"] = (
                test_df.loc[valid, "match_datetime_utc"].astype(str).to_numpy()
            )
        holdout_prediction_frames.append(
            build_prediction_frame(
                label_key="market",
                label_value=market,
                fixture_ids=(
                    test_df.loc[valid, "fixture_id"].to_numpy()
                    if "fixture_id" in test_df.columns
                    else test_df.index.to_numpy()[valid]
                ),
                y_true=y_true,
                p_true=p_true,
                extra_columns=extra_columns,
            )
        )
        market_metrics.append(_market_metrics(market, y_true, p_true))
        if "league_code" in test_df.columns:
            holdout_league_rows.extend(
                group_binary_classification_rows(
                    market=market,
                    group_values=test_df.loc[valid, "league_code"],
                    y_true=y_true,
                    p_true=p_true,
                )
            )

    holdout_prediction_frame = (
        pd.concat(holdout_prediction_frames, ignore_index=True)
        if holdout_prediction_frames
        else pd.DataFrame(columns=["market", "fixture_id", "y_true", "p_model"])
    )

    diagnostics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": features,
        "model_name": MODEL_NAME,
        "model_version": str(args.model_version),
        "data_source": data_source,
        "model_type_requested": model_type_requested,
        "model_type_selected": model_type_selected,
        "path_version": path_version,
        "home_poisson_alpha": float(args.home_poisson_alpha),
        "away_poisson_alpha": float(args.away_poisson_alpha),
        "state_ladder_prior_version": str(args.state_ladder_prior_version),
        "model_selection_summary": model_selection_summary,
        "max_goals": int(args.max_goals),
        "walkforward_folds": int(args.folds),
        "walkforward_min_fold_test_n": int(args.min_fold_test_n),
        "walkforward_markets_scored": int(len(walkforward_summary)),
        "walkforward_league_rows": int(len(walkforward_league_rows)),
        "holdout_league_rows": int(len(holdout_league_rows)),
        "holdout_prediction_rows": int(len(holdout_prediction_frame)),
        "feature_coverage_snapshot_rows": int(len(features)),
        "trained_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    feature_coverage_snapshot = build_feature_coverage_snapshot(df, features)
    if path_version in {"phase_split", "state_ladder"}:
        diagnostics["home_goals_p1_mae"] = float(
            mean_absolute_error(test_df["home_goals_p1"].astype(float), preds["home_p1"])
        )
        diagnostics["away_goals_p1_mae"] = float(
            mean_absolute_error(test_df["away_goals_p1"].astype(float), preds["away_p1"])
        )
        diagnostics["home_goals_p2_mae"] = float(
            mean_absolute_error(test_df["home_goals_p2"].astype(float), preds["home_p2"])
        )
        diagnostics["away_goals_p2_mae"] = float(
            mean_absolute_error(test_df["away_goals_p2"].astype(float), preds["away_p2"])
        )
        if path_version == "state_ladder":
            diagnostics["state_ladder_home_2up_cond_train_n"] = int(
                test_df.get("target_h_2up_given_h_1up", pd.Series(dtype=float)).notna().sum()
            )
            diagnostics["state_ladder_away_2up_cond_train_n"] = int(
                test_df.get("target_a_2up_given_a_1up", pd.Series(dtype=float)).notna().sum()
            )
            diagnostics["state_ladder_blend"] = models.get("state_ladder_blend", {})
            diagnostics["state_ladder_head_calibration_methods"] = {
                key: str((value or {}).get("method") or "identity")
                for key, value in (models.get("state_ladder_head_calibrators") or {}).items()
            }
    elif path_version == "direct_monotone":
        diagnostics["home_goals_mae"] = float(
            mean_absolute_error(test_df["home_goals"].astype(float), preds["home"])
        )
        diagnostics["away_goals_mae"] = float(
            mean_absolute_error(test_df["away_goals"].astype(float), preds["away"])
        )
        diagnostics["direct_monotone_blend"] = models.get("direct_monotone_blend", {})
        diagnostics["direct_monotone_head_calibration_methods"] = {
            key: str((value or {}).get("method") or "identity")
            for key, value in (models.get("direct_monotone_head_calibrators") or {}).items()
        }
    else:
        diagnostics["home_goals_mae"] = float(
            mean_absolute_error(test_df["home_goals"].astype(float), preds["home"])
        )
        diagnostics["away_goals_mae"] = float(
            mean_absolute_error(test_df["away_goals"].astype(float), preds["away"])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_metadata(
        args.output_dir,
        build_artifact_metadata(
            family="anytime",
            model_name=MODEL_NAME,
            model_version=str(args.model_version),
            artifact_dir=args.output_dir,
            trained_at_utc=str(diagnostics["trained_at_utc"]),
            extra={
                "model_type_selected": model_type_selected,
                "path_version": path_version,
                "data_source": data_source,
            },
        ),
    )
    if path_version in {"phase_split", "state_ladder"}:
        joblib.dump(models["home_p1"], args.output_dir / "home_goals_p1_model.pkl")
        joblib.dump(models["away_p1"], args.output_dir / "away_goals_p1_model.pkl")
        joblib.dump(models["home_p2"], args.output_dir / "home_goals_p2_model.pkl")
        joblib.dump(models["away_p2"], args.output_dir / "away_goals_p2_model.pkl")
        if path_version == "state_ladder":
            joblib.dump(models["home_1up_head"], args.output_dir / "home_1up_head.pkl")
            joblib.dump(models["away_1up_head"], args.output_dir / "away_1up_head.pkl")
            joblib.dump(models["home_2up_cond_head"], args.output_dir / "home_2up_cond_head.pkl")
            joblib.dump(models["away_2up_cond_head"], args.output_dir / "away_2up_cond_head.pkl")
            if str(args.state_ladder_prior_version) == "constant_model":
                joblib.dump(models["home_prior"], args.output_dir / "home_prior_model.pkl")
                joblib.dump(models["away_prior"], args.output_dir / "away_prior_model.pkl")
            joblib.dump(
                models.get("state_ladder_head_calibrators", {}),
                args.output_dir / "state_ladder_head_calibrators.joblib",
            )
    elif path_version == "direct_monotone":
        joblib.dump(models["home"], args.output_dir / "home_goals_model.pkl")
        joblib.dump(models["away"], args.output_dir / "away_goals_model.pkl")
        joblib.dump(models["home_1up_head"], args.output_dir / "home_1up_head.pkl")
        joblib.dump(models["away_1up_head"], args.output_dir / "away_1up_head.pkl")
        joblib.dump(models["home_2up_head"], args.output_dir / "home_2up_head.pkl")
        joblib.dump(models["away_2up_head"], args.output_dir / "away_2up_head.pkl")
        joblib.dump(
            models.get("direct_monotone_head_calibrators", {}),
            args.output_dir / "direct_monotone_head_calibrators.joblib",
        )
    else:
        joblib.dump(models["home"], args.output_dir / "home_goals_model.pkl")
        joblib.dump(models["away"], args.output_dir / "away_goals_model.pkl")
    (args.output_dir / "features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )
    (args.output_dir / "imputation.json").write_text(
        json.dumps({"global_medians": imputation}, indent=2), encoding="utf-8"
    )
    (args.output_dir / "model_config.json").write_text(
        json.dumps(
            {
                "max_goals": int(args.max_goals),
                "path_version": path_version,
                "state_ladder_prior_version": str(args.state_ladder_prior_version),
                "state_ladder_blend": models.get("state_ladder_blend", {}) if path_version == "state_ladder" else {},
                "direct_monotone_blend": models.get("direct_monotone_blend", {}) if path_version == "direct_monotone" else {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "metrics_holdout.json").write_text(
        json.dumps(market_metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics_holdout_by_league.json").write_text(
        json.dumps(holdout_league_rows, indent=2), encoding="utf-8"
    )
    holdout_prediction_frame.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    (args.output_dir / "metrics_walkforward_folds.json").write_text(
        json.dumps(walkforward_rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics_walkforward_folds_by_league.json").write_text(
        json.dumps(walkforward_league_rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics_walkforward.json").write_text(
        json.dumps(walkforward_summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "feature_coverage_snapshot.json").write_text(
        json.dumps(feature_coverage_snapshot, indent=2), encoding="utf-8"
    )
    (args.output_dir / "training_report.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    print(f"Saved anytime artifacts to {args.output_dir}")
    if path_version in {"phase_split", "state_ladder"}:
        print(
            "Phase Goal MAE "
            f"home_p1={diagnostics['home_goals_p1_mae']:.3f} "
            f"away_p1={diagnostics['away_goals_p1_mae']:.3f} "
            f"home_p2={diagnostics['home_goals_p2_mae']:.3f} "
            f"away_p2={diagnostics['away_goals_p2_mae']:.3f}"
        )
    elif path_version == "direct_monotone":
        print(
            "Goal MAE "
            f"home={diagnostics['home_goals_mae']:.3f} away={diagnostics['away_goals_mae']:.3f}"
        )
    else:
        print(
            "Goal MAE "
            f"home={diagnostics['home_goals_mae']:.3f} away={diagnostics['away_goals_mae']:.3f}"
        )
    if model_type_requested == "auto":
        print(f"Auto-selected model_type={model_type_selected}")
    for row in market_metrics:
        print(
            f"{row['market']}: AUC={row['auc']}, Brier={row['brier']:.4f}, "
            f"Acc={row['accuracy']:.3f}, n={row['n']}"
        )


if __name__ == "__main__":
    main()
