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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error


ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.layer2_markets import market_outcome_calibrator as legacy_calibrator
from src.modeling.v2.calibration.methods import apply_binary_calibrator, evaluate_market_calibration
from src.modeling.v2.eval.metrics import (
    aggregate_market_summary,
    binary_classification_row,
    build_prediction_frame,
    group_binary_classification_rows,
    summarize_binary_metric_rows,
)
from src.modeling.v2.families.corners.derive_lines import (
    AWAY_MARKETS,
    derive_and_validate_corners,
    estimate_nb_dispersion,
    HOME_MARKETS,
    over_probability,
    reconcile_team_means_from_total_home_mu,
    reconcile_team_means_from_total_share,
    TOTAL_MARKETS,
)
from src.modeling.v2.families.corners.features import (
    add_corners_context_features,
    apply_league_regime,
    fit_league_regime,
    LEAGUE_REGIME_FEATURES,
)
from src.modeling.v2.families.corners.neural_residual import (
    NEURAL_SHARE_RESIDUAL_ARTIFACT_FORMAT,
    NEURAL_TOTAL_RESIDUAL_BUNDLE_FILENAME,
    NEURAL_TOTAL_RESIDUAL_DEFAULT_BOUND,
    NEURAL_TOTAL_RESIDUAL_TARGET_KIND,
    NEURAL_TOTAL_SHARE_RESIDUAL_PATH_VERSION,
    NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME,
    NEURAL_SHARE_RESIDUAL_DEFAULT_BOUND,
    NEURAL_SHARE_RESIDUAL_DROPOUT,
    NEURAL_SHARE_RESIDUAL_HIDDEN_DIMS,
    NEURAL_SHARE_RESIDUAL_PATH_VERSION,
    NEURAL_SHARE_RESIDUAL_TARGET_KIND,
    fit_neural_total_residual_bundle,
    predict_neural_total_residual_delta,
    save_neural_total_residual_bundle,
    fit_neural_share_residual_bundle,
    predict_neural_share_residual_delta,
    save_neural_share_residual_bundle,
)
from src.modeling.v2.families.corners.neural_total_ladder import (
    NEURAL_TOTAL_MARKET_LADDER_ARTIFACT_FORMAT,
    NEURAL_TOTAL_MARKET_LADDER_BUNDLE_FILENAME,
    NEURAL_TOTAL_MARKET_LADDER_DROPOUT,
    NEURAL_TOTAL_MARKET_LADDER_HIDDEN_DIMS,
    NEURAL_TOTAL_MARKET_LADDER_PATH_VERSION,
    NEURAL_TOTAL_MARKET_LADDER_TARGET_KIND,
    fit_neural_total_market_ladder_bundle,
    predict_neural_total_market_ladder_probs,
    save_neural_total_market_ladder_bundle,
)
from src.modeling.v2.io.artifact_identity import build_artifact_metadata, write_artifact_metadata
from src.modeling.v2.io.baseline_registry import load_scope_markets
from src.modeling.v2.io.contracts import load_feature_contract


DEFAULT_CONTRACT = ROOT_DIR / "model_v2" / "feature_contracts" / "corners.yaml"
DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "corners"
MODEL_NAME = "corners_v2"
MODEL_VERSION = "distribution_head_v1"
TOTAL_SURFACE_PATHS = {"totals_surface_calibrated"}
NEURAL_TOTAL_LADDER_PATHS = {NEURAL_TOTAL_MARKET_LADDER_PATH_VERSION}
PMF_SURFACE_PATHS = {"pmf_surface_blended"}
SHARE_PRIOR_PATHS = {"totals_first_league_share_residual"}
NEURAL_SHARE_RESIDUAL_PATHS = {NEURAL_SHARE_RESIDUAL_PATH_VERSION}
NEURAL_TOTAL_SHARE_RESIDUAL_PATHS = {NEURAL_TOTAL_SHARE_RESIDUAL_PATH_VERSION}
CALIBRATED_TEAM_HEAD_PATHS = {"totals_first_team_market_calibrated"}
MARKET_HEAD_PATHS = {"totals_first_market_heads", *CALIBRATED_TEAM_HEAD_PATHS}
TOTALS_FIRST_PATHS = {
    "totals_first",
    "totals_first_residual",
    *SHARE_PRIOR_PATHS,
    *NEURAL_SHARE_RESIDUAL_PATHS,
    *NEURAL_TOTAL_SHARE_RESIDUAL_PATHS,
    *MARKET_HEAD_PATHS,
    *TOTAL_SURFACE_PATHS,
    *NEURAL_TOTAL_LADDER_PATHS,
    *PMF_SURFACE_PATHS,
}
TOTAL_HEAD_MARKETS = tuple(TOTAL_MARKETS.keys())
TEAM_HEAD_MARKETS = tuple(HOME_MARKETS.keys()) + tuple(AWAY_MARKETS.keys())
PMF_MAX_COUNT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v2 corners family head.")
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
        help="Path to corners feature contract.",
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
        help="Output directory for corners artifacts.",
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
        help="Regressor family for home/away corners. Use auto to select by walk-forward quality.",
    )
    parser.add_argument(
        "--path-version",
        type=str,
        choices=(
            "sum_heads",
            "totals_first",
            "totals_first_residual",
            "totals_first_league_share_residual",
            NEURAL_SHARE_RESIDUAL_PATH_VERSION,
            NEURAL_TOTAL_SHARE_RESIDUAL_PATH_VERSION,
            "pmf_surface_blended",
            "totals_first_market_heads",
            "totals_first_team_market_calibrated",
            "totals_surface_calibrated",
            NEURAL_TOTAL_MARKET_LADDER_PATH_VERSION,
        ),
        default="sum_heads",
        help="Corners derivation path. `sum_heads` keeps legacy home/away heads; `totals_first` models total corners directly then splits to team means; `totals_first_residual` adds a residual home-team correction while preserving the direct total backbone; `totals_first_league_share_residual` anchors home-share to a smoothed league prior and learns residual deviations plus a conservative blend; `totals_first_neural_share_residual` keeps the classical totals-first backbone but learns a bounded PyTorch residual on home share; `totals_first_neural_total_share_residual` adds bounded PyTorch residuals on both total corners and home share while preserving the totals-first backbone contract; `pmf_surface_blended` trains home/away count PMFs and blends the coherent hc*/ac*/c* surface back toward the stronger totals-first prior backbone; `totals_first_market_heads` keeps the direct total backbone but predicts hc*/ac* markets with direct binary heads; `totals_first_team_market_calibrated` adds calibrated/blended direct team-market heads on top of the totals-first backbone; `totals_surface_calibrated` models c75/c85/c95/c105 directly with calibrated monotone totals heads while keeping the stable totals-first team prior path; `totals_surface_neural_ladder_calibrated` keeps the stable totals-first backbone but trains a PyTorch totals ladder for c75/c85/c95/c105 only, then calibrates/blends those totals markets against the prior without widening team-market scope.",
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
    return parser.parse_args()


def _build_regressor(model_type: str) -> Any:
    if model_type == "poisson_glm":
        return PoissonRegressor(alpha=1.0, max_iter=500)
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=40,
        random_state=42,
    )


def _build_share_regressor() -> Any:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=4,
        max_iter=250,
        min_samples_leaf=40,
        random_state=42,
    )


def _build_residual_regressor() -> Any:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=4,
        max_iter=250,
        min_samples_leaf=40,
        random_state=42,
    )


def _build_binary_head_model() -> Any:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_depth=4,
        max_iter=250,
        min_samples_leaf=40,
        random_state=42,
    )


def _positive_class_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (int, float, np.integer, np.floating)):
        return np.full(len(x), float(np.clip(float(model), 0.001, 0.999)), dtype=float)
    probs = model.predict_proba(x)
    classes = [int(c) for c in getattr(model, "classes_", [0, 1])]
    if probs.ndim != 2:
        return np.full(len(x), 0.5, dtype=float)
    if probs.shape[1] == 1:
        value = 1.0 if classes and classes[0] == 1 else 0.0
        return np.full(len(x), value, dtype=float)
    idx = classes.index(1) if 1 in classes else probs.shape[1] - 1
    return np.clip(probs[:, idx], 0.001, 0.999)


def _build_count_distribution_model() -> Any:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=30,
        random_state=42,
    )


def _count_target(frame: pd.DataFrame, column: str, max_count: int = PMF_MAX_COUNT) -> np.ndarray:
    raw = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.clip(np.rint(raw), 0, max_count).astype(int)


def _full_count_probability_matrix(model: Any, x: pd.DataFrame, max_count: int = PMF_MAX_COUNT) -> np.ndarray:
    probs = model.predict_proba(x)
    classes = [int(c) for c in getattr(model, "classes_", [])]
    out = np.zeros((len(x), max_count + 1), dtype=float)
    if probs.ndim != 2 or probs.shape[0] != len(x):
        out[:, 0] = 1.0
        return out
    for idx, cls in enumerate(classes):
        out[:, min(int(cls), max_count)] += probs[:, idx]
    row_sums = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out / row_sums


def _pmf_expectation(pmf: np.ndarray) -> np.ndarray:
    support = np.arange(pmf.shape[1], dtype=float)
    return pmf @ support


def _derive_distribution_surface_probs(
    *,
    home_pmf: np.ndarray,
    away_pmf: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    total_pmf = np.zeros((home_pmf.shape[0], home_pmf.shape[1] + away_pmf.shape[1] - 1), dtype=float)
    for idx in range(home_pmf.shape[0]):
        total_pmf[idx] = np.convolve(home_pmf[idx], away_pmf[idx])
    total_probs: dict[str, np.ndarray] = {}
    for market, line in TOTAL_MARKETS.items():
        threshold = int(np.floor(float(line))) + 1
        total_probs[market] = total_pmf[:, threshold:].sum(axis=1)
    team_probs: dict[str, np.ndarray] = {}
    for market, line in HOME_MARKETS.items():
        threshold = int(np.floor(float(line))) + 1
        team_probs[market] = home_pmf[:, threshold:].sum(axis=1)
    for market, line in AWAY_MARKETS.items():
        threshold = int(np.floor(float(line))) + 1
        team_probs[market] = away_pmf[:, threshold:].sum(axis=1)
    return (
        _project_total_head_prob_arrays(total_probs),
        _project_team_head_prob_arrays(team_probs),
        _pmf_expectation(home_pmf),
        _pmf_expectation(away_pmf),
        _pmf_expectation(total_pmf),
    )


def _derive_full_market_prior_probs(
    *,
    total_mu: np.ndarray,
    home_mu: np.ndarray,
    away_mu: np.ndarray,
    total_r: float | None,
    home_r: float | None,
    away_r: float | None,
) -> dict[str, np.ndarray]:
    prior: dict[str, np.ndarray] = {}
    prior.update(_derive_total_market_prior_probs(total_mu=total_mu, total_r=total_r))
    prior.update(
        _derive_team_market_prior_probs(
            home_mu=home_mu,
            away_mu=away_mu,
            home_r=home_r,
            away_r=away_r,
        )
    )
    return prior


def _fit_blend_weight(
    *,
    prior_probs: np.ndarray | list[float],
    head_probs: np.ndarray | list[float],
    y_true: pd.Series | np.ndarray,
) -> float:
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


def _home_share_target(frame: pd.DataFrame) -> np.ndarray:
    total = frame["total_corners"].astype(float).to_numpy(dtype=float)
    home = frame["home_corners"].astype(float).to_numpy(dtype=float)
    share = (home + 0.5) / np.clip(total + 1.0, 1.0, None)
    return np.clip(np.nan_to_num(share, nan=0.5, posinf=0.95, neginf=0.05), 0.05, 0.95)


def _league_home_share_prior(frame: pd.DataFrame) -> np.ndarray:
    if "league_home_share_mean" not in frame.columns:
        return np.full(len(frame), 0.5, dtype=float)
    values = pd.to_numeric(frame["league_home_share_mean"], errors="coerce").to_numpy(dtype=float)
    return np.clip(np.nan_to_num(values, nan=0.5, posinf=0.95, neginf=0.05), 0.05, 0.95)


def _predict_share_with_prior(
    *,
    residual_model: Any,
    x: pd.DataFrame,
    blend_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior_share = _league_home_share_prior(x)
    raw_share = np.clip(prior_share + residual_model.predict(x), 0.05, 0.95)
    home_share = np.clip(prior_share + float(blend_alpha) * (raw_share - prior_share), 0.05, 0.95)
    return home_share, prior_share, raw_share


def _total_market_target(frame: pd.DataFrame, market: str) -> np.ndarray:
    target_col = f"target_{market}"
    if target_col in frame.columns:
        values = pd.to_numeric(frame[target_col], errors="coerce")
        if values.notna().all():
            return values.astype(int).to_numpy(dtype=int)
    raw = frame["total_corners"].astype(float).to_numpy(dtype=float)
    line = float(TOTAL_MARKETS[market])
    return (raw > line).astype(int)


def _team_market_target(frame: pd.DataFrame, market: str) -> np.ndarray:
    target_col = f"target_{market}"
    if target_col in frame.columns:
        values = pd.to_numeric(frame[target_col], errors="coerce")
        if values.notna().all():
            return values.astype(int).to_numpy(dtype=int)
    if market in HOME_MARKETS:
        raw = frame["home_corners"].astype(float).to_numpy(dtype=float)
        line = float(HOME_MARKETS[market])
    elif market in AWAY_MARKETS:
        raw = frame["away_corners"].astype(float).to_numpy(dtype=float)
        line = float(AWAY_MARKETS[market])
    else:
        raise KeyError(f"Unsupported team corners market: {market}")
    return (raw > line).astype(int)


def _fit_team_market_head_models(
    *,
    head_x: pd.DataFrame,
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for market in TEAM_HEAD_MARKETS:
        target = _team_market_target(train_df, market)
        if np.unique(target).size < 2:
            models[market] = float(np.clip(float(np.mean(target)), 0.001, 0.999))
            continue
        model = _build_binary_head_model()
        model.fit(head_x, target)
        models[market] = model
    return models


def _predict_team_market_head_probs(
    *,
    models: dict[str, Any],
    head_x: pd.DataFrame,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for market in TEAM_HEAD_MARKETS:
        out[market] = _positive_class_probability(models[market], head_x)
    return out


def _predict_team_means_from_backbone(
    *,
    total_model: Any,
    share_model: Any,
    x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    total_mu = np.clip(total_model.predict(x), 0.1, 30.0)
    home_share = np.clip(share_model.predict(x), 0.05, 0.95)
    home_mu: list[float] = []
    away_mu: list[float] = []
    for total, share in zip(total_mu, home_share, strict=True):
        hm, am = reconcile_team_means_from_total_share(
            total_mu=float(total),
            home_share=float(share),
        )
        home_mu.append(hm)
        away_mu.append(am)
    return np.asarray(home_mu, dtype=float), np.asarray(away_mu, dtype=float)


def _derive_total_market_prior_probs(
    *,
    total_mu: np.ndarray,
    total_r: float | None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    total_arr = np.asarray(total_mu, dtype=float)
    for market, line in TOTAL_MARKETS.items():
        out[market] = np.asarray(
            [over_probability(float(mu), line, total_r) for mu in total_arr],
            dtype=float,
        )
    return _project_total_head_prob_arrays(out)


def _build_total_head_frame(
    x: pd.DataFrame,
    prior_probs: dict[str, np.ndarray],
) -> pd.DataFrame:
    prior_frame = pd.DataFrame(
        {f"prior_{market}": np.asarray(values, dtype=float) for market, values in prior_probs.items()}
    )
    return pd.concat([x.reset_index(drop=True), prior_frame.reset_index(drop=True)], axis=1)


def _derive_team_market_prior_probs(
    *,
    home_mu: np.ndarray,
    away_mu: np.ndarray,
    home_r: float | None,
    away_r: float | None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    home_arr = np.asarray(home_mu, dtype=float)
    away_arr = np.asarray(away_mu, dtype=float)
    for market, line in HOME_MARKETS.items():
        out[market] = np.asarray(
            [over_probability(float(mu), line, home_r) for mu in home_arr],
            dtype=float,
        )
    for market, line in AWAY_MARKETS.items():
        out[market] = np.asarray(
            [over_probability(float(mu), line, away_r) for mu in away_arr],
            dtype=float,
        )
    return _project_team_head_prob_arrays(out)


def _build_team_head_frame(
    x: pd.DataFrame,
    prior_probs: dict[str, np.ndarray],
) -> pd.DataFrame:
    prior_frame = pd.DataFrame(
        {f"prior_{market}": np.asarray(values, dtype=float) for market, values in prior_probs.items()}
    )
    return pd.concat([x.reset_index(drop=True), prior_frame.reset_index(drop=True)], axis=1)


def _mean_team_market_brier(
    *,
    home_mu: np.ndarray,
    away_mu: np.ndarray,
    home_r: float | None,
    away_r: float | None,
    frame: pd.DataFrame,
) -> float:
    rows: list[float] = []
    home_arr = np.asarray(home_mu, dtype=float)
    away_arr = np.asarray(away_mu, dtype=float)
    for market, line in HOME_MARKETS.items():
        probs = np.asarray([over_probability(float(mu), line, home_r) for mu in home_arr], dtype=float)
        y_true = _team_market_target(frame, market)
        rows.append(float(np.mean((probs - y_true) ** 2)))
    for market, line in AWAY_MARKETS.items():
        probs = np.asarray([over_probability(float(mu), line, away_r) for mu in away_arr], dtype=float)
        y_true = _team_market_target(frame, market)
        rows.append(float(np.mean((probs - y_true) ** 2)))
    return float(np.mean(rows)) if rows else float("inf")


def _fit_share_prior_blend_weight(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
) -> float:
    sort_cols = [col for col in ["match_datetime_utc", "fixture_id"] if col in train_df.columns]
    ordered = (
        train_df.sort_values(sort_cols).reset_index(drop=True)
        if sort_cols
        else train_df.reset_index(drop=True)
    )
    oof_total = np.full(len(ordered), np.nan, dtype=float)
    oof_prior = _league_home_share_prior(ordered)
    oof_raw = np.full(len(ordered), np.nan, dtype=float)
    for train_end, test_end in _walkforward_ranges(len(ordered), folds=4):
        fold_train = ordered.iloc[:train_end].copy()
        fold_test = ordered.iloc[train_end:test_end].copy()
        if fold_train.empty or fold_test.empty:
            continue
        fold_total_model = _build_regressor(model_type)
        fold_total_model.fit(fold_train[features], fold_train["total_corners"].astype(float))
        fold_share_residual = _build_share_regressor()
        fold_share_residual.fit(
            fold_train[features],
            np.clip(_home_share_target(fold_train) - _league_home_share_prior(fold_train), -0.45, 0.45),
        )
        oof_total[train_end:test_end] = np.clip(fold_total_model.predict(fold_test[features]), 0.1, 30.0)
        _, _, raw_share = _predict_share_with_prior(
            residual_model=fold_share_residual,
            x=fold_test[features],
            blend_alpha=1.0,
        )
        oof_raw[train_end:test_end] = raw_share
    if not np.isfinite(oof_total).all() or not np.isfinite(oof_raw).all():
        total_model = _build_regressor(model_type)
        total_model.fit(ordered[features], ordered["total_corners"].astype(float))
        share_residual = _build_share_regressor()
        share_residual.fit(
            ordered[features],
            np.clip(_home_share_target(ordered) - _league_home_share_prior(ordered), -0.45, 0.45),
        )
        fill_total = np.clip(total_model.predict(ordered[features]), 0.1, 30.0)
        _, _, fill_raw = _predict_share_with_prior(
            residual_model=share_residual,
            x=ordered[features],
            blend_alpha=1.0,
        )
        oof_total = np.where(np.isfinite(oof_total), oof_total, fill_total)
        oof_raw = np.where(np.isfinite(oof_raw), oof_raw, fill_raw)
    home_r = estimate_nb_dispersion(ordered["home_corners"].to_numpy(dtype=float))
    away_r = estimate_nb_dispersion(ordered["away_corners"].to_numpy(dtype=float))
    best_alpha = 0.75
    best_brier = float("inf")
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0):
        share = np.clip(oof_prior + float(alpha) * (oof_raw - oof_prior), 0.05, 0.95)
        home_mu: list[float] = []
        away_mu: list[float] = []
        for total_value, share_value in zip(oof_total, share, strict=True):
            hm, am = reconcile_team_means_from_total_share(
                total_mu=float(total_value),
                home_share=float(share_value),
            )
            home_mu.append(hm)
            away_mu.append(am)
        brier = _mean_team_market_brier(
            home_mu=np.asarray(home_mu, dtype=float),
            away_mu=np.asarray(away_mu, dtype=float),
            home_r=home_r,
            away_r=away_r,
            frame=ordered,
        )
        if brier < best_brier - 1e-12:
            best_brier = brier
            best_alpha = float(alpha)
    return best_alpha


def _fit_total_market_head_models(
    *,
    head_x: pd.DataFrame,
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for market in TOTAL_HEAD_MARKETS:
        target = _total_market_target(train_df, market)
        if np.unique(target).size < 2:
            models[market] = float(np.clip(float(np.mean(target)), 0.001, 0.999))
            continue
        model = _build_binary_head_model()
        model.fit(head_x, target)
        models[market] = model
    return models


def _predict_total_market_head_probs(
    *,
    models: dict[str, Any],
    head_x: pd.DataFrame,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for market in TOTAL_HEAD_MARKETS:
        out[market] = _positive_class_probability(models[market], head_x)
    return out


def _project_total_head_prob_arrays(total_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {market: np.asarray(values, dtype=float).copy() for market, values in total_probs.items()}
    matrix = np.column_stack([out[market] for market in TOTAL_HEAD_MARKETS])
    matrix = np.minimum.accumulate(np.clip(matrix, 0.001, 0.999), axis=1)
    for idx, market in enumerate(TOTAL_HEAD_MARKETS):
        out[market] = matrix[:, idx]
    return out


def _fit_total_surface_models(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    total_model: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], dict[str, Any]]:
    ordered = train_df.copy()
    sort_cols = [col for col in ["match_datetime_utc", "fixture_id"] if col in ordered.columns]
    if sort_cols:
        ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)

    oof_raw = {market: np.full(len(ordered), np.nan, dtype=float) for market in TOTAL_HEAD_MARKETS}
    oof_prior = {market: np.full(len(ordered), np.nan, dtype=float) for market in TOTAL_HEAD_MARKETS}
    for train_end, test_end in _walkforward_ranges(len(ordered), folds=4):
        fold_train = ordered.iloc[:train_end].copy()
        fold_test = ordered.iloc[train_end:test_end].copy()
        if fold_train.empty or fold_test.empty:
            continue
        fold_total_model = _build_regressor(model_type)
        fold_total_model.fit(fold_train[features], fold_train["total_corners"].astype(float))
        fold_total_r = estimate_nb_dispersion(fold_train["total_corners"].to_numpy(dtype=float))
        fold_train_total = np.clip(fold_total_model.predict(fold_train[features]), 0.1, 30.0)
        fold_test_total = np.clip(fold_total_model.predict(fold_test[features]), 0.1, 30.0)
        fold_train_prior = _derive_total_market_prior_probs(total_mu=fold_train_total, total_r=fold_total_r)
        fold_test_prior = _derive_total_market_prior_probs(total_mu=fold_test_total, total_r=fold_total_r)
        fold_head_models = _fit_total_market_head_models(
            head_x=_build_total_head_frame(fold_train[features], fold_train_prior),
            train_df=fold_train,
        )
        fold_test_head_x = _build_total_head_frame(fold_test[features], fold_test_prior)
        fold_test_raw = _predict_total_market_head_probs(models=fold_head_models, head_x=fold_test_head_x)
        for market in TOTAL_HEAD_MARKETS:
            oof_prior[market][train_end:test_end] = fold_test_prior[market]
            oof_raw[market][train_end:test_end] = fold_test_raw[market]

    total_r_full = estimate_nb_dispersion(ordered["total_corners"].to_numpy(dtype=float))
    full_total = np.clip(total_model.predict(ordered[features]), 0.1, 30.0)
    full_prior = _derive_total_market_prior_probs(total_mu=full_total, total_r=total_r_full)
    full_head_x = _build_total_head_frame(ordered[features], full_prior)
    final_head_models = _fit_total_market_head_models(head_x=full_head_x, train_df=ordered)
    full_raw = _predict_total_market_head_probs(models=final_head_models, head_x=full_head_x)
    for market in TOTAL_HEAD_MARKETS:
        missing = ~np.isfinite(oof_raw[market])
        if np.any(missing):
            oof_raw[market][missing] = full_raw[market][missing]
            oof_prior[market][missing] = full_prior[market][missing]

    calibrators: dict[str, Any] = {}
    blend: dict[str, float] = {}
    calibration_report: dict[str, Any] = {}
    for market in TOTAL_HEAD_MARKETS:
        ordered_frame = pd.DataFrame(
            {
                "match_datetime_utc": ordered.get("match_datetime_utc"),
                "p_model": oof_raw[market],
                "y_true": _total_market_target(ordered, market),
            }
        )
        report, calibrator = evaluate_market_calibration(
            ordered_frame,
            fit_fraction=0.6,
            min_fit_rows=60,
            min_eval_rows=20,
            max_auc_drop=0.01,
        )
        selected = calibrator or {"method": "identity"}
        calibrators[market] = selected
        calibration_report[market] = report
        calibrated = apply_binary_calibrator(selected, oof_raw[market])
        blend[market] = _fit_blend_weight(
            prior_probs=oof_prior[market],
            head_probs=calibrated,
            y_true=_total_market_target(ordered, market),
        )
    return final_head_models, calibrators, blend, calibration_report


def _fit_neural_total_surface_models(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    total_model: Any,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    ordered = train_df.copy()
    sort_cols = [col for col in ["match_datetime_utc", "fixture_id"] if col in ordered.columns]
    if sort_cols:
        ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)

    oof_raw = {market: np.full(len(ordered), np.nan, dtype=float) for market in TOTAL_HEAD_MARKETS}
    oof_prior = {market: np.full(len(ordered), np.nan, dtype=float) for market in TOTAL_HEAD_MARKETS}
    for train_end, test_end in _walkforward_ranges(len(ordered), folds=4):
        fold_train = ordered.iloc[:train_end].copy()
        fold_test = ordered.iloc[train_end:test_end].copy()
        if fold_train.empty or fold_test.empty:
            continue
        fold_total_model = _build_regressor(model_type)
        fold_total_model.fit(fold_train[features], fold_train["total_corners"].astype(float))
        fold_total_r = estimate_nb_dispersion(fold_train["total_corners"].to_numpy(dtype=float))
        fold_train_total = np.clip(fold_total_model.predict(fold_train[features]), 0.1, 30.0)
        fold_test_total = np.clip(fold_total_model.predict(fold_test[features]), 0.1, 30.0)
        fold_train_prior = _derive_total_market_prior_probs(total_mu=fold_train_total, total_r=fold_total_r)
        fold_test_prior = _derive_total_market_prior_probs(total_mu=fold_test_total, total_r=fold_total_r)
        fold_bundle = fit_neural_total_market_ladder_bundle(
            fold_train[features],
            prior_total_probs=fold_train_prior,
            total_corners=fold_train["total_corners"].astype(float).to_numpy(dtype=float),
        )
        fold_test_raw = predict_neural_total_market_ladder_probs(
            fold_bundle,
            fold_test[features],
            prior_total_probs=fold_test_prior,
        )
        for market in TOTAL_HEAD_MARKETS:
            oof_prior[market][train_end:test_end] = fold_test_prior[market]
            oof_raw[market][train_end:test_end] = fold_test_raw[market]

    total_r_full = estimate_nb_dispersion(ordered["total_corners"].to_numpy(dtype=float))
    full_total = np.clip(total_model.predict(ordered[features]), 0.1, 30.0)
    full_prior = _derive_total_market_prior_probs(total_mu=full_total, total_r=total_r_full)
    final_bundle = fit_neural_total_market_ladder_bundle(
        ordered[features],
        prior_total_probs=full_prior,
        total_corners=ordered["total_corners"].astype(float).to_numpy(dtype=float),
    )

    calibrators: dict[str, Any] = {}
    blend: dict[str, float] = {}
    calibration_report: dict[str, Any] = {}
    for market in TOTAL_HEAD_MARKETS:
        y_true = _total_market_target(ordered, market)
        valid = np.isfinite(oof_raw[market]) & np.isfinite(oof_prior[market])
        if int(valid.sum()) >= 20:
            ordered_frame = pd.DataFrame(
                {
                    "p_model": oof_raw[market][valid],
                    "y_true": y_true[valid],
                }
            )
            if "match_datetime_utc" in ordered.columns:
                ordered_frame["match_datetime_utc"] = ordered.loc[valid, "match_datetime_utc"].to_numpy()
            report, calibrator = evaluate_market_calibration(
                ordered_frame,
                fit_fraction=0.6,
                min_fit_rows=60,
                min_eval_rows=20,
                max_auc_drop=0.01,
            )
        else:
            report, calibrator = ({"status": "insufficient_oof_rows", "rows_total": int(valid.sum())}, None)
        selected = calibrator or {"method": "identity"}
        calibrators[market] = selected
        report["oof_rows"] = int(valid.sum())
        calibration_report[market] = report
        calibrated = apply_binary_calibrator(selected, oof_raw[market][valid]) if np.any(valid) else np.asarray([], dtype=float)
        blend[market] = _fit_blend_weight(
            prior_probs=oof_prior[market][valid],
            head_probs=calibrated,
            y_true=y_true[valid],
        )
    return final_bundle, calibrators, blend, calibration_report


def _project_team_head_prob_arrays(team_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {market: np.asarray(values, dtype=float).copy() for market, values in team_probs.items()}
    for markets in (tuple(HOME_MARKETS.keys()), tuple(AWAY_MARKETS.keys())):
        matrix = np.column_stack([out[market] for market in markets])
        matrix = np.minimum.accumulate(np.clip(matrix, 0.001, 0.999), axis=1)
        for idx, market in enumerate(markets):
            out[market] = matrix[:, idx]
    return out


def _fit_team_surface_models(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    total_model: Any,
    share_model: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], dict[str, Any]]:
    ordered = train_df.copy()
    sort_cols = [col for col in ["match_datetime_utc", "fixture_id"] if col in ordered.columns]
    if sort_cols:
        ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)

    oof_raw = {market: np.full(len(ordered), np.nan, dtype=float) for market in TEAM_HEAD_MARKETS}
    oof_prior = {market: np.full(len(ordered), np.nan, dtype=float) for market in TEAM_HEAD_MARKETS}
    for train_end, test_end in _walkforward_ranges(len(ordered), folds=4):
        fold_train = ordered.iloc[:train_end].copy()
        fold_test = ordered.iloc[train_end:test_end].copy()
        if fold_train.empty or fold_test.empty:
            continue
        fold_total_model = _build_regressor(model_type)
        fold_share_model = _build_share_regressor()
        fold_total_model.fit(fold_train[features], fold_train["total_corners"].astype(float))
        fold_share_model.fit(fold_train[features], _home_share_target(fold_train))
        fold_home_r = estimate_nb_dispersion(fold_train["home_corners"].to_numpy(dtype=float))
        fold_away_r = estimate_nb_dispersion(fold_train["away_corners"].to_numpy(dtype=float))
        fold_train_home, fold_train_away = _predict_team_means_from_backbone(
            total_model=fold_total_model,
            share_model=fold_share_model,
            x=fold_train[features],
        )
        fold_test_home, fold_test_away = _predict_team_means_from_backbone(
            total_model=fold_total_model,
            share_model=fold_share_model,
            x=fold_test[features],
        )
        fold_train_prior = _derive_team_market_prior_probs(
            home_mu=fold_train_home,
            away_mu=fold_train_away,
            home_r=fold_home_r,
            away_r=fold_away_r,
        )
        fold_test_prior = _derive_team_market_prior_probs(
            home_mu=fold_test_home,
            away_mu=fold_test_away,
            home_r=fold_home_r,
            away_r=fold_away_r,
        )
        fold_head_models = _fit_team_market_head_models(
            head_x=_build_team_head_frame(fold_train[features], fold_train_prior),
            train_df=fold_train,
        )
        fold_test_head_x = _build_team_head_frame(fold_test[features], fold_test_prior)
        fold_test_raw = _predict_team_market_head_probs(models=fold_head_models, head_x=fold_test_head_x)
        for market in TEAM_HEAD_MARKETS:
            oof_prior[market][train_end:test_end] = fold_test_prior[market]
            oof_raw[market][train_end:test_end] = fold_test_raw[market]

    home_r_full = estimate_nb_dispersion(ordered["home_corners"].to_numpy(dtype=float))
    away_r_full = estimate_nb_dispersion(ordered["away_corners"].to_numpy(dtype=float))
    full_home, full_away = _predict_team_means_from_backbone(
        total_model=total_model,
        share_model=share_model,
        x=ordered[features],
    )
    full_prior = _derive_team_market_prior_probs(
        home_mu=full_home,
        away_mu=full_away,
        home_r=home_r_full,
        away_r=away_r_full,
    )
    full_head_x = _build_team_head_frame(ordered[features], full_prior)
    final_head_models = _fit_team_market_head_models(head_x=full_head_x, train_df=ordered)
    full_raw = _predict_team_market_head_probs(models=final_head_models, head_x=full_head_x)
    for market in TEAM_HEAD_MARKETS:
        missing = ~np.isfinite(oof_raw[market])
        if np.any(missing):
            oof_raw[market][missing] = full_raw[market][missing]
            oof_prior[market][missing] = full_prior[market][missing]

    calibrators: dict[str, Any] = {}
    blend: dict[str, float] = {}
    calibration_report: dict[str, Any] = {}
    for market in TEAM_HEAD_MARKETS:
        ordered_frame = pd.DataFrame(
            {
                "match_datetime_utc": ordered.get("match_datetime_utc"),
                "p_model": oof_raw[market],
                "y_true": _team_market_target(ordered, market),
            }
        )
        report, calibrator = evaluate_market_calibration(
            ordered_frame,
            fit_fraction=0.6,
            min_fit_rows=60,
            min_eval_rows=20,
            max_auc_drop=0.01,
        )
        selected = calibrator or {"method": "identity"}
        calibrators[market] = selected
        calibration_report[market] = report
        calibrated = apply_binary_calibrator(selected, oof_raw[market])
        blend[market] = _fit_blend_weight(
            prior_probs=oof_prior[market],
            head_probs=calibrated,
            y_true=_team_market_target(ordered, market),
        )
    return final_head_models, calibrators, blend, calibration_report


def _fit_pmf_surface_models(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    total_model: Any,
    share_model: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], dict[str, Any]]:
    ordered = train_df.copy()
    sort_cols = [col for col in ["match_datetime_utc", "fixture_id"] if col in ordered.columns]
    if sort_cols:
        ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)

    all_markets = TOTAL_HEAD_MARKETS + TEAM_HEAD_MARKETS
    oof_raw = {market: np.full(len(ordered), np.nan, dtype=float) for market in all_markets}
    oof_prior = {market: np.full(len(ordered), np.nan, dtype=float) for market in all_markets}
    for train_end, test_end in _walkforward_ranges(len(ordered), folds=4):
        fold_train = ordered.iloc[:train_end].copy()
        fold_test = ordered.iloc[train_end:test_end].copy()
        if fold_train.empty or fold_test.empty:
            continue
        fold_total_model = _build_regressor(model_type)
        fold_share_model = _build_share_regressor()
        fold_total_model.fit(fold_train[features], fold_train["total_corners"].astype(float))
        fold_share_model.fit(fold_train[features], _home_share_target(fold_train))
        fold_total_mu = np.clip(fold_total_model.predict(fold_test[features]), 0.1, 30.0)
        fold_share = np.clip(fold_share_model.predict(fold_test[features]), 0.05, 0.95)
        fold_home_mu: list[float] = []
        fold_away_mu: list[float] = []
        for total_value, share_value in zip(fold_total_mu, fold_share, strict=True):
            hm, am = reconcile_team_means_from_total_share(
                total_mu=float(total_value),
                home_share=float(share_value),
            )
            fold_home_mu.append(hm)
            fold_away_mu.append(am)
        fold_total_r = estimate_nb_dispersion(fold_train["total_corners"].to_numpy(dtype=float))
        fold_home_r = estimate_nb_dispersion(fold_train["home_corners"].to_numpy(dtype=float))
        fold_away_r = estimate_nb_dispersion(fold_train["away_corners"].to_numpy(dtype=float))
        fold_prior = _derive_full_market_prior_probs(
            total_mu=np.asarray(fold_total_mu, dtype=float),
            home_mu=np.asarray(fold_home_mu, dtype=float),
            away_mu=np.asarray(fold_away_mu, dtype=float),
            total_r=fold_total_r,
            home_r=fold_home_r,
            away_r=fold_away_r,
        )
        fold_home_model = _build_count_distribution_model()
        fold_away_model = _build_count_distribution_model()
        fold_home_model.fit(fold_train[features], _count_target(fold_train, "home_corners"))
        fold_away_model.fit(fold_train[features], _count_target(fold_train, "away_corners"))
        fold_home_pmf = _full_count_probability_matrix(fold_home_model, fold_test[features])
        fold_away_pmf = _full_count_probability_matrix(fold_away_model, fold_test[features])
        fold_total_probs, fold_team_probs, _, _, _ = _derive_distribution_surface_probs(
            home_pmf=fold_home_pmf,
            away_pmf=fold_away_pmf,
        )
        for market in TOTAL_HEAD_MARKETS:
            oof_prior[market][train_end:test_end] = fold_prior[market]
            oof_raw[market][train_end:test_end] = fold_total_probs[market]
        for market in TEAM_HEAD_MARKETS:
            oof_prior[market][train_end:test_end] = fold_prior[market]
            oof_raw[market][train_end:test_end] = fold_team_probs[market]

    final_home_model = _build_count_distribution_model()
    final_away_model = _build_count_distribution_model()
    final_home_model.fit(ordered[features], _count_target(ordered, "home_corners"))
    final_away_model.fit(ordered[features], _count_target(ordered, "away_corners"))
    full_home_pmf = _full_count_probability_matrix(final_home_model, ordered[features])
    full_away_pmf = _full_count_probability_matrix(final_away_model, ordered[features])
    full_total_probs, full_team_probs, _, _, _ = _derive_distribution_surface_probs(
        home_pmf=full_home_pmf,
        away_pmf=full_away_pmf,
    )
    total_r_full = estimate_nb_dispersion(ordered["total_corners"].to_numpy(dtype=float))
    home_r_full = estimate_nb_dispersion(ordered["home_corners"].to_numpy(dtype=float))
    away_r_full = estimate_nb_dispersion(ordered["away_corners"].to_numpy(dtype=float))
    full_total_mu = np.clip(total_model.predict(ordered[features]), 0.1, 30.0)
    full_share = np.clip(share_model.predict(ordered[features]), 0.05, 0.95)
    full_home_mu: list[float] = []
    full_away_mu: list[float] = []
    for total_value, share_value in zip(full_total_mu, full_share, strict=True):
        hm, am = reconcile_team_means_from_total_share(
            total_mu=float(total_value),
            home_share=float(share_value),
        )
        full_home_mu.append(hm)
        full_away_mu.append(am)
    full_prior = _derive_full_market_prior_probs(
        total_mu=np.asarray(full_total_mu, dtype=float),
        home_mu=np.asarray(full_home_mu, dtype=float),
        away_mu=np.asarray(full_away_mu, dtype=float),
        total_r=total_r_full,
        home_r=home_r_full,
        away_r=away_r_full,
    )
    for market in all_markets:
        full_market_raw = full_total_probs[market] if market in TOTAL_HEAD_MARKETS else full_team_probs[market]
        missing = ~np.isfinite(oof_raw[market])
        if np.any(missing):
            oof_raw[market][missing] = full_market_raw[missing]
            oof_prior[market][missing] = full_prior[market][missing]

    calibrators: dict[str, Any] = {}
    blend: dict[str, float] = {}
    calibration_report: dict[str, Any] = {}
    for market in all_markets:
        y_true = _total_market_target(ordered, market) if market in TOTAL_HEAD_MARKETS else _team_market_target(ordered, market)
        ordered_frame = pd.DataFrame(
            {
                "match_datetime_utc": ordered.get("match_datetime_utc"),
                "p_model": oof_raw[market],
                "y_true": y_true,
            }
        )
        report, calibrator = evaluate_market_calibration(
            ordered_frame,
            fit_fraction=0.6,
            min_fit_rows=60,
            min_eval_rows=20,
            max_auc_drop=0.01,
        )
        selected = calibrator or {"method": "identity"}
        calibrators[market] = selected
        calibration_report[market] = report
        calibrated = apply_binary_calibrator(selected, oof_raw[market])
        blend[market] = _fit_blend_weight(
            prior_probs=oof_prior[market],
            head_probs=calibrated,
            y_true=y_true,
        )
    return (
        {"home": final_home_model, "away": final_away_model, "max_count": PMF_MAX_COUNT},
        calibrators,
        blend,
        calibration_report,
    )


def _fit_home_delta_model(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    fallback_total_model: Any,
    fallback_share_model: Any,
) -> Any:
    ordered = train_df.copy()
    sort_cols = [col for col in ["match_datetime_utc", "fixture_id"] if col in ordered.columns]
    if sort_cols:
        ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)

    oof_base_home = np.full(len(ordered), np.nan, dtype=float)
    for train_end, test_end in _walkforward_ranges(len(ordered), folds=4):
        fold_train = ordered.iloc[:train_end].copy()
        fold_test = ordered.iloc[train_end:test_end].copy()
        if fold_train.empty or fold_test.empty:
            continue
        fold_total_model = _build_regressor(model_type)
        fold_share_model = _build_share_regressor()
        fold_total_model.fit(fold_train[features], fold_train["total_corners"].astype(float))
        fold_share_model.fit(fold_train[features], _home_share_target(fold_train))
        fold_total = np.clip(fold_total_model.predict(fold_test[features]), 0.1, 30.0)
        fold_share = np.clip(fold_share_model.predict(fold_test[features]), 0.05, 0.95)
        for idx, (total_value, share_value) in enumerate(zip(fold_total, fold_share, strict=True), start=train_end):
            home_mu, _ = reconcile_team_means_from_total_share(
                total_mu=float(total_value),
                home_share=float(share_value),
            )
            oof_base_home[idx] = home_mu

    if not np.isfinite(oof_base_home).all():
        fallback_total = np.clip(fallback_total_model.predict(ordered[features]), 0.1, 30.0)
        fallback_share = np.clip(fallback_share_model.predict(ordered[features]), 0.05, 0.95)
        for idx, (total_value, share_value) in enumerate(zip(fallback_total, fallback_share, strict=True)):
            if np.isfinite(oof_base_home[idx]):
                continue
            home_mu, _ = reconcile_team_means_from_total_share(
                total_mu=float(total_value),
                home_share=float(share_value),
            )
            oof_base_home[idx] = home_mu

    residual_target = np.clip(
        ordered["home_corners"].astype(float).to_numpy(dtype=float) - oof_base_home,
        -8.0,
        8.0,
    )
    residual_model = _build_residual_regressor()
    residual_model.fit(ordered[features], residual_target)
    return residual_model


def _fit_corner_models(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    path_version: str,
) -> dict[str, Any]:
    x_train = train_df[features]
    if path_version in TOTALS_FIRST_PATHS:
        total_model = _build_regressor(model_type)
        share_model = _build_share_regressor()
        total_model.fit(x_train, train_df["total_corners"].astype(float))
        share_model.fit(x_train, _home_share_target(train_df))
        models: dict[str, Any] = {"total": total_model, "home_share": share_model}
        if path_version == "totals_first_residual":
            models["home_delta"] = _fit_home_delta_model(
                train_df=train_df,
                features=features,
                model_type=model_type,
                fallback_total_model=total_model,
                fallback_share_model=share_model,
            )
        elif path_version in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS:
            base_total = np.clip(total_model.predict(x_train), 0.1, 30.0)
            base_share = np.clip(share_model.predict(x_train), 0.05, 0.95)
            models["neural_total_residual"] = fit_neural_total_residual_bundle(
                x_train,
                base_total_mu=base_total,
                base_home_share=base_share,
                target_total_mu=train_df["total_corners"].astype(float).to_numpy(dtype=float),
            )
            adjusted_total = np.clip(
                base_total
                + predict_neural_total_residual_delta(
                    models["neural_total_residual"],
                    x_train,
                    base_total_mu=base_total,
                    base_home_share=base_share,
                ),
                0.1,
                30.0,
            )
            models["neural_share_residual"] = fit_neural_share_residual_bundle(
                x_train,
                base_total_mu=adjusted_total,
                base_home_share=base_share,
                target_home_share=np.clip(_home_share_target(train_df), 0.05, 0.95),
            )
        elif path_version in NEURAL_SHARE_RESIDUAL_PATHS:
            base_total = np.clip(total_model.predict(x_train), 0.1, 30.0)
            base_share = np.clip(share_model.predict(x_train), 0.05, 0.95)
            models["neural_share_residual"] = fit_neural_share_residual_bundle(
                x_train,
                base_total_mu=base_total,
                base_home_share=base_share,
                target_home_share=np.clip(_home_share_target(train_df), 0.05, 0.95),
            )
        elif path_version in SHARE_PRIOR_PATHS:
            share_residual = _build_share_regressor()
            share_residual.fit(
                x_train,
                np.clip(_home_share_target(train_df) - _league_home_share_prior(train_df), -0.45, 0.45),
            )
            models["home_share_residual"] = share_residual
            models["home_share_prior_blend"] = _fit_share_prior_blend_weight(
                train_df=train_df,
                features=features,
                model_type=model_type,
            )
        elif path_version in PMF_SURFACE_PATHS:
            (
                models["pmf_surface_models"],
                models["pmf_market_calibrators"],
                models["pmf_market_blend"],
                models["pmf_market_calibration_report"],
            ) = _fit_pmf_surface_models(
                train_df=train_df,
                features=features,
                model_type=model_type,
                total_model=total_model,
                share_model=share_model,
            )
        elif path_version in TOTAL_SURFACE_PATHS:
            (
                models["total_market_heads"],
                models["total_market_head_calibrators"],
                models["total_market_head_blend"],
                models["total_market_head_calibration_report"],
            ) = _fit_total_surface_models(
                train_df=train_df,
                features=features,
                model_type=model_type,
                total_model=total_model,
            )
        elif path_version in NEURAL_TOTAL_LADDER_PATHS:
            (
                models["neural_total_market_ladder"],
                models["total_market_ladder_calibrators"],
                models["total_market_ladder_blend"],
                models["total_market_ladder_calibration_report"],
            ) = _fit_neural_total_surface_models(
                train_df=train_df,
                features=features,
                model_type=model_type,
                total_model=total_model,
            )
        elif path_version in CALIBRATED_TEAM_HEAD_PATHS:
            (
                models["team_market_heads"],
                models["team_market_head_calibrators"],
                models["team_market_head_blend"],
                models["team_market_head_calibration_report"],
            ) = _fit_team_surface_models(
                train_df=train_df,
                features=features,
                model_type=model_type,
                total_model=total_model,
                share_model=share_model,
            )
        elif path_version in MARKET_HEAD_PATHS:
            models["team_market_heads"] = _fit_team_market_head_models(
                head_x=train_df[features],
                train_df=train_df,
            )
        return models
    home_model = _build_regressor(model_type)
    away_model = _build_regressor(model_type)
    home_model.fit(x_train, train_df["home_corners"].astype(float))
    away_model.fit(x_train, train_df["away_corners"].astype(float))
    return {"home": home_model, "away": away_model}


def _predict_corner_rates(
    *,
    models: dict[str, Any],
    x: pd.DataFrame,
    path_version: str,
    total_r: float | None = None,
    home_r: float | None = None,
    away_r: float | None = None,
) -> dict[str, Any]:
    if path_version in TOTALS_FIRST_PATHS:
        total_mu = np.clip(models["total"].predict(x), 0.1, 30.0)
        total_mu_base = np.asarray(total_mu, dtype=float)
        home_share_prior = None
        home_share_raw = None
        total_raw = np.asarray(total_mu, dtype=float)
        total_delta = None
        neural_share_delta = None
        if path_version in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS:
            base_share_for_total = np.clip(models["home_share"].predict(x), 0.05, 0.95)
            total_delta = np.asarray(
                predict_neural_total_residual_delta(
                    models["neural_total_residual"],
                    x,
                    base_total_mu=total_mu_base,
                    base_home_share=base_share_for_total,
                ),
                dtype=float,
            )
            total_raw = total_mu_base + total_delta
            total_mu = np.clip(total_raw, 0.1, 30.0)
        if path_version in SHARE_PRIOR_PATHS:
            home_share, home_share_prior, home_share_raw = _predict_share_with_prior(
                residual_model=models["home_share_residual"],
                x=x,
                blend_alpha=float(models.get("home_share_prior_blend", 1.0)),
            )
        else:
            home_share = np.clip(models["home_share"].predict(x), 0.05, 0.95)
        base_share = np.asarray(home_share, dtype=float)
        if path_version in (NEURAL_SHARE_RESIDUAL_PATHS | NEURAL_TOTAL_SHARE_RESIDUAL_PATHS):
            neural_share_delta = np.asarray(
                predict_neural_share_residual_delta(
                    models["neural_share_residual"],
                    x,
                    base_total_mu=np.asarray(total_mu, dtype=float),
                    base_home_share=base_share,
                ),
                dtype=float,
            )
            home_share_raw = base_share + neural_share_delta
            home_share = np.clip(home_share_raw, 0.05, 0.95)
        base_home_mu: list[float] = []
        home_mu: list[float] = []
        away_mu: list[float] = []
        for total_value, base_share_value, share_value in zip(total_mu, base_share, home_share, strict=True):
            base_hm, _ = reconcile_team_means_from_total_share(
                total_mu=float(total_value),
                home_share=float(base_share_value),
            )
            hm, am = reconcile_team_means_from_total_share(
                total_mu=float(total_value),
                home_share=float(share_value),
            )
            base_home_mu.append(base_hm)
            home_mu.append(hm)
            away_mu.append(am)
        home_delta = None
        if path_version == "totals_first_residual":
            home_delta = np.clip(models["home_delta"].predict(x), -8.0, 8.0)
            adjusted_home: list[float] = []
            adjusted_away: list[float] = []
            for total_value, base_home_value, delta_value in zip(
                total_mu,
                base_home_mu,
                home_delta,
                strict=True,
            ):
                hm, am = reconcile_team_means_from_total_home_mu(
                    total_mu=float(total_value),
                    proposed_home_mu=float(base_home_value + delta_value),
                )
                adjusted_home.append(hm)
                adjusted_away.append(am)
            home_mu = adjusted_home
            away_mu = adjusted_away
        out: dict[str, Any] = {
            "total": np.asarray(total_mu, dtype=float),
            "base_total": total_mu_base,
            "total_raw": np.asarray(total_raw, dtype=float),
            "total_residual_delta": (
                np.asarray(total_delta, dtype=float)
                if total_delta is not None
                else np.zeros(len(total_mu), dtype=float)
            ),
            "home_share": np.asarray(home_share, dtype=float),
            "home_share_prior": (
                np.asarray(home_share_prior, dtype=float)
                if home_share_prior is not None
                else np.full(len(total_mu), np.nan, dtype=float)
            ),
            "home_share_raw": (
                np.asarray(home_share_raw, dtype=float)
                if home_share_raw is not None
                else np.asarray(home_share, dtype=float)
            ),
            "base_home": np.asarray(base_home_mu, dtype=float),
            "home_delta": (
                np.asarray(home_delta, dtype=float)
                if home_delta is not None
                else np.zeros(len(total_mu), dtype=float)
            ),
            "home_share_residual_delta": (
                np.asarray(neural_share_delta, dtype=float)
                if neural_share_delta is not None
                else np.zeros(len(total_mu), dtype=float)
            ),
            "home": np.asarray(home_mu, dtype=float),
            "away": np.asarray(away_mu, dtype=float),
        }
        if path_version in PMF_SURFACE_PATHS:
            pmf_models = models["pmf_surface_models"]
            max_count = int(pmf_models.get("max_count", PMF_MAX_COUNT))
            home_pmf = _full_count_probability_matrix(pmf_models["home"], x, max_count)
            away_pmf = _full_count_probability_matrix(pmf_models["away"], x, max_count)
            raw_total_probs, raw_team_probs, raw_home_mu, raw_away_mu, raw_total_mu = _derive_distribution_surface_probs(
                home_pmf=home_pmf,
                away_pmf=away_pmf,
            )
            prior_all = _derive_full_market_prior_probs(
                total_mu=np.asarray(total_mu, dtype=float),
                home_mu=np.asarray(home_mu, dtype=float),
                away_mu=np.asarray(away_mu, dtype=float),
                total_r=total_r,
                home_r=home_r,
                away_r=away_r,
            )
            total_market_probs: dict[str, np.ndarray] = {}
            team_market_probs: dict[str, np.ndarray] = {}
            calibrators = models.get("pmf_market_calibrators") or {}
            blend = models.get("pmf_market_blend") or {}
            for market in TOTAL_HEAD_MARKETS:
                calibrated = apply_binary_calibrator(
                    calibrators.get(market, {"method": "identity"}),
                    raw_total_probs[market],
                )
                alpha = float(blend.get(market, 0.35))
                total_market_probs[market] = np.clip(
                    prior_all[market] + alpha * (calibrated - prior_all[market]),
                    0.001,
                    0.999,
                )
            for market in TEAM_HEAD_MARKETS:
                calibrated = apply_binary_calibrator(
                    calibrators.get(market, {"method": "identity"}),
                    raw_team_probs[market],
                )
                alpha = float(blend.get(market, 0.35))
                team_market_probs[market] = np.clip(
                    prior_all[market] + alpha * (calibrated - prior_all[market]),
                    0.001,
                    0.999,
                )
            out["total_market_probs"] = _project_total_head_prob_arrays(total_market_probs)
            out["team_market_probs"] = _project_team_head_prob_arrays(team_market_probs)
            out["pmf_home"] = np.asarray(raw_home_mu, dtype=float)
            out["pmf_away"] = np.asarray(raw_away_mu, dtype=float)
            out["pmf_total"] = np.asarray(raw_total_mu, dtype=float)
        if path_version in TOTAL_SURFACE_PATHS:
            prior_total_probs = _derive_total_market_prior_probs(
                total_mu=np.asarray(total_mu, dtype=float),
                total_r=total_r,
            )
            head_x = _build_total_head_frame(x, prior_total_probs)
            raw_total_head_probs = _predict_total_market_head_probs(
                models=models["total_market_heads"],
                head_x=head_x,
            )
            total_market_probs: dict[str, np.ndarray] = {}
            calibrators = models.get("total_market_head_calibrators") or {}
            blend = models.get("total_market_head_blend") or {}
            for market in TOTAL_HEAD_MARKETS:
                calibrated = apply_binary_calibrator(
                    calibrators.get(market, {"method": "identity"}),
                    raw_total_head_probs[market],
                )
                alpha = float(blend.get(market, 0.35))
                total_market_probs[market] = np.clip(
                    prior_total_probs[market] + alpha * (calibrated - prior_total_probs[market]),
                    0.001,
                    0.999,
                )
            out["total_market_probs"] = _project_total_head_prob_arrays(total_market_probs)
        if path_version in NEURAL_TOTAL_LADDER_PATHS:
            prior_total_probs = _derive_total_market_prior_probs(
                total_mu=np.asarray(total_mu, dtype=float),
                total_r=total_r,
            )
            raw_total_probs = predict_neural_total_market_ladder_probs(
                models["neural_total_market_ladder"],
                x,
                prior_total_probs=prior_total_probs,
            )
            calibrators = models.get("total_market_ladder_calibrators") or {}
            blend = models.get("total_market_ladder_blend") or {}
            total_market_probs: dict[str, np.ndarray] = {}
            for market in TOTAL_HEAD_MARKETS:
                calibrated = apply_binary_calibrator(
                    calibrators.get(market, {"method": "identity"}),
                    raw_total_probs[market],
                )
                alpha = float(blend.get(market, 0.35))
                total_market_probs[market] = np.clip(
                    prior_total_probs[market] + alpha * (calibrated - prior_total_probs[market]),
                    0.001,
                    0.999,
                )
            out["total_market_probs"] = _project_total_head_prob_arrays(total_market_probs)
        if path_version in MARKET_HEAD_PATHS:
            if path_version in CALIBRATED_TEAM_HEAD_PATHS:
                prior_team_probs = _derive_team_market_prior_probs(
                    home_mu=np.asarray(home_mu, dtype=float),
                    away_mu=np.asarray(away_mu, dtype=float),
                    home_r=home_r,
                    away_r=away_r,
                )
                head_x = _build_team_head_frame(x, prior_team_probs)
                raw_team_probs = _predict_team_market_head_probs(
                    models=models["team_market_heads"],
                    head_x=head_x,
                )
                team_market_probs: dict[str, np.ndarray] = {}
                calibrators = models.get("team_market_head_calibrators") or {}
                blend = models.get("team_market_head_blend") or {}
                for market in TEAM_HEAD_MARKETS:
                    calibrated = apply_binary_calibrator(
                        calibrators.get(market, {"method": "identity"}),
                        raw_team_probs[market],
                    )
                    alpha = float(blend.get(market, 0.35))
                    team_market_probs[market] = np.clip(
                        prior_team_probs[market] + alpha * (calibrated - prior_team_probs[market]),
                        0.001,
                        0.999,
                    )
                out["team_market_probs"] = _project_team_head_prob_arrays(team_market_probs)
            else:
                out["team_market_probs"] = _project_team_head_prob_arrays(
                    _predict_team_market_head_probs(models=models["team_market_heads"], head_x=x)
                )
        return out
    return {
        "home": np.clip(models["home"].predict(x), 0.05, 20.0),
        "away": np.clip(models["away"].predict(x), 0.05, 20.0),
    }


def _derive_corner_frame(
    *,
    preds: dict[str, Any],
    total_r: float | None,
    home_r: float | None,
    away_r: float | None,
    path_version: str,
) -> pd.DataFrame:
    derived_rows: list[dict[str, float]] = []
    total_override = preds.get("total") if path_version in TOTALS_FIRST_PATHS else None
    total_market_probs = preds.get("total_market_probs")
    team_market_probs = preds.get("team_market_probs")
    for idx, (home_mu, away_mu) in enumerate(zip(preds["home"], preds["away"], strict=True)):
        total_market_overrides = None
        if total_market_probs is not None:
            total_market_overrides = {
                market: float(values[idx])
                for market, values in total_market_probs.items()
            }
        team_overrides = None
        if team_market_probs is not None:
            team_overrides = {
                market: float(values[idx])
                for market, values in team_market_probs.items()
            }
        derived_rows.append(
            derive_and_validate_corners(
                home_mu=float(home_mu),
                away_mu=float(away_mu),
                total_r=total_r,
                home_r=home_r,
                away_r=away_r,
                total_mu_override=(float(total_override[idx]) if total_override is not None else None),
                total_market_overrides=total_market_overrides,
                team_market_overrides=team_overrides,
            )
        )
    return pd.DataFrame(derived_rows)


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
        frame = pd.read_csv(dataset_path)
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
            "Missing required corners contract features: "
            + ", ".join(sorted(missing_required))
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
    scope_markets: set[str],
    model_type: str,
    path_version: str,
    folds: int,
    min_fold_test_n: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float | int | None]],
    list[dict[str, Any]],
]:
    ordered = add_corners_context_features(frame).sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    fold_rows: list[dict[str, Any]] = []
    fold_league_rows: list[dict[str, Any]] = []
    for fold_idx, (train_end, test_end) in enumerate(
        _walkforward_ranges(len(ordered), folds), start=1
    ):
        train_df = ordered.iloc[:train_end].copy()
        test_df = ordered.iloc[train_end:test_end].copy()
        if train_df.empty or test_df.empty:
            continue

        league_regime = fit_league_regime(train_df)
        train_df = apply_league_regime(train_df, league_regime)
        test_df = apply_league_regime(test_df, league_regime)

        imputation = _fit_imputation(train_df, features)
        train_df = _apply_imputation(train_df, imputation)
        test_df = _apply_imputation(test_df, imputation)

        x_test = test_df[features]
        models = _fit_corner_models(
            train_df=train_df,
            features=features,
            model_type=model_type,
            path_version=path_version,
        )
        total_r = estimate_nb_dispersion(train_df["total_corners"].to_numpy(dtype=float))
        home_r = estimate_nb_dispersion(train_df["home_corners"].to_numpy(dtype=float))
        away_r = estimate_nb_dispersion(train_df["away_corners"].to_numpy(dtype=float))
        preds = _predict_corner_rates(
            models=models,
            x=x_test,
            path_version=path_version,
            total_r=total_r,
            home_r=home_r,
            away_r=away_r,
        )

        pred_frame = _derive_corner_frame(
            preds=preds,
            total_r=total_r,
            home_r=home_r,
            away_r=away_r,
            path_version=path_version,
        )

        for market in sorted(pred_frame.columns):
            if market not in scope_markets:
                continue
            target_col = f"target_{market}"
            if target_col not in test_df.columns:
                continue
            valid = test_df[target_col].notna().to_numpy(dtype=bool)
            valid_n = int(valid.sum())
            if valid_n < int(min_fold_test_n):
                continue
            y_true = test_df[target_col].to_numpy(dtype=float)[valid].astype(int)
            p_series = pred_frame[market]
            if isinstance(p_series, pd.DataFrame):
                p_values = p_series.iloc[:, 0].to_numpy(dtype=float)
            else:
                p_values = p_series.to_numpy(dtype=float)
            p_true = np.clip(p_values[valid], 0.001, 0.999)
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
    scope_markets: set[str],
    path_version: str,
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
            scope_markets=scope_markets,
            model_type=model_type,
            path_version=path_version,
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
        raise RuntimeError("No rows available for corners training.")
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
    df = add_corners_context_features(df)

    # Require actual corners labels for model fitting.
    df = df[df["home_corners"].notna() & df["away_corners"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rows with home/away corners labels available.")

    features = _select_features(df, args.contract)
    scope_markets = set(load_scope_markets(args.scope))
    model_type_requested = str(args.model_type)
    model_selection_summary: dict[str, dict[str, float | int | None]]
    walkforward_league_rows: list[dict[str, Any]]
    if model_type_requested == "auto":
        model_type_selected, walkforward_rows, walkforward_summary, model_selection_summary, walkforward_league_rows = (
            _select_model_type(
                frame=df,
                features=features,
                scope_markets=scope_markets,
                path_version=str(args.path_version),
                folds=int(args.folds),
                min_fold_test_n=int(args.min_fold_test_n),
            )
        )
    else:
        model_type_selected = model_type_requested
        walkforward_rows, walkforward_summary, walkforward_league_rows = _evaluate_walkforward(
            frame=df,
            features=features,
            scope_markets=scope_markets,
            model_type=model_type_selected,
            path_version=str(args.path_version),
            folds=int(args.folds),
            min_fold_test_n=int(args.min_fold_test_n),
        )
        model_selection_summary = {
            model_type_selected: _aggregate_walkforward_quality(walkforward_summary)
        }

    train_df_raw, test_df_raw = legacy_calibrator.split_time_respecting(df)
    league_regime = fit_league_regime(train_df_raw)
    train_df = apply_league_regime(train_df_raw, league_regime)
    test_df = apply_league_regime(test_df_raw, league_regime)
    imputation = _fit_imputation(train_df, features)
    train_df = _apply_imputation(train_df, imputation)
    test_df = _apply_imputation(test_df, imputation)

    x_test = test_df[features]
    y_home_test = test_df["home_corners"].astype(float)
    y_away_test = test_df["away_corners"].astype(float)

    models = _fit_corner_models(
        train_df=train_df,
        features=features,
        model_type=model_type_selected,
        path_version=str(args.path_version),
    )
    total_r = estimate_nb_dispersion(train_df["total_corners"].to_numpy(dtype=float))
    home_r = estimate_nb_dispersion(train_df["home_corners"].to_numpy(dtype=float))
    away_r = estimate_nb_dispersion(train_df["away_corners"].to_numpy(dtype=float))
    preds = _predict_corner_rates(
        models=models,
        x=x_test,
        path_version=str(args.path_version),
        total_r=total_r,
        home_r=home_r,
        away_r=away_r,
    )
    pred_frame = _derive_corner_frame(
        preds=preds,
        total_r=total_r,
        home_r=home_r,
        away_r=away_r,
        path_version=str(args.path_version),
    )

    market_metrics: list[dict[str, Any]] = []
    holdout_league_rows: list[dict[str, Any]] = []
    holdout_prediction_frames: list[pd.DataFrame] = []
    for market in sorted(pred_frame.columns):
        if market not in scope_markets:
            continue
        target_col = f"target_{market}"
        if target_col not in test_df.columns:
            continue
        valid = test_df[target_col].notna().to_numpy(dtype=bool)
        if int(valid.sum()) < 50:
            continue
        y_true = test_df[target_col].to_numpy(dtype=float)[valid].astype(int)
        p_series = pred_frame[market]
        if isinstance(p_series, pd.DataFrame):
            # Defensive guard if duplicate columns are introduced by future edits.
            p_values = p_series.iloc[:, 0].to_numpy(dtype=float)
        else:
            p_values = p_series.to_numpy(dtype=float)
        p_true = np.clip(p_values[valid], 0.001, 0.999)
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
        "path_version": str(args.path_version),
        "model_type_requested": model_type_requested,
        "model_type_selected": model_type_selected,
        "model_selection_summary": model_selection_summary,
        "dispersion": {
            "total_r": total_r,
            "home_r": home_r,
            "away_r": away_r,
        },
        "home_corners_mae": float(mean_absolute_error(y_home_test, preds["home"])),
        "away_corners_mae": float(mean_absolute_error(y_away_test, preds["away"])),
        "total_corners_mae": float(
            mean_absolute_error(
                test_df["total_corners"].astype(float),
                preds.get("total", preds["home"] + preds["away"]),
            )
        ),
        "walkforward_folds": int(args.folds),
        "walkforward_min_fold_test_n": int(args.min_fold_test_n),
        "walkforward_markets_scored": int(len(walkforward_summary)),
        "walkforward_league_rows": int(len(walkforward_league_rows)),
        "holdout_league_rows": int(len(holdout_league_rows)),
        "holdout_prediction_rows": int(len(holdout_prediction_frame)),
        "league_regime_features": list(LEAGUE_REGIME_FEATURES),
        "league_regime_prior_strength": float(league_regime.get("prior_strength", 0.0)),
        "home_share_prior_blend": float(models.get("home_share_prior_blend", 1.0)),
        "pmf_market_blend": (models.get("pmf_market_blend") or {}),
        "pmf_market_calibration": (models.get("pmf_market_calibration_report") or {}),
        "total_market_head_blend": (models.get("total_market_head_blend") or {}),
        "total_market_head_calibration": (models.get("total_market_head_calibration_report") or {}),
        "total_market_ladder_blend": (models.get("total_market_ladder_blend") or {}),
        "total_market_ladder_calibration": (models.get("total_market_ladder_calibration_report") or {}),
        "team_market_head_blend": (models.get("team_market_head_blend") or {}),
        "team_market_head_calibration": (models.get("team_market_head_calibration_report") or {}),
        "neural_residual_kind": (
            NEURAL_SHARE_RESIDUAL_TARGET_KIND if str(args.path_version) in NEURAL_SHARE_RESIDUAL_PATHS else None
        ),
        "neural_total_residual_kind": (
            NEURAL_TOTAL_RESIDUAL_TARGET_KIND
            if str(args.path_version) in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS
            else None
        ),
        "neural_residual_bound": (
            float(models["neural_share_residual"].get("delta_bound", NEURAL_SHARE_RESIDUAL_DEFAULT_BOUND))
            if str(args.path_version) in (NEURAL_SHARE_RESIDUAL_PATHS | NEURAL_TOTAL_SHARE_RESIDUAL_PATHS)
            else None
        ),
        "neural_total_residual_bound": (
            float(models["neural_total_residual"].get("delta_bound", NEURAL_TOTAL_RESIDUAL_DEFAULT_BOUND))
            if str(args.path_version) in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS
            else None
        ),
        "neural_hidden_dims": (
            models["neural_share_residual"].get("hidden_dims", list(NEURAL_SHARE_RESIDUAL_HIDDEN_DIMS))
            if str(args.path_version) in (NEURAL_SHARE_RESIDUAL_PATHS | NEURAL_TOTAL_SHARE_RESIDUAL_PATHS)
            else []
        ),
        "neural_dropout": (
            float(models["neural_share_residual"].get("dropout", NEURAL_SHARE_RESIDUAL_DROPOUT))
            if str(args.path_version) in (NEURAL_SHARE_RESIDUAL_PATHS | NEURAL_TOTAL_SHARE_RESIDUAL_PATHS)
            else None
        ),
        "neural_total_market_ladder_kind": (
            NEURAL_TOTAL_MARKET_LADDER_TARGET_KIND if str(args.path_version) in NEURAL_TOTAL_LADDER_PATHS else None
        ),
        "neural_total_market_ladder_hidden_dims": (
            models["neural_total_market_ladder"].get("hidden_dims", list(NEURAL_TOTAL_MARKET_LADDER_HIDDEN_DIMS))
            if str(args.path_version) in NEURAL_TOTAL_LADDER_PATHS
            else []
        ),
        "neural_total_market_ladder_dropout": (
            float(models["neural_total_market_ladder"].get("dropout", NEURAL_TOTAL_MARKET_LADDER_DROPOUT))
            if str(args.path_version) in NEURAL_TOTAL_LADDER_PATHS
            else None
        ),
        "trained_at_utc": datetime.now(tz=UTC).isoformat(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_metadata(
        args.output_dir,
        build_artifact_metadata(
            family="corners",
            model_name=MODEL_NAME,
            model_version=str(args.model_version),
            artifact_dir=args.output_dir,
            trained_at_utc=str(diagnostics["trained_at_utc"]),
            extra={
                "model_type_selected": model_type_selected,
                "data_source": data_source,
                "path_version": str(args.path_version),
            },
        ),
    )
    if str(args.path_version) in TOTALS_FIRST_PATHS:
        joblib.dump(models["total"], args.output_dir / "total_corners_model.pkl")
        joblib.dump(models["home_share"], args.output_dir / "home_share_model.pkl")
        if str(args.path_version) == "totals_first_residual":
            joblib.dump(models["home_delta"], args.output_dir / "home_delta_model.pkl")
        if str(args.path_version) in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS:
            save_neural_total_residual_bundle(
                models["neural_total_residual"],
                args.output_dir / NEURAL_TOTAL_RESIDUAL_BUNDLE_FILENAME,
            )
        if str(args.path_version) in NEURAL_SHARE_RESIDUAL_PATHS:
            save_neural_share_residual_bundle(
                models["neural_share_residual"],
                args.output_dir / NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME,
            )
        if str(args.path_version) in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS:
            save_neural_share_residual_bundle(
                models["neural_share_residual"],
                args.output_dir / NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME,
            )
        if str(args.path_version) in SHARE_PRIOR_PATHS:
            joblib.dump(models["home_share_residual"], args.output_dir / "home_share_residual_model.pkl")
        if str(args.path_version) in PMF_SURFACE_PATHS:
            joblib.dump(models["pmf_surface_models"]["home"], args.output_dir / "home_count_distribution_model.pkl")
            joblib.dump(models["pmf_surface_models"]["away"], args.output_dir / "away_count_distribution_model.pkl")
            joblib.dump(
                models["pmf_market_calibrators"],
                args.output_dir / "pmf_market_calibrators.joblib",
            )
        if str(args.path_version) in TOTAL_SURFACE_PATHS:
            joblib.dump(models["total_market_heads"], args.output_dir / "total_market_heads.pkl")
            joblib.dump(
                models["total_market_head_calibrators"],
                args.output_dir / "total_market_head_calibrators.joblib",
            )
        if str(args.path_version) in NEURAL_TOTAL_LADDER_PATHS:
            save_neural_total_market_ladder_bundle(
                models["neural_total_market_ladder"],
                args.output_dir / NEURAL_TOTAL_MARKET_LADDER_BUNDLE_FILENAME,
            )
            joblib.dump(
                models["total_market_ladder_calibrators"],
                args.output_dir / "total_market_ladder_calibrators.joblib",
            )
        if str(args.path_version) in MARKET_HEAD_PATHS:
            joblib.dump(models["team_market_heads"], args.output_dir / "team_market_heads.pkl")
            if str(args.path_version) in CALIBRATED_TEAM_HEAD_PATHS:
                joblib.dump(
                    models["team_market_head_calibrators"],
                    args.output_dir / "team_market_head_calibrators.joblib",
                )
    else:
        joblib.dump(models["home"], args.output_dir / "home_corners_model.pkl")
        joblib.dump(models["away"], args.output_dir / "away_corners_model.pkl")
    (args.output_dir / "features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )
    model_config = {
        "path_version": str(args.path_version),
        "model_type_selected": model_type_selected,
        "direct_total_markets": (list(TOTAL_HEAD_MARKETS) if str(args.path_version) in (TOTAL_SURFACE_PATHS | NEURAL_TOTAL_LADDER_PATHS | PMF_SURFACE_PATHS) else []),
        "direct_team_markets": (list(TEAM_HEAD_MARKETS) if str(args.path_version) in (MARKET_HEAD_PATHS | PMF_SURFACE_PATHS) else []),
        "pmf_max_count": int(PMF_MAX_COUNT),
        "home_share_prior_blend": float(models.get("home_share_prior_blend", 1.0)),
        "pmf_market_blend": (models.get("pmf_market_blend") or {}),
        "total_market_head_blend": (models.get("total_market_head_blend") or {}),
        "total_market_ladder_blend": (models.get("total_market_ladder_blend") or {}),
        "team_market_head_blend": (models.get("team_market_head_blend") or {}),
        "league_regime_features": list(LEAGUE_REGIME_FEATURES),
    }
    if str(args.path_version) in NEURAL_SHARE_RESIDUAL_PATHS:
        neural_bundle = models["neural_share_residual"]
        model_config.update(
            {
                "neural_residual_kind": NEURAL_SHARE_RESIDUAL_TARGET_KIND,
                "neural_residual_sidecar": NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME,
                "neural_residual_bound": float(neural_bundle.get("delta_bound", NEURAL_SHARE_RESIDUAL_DEFAULT_BOUND)),
                "neural_hidden_dims": neural_bundle.get("hidden_dims", list(NEURAL_SHARE_RESIDUAL_HIDDEN_DIMS)),
                "neural_dropout": float(neural_bundle.get("dropout", NEURAL_SHARE_RESIDUAL_DROPOUT)),
                "neural_artifact_format": NEURAL_SHARE_RESIDUAL_ARTIFACT_FORMAT,
            }
        )
    if str(args.path_version) in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS:
        neural_total_bundle = models["neural_total_residual"]
        neural_share_bundle = models["neural_share_residual"]
        model_config.update(
            {
                "neural_total_residual_kind": NEURAL_TOTAL_RESIDUAL_TARGET_KIND,
                "neural_total_residual_sidecar": NEURAL_TOTAL_RESIDUAL_BUNDLE_FILENAME,
                "neural_total_residual_bound": float(
                    neural_total_bundle.get("delta_bound", NEURAL_TOTAL_RESIDUAL_DEFAULT_BOUND)
                ),
                "neural_share_residual_kind": NEURAL_SHARE_RESIDUAL_TARGET_KIND,
                "neural_share_residual_sidecar": NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME,
                "neural_share_residual_bound": float(
                    neural_share_bundle.get("delta_bound", NEURAL_SHARE_RESIDUAL_DEFAULT_BOUND)
                ),
                "neural_hidden_dims": neural_share_bundle.get("hidden_dims", list(NEURAL_SHARE_RESIDUAL_HIDDEN_DIMS)),
                "neural_dropout": float(neural_share_bundle.get("dropout", NEURAL_SHARE_RESIDUAL_DROPOUT)),
                "neural_artifact_format": NEURAL_SHARE_RESIDUAL_ARTIFACT_FORMAT,
            }
        )
    if str(args.path_version) in NEURAL_TOTAL_LADDER_PATHS:
        neural_ladder_bundle = models["neural_total_market_ladder"]
        model_config.update(
            {
                "neural_total_market_ladder_kind": NEURAL_TOTAL_MARKET_LADDER_TARGET_KIND,
                "neural_total_market_ladder_sidecar": NEURAL_TOTAL_MARKET_LADDER_BUNDLE_FILENAME,
                "neural_total_market_ladder_hidden_dims": neural_ladder_bundle.get(
                    "hidden_dims", list(NEURAL_TOTAL_MARKET_LADDER_HIDDEN_DIMS)
                ),
                "neural_total_market_ladder_dropout": float(
                    neural_ladder_bundle.get("dropout", NEURAL_TOTAL_MARKET_LADDER_DROPOUT)
                ),
                "neural_total_market_ladder_artifact_format": NEURAL_TOTAL_MARKET_LADDER_ARTIFACT_FORMAT,
            }
        )
    (args.output_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "league_regime.json").write_text(
        json.dumps(league_regime, indent=2), encoding="utf-8"
    )
    (args.output_dir / "imputation.json").write_text(
        json.dumps({"global_medians": imputation}, indent=2), encoding="utf-8"
    )
    (args.output_dir / "dispersion.json").write_text(
        json.dumps(diagnostics["dispersion"], indent=2), encoding="utf-8"
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
    (args.output_dir / "training_report.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    print(f"Saved corners artifacts to {args.output_dir}")
    print(
        "Corners MAE "
        f"home={diagnostics['home_corners_mae']:.3f} away={diagnostics['away_corners_mae']:.3f} total={diagnostics['total_corners_mae']:.3f}"
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
