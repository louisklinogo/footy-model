from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ONE_X_TWO_MARKETS = ("1x2_h", "1x2_d", "1x2_a")
DOUBLE_CHANCE_MARKETS = ("dc_1x", "dc_x2", "dc_12")
BINARY_RESIDUAL_MARKETS = ("o15", "u35")
CORE_BASE_FEATURES = (
    "base_lambda_home",
    "base_lambda_away",
    "base_lambda_total",
    "base_1x2_h",
    "base_1x2_d",
    "base_1x2_a",
    "base_dc_1x",
    "base_dc_x2",
    "base_dc_12",
    "base_o15",
    "base_u35",
)
RAW_RESIDUAL_FEATURES = (
    "odds_over_15",
    "odds_under_15",
    "odds_over_35",
    "odds_under_35",
    "implied_over15",
    "implied_under15",
    "implied_over35",
    "implied_under35",
    "odds_gap_15",
    "odds_gap_35",
    "adj_lambda_home_final",
    "adj_lambda_away_final",
    "rule_fired_home",
    "rule_fired_away",
    "home_availability_known",
    "away_availability_known",
    "home_lineup_known",
    "away_lineup_known",
    "home_missing_players",
    "away_missing_players",
    "style_delta",
    "xg_net_diff",
    "goal_diff_proxy",
    "home_sample_size",
    "away_sample_size",
)
CLIP_EPS = 0.001


def build_residual_source_frame(
    *,
    raw_frame: pd.DataFrame,
    base_market_frame: pd.DataFrame,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
) -> pd.DataFrame:
    out = raw_frame.reset_index(drop=True).copy()
    base = base_market_frame.reset_index(drop=True)
    for column in ONE_X_TWO_MARKETS + DOUBLE_CHANCE_MARKETS + BINARY_RESIDUAL_MARKETS:
        if column in base.columns:
            out[f"base_{column}"] = base[column].to_numpy(dtype=float)
    out["base_lambda_home"] = np.asarray(lambda_home, dtype=float)
    out["base_lambda_away"] = np.asarray(lambda_away, dtype=float)
    out["base_lambda_total"] = np.asarray(lambda_home, dtype=float) + np.asarray(
        lambda_away, dtype=float
    )
    return out


def normalize_one_x_two_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for market in ONE_X_TWO_MARKETS:
        if market not in out.columns:
            out[market] = 1.0 / 3.0
    probs = np.clip(out[list(ONE_X_TWO_MARKETS)].to_numpy(dtype=float), CLIP_EPS, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    zero_mask = row_sums[:, 0] <= 0.0
    if np.any(zero_mask):
        probs[zero_mask, :] = 1.0 / 3.0
        row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums
    out["1x2_h"] = probs[:, 0]
    out["1x2_d"] = probs[:, 1]
    out["1x2_a"] = probs[:, 2]
    out["dc_1x"] = np.clip(out["1x2_h"] + out["1x2_d"], CLIP_EPS, 1.0 - CLIP_EPS)
    out["dc_x2"] = np.clip(out["1x2_d"] + out["1x2_a"], CLIP_EPS, 1.0 - CLIP_EPS)
    out["dc_12"] = np.clip(out["1x2_h"] + out["1x2_a"], CLIP_EPS, 1.0 - CLIP_EPS)
    return out


def predict_binary_probabilities(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(x), dtype=float)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return np.clip(probs[:, 1], CLIP_EPS, 1.0 - CLIP_EPS)
        if probs.ndim == 1:
            return np.clip(probs, CLIP_EPS, 1.0 - CLIP_EPS)
    if hasattr(model, "decision_function"):
        logits = np.asarray(model.decision_function(x), dtype=float)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(probs, CLIP_EPS, 1.0 - CLIP_EPS)
    raw = np.asarray(model.predict(x), dtype=float)
    return np.clip(raw, CLIP_EPS, 1.0 - CLIP_EPS)


def prepare_residual_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    imputation: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    out = frame.reset_index(drop=True).copy()
    selected = list(feature_columns or [])
    if not selected:
        selected.extend([column for column in CORE_BASE_FEATURES if column in out.columns])
        selected.extend(
            column
            for column in RAW_RESIDUAL_FEATURES
            if column in out.columns and column not in selected
        )
        generated_missingness: list[str] = []
        for column in list(selected):
            if column.startswith("base_"):
                continue
            indicator = f"{column}__missing"
            if indicator in out.columns:
                generated_missingness.append(indicator)
                continue
            out[indicator] = out[column].isna().astype(float)
            generated_missingness.append(indicator)
        selected.extend(ind for ind in generated_missingness if ind not in selected)

    for column in selected:
        if column not in out.columns:
            out[column] = np.nan

    medians = dict(imputation or {})
    if not medians:
        for column in selected:
            values = pd.to_numeric(out[column], errors="coerce")
            median = values.median()
            medians[column] = 0.0 if pd.isna(median) else float(median)

    prepared = pd.DataFrame(index=out.index)
    for column in selected:
        values = pd.to_numeric(out[column], errors="coerce")
        prepared[column] = values.fillna(float(medians.get(column, 0.0))).astype(float)
    return prepared, selected, medians


def apply_residual_bundle(
    *,
    base_market_frame: pd.DataFrame,
    residual_features: pd.DataFrame,
    residual_bundle: dict[str, Any] | None,
    direct_market_overrides: bool = False,
) -> pd.DataFrame:
    if not residual_bundle or not bool(direct_market_overrides):
        return base_market_frame.reset_index(drop=True).copy()

    models = residual_bundle.get("models") or {}
    out = base_market_frame.reset_index(drop=True).copy()
    one_x_two_ready = all(str(market) in models for market in ONE_X_TWO_MARKETS)
    if one_x_two_ready:
        probs = np.column_stack(
            [
                predict_binary_probabilities(models[str(market)], residual_features)
                for market in ONE_X_TWO_MARKETS
            ]
        )
        out["1x2_h"] = probs[:, 0]
        out["1x2_d"] = probs[:, 1]
        out["1x2_a"] = probs[:, 2]
        out = normalize_one_x_two_probabilities(out)

    for market in BINARY_RESIDUAL_MARKETS:
        if str(market) not in models:
            continue
        out[market] = predict_binary_probabilities(models[str(market)], residual_features)

    for column in out.columns:
        out[column] = np.clip(pd.to_numeric(out[column], errors="coerce"), CLIP_EPS, 1.0 - CLIP_EPS)
    return out