from __future__ import annotations

from typing import Any

import numpy as np


IDENTITY_MULTIPLIER_TOLERANCE = 1e-5


def build_identity_total_intensity_correction() -> dict[str, Any]:
    return {
        "method": "identity",
        "multiplier": 1.0,
        "fit_rows": 0,
        "raw_total_goals_mean_pred": None,
        "corrected_total_goals_mean_pred": None,
        "actual_total_goals_mean": None,
        "raw_total_goals_mae": None,
        "corrected_total_goals_mae": None,
        "clamp_min": 1.0,
        "clamp_max": 1.0,
    }


def fit_total_intensity_correction(
    *,
    actual_totals: np.ndarray,
    pred_home: np.ndarray,
    pred_away: np.ndarray,
    clamp_min: float = 0.9,
    clamp_max: float = 1.1,
) -> dict[str, Any]:
    actual = np.asarray(actual_totals, dtype=float)
    raw_home = np.asarray(pred_home, dtype=float)
    raw_away = np.asarray(pred_away, dtype=float)
    raw_total = raw_home + raw_away
    mask = np.isfinite(actual) & np.isfinite(raw_total)
    if not mask.any():
        payload = build_identity_total_intensity_correction()
        payload.update({"clamp_min": float(clamp_min), "clamp_max": float(clamp_max)})
        return payload

    actual = actual[mask]
    raw_total = raw_total[mask]
    raw_mean = float(np.mean(raw_total)) if len(raw_total) else 0.0
    actual_mean = float(np.mean(actual)) if len(actual) else 0.0
    multiplier = 1.0
    if raw_mean > 1e-9:
        multiplier = float(np.clip(actual_mean / raw_mean, clamp_min, clamp_max))
    if abs(multiplier - 1.0) <= IDENTITY_MULTIPLIER_TOLERANCE:
        multiplier = 1.0
    corrected_total = raw_total * multiplier
    return {
        "method": "mean_total_ratio_v1",
        "multiplier": float(multiplier),
        "fit_rows": int(len(actual)),
        "raw_total_goals_mean_pred": raw_mean,
        "corrected_total_goals_mean_pred": float(np.mean(corrected_total)) if len(corrected_total) else None,
        "actual_total_goals_mean": actual_mean,
        "raw_total_goals_mae": float(np.mean(np.abs(actual - raw_total))) if len(raw_total) else None,
        "corrected_total_goals_mae": float(np.mean(np.abs(actual - corrected_total))) if len(corrected_total) else None,
        "clamp_min": float(clamp_min),
        "clamp_max": float(clamp_max),
    }


def apply_total_intensity_correction(
    *,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    correction: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_home = np.asarray(lambda_home, dtype=float)
    raw_away = np.asarray(lambda_away, dtype=float)
    raw_total = raw_home + raw_away
    multiplier = 1.0
    if isinstance(correction, dict):
        try:
            multiplier = float(correction.get("multiplier") or 1.0)
        except (TypeError, ValueError):
            multiplier = 1.0
    safe_total = np.where(raw_total > 1e-9, raw_total, 1.0)
    home_share = raw_home / safe_total
    corrected_total = raw_total * multiplier
    corrected_home = corrected_total * home_share
    corrected_away = corrected_total - corrected_home
    return np.clip(corrected_home, 0.05, 8.0), np.clip(corrected_away, 0.05, 8.0)