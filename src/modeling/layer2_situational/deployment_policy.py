from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POLICY_FILENAME = "layer2_deployment_policy.json"
ALPHA_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)  # finer resolution at low end
MAX_ENABLED_ALPHA = 0.5


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def lift_vs_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    baseline = rmse(y_true, np.zeros_like(y_true))
    if baseline <= 0.0:
        return 0.0
    return float((baseline - rmse(y_true, y_pred)) / baseline)


def tune_alpha(
    y_home_true: np.ndarray,
    y_away_true: np.ndarray,
    y_home_pred: np.ndarray,
    y_away_pred: np.ndarray,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
) -> tuple[float, dict[str, float]]:
    best_alpha = 0.0
    best_score = float("inf")
    best_metrics: dict[str, float] = {
        "home_rmse": rmse(y_home_true, np.zeros_like(y_home_true)),
        "away_rmse": rmse(y_away_true, np.zeros_like(y_away_true)),
        "objective": float("inf"),
    }
    for alpha in alpha_grid:
        home_hat = alpha * y_home_pred
        away_hat = alpha * y_away_pred
        home_rmse = rmse(y_home_true, home_hat)
        away_rmse = rmse(y_away_true, away_hat)
        objective = float(np.mean([home_rmse, away_rmse]))
        # Tie-break toward safer shrinkage (lower alpha).
        if objective < best_score - 1e-12 or (
            abs(objective - best_score) <= 1e-12 and alpha < best_alpha
        ):
            best_alpha = float(alpha)
            best_score = objective
            best_metrics = {
                "home_rmse": home_rmse,
                "away_rmse": away_rmse,
                "objective": objective,
            }
    return best_alpha, best_metrics


def _cv_summary(cv_league_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_league: dict[str, list[dict[str, Any]]] = {}
    for row in cv_league_rows:
        league = str(row.get("league_code", ""))
        if not league:
            continue
        by_league.setdefault(league, []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for league, rows in by_league.items():
        valid = [r for r in rows if r.get("test_n", 0) > 0]
        pos = sum(1 for r in valid if float(r.get("mean_lift", 0.0)) > 0.0)
        out[league] = {
            "cv_positive_folds": int(pos),
            "cv_total_folds": int(len(valid)),
        }
    return out


def build_layer2_deployment_policy(
    holdout_df: pd.DataFrame,
    cv_league_rows: list[dict[str, Any]],
    *,
    min_recent_n: int = 50,
    required_positive_folds: int = 2,
    required_cv_folds: int = 3,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    max_enabled_alpha: float = MAX_ENABLED_ALPHA,
    model_version: str = "v2",
) -> dict[str, Any]:
    required_cols = {
        "league_code",
        "y_home_true",
        "y_away_true",
        "y_home_pred",
        "y_away_pred",
    }
    missing = required_cols - set(holdout_df.columns)
    if missing:
        raise ValueError(f"holdout_df missing required columns: {sorted(missing)}")

    y_home_true = holdout_df["y_home_true"].to_numpy(dtype=float)
    y_away_true = holdout_df["y_away_true"].to_numpy(dtype=float)
    y_home_pred = holdout_df["y_home_pred"].to_numpy(dtype=float)
    y_away_pred = holdout_df["y_away_pred"].to_numpy(dtype=float)
    global_alpha, _ = tune_alpha(
        y_home_true,
        y_away_true,
        y_home_pred,
        y_away_pred,
        alpha_grid=alpha_grid,
    )

    cv_summary = _cv_summary(cv_league_rows)
    leagues = sorted(
        set(holdout_df["league_code"].astype(str).tolist()) | set(cv_summary.keys())
    )
    league_policies: dict[str, Any] = {}

    for league in leagues:
        league_holdout = holdout_df[holdout_df["league_code"].astype(str) == league]
        recent_n = int(len(league_holdout))
        if recent_n > 0:
            y_h_t = league_holdout["y_home_true"].to_numpy(dtype=float)
            y_a_t = league_holdout["y_away_true"].to_numpy(dtype=float)
            y_h_p = league_holdout["y_home_pred"].to_numpy(dtype=float)
            y_a_p = league_holdout["y_away_pred"].to_numpy(dtype=float)
        else:
            y_h_t = np.array([], dtype=float)
            y_a_t = np.array([], dtype=float)
            y_h_p = np.array([], dtype=float)
            y_a_p = np.array([], dtype=float)

        if recent_n >= min_recent_n:
            alpha, _ = tune_alpha(y_h_t, y_a_t, y_h_p, y_a_p, alpha_grid=alpha_grid)
        else:
            alpha = float(global_alpha)
        alpha = float(max(0.0, min(alpha, max_enabled_alpha)))

        recent_lift_home = lift_vs_zero(y_h_t, alpha * y_h_p) if recent_n > 0 else 0.0
        recent_lift_away = lift_vs_zero(y_a_t, alpha * y_a_p) if recent_n > 0 else 0.0
        recent_lift_mean = float(np.mean([recent_lift_home, recent_lift_away]))

        cv_info = cv_summary.get(
            league, {"cv_positive_folds": 0, "cv_total_folds": 0}
        )
        fail_reasons: list[str] = []
        if recent_n < min_recent_n:
            fail_reasons.append("insufficient_recent_sample")
        if int(cv_info["cv_total_folds"]) < required_cv_folds:
            fail_reasons.append("insufficient_cv_folds")
        if int(cv_info["cv_positive_folds"]) < required_positive_folds:
            fail_reasons.append("cv_gate_failed")
        if recent_lift_mean <= 0.0:
            fail_reasons.append("recent_holdout_non_positive")

        enabled = len(fail_reasons) == 0
        reason = "enabled" if enabled else ";".join(fail_reasons)

        league_policies[league] = {
            "enabled": bool(enabled),
            "alpha": float(alpha if enabled else 0.0),
            "reason": reason,
            "gate": {
                "cv_positive_folds": int(cv_info["cv_positive_folds"]),
                "cv_total_folds": int(cv_info["cv_total_folds"]),
                "recent_lift_home": float(recent_lift_home),
                "recent_lift_away": float(recent_lift_away),
                "recent_lift_mean": float(recent_lift_mean),
                "recent_n": int(recent_n),
                "pass": bool(enabled),
                "fail_reasons": fail_reasons,
            },
        }

    policy = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_version": model_version,
        "defaults": {
            "enabled": False,
            "alpha": 0.0,
            "reason": "default_disabled",
            "global_alpha_candidate": float(global_alpha),
        },
        "leagues": league_policies,
    }
    return policy


def load_layer2_deployment_policy(model_dir: Path) -> dict[str, Any]:
    path = model_dir / POLICY_FILENAME
    if not path.exists():
        return {
            "defaults": {
                "enabled": False,
                "alpha": 0.0,
                "reason": "policy_missing",
            },
            "leagues": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "defaults": {
                "enabled": False,
                "alpha": 0.0,
                "reason": "policy_invalid",
            },
            "leagues": {},
        }
    if not isinstance(payload, dict):
        return {
            "defaults": {
                "enabled": False,
                "alpha": 0.0,
                "reason": "policy_invalid_shape",
            },
            "leagues": {},
        }
    payload.setdefault(
        "defaults", {"enabled": False, "alpha": 0.0, "reason": "policy_defaulted"}
    )
    payload.setdefault("leagues", {})
    return payload


def resolve_league_policy(policy: dict[str, Any], league_code: str | None) -> dict[str, Any]:
    league = str(league_code or "")
    defaults = policy.get("defaults", {})
    per_league = policy.get("leagues", {})
    if league and isinstance(per_league, dict) and league in per_league:
        league_payload = per_league.get(league, {})
        return {
            "enabled": bool(league_payload.get("enabled", False)),
            "alpha": float(league_payload.get("alpha", 0.0)),
            "reason": str(league_payload.get("reason", "league_policy")),
            "gate": league_payload.get("gate", {}),
        }
    return {
        "enabled": bool(defaults.get("enabled", False)),
        "alpha": float(defaults.get("alpha", 0.0)),
        "reason": str(defaults.get("reason", "default_policy")),
        "gate": {},
    }
