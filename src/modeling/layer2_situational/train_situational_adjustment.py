"""
Layer 2: Situational ML Adjustment Model (v3 - Bounded Lambda Adjustment)

Instead of predicting residuals (which are noise by definition), this model
predicts a bounded multiplicative adjustment factor (alpha) to apply to the
base lambda predictions.

Key improvements over residual model:
1. Target: alpha = home_goals / lambda_home, bounded to [0.5, 2.0]
2. Loss: Poisson deviance instead of MSE
3. Evaluation: Poisson log-likelihood, Brier score on match outcomes
4. Output: log_alpha (symmetric around 1.0 in log space)

Usage:
  python train_situational_adjustment.py          # Save features parquet
  python train_situational_adjustment.py --train   # Train the model
  python train_situational_adjustment.py --compare # Compare all approaches
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import poisson as sp_poisson

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.layer2_situational import situational_utils
from src.modeling.layer2_situational.train_situational_residual import (
    FEATURE_COLS,
    FEATURE_COLS_PREMIUM,
    ODDS_FEATURE_COLS,
    load_feature_data,
    add_odds_model_gap,
)

warnings.filterwarnings("ignore")

# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnusedImport=false, reportUnusedVariable=false, reportUnusedParameter=false, reportCallIssue=false, reportArgumentType=false, reportUndefinedVariable=false, reportMissingTypeArgument=false, reportUnknownLambdaType=false, reportIndexIssue=false, reportGeneralTypeIssues=false

OUT_DIR = ROOT_DIR / "model_artifacts" / "situational_model"

# Bounds for alpha multiplier
ALPHA_MIN = 0.5
ALPHA_MAX = 2.0


# ---------------------------------------------------------------------------
# Poisson Loss Functions
# ---------------------------------------------------------------------------


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Poisson deviance (scaled deviance).
    
    D = 2 * sum(y * log(y / y_hat) - (y - y_hat))
    
    Handles edge cases where y=0 or y_hat=0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, None)
    
    # For y=0, the term y * log(y/y_hat) = 0
    deviance = 0.0
    for yt, yp in zip(y_true, y_pred):
        if yt > 0:
            deviance += yt * np.log(yt / yp) - (yt - yp)
        else:
            deviance += -(0 - yp)  # Just the (y - y_hat) term when y=0
    
    return float(2.0 * deviance)


def poisson_log_likelihood(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Poisson log-likelihood.
    
    LL = sum(y * log(lambda) - lambda - log(y!))
    
    Higher is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, None)
    
    ll = 0.0
    for yt, yp in zip(y_true, y_pred):
        ll += yt * np.log(yp) - yp - special.gammaln(yt + 1)
    
    return float(ll)


def mean_poisson_log_likelihood(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson log-likelihood per observation."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return poisson_log_likelihood(y_true, y_pred) / n


def compute_outcome_probs(lh: float, la: float, max_goals: int = 8) -> tuple[float, float, float]:
    """Compute home/draw/away probabilities from Poisson lambdas."""
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = sp_poisson.pmf(h, lh) * sp_poisson.pmf(a, la)
            if h > a:
                p_home += prob
            elif h == a:
                p_draw += prob
            else:
                p_away += prob
    return p_home, p_draw, p_away


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score for binary outcomes.
    
    BS = mean((y_true - y_prob)^2)
    
    Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    
    if y_true.size == 0:
        return 0.0
    
    return float(np.mean((y_true - y_prob) ** 2))


def multi_brier_score(
    outcomes: np.ndarray,
    probs: np.ndarray,
) -> float:
    """
    Compute Brier score for multi-class outcomes (1X2).
    
    outcomes: shape (n, 3) - one-hot encoded outcomes
    probs: shape (n, 3) - predicted probabilities
    
    Lower is better.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    probs = np.asarray(probs, dtype=float)
    
    if outcomes.size == 0:
        return 0.0
    
    return float(np.mean((outcomes - probs) ** 2))


# ---------------------------------------------------------------------------
# Target Engineering
# ---------------------------------------------------------------------------


def compute_alpha_target(
    goals: np.ndarray,
    lambda_base: np.ndarray,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> np.ndarray:
    """
    Compute bounded alpha target: alpha = goals / lambda_base
    
    alpha is clipped to [alpha_min, alpha_max] and then log-transformed
    for symmetry around 1.0.
    
    Returns: log_alpha (ready for regression)
    """
    goals = np.asarray(goals, dtype=float)
    lambda_base = np.asarray(lambda_base, dtype=float)
    
    # Avoid division by zero
    lambda_safe = np.maximum(lambda_base, 0.1)
    
    # Compute raw alpha
    alpha = goals / lambda_safe
    
    # Clip to bounds
    alpha_clipped = np.clip(alpha, alpha_min, alpha_max)
    
    # Log transform for symmetry
    log_alpha = np.log(alpha_clipped)
    
    return log_alpha


def alpha_to_lambda(
    log_alpha: np.ndarray,
    lambda_base: np.ndarray,
) -> np.ndarray:
    """
    Convert predicted log_alpha to adjusted lambda.
    
    lambda_adjusted = lambda_base * exp(log_alpha)
    """
    log_alpha = np.asarray(log_alpha, dtype=float)
    lambda_base = np.asarray(lambda_base, dtype=float)
    
    alpha = np.exp(log_alpha)
    return lambda_base * alpha


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------


def train_adjustment_model(
    df: pd.DataFrame,
    out_dir: Path,
    feature_override: list[str] | None = None,
    model_name: str = "situational_adjustment",
) -> dict[str, Any]:
    """
    Train a model to predict bounded lambda adjustment (log_alpha).
    
    Uses Poisson deviance as the loss function and evaluates with
    proper probabilistic metrics.
    """
    import joblib
    import hashlib
    import subprocess
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    
    all_features = feature_override or (FEATURE_COLS + ODDS_FEATURE_COLS)
    
    df = df.sort_values("match_datetime_utc").reset_index(drop=True)
    
    # Time-respecting split by timestamp (80th percentile)
    split_time = df["match_datetime_utc"].dropna().quantile(0.8)
    train_df = df[df["match_datetime_utc"] <= split_time].copy()
    test_df = df[df["match_datetime_utc"] > split_time].copy()
    
    if not train_df.empty and not test_df.empty:
        max_train_dt = train_df["match_datetime_utc"].max()
        min_test_dt = test_df["match_datetime_utc"].min()
        if pd.notna(max_train_dt) and pd.notna(min_test_dt) and max_train_dt >= min_test_dt:
            raise RuntimeError("Chronology violation: train set is not strictly before test set.")
    
    MIN_GAMES = 4
    
    def _apply_filters(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        pre_filter = len(frame)
        frame = frame[
            (frame["home_played"] >= MIN_GAMES) & (frame["away_played"] >= MIN_GAMES)
        ]
        print(f"Min-{MIN_GAMES}-games filter ({label}): {pre_filter} -> {len(frame)} rows")
        
        pre_lambda = len(frame)
        frame = frame[frame["lambda_home"].notna() & frame["lambda_away"].notna()]
        print(f"Lambda-availability filter ({label}): {pre_lambda} -> {len(frame)} rows")
        return frame
    
    train_df = _apply_filters(train_df, "Train")
    test_df = _apply_filters(test_df, "Test")
    
    if train_df.empty:
        raise RuntimeError("No training rows remain after lambda availability filter.")
    if test_df.empty:
        raise RuntimeError("No test rows remain after lambda availability filter.")
    
    # Compute bounded alpha targets
    train_df["log_alpha_home"] = compute_alpha_target(
        train_df["home_goals"].values,
        train_df["lambda_home"].values,
    )
    train_df["log_alpha_away"] = compute_alpha_target(
        train_df["away_goals"].values,
        train_df["lambda_away"].values,
    )
    test_df["log_alpha_home"] = compute_alpha_target(
        test_df["home_goals"].values,
        test_df["lambda_home"].values,
    )
    test_df["log_alpha_away"] = compute_alpha_target(
        test_df["away_goals"].values,
        test_df["lambda_away"].values,
    )
    
    # Feature coverage report
    coverage = train_df[all_features].notna().mean()
    COVERAGE_THRESHOLD = 0.10
    use_features = [f for f in all_features if coverage[f] > COVERAGE_THRESHOLD]
    print(f"\nFeature coverage (train set):")
    for feat in all_features:
        pct = coverage.get(feat, 0)
        flag = "[OK]" if pct > COVERAGE_THRESHOLD else "[NO]"
        print(f"  {flag} {feat}: {pct:.1%}")
    
    print(f"\nUsing {len(use_features)}/{len(all_features)} features with >{COVERAGE_THRESHOLD:.0%} coverage.")
    
    X_train = train_df[use_features].fillna(0).values
    X_test = test_df[use_features].fillna(0).values
    
    y_home_train = train_df["log_alpha_home"].values
    y_away_train = train_df["log_alpha_away"].values
    y_home_test = test_df["log_alpha_home"].values
    y_away_test = test_df["log_alpha_away"].values
    
    # Sample weights: inverse variance weighting based on lambda
    # Higher lambda = more reliable target
    w_home_train = np.sqrt(np.maximum(train_df["lambda_home"].values, 0.1))
    w_away_train = np.sqrt(np.maximum(train_df["lambda_away"].values, 0.1))
    
    # GBM parameters - conservative to avoid overfitting
    gbm_params = dict(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=30,
        learning_rate=0.05,
        random_state=42,
    )
    
    # Train models
    home_model = GradientBoostingRegressor(**gbm_params)
    home_model.fit(X_train, y_home_train, sample_weight=w_home_train)
    
    away_model = GradientBoostingRegressor(**gbm_params)
    away_model.fit(X_train, y_away_train, sample_weight=w_away_train)
    
    # Predictions
    home_pred_log_alpha = home_model.predict(X_test)
    away_pred_log_alpha = away_model.predict(X_test)
    
    # Convert to adjusted lambdas
    home_lambda_base = test_df["lambda_home"].values
    away_lambda_base = test_df["lambda_away"].values
    
    home_lambda_adjusted = alpha_to_lambda(home_pred_log_alpha, home_lambda_base)
    away_lambda_adjusted = alpha_to_lambda(away_pred_log_alpha, away_lambda_base)
    
    home_goals_true = test_df["home_goals"].values
    away_goals_true = test_df["away_goals"].values
    
    # --- Evaluation Metrics ---
    
    # 1. Poisson Log-Likelihood
    base_home_pll = mean_poisson_log_likelihood(home_goals_true, home_lambda_base)
    base_away_pll = mean_poisson_log_likelihood(away_goals_true, away_lambda_base)
    adj_home_pll = mean_poisson_log_likelihood(home_goals_true, home_lambda_adjusted)
    adj_away_pll = mean_poisson_log_likelihood(away_goals_true, away_lambda_adjusted)
    
    # 2. Poisson Deviance
    base_home_dev = poisson_deviance(home_goals_true, home_lambda_base)
    base_away_dev = poisson_deviance(away_goals_true, away_lambda_base)
    adj_home_dev = poisson_deviance(home_goals_true, home_lambda_adjusted)
    adj_away_dev = poisson_deviance(away_goals_true, away_lambda_adjusted)
    
    # 3. Brier Score on Match Outcomes
    base_outcomes = []
    adj_outcomes = []
    true_outcomes = []
    
    for i in range(len(test_df)):
        # True outcome (one-hot)
        hg = home_goals_true[i]
        ag = away_goals_true[i]
        if hg > ag:
            true_outcomes.append([1, 0, 0])  # Home win
        elif hg == ag:
            true_outcomes.append([0, 1, 0])  # Draw
        else:
            true_outcomes.append([0, 0, 1])  # Away win
        
        # Base lambda probabilities
        base_outcomes.append(compute_outcome_probs(home_lambda_base[i], away_lambda_base[i]))
        # Adjusted lambda probabilities
        adj_outcomes.append(compute_outcome_probs(home_lambda_adjusted[i], away_lambda_adjusted[i]))
    
    true_outcomes = np.array(true_outcomes)
    base_outcomes = np.array(base_outcomes)
    adj_outcomes = np.array(adj_outcomes)
    
    base_brier = multi_brier_score(true_outcomes, base_outcomes)
    adj_brier = multi_brier_score(true_outcomes, adj_outcomes)
    
    # 4. MSE on log_alpha (for model quality check)
    home_mse = mean_squared_error(y_home_test, home_pred_log_alpha)
    away_mse = mean_squared_error(y_away_test, away_pred_log_alpha)
    
    # --- Print Results ---
    print(f"\n{'='*60}")
    print("EVALUATION: Bounded Lambda Adjustment Model")
    print(f"{'='*60}")
    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
    
    print(f"\n--- Poisson Log-Likelihood (higher is better) ---")
    print(f"  Home: Base={base_home_pll:.4f}, Adjusted={adj_home_pll:.4f}, Delta={adj_home_pll - base_home_pll:+.4f}")
    print(f"  Away: Base={base_away_pll:.4f}, Adjusted={adj_away_pll:.4f}, Delta={adj_away_pll - base_away_pll:+.4f}")
    
    print(f"\n--- Poisson Deviance (lower is better) ---")
    print(f"  Home: Base={base_home_dev:.4f}, Adjusted={adj_home_dev:.4f}, Delta={adj_home_dev - base_home_dev:+.4f}")
    print(f"  Away: Base={base_away_dev:.4f}, Adjusted={adj_away_dev:.4f}, Delta={adj_away_dev - base_away_dev:+.4f}")
    
    print(f"\n--- Brier Score on 1X2 Outcomes (lower is better) ---")
    print(f"  Base: {base_brier:.4f}, Adjusted: {adj_brier:.4f}, Delta={adj_brier - base_brier:+.4f}")
    
    print(f"\n--- Model Quality (MSE on log_alpha) ---")
    print(f"  Home MSE: {home_mse:.4f}")
    print(f"  Away MSE: {away_mse:.4f}")
    
    # Feature importance
    print(f"\n--- Feature Importance (Home model) ---")
    importances = sorted(
        zip(use_features, home_model.feature_importances_), key=lambda x: -x[1]
    )
    for feat, imp in importances:
        print(f"  {feat}: {imp:.4f}")
    
    # Save model
    result = {
        "features": use_features,
        "home_model": home_model,
        "away_model": away_model,
        "train_n": len(train_df),
        "test_n": len(test_df),
        "alpha_bounds": {"min": ALPHA_MIN, "max": ALPHA_MAX},
        "metrics": {
            "poisson_log_likelihood": {
                "home": {"base": base_home_pll, "adjusted": adj_home_pll},
                "away": {"base": base_away_pll, "adjusted": adj_away_pll},
            },
            "poisson_deviance": {
                "home": {"base": base_home_dev, "adjusted": adj_home_dev},
                "away": {"base": base_away_dev, "adjusted": adj_away_dev},
            },
            "brier_score_1x2": {
                "base": base_brier,
                "adjusted": adj_brier,
            },
            "log_alpha_mse": {
                "home": home_mse,
                "away": away_mse,
            },
        },
    }
    
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{model_name}.pkl"
    joblib.dump(result, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save metadata
    feature_hash = hashlib.sha256(",".join(use_features).encode("utf-8")).hexdigest()
    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = None
    
    sidecar = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "bounded_lambda_adjustment",
        "alpha_bounds": {"min": ALPHA_MIN, "max": ALPHA_MAX},
        "train_range": {
            "start": str(train_df["match_datetime_utc"].min()),
            "end": str(train_df["match_datetime_utc"].max()),
        },
        "test_range": {
            "start": str(test_df["match_datetime_utc"].min()),
            "end": str(test_df["match_datetime_utc"].max()),
        },
        "train_n": len(train_df),
        "test_n": len(test_df),
        "features": use_features,
        "feature_hash": feature_hash,
        "hyperparams": gbm_params,
        "git_commit": git_commit,
        "metrics": result["metrics"],
    }
    sidecar_path = out_dir / f"{model_name}.meta.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Metadata saved to {sidecar_path}")
    
    return result


# ---------------------------------------------------------------------------
# Comparison Framework
# ---------------------------------------------------------------------------


def compare_approaches(
    df: pd.DataFrame,
    model_dir: Path,
    feature_override: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare three approaches:
    1. Base lambda (no adjustment)
    2. Rule layer only (current fix)
    3. ML adjustment model (new)
    
    Returns a comparison report with metrics for each approach.
    """
    import joblib
    from src.modeling.layer2_situational.rule_layer import (
        apply_rule_adjustment,
        load_rule_layer_config,
    )
    
    all_features = feature_override or (FEATURE_COLS + ODDS_FEATURE_COLS)
    
    df = df.sort_values("match_datetime_utc").reset_index(drop=True)
    
    # Use last 20% as test set
    split_time = df["match_datetime_utc"].dropna().quantile(0.8)
    test_df = df[df["match_datetime_utc"] > split_time].copy()
    
    # Apply filters
    MIN_GAMES = 4
    test_df = test_df[
        (test_df["home_played"] >= MIN_GAMES) & (test_df["away_played"] >= MIN_GAMES)
    ]
    test_df = test_df[test_df["lambda_home"].notna() & test_df["lambda_away"].notna()]
    
    if test_df.empty:
        raise RuntimeError("No test rows available for comparison.")
    
    test_df = test_df.reset_index(drop=True)
    
    # Ground truth
    home_goals = test_df["home_goals"].values
    away_goals = test_df["away_goals"].values
    home_lambda_base = test_df["lambda_home"].values
    away_lambda_base = test_df["lambda_away"].values
    
    # True outcomes (one-hot)
    true_outcomes = []
    for hg, ag in zip(home_goals, away_goals):
        if hg > ag:
            true_outcomes.append([1, 0, 0])
        elif hg == ag:
            true_outcomes.append([0, 1, 0])
        else:
            true_outcomes.append([0, 0, 1])
    true_outcomes = np.array(true_outcomes)
    
    # --- Approach 1: Base Lambda ---
    base_home_pll = mean_poisson_log_likelihood(home_goals, home_lambda_base)
    base_away_pll = mean_poisson_log_likelihood(away_goals, away_lambda_base)
    base_home_dev = poisson_deviance(home_goals, home_lambda_base)
    base_away_dev = poisson_deviance(away_goals, away_lambda_base)
    
    base_probs = np.array([
        compute_outcome_probs(lh, la)
        for lh, la in zip(home_lambda_base, away_lambda_base)
    ])
    base_brier = multi_brier_score(true_outcomes, base_probs)
    
    # --- Approach 2: Rule Layer Only ---
    rule_config_path = model_dir / "rule_layer_config.json"
    rule_config = load_rule_layer_config(rule_config_path)
    
    home_lambda_rule = np.zeros(len(test_df))
    away_lambda_rule = np.zeros(len(test_df))
    
    for i, row in test_df.iterrows():
        home_rule = apply_rule_adjustment(
            home_lambda_base[i],
            row,
            side="home",
            config=rule_config,
            scope="enabled_league",
        )
        away_rule = apply_rule_adjustment(
            away_lambda_base[i],
            row,
            side="away",
            config=rule_config,
            scope="enabled_league",
        )
        home_lambda_rule[i] = home_rule["lambda_after"]
        away_lambda_rule[i] = away_rule["lambda_after"]
    
    rule_home_pll = mean_poisson_log_likelihood(home_goals, home_lambda_rule)
    rule_away_pll = mean_poisson_log_likelihood(away_goals, away_lambda_rule)
    rule_home_dev = poisson_deviance(home_goals, home_lambda_rule)
    rule_away_dev = poisson_deviance(away_goals, away_lambda_rule)
    
    rule_probs = np.array([
        compute_outcome_probs(lh, la)
        for lh, la in zip(home_lambda_rule, away_lambda_rule)
    ])
    rule_brier = multi_brier_score(true_outcomes, rule_probs)
    
    # --- Approach 3: ML Adjustment Model ---
    # Load model if it exists
    ml_model_path = model_dir / "situational_adjustment.pkl"
    if ml_model_path.exists():
        ml_blob = joblib.load(ml_model_path)
        ml_features = ml_blob["features"]
        ml_home_model = ml_blob["home_model"]
        ml_away_model = ml_blob["away_model"]
        
        # Determine which features to use
        use_features = [f for f in ml_features if f in test_df.columns]
        X_test = test_df[use_features].fillna(0).values
        
        home_log_alpha = ml_home_model.predict(X_test)
        away_log_alpha = ml_away_model.predict(X_test)
        
        home_lambda_ml = alpha_to_lambda(home_log_alpha, home_lambda_base)
        away_lambda_ml = alpha_to_lambda(away_log_alpha, away_lambda_base)
        
        ml_home_pll = mean_poisson_log_likelihood(home_goals, home_lambda_ml)
        ml_away_pll = mean_poisson_log_likelihood(away_goals, away_lambda_ml)
        ml_home_dev = poisson_deviance(home_goals, home_lambda_ml)
        ml_away_dev = poisson_deviance(away_goals, away_lambda_ml)
        
        ml_probs = np.array([
            compute_outcome_probs(lh, la)
            for lh, la in zip(home_lambda_ml, away_lambda_ml)
        ])
        ml_brier = multi_brier_score(true_outcomes, ml_probs)
        
        ml_available = True
    else:
        print(f"ML adjustment model not found at {ml_model_path}")
        ml_home_pll = ml_away_pll = None
        ml_home_dev = ml_away_dev = None
        ml_brier = None
        ml_available = False
    
    # --- Build Comparison Report ---
    comparison = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "test_n": len(test_df),
        "test_range": {
            "start": str(test_df["match_datetime_utc"].min()),
            "end": str(test_df["match_datetime_utc"].max()),
        },
        "approaches": {
            "base_lambda": {
                "poisson_log_likelihood": {
                    "home": base_home_pll,
                    "away": base_away_pll,
                    "mean": (base_home_pll + base_away_pll) / 2,
                },
                "poisson_deviance": {
                    "home": base_home_dev,
                    "away": base_away_dev,
                    "total": base_home_dev + base_away_dev,
                },
                "brier_score_1x2": base_brier,
            },
            "rule_layer": {
                "poisson_log_likelihood": {
                    "home": rule_home_pll,
                    "away": rule_away_pll,
                    "mean": (rule_home_pll + rule_away_pll) / 2,
                },
                "poisson_deviance": {
                    "home": rule_home_dev,
                    "away": rule_away_dev,
                    "total": rule_home_dev + rule_away_dev,
                },
                "brier_score_1x2": rule_brier,
            },
        },
    }
    
    if ml_available:
        comparison["approaches"]["ml_adjustment"] = {
            "poisson_log_likelihood": {
                "home": ml_home_pll,
                "away": ml_away_pll,
                "mean": (ml_home_pll + ml_away_pll) / 2,
            },
            "poisson_deviance": {
                "home": ml_home_dev,
                "away": ml_away_dev,
                "total": ml_home_dev + ml_away_dev,
            },
            "brier_score_1x2": ml_brier,
        }
    
    # Determine winner
    approaches = ["base_lambda", "rule_layer"]
    if ml_available:
        approaches.append("ml_adjustment")
    
    # By Poisson LL (higher is better)
    pll_ranking = sorted(
        approaches,
        key=lambda a: comparison["approaches"][a]["poisson_log_likelihood"]["mean"],
        reverse=True,
    )
    
    # By Brier score (lower is better)
    brier_ranking = sorted(
        approaches,
        key=lambda a: comparison["approaches"][a]["brier_score_1x2"],
    )
    
    comparison["ranking"] = {
        "by_poisson_ll": pll_ranking,
        "by_brier_score": brier_ranking,
        "best_overall": pll_ranking[0],  # Use PLL as primary metric
    }
    
    # --- Print Report ---
    print(f"\n{'='*70}")
    print("COMPARISON: Base Lambda vs Rule Layer vs ML Adjustment")
    print(f"{'='*70}")
    print(f"\nTest set: {len(test_df)} fixtures")
    
    print(f"\n{'─'*70}")
    print(f"{'Approach':<20} {'PLL (mean)':<12} {'Deviance':<12} {'Brier 1X2':<12}")
    print(f"{'─'*70}")
    
    for approach in approaches:
        data = comparison["approaches"][approach]
        pll = data["poisson_log_likelihood"]["mean"]
        dev = data["poisson_deviance"]["total"]
        brier = data["brier_score_1x2"]
        print(f"{approach:<20} {pll:>10.4f} {dev:>12.2f} {brier:>12.4f}")
    
    print(f"{'─'*70}")
    print(f"\nBest by Poisson LL: {pll_ranking[0]}")
    print(f"Best by Brier Score: {brier_ranking[0]}")
    
    # Save report
    report_path = model_dir / "approach_comparison.json"
    report_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nComparison saved to {report_path}")
    
    return comparison


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the adjustment model")
    parser.add_argument("--compare", action="store_true", help="Compare all approaches")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--drop-features",
        type=str,
        default="",
        help="Comma-separated feature list to drop before coverage filtering.",
    )
    args = parser.parse_args()
    
    df = load_feature_data()
    df = add_odds_model_gap(df)
    
    all_features = FEATURE_COLS + ODDS_FEATURE_COLS
    all_features_premium = FEATURE_COLS_PREMIUM + ODDS_FEATURE_COLS
    
    drop_features = [f.strip() for f in args.drop_features.split(",") if f.strip()]
    unknown = sorted(set(drop_features) - set(all_features + all_features_premium))
    if unknown:
        raise ValueError(f"Unknown features in --drop-features: {unknown}")
    
    if args.train:
        # Train BASE model
        selected_base = [f for f in all_features if f not in set(drop_features)]
        if not selected_base:
            raise ValueError("No features remain in BASE after applying --drop-features.")
        print("\n" + "=" * 50)
        print("TRAINING BASE ADJUSTMENT MODEL (All Rows)")
        print("=" * 50)
        train_adjustment_model(df, args.out_dir, feature_override=selected_base, model_name="situational_adjustment_base")
        
        # Train PREMIUM model
        selected_premium = [f for f in all_features_premium if f not in set(drop_features)]
        if not selected_premium:
            raise ValueError("No features remain in PREMIUM after applying --drop-features.")
        print("\n" + "=" * 50)
        print("TRAINING PREMIUM ADJUSTMENT MODEL (H1/H2 Rows)")
        print("=" * 50)
        train_adjustment_model(df, args.out_dir, feature_override=selected_premium, model_name="situational_adjustment_premium")
    
    if args.compare:
        print("\n" + "=" * 50)
        print("COMPARING APPROACHES")
        print("=" * 50)
        selected_base = [f for f in all_features if f not in set(drop_features)]
        compare_approaches(df, args.out_dir, feature_override=selected_base)
    
    if not args.train and not args.compare:
        # Default: save features
        out = args.out_dir / "situational_features.parquet"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"Feature data saved to {out}")
        print(f"Columns: {list(df.columns)}")
        print(df[all_features_premium].describe())


if __name__ == "__main__":
    main()
