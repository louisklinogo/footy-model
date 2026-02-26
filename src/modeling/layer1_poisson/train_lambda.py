import sys
import argparse
import json
import numpy as np
from pathlib import Path
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.features.build_features import build_point_in_time_features
from src.pricing.poisson import PoissonPricer


def compute_time_decay_weights(match_dates: pd.Series, as_of: pd.Timestamp = None, xi: float = 0.002) -> np.ndarray:
    """
    Compute Dixon-Coles time decay weights.
    
    Formula: w = exp(-xi * days_ago)
    
    Recommended xi values:
    - xi=0.001: Slow decay (90-day half-life)
    - xi=0.002: Moderate decay (45-day half-life) - DEFAULT
    - xi=0.003: Fast decay (30-day half-life)
    
    Args:
        match_dates: Series of match timestamps
        as_of: Reference date (default: latest match date)
        xi: Decay rate (default: 0.002)
    
    Returns:
        Array of sample weights
    """
    if as_of is None:
        as_of = pd.to_datetime(match_dates).max()
    
    dates = pd.to_datetime(match_dates)
    days_ago = (as_of - dates).dt.days.astype(float)
    
    # Handle any negative values (future dates)
    days_ago = days_ago.clip(lower=0)
    
    weights = np.exp(-xi * days_ago)
    
    # Normalize to sum to n (so average weight = 1)
    weights = weights * len(weights) / weights.sum()
    
    return weights

def get_training_fixture_ids(leagues: list[str] = None, limit: int = None) -> list[int]:
    """Fetch fixtures that have finished, have results, and have generated snapshots."""
    conn = connect_db()
    
    league_filter = ""
    limit_sql = ""
    params = []
    
    if leagues:
        league_filter = "AND f.league_code = ANY(%s)"
        params.append(leagues)
        
    if limit:
        limit_sql = "LIMIT %s"
        params.append(limit)
        
    query = f"""
        SELECT DISTINCT f.fixture_id, f.match_datetime_utc
        FROM fixtures f
        JOIN fixture_results r ON f.fixture_id = r.fixture_id
        JOIN team_premium_snapshots s ON f.fixture_id = s.fixture_id
        WHERE r.home_goals IS NOT NULL 
          AND r.away_goals IS NOT NULL
          AND f.status = 'ft'
          {league_filter}
        ORDER BY f.match_datetime_utc DESC
        {limit_sql}
    """
    
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        
    conn.close()
    return [r[0] for r in rows]

def calibrate_dixon_coles(lambdas_h: np.ndarray, lambdas_a: np.ndarray, actual_draws: np.ndarray):
    """
    Finds the optimal rho for the Dixon-Coles adjustment.
    rho adjusts for the higher-than-expected frequency of low-scoring draws (0-0, 1-1).
    """
    pricer = PoissonPricer()
    
    def objective(rho):
        # rho is passed as a 1-element array by scipy minimize
        rho_val = float(rho[0])
        draw_probs = []
        for lh, la in zip(lambdas_h, lambdas_a):
            matrix = pricer.generate_matrix(lh, la, rho=rho_val)
            draw_probs.append(np.sum(np.diag(matrix)))
        
        # Minimize the squared error between actual draw frequency and predicted draw probs
        return (np.mean(draw_probs) - np.mean(actual_draws))**2

    res = minimize(objective, x0=[-0.1], bounds=[(-0.3, 0.0)])
    return float(res.x[0])

def train_and_evaluate(
    target_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    train_weights: np.ndarray | None,
    booster_params: dict[str, object],
    num_boost_round: int,
    early_stopping_rounds: int,
):
    """
    Train XGBoost using Poisson regression objective.
    Early-stops on the eval (calibration) set.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    deval = xgb.DMatrix(X_eval, label=y_eval)
    
    print(f"\n[Training {target_name}]")
    print(f"  Train: {len(X_train)} rows | Eval: {len(X_eval)} rows")
    model = xgb.train(
        booster_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (deval, 'eval')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100
    )
    
    # Artifact Save
    # Canonical path consumed by predict_lambda.py
    poisson_dir = ROOT_DIR / "model_artifacts" / "poisson_model"
    poisson_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(poisson_dir / f"xgb_{target_name}_v1.json")

    # Keep legacy mirror while other tooling migrates.
    legacy_dir = ROOT_DIR / "model_artifacts" / "artifacts"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(legacy_dir / f"xgb_{target_name}_v1.json")
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", type=str, default=None)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--xi",
        type=float,
        default=0.001,
        help="Time-decay rate for training sample weights (0 disables decay).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.025,
        help="XGBoost learning rate.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="XGBoost max tree depth.",
    )
    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=25.0,
        help="XGBoost minimum child weight.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.7,
        help="XGBoost row subsample fraction.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.7,
        help="XGBoost column subsample fraction.",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=2000,
        help="Maximum XGBoost boosting rounds.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=100,
        help="Early stopping rounds on calibration split.",
    )
    args = parser.parse_args()
    if args.xi < 0:
        raise ValueError("--xi must be >= 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if args.max_depth <= 0:
        raise ValueError("--max-depth must be > 0")
    if args.min_child_weight <= 0:
        raise ValueError("--min-child-weight must be > 0")
    if not 0 < args.subsample <= 1:
        raise ValueError("--subsample must be in (0, 1]")
    if not 0 < args.colsample_bytree <= 1:
        raise ValueError("--colsample-bytree must be in (0, 1]")
    if args.num_boost_round <= 0:
        raise ValueError("--num-boost-round must be > 0")
    if args.early_stopping_rounds <= 0:
        raise ValueError("--early-stopping-rounds must be > 0")

    booster_params: dict[str, object] = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "random_state": 42,
    }
    
    leagues_list = None
    if args.leagues:
        leagues_list = [l.strip() for l in args.leagues.split(",")]
    
    fixture_ids = get_training_fixture_ids(leagues=leagues_list, limit=args.limit)
    if not fixture_ids:
        print("No fixtures found. Exiting.")
        return
    
    print(f"Building Dataset (T-60m) for {len(fixture_ids)} fixtures...")
    df = build_point_in_time_features(fixture_ids, target_window_mins=60)
    
    if df.empty:
        print("Feature dataframe is empty.")
        return
        
    print(f"Raw built dataframe: {len(df)} rows")
    # Drop rows without snapshot data (prevents training on noise)
    df = df.dropna(subset=['h_xg', 'a_xg'])
    print(f"After dropping missing team snapshots: {len(df)} rows")
    
    df = df.dropna(subset=['league_avg_xg'])
    print(f"After dropping missing league intensity: {len(df)} rows")

    # --- Min 6 Games Filter ---
    # Both teams must have played >= 6 games in the current season
    # Note: h_sample_size and a_sample_size in build_features refer to games played
    df = df[(df['h_sample_size'] >= 6) & (df['a_sample_size'] >= 6)]
    print(f"After filtering for min 6 games played: {len(df)} rows")

    # Enforce chronological order (defensive - SQL already returns ordered)
    df = df.sort_values(['kickoff_time', 'fixture_id']).reset_index(drop=True)

    # Feature subset including League Intensity anchors
    # We MUST exclude 'league_code' and other string meta-data
    feature_cols = [
        c for c in df.columns 
        if (c.startswith('h_') or c.startswith('a_') or c.startswith('league_')) 
        and c not in ['league_code', 'fidelity_score', 'h_sample_size', 'a_sample_size']
    ]
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # --- Chronological 3-way split: Train (70%) / Calibration (10%) / Test (20%) ---
    n = len(df)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.80)
    
    train_df = df.iloc[:train_end]
    cal_df = df.iloc[train_end:cal_end]
    test_df = df.iloc[cal_end:]
    
    print(f"\nSplit: Train={len(train_df)} | Calibration={len(cal_df)} | Test={len(test_df)}")
    print(f"  Train period: {train_df['kickoff_time'].min()} -> {train_df['kickoff_time'].max()}")
    print(f"  Cal period:   {cal_df['kickoff_time'].min()} -> {cal_df['kickoff_time'].max()}")
    print(f"  Test period:  {test_df['kickoff_time'].min()} -> {test_df['kickoff_time'].max()}")

    # Compute time decay weights for the training period
    # We calibrate weights as-of the end of the training period for consistency
    train_as_of = train_df['kickoff_time'].max()
    train_weights = compute_time_decay_weights(
        train_df['kickoff_time'],
        as_of=train_as_of,
        xi=args.xi,
    )
    print(f"Applying time decay weights (xi={args.xi:.4f}, train_as_of={train_as_of})")
    print(
        "Using XGBoost params: "
        f"lr={args.learning_rate}, max_depth={args.max_depth}, "
        f"min_child_weight={args.min_child_weight}, subsample={args.subsample}, "
        f"colsample_bytree={args.colsample_bytree}, "
        f"num_boost_round={args.num_boost_round}, "
        f"early_stopping_rounds={args.early_stopping_rounds}"
    )

    # Train Lambdas (early-stopping on calibration set)
    model_h = train_and_evaluate(
        "lambda_home",
        train_df[feature_cols],
        train_df['home_goals'],
        cal_df[feature_cols],
        cal_df['home_goals'],
        train_weights=train_weights,
        booster_params=booster_params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    model_a = train_and_evaluate(
        "lambda_away",
        train_df[feature_cols],
        train_df['away_goals'],
        cal_df[feature_cols],
        cal_df['away_goals'],
        train_weights=train_weights,
        booster_params=booster_params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    # --- Dixon-Coles Rho: calibrate on the CALIBRATION set (out-of-sample) ---
    print("\n--- DIXON-COLES RHO CALIBRATION (on held-out calibration set) ---")
    cal_preds_h = model_h.predict(xgb.DMatrix(cal_df[feature_cols])).flatten()
    cal_preds_a = model_a.predict(xgb.DMatrix(cal_df[feature_cols])).flatten()
    cal_actual_draws = (cal_df['home_goals'] == cal_df['away_goals']).astype(int).values.flatten()
    
    optimal_rho = calibrate_dixon_coles(cal_preds_h, cal_preds_a, cal_actual_draws)
    print(f"Calibrated Dixon-Coles Rho: {optimal_rho:.4f}")
    print(f"  Calibration draw rate: {cal_actual_draws.mean():.3f}")

    # --- Final evaluation on held-out TEST set ---
    print("\n--- TEST SET EVALUATION ---")
    test_preds_h = model_h.predict(xgb.DMatrix(test_df[feature_cols])).flatten()
    test_preds_a = model_a.predict(xgb.DMatrix(test_df[feature_cols])).flatten()
    
    pricer = PoissonPricer()
    test_draw_probs = []
    for lh, la in zip(test_preds_h, test_preds_a):
        matrix = pricer.generate_matrix(lh, la, rho=optimal_rho)
        test_draw_probs.append(np.sum(np.diag(matrix)))
    
    test_actual_draws = (test_df['home_goals'] == test_df['away_goals']).astype(int).values.flatten()
    print(f"  Test draw rate (actual):    {test_actual_draws.mean():.3f}")
    print(f"  Test draw rate (predicted): {np.mean(test_draw_probs):.3f}")
    print(f"  Lambda Home - mean pred: {test_preds_h.mean():.3f}, actual: {test_df['home_goals'].mean():.3f}")
    print(f"  Lambda Away - mean pred: {test_preds_a.mean():.3f}, actual: {test_df['away_goals'].mean():.3f}")
    
    # Save parameters
    artifacts_dir = ROOT_DIR / 'model_artifacts' / 'poisson_model'
    with open(artifacts_dir / 'calibration_params.json', 'w') as f:
        json.dump({
            "rho": optimal_rho, 
            "xi": args.xi,
            "booster_params": {
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "min_child_weight": args.min_child_weight,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "num_boost_round": args.num_boost_round,
                "early_stopping_rounds": args.early_stopping_rounds,
            },
            "features": feature_cols,
            "split": {
                "train_n": len(train_df),
                "cal_n": len(cal_df),
                "test_n": len(test_df),
            },
        }, f, indent=2)

if __name__ == "__main__":
    main()
