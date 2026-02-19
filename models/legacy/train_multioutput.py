"""
Multi-output XGBoost model with Elite Features.
Trains on 12 markets simultaneously with isotonic calibration.
Includes: Volatility, Adjusted Form, Congestion, Luck, and Handicap Supremacy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upgraded Feature Set
FEATURE_COLS = [
    'home_rgs5', 'home_rgc5', 'home_attack', 'home_defense', 'home_rest',
    'away_rgs5', 'away_rgc5', 'away_attack', 'away_defense', 'away_rest',
    'home_vol', 'away_vol', 'home_adj_gf', 'away_adj_gf', 
    'home_congestion', 'away_congestion', 'home_luck', 'away_luck',
    'home_rgd5', 'away_rgd5',
    'league_avg_goals', 'league_tier',
    'total_volatility', 'congestion_diff', 'adj_attack_diff', 'luck_diff',
    'supremacy_form', 'supremacy_strength',
    'implied_home', 'implied_draw', 'implied_away', 'implied_over25'
]

TARGET_COLS = [
    'target_o15', 'target_u35', 'target_btts_yes', 'target_c85',
    'target_o25', 'target_u45', 'target_c95', 'target_btts_no',
    'target_h_win_by_1', 'target_h_win_by_2', 'target_a_win_by_1', 'target_a_win_by_2'
]

MARKET_NAMES = {
    'target_o15': 'Over 1.5 Goals',
    'target_u35': 'Under 3.5 Goals',
    'target_btts_yes': 'BTTS Yes',
    'target_c85': 'Over 8.5 Corners',
    'target_o25': 'Over 2.5 Goals',
    'target_u45': 'Under 4.5 Goals',
    'target_c95': 'Over 9.5 Corners',
    'target_btts_no': 'BTTS No',
    'target_h_win_by_1': 'Home -0.5 AH',
    'target_h_win_by_2': 'Home -1.5 AH',
    'target_a_win_by_1': 'Away -0.5 AH',
    'target_a_win_by_2': 'Away -1.5 AH'
}

def load_data():
    """Load and prepare training data."""
    data_path = Path(__file__).parent.parent / 'data' / 'training' / 'features_22k.csv'
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'match_id']).reset_index(drop=True)
    
    # Handle missing values
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def walk_forward_cv(df, n_splits=4):
    """Walk-forward time series cross-validation."""
    season_2425_end = df[df['season'] == 2425].index[-1]
    season_2526_matches = df[df['season'] == 2526]
    
    n_2526 = len(season_2526_matches)
    fold_size = n_2526 // n_splits
    
    fold_indices = []
    for i in range(n_splits):
        test_start_idx = season_2425_end + 1 + (i * fold_size)
        test_end_idx = min(test_start_idx + fold_size, len(df))
        
        train_idx = list(range(0, test_start_idx))
        test_idx = list(range(test_start_idx, test_end_idx))
        
        fold_indices.append((train_idx, test_idx))
    
    return fold_indices

def train_market_model(X_train, y_train, X_test, target_name):
    """Train XGBoost model for a single market."""
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Probabilities
    y_prob_raw = model.predict_proba(X_test)[:, 1]
    
    # Calibration
    calibrator = IsotonicRegression(out_of_bounds='clip')
    cal_idx = np.random.choice(len(X_train), min(2000, len(X_train)), replace=False)
    calibrator.fit(model.predict_proba(X_train.iloc[cal_idx])[:, 1], y_train.iloc[cal_idx])
    
    y_prob_calibrated = calibrator.transform(y_prob_raw)
    y_prob_calibrated = np.minimum(y_prob_calibrated, 0.92)
    
    return model, calibrator, y_prob_calibrated

def calculate_kelly_stake(prob, odds, bankroll, fraction=0.5):
    if odds <= 1: return 0
    edge = prob - (1 / odds)
    if edge <= 0: return 0
    stake = bankroll * (edge / (odds - 1)) * fraction
    return max(50, min(stake, bankroll * 0.1))

def simulate_betting_o25(df_test, probs, bankroll_start=50):
    bankroll = bankroll_start
    bets = []
    
    for idx, prob in zip(df_test.index, probs):
        row = df_test.loc[idx]
        if pd.isna(row['b365_over25']) or row['b365_over25'] <= 1:
            continue
            
        implied_prob = 1 / row['b365_over25']
        edge = prob - implied_prob
        
        if edge > 0.03:
            stake = calculate_kelly_stake(prob, row['b365_over25'], bankroll)
            won = row['target_o25'] == 1
            profit = stake * (row['b365_over25'] - 1) if won else -stake
            bankroll += profit
            bets.append({'won': won, 'profit': profit, 'edge': edge, 'stake': stake})
            
    return pd.DataFrame(bets), bankroll

def main():
    logger.info("="*60)
    logger.info("Elite Multi-Output XGBoost Training")
    logger.info("="*60)
    
    df = load_data()
    X = df[FEATURE_COLS]
    
    fold_indices = walk_forward_cv(df)
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        logger.info(f"\nFold {fold_idx+1}/4")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        df_test = df.iloc[test_idx]
        
        fold_results = {}
        for target in TARGET_COLS:
            y_train = df.iloc[train_idx][target]
            y_test = df.iloc[test_idx][target]
            model, cal, y_prob = train_market_model(X_train, y_train, X_test, target)
            auc = roc_auc_score(y_test, y_prob)
            fold_results[target] = {'auc': auc, 'probs': y_prob}
            logger.info(f"  {MARKET_NAMES[target]:20s} AUC: {auc:.4f}")
            
        # O2.5 ROI
        bets_df, final_br = simulate_betting_o25(df_test, fold_results['target_o25']['probs'])
        if len(bets_df) > 0:
            roi = (final_br - 50) / bets_df['stake'].sum() * 100
            logger.info(f"  >>> O2.5 ROI: {roi:+.2f}% | Win Rate: {bets_df['won'].mean():.1%} | Bets: {len(bets_df)}")
        
    # Final training
    logger.info("\nTraining Final Elite Model on All Data...")
    final_models = {}
    for target in TARGET_COLS:
        model, cal, _ = train_market_model(X, df[target], X.iloc[:10], target)
        final_models[target] = {'model': model, 'calibrator': cal}
        
    # Save
    model_save_path = Path(__file__).parent / 'xgboost_multi_v1.pkl'
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'models': final_models,
            'feature_cols': FEATURE_COLS,
            'target_cols': TARGET_COLS,
            'market_names': MARKET_NAMES
        }, f)
    
    logger.info(f"Final Elite Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
