
"""
V3 Hierarchical Training Pipeline.
Prioritizes:
1. Supremacy (AH) - Strongest Signal (0.59 correlation)
2. Goals (O1.5) - Reliable Signal (0.53 correlation)
3. Corners - Experimental (Low correlation, requires interaction features)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
from pathlib import Path
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

# Paths
DATA_PATH = Path('footy-model-v2/data/training/features_v3.csv')
MODEL_DIR = Path('footy-model-v2/models/v3')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_v3_model():
    print("Loading V3 Elite Dataset...")
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Run extract_features_v3.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(['date', 'match_id'])
    
    # Features (The Signal - Rolling Averages to prevent leak)
    features = [
        # Rolling Premium Tactical (No-Leak)
        'home_roll_xg', 'away_roll_xg', 
        'home_roll_xgot', 'away_roll_xgot',
        'home_roll_touches', 'away_roll_touches',
        'home_roll_crosses', 'away_roll_crosses',
        'home_roll_blocked', 'away_roll_blocked',
        'home_roll_big_chances', 'away_roll_big_chances',
        
        # Context
        'ah_line', 'fidelity_score',
        'h_strength', 'a_strength', 'h_def_strength', 'a_def_strength',
        
        # Calculated from Rolling
        'total_exp_goals', 'total_exp_touches', 'total_exp_crosses'
    ]
    
    # Targets (The Full Market Spectrum)
    targets = {
        'target_o15': 'Over 1.5 Goals',
        'target_o25': 'Over 2.5 Goals',
        'target_u35': 'Under 3.5 Goals',
        'target_u45': 'Under 4.5 Goals',
        'target_btts_yes': 'BTTS: Yes',
        'target_btts_no': 'BTTS: No',
        'target_c85': 'Corners Over 8.5',
        'target_c95': 'Corners Over 9.5',
        'target_h_win': 'Match Winner (Home)',
        'target_a_win': 'Match Winner (Away)',
        'target_h_win_by_2': 'Home -1.5 AH', # Supremacy
        'target_a_win_by_2': 'Away -1.5 AH'  # Supremacy
    }
    
    # Time-Series Split (Respecting Time)
    tscv = TimeSeriesSplit(n_splits=4)
    X = df[features]
    
    # Store trained models
    model_suite = {}
    
    print(f"\n--- Training V3 Bayesian Hierarchy (N={len(df)}) ---")
    
    for target_col, target_name in targets.items():
        y = df[target_col]
        print(f"\nTraining: {target_name} ({target_col})")
        
        scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # XGBoost with Monotonic Constraints (Forcing Logic)
            # Removed early_stopping to be compatible with CalibratedClassifierCV
            clf = xgb.XGBClassifier(
                n_estimators=500, # Reduced slightly as we aren't early stopping
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8, # Correct parameter name
                random_state=42,
                n_jobs=-1
            )
            
            # Calibrate Probabilities (Isotonic)
            calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=3)
            calibrated_clf.fit(X_train, y_train)
            
            # Evaluate
            probs = calibrated_clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
            brier = brier_score_loss(y_val, probs)
            
            scores.append(auc)
            models.append(calibrated_clf)
            print(f"  Fold {fold+1}: AUC = {auc:.3f} | Brier = {brier:.3f}")
            
        print(f"  > Average AUC: {np.mean(scores):.3f}")
        
        # Save best model (trained on full data for production)
        final_model = CalibratedClassifierCV(clf, method='isotonic', cv=3)
        final_model.fit(X, y)
        
        model_path = MODEL_DIR / f"xgb_v3_{target_col}.pkl"
        joblib.dump(final_model, model_path)
        model_suite[target_col] = {
            'model': final_model,
            'auc': np.mean(scores)
        }

    # Save feature list for inference
    joblib.dump(features, MODEL_DIR / "v3_features.pkl")
    print("\n✅ V3 Training Complete. Models saved to footy-model-v2/models/v3/")

if __name__ == "__main__":
    train_v3_model()
