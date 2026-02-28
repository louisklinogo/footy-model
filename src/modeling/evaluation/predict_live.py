import sys
import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.features.build_features import build_point_in_time_features
from src.pricing.poisson import PoissonPricer

def get_upcoming_fixture_ids(days_ahead: int = 1) -> list[int]:
    """Fetch fixtures that are pending in the next few days."""
    conn = connect_db()
    query = """
        SELECT fixture_id 
        FROM fixtures 
        WHERE status != 'ft' 
          AND match_datetime_utc > NOW() 
          AND match_datetime_utc < NOW() + interval '%s days'
        ORDER BY match_datetime_utc ASC
    """
    with conn.cursor() as cur:
        cur.execute(query, (days_ahead,))
        rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_latest_bookie_odds(fixture_ids: list[int]) -> pd.DataFrame:
    """Fetch the most recent pre-match 1X2 odds for the given fixtures."""
    conn = connect_db()
    query = """
        SELECT DISTINCT ON (fom.fixture_id)
            fom.fixture_id,
            (fom.odds_json -> 'prices_latest') AS one_x_two_json,
            fom.snapshot_time_utc
        FROM fixture_odds_markets fom
        JOIN fixtures f
          ON f.fixture_id = fom.fixture_id
        WHERE fom.fixture_id = ANY(%s)
          AND fom.provider = 'sofascore'
          AND fom.market_code = '1x2'
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND f.match_datetime_utc IS NOT NULL
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY
            fom.fixture_id,
            (fom.snapshot_type = 'latest_pre_match') DESC,
            fom.snapshot_time_utc DESC
    """
    df = pd.read_sql_query(query, conn, params=(fixture_ids,))
    conn.close()
    return df

def predict_live():
    # 1. Loading Artifacts
    artifacts_dir = ROOT_DIR / 'model_artifacts' / 'poisson_model'
    with open(artifacts_dir / 'calibration_params.json', 'r') as f:
        params = json.load(f)
    
    rho = params['rho']
    feature_cols = params['features']
    
    model_h = xgb.Booster()
    model_h.load_model(artifacts_dir / 'xgb_lambda_home_v1.json')
    
    model_a = xgb.Booster()
    model_a.load_model(artifacts_dir / 'xgb_lambda_away_v1.json')
    
    # 2. Fetching upcoming matches
    print("Fetching upcoming fixtures...")
    fixture_ids = get_upcoming_fixture_ids(days_ahead=2)
    if not fixture_ids:
        print("No upcoming fixtures found.")
        return

    # 3. Building Features
    print(f"Building features for {len(fixture_ids)} matches...")
    # NOTE: building snapshots for LIVE fixtures is tricky if we don't have the T-snapshot yet.
    # In a real production loop, we call build_team_premium_snapshots BEFORE this.
    df = build_point_in_time_features(fixture_ids, target_window_mins=0) 
    
    if df.empty:
        print("Could not build features (lack of snapshots).")
        return

    # Filter for matches that actually have valid features
    df = df.dropna(subset=['h_xg', 'a_xg', 'league_avg_xg'])
    if df.empty:
        print("No matches with sufficient historical snapshots.")
        return

    # 4. Generating Lambdas
    d_matrix = xgb.DMatrix(df[feature_cols])
    df['pred_lambda_h'] = model_h.predict(d_matrix)
    df['pred_lambda_a'] = model_a.predict(d_matrix)
    
    # 5. Pricing and EV Calculation
    pricer = PoissonPricer()
    print("\nDaily Edge Report:")
    print("-" * 50)
    
    # We fetch odds for these fixtures
    print(f"Fetching latest odds for {len(fixture_ids)} fixtures...")
    odds_df = get_latest_bookie_odds(fixture_ids)
    
    conn = connect_db()
    cur = conn.cursor()
    
    edges_found = 0
    for _, row in df.iterrows():
        lh, la = row['pred_lambda_h'], row['pred_lambda_a']
        matrix = pricer.generate_matrix(lh, la, rho=rho)
        prob_dict = pricer.get_1x2(matrix)
        
        # Match specific odds
        match_odds_rows = odds_df[odds_df['fixture_id'] == row['fixture_id']]
        if match_odds_rows.empty:
            continue
            
        odds_snapshot = match_odds_rows.iloc[0]['one_x_two_json']
        if not odds_snapshot:
            continue
            
        # prices_latest map: {"home": 2.1, "draw": 3.4, "away": 3.1}
        mapping = {"home": "home", "draw": "draw", "away": "away"}
        
        for bookie_key, model_key in mapping.items():
            if bookie_key in odds_snapshot:
                odds = float(odds_snapshot[bookie_key])
                prob = prob_dict[model_key]
                ev = (prob * odds) - 1
                
                if ev > 0.0: # Show all positive EV for debugging
                    edges_found += 1
                    print(f"[{row['league_code']}] {row['kickoff_time']} | {model_key.upper()} EDGE: {ev:.1%} | Odds: {odds:.2f} (Prob: {prob:.1%})")
                    
                    # Log to DB
                    insert_query = """
                        INSERT INTO execution_logs 
                        (fixture_id, league_code, match_datetime_utc, predicted_h_lambda, predicted_a_lambda, rho, market_type, market_key, model_prob, bookie_odds, ev, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')
                    """
                    cur.execute(insert_query, (
                        row['fixture_id'], row['league_code'], row['kickoff_time'],
                        lh, la, rho, '1x2', model_key, prob, odds, ev
                    ))

    conn.commit()
    conn.close()
    print(f"\nScan complete. Found and logged {edges_found} edges.")

if __name__ == "__main__":
    predict_live()
