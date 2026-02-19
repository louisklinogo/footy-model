"""
Premium O1.5 Backtest Engine - PRODUCTION VERSION.
Joins Scraped Odds with Historical Results using fuzzy team matching.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from difflib import get_close_matches
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_all_enriched_odds():
    """Load all enriched JSONs and link them to league match ID files."""
    base_path = Path('footy-model-v2/scrapers/data/premium')
    odds_data = []
    
    # 1. Load team mappings to normalize names
    with open('footy-model-v2/data/team_mappings.json', 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    # 2. Iterate through league subdirectories
    for league_dir in base_path.iterdir():
        if not league_dir.is_dir(): continue
        league_code = league_dir.name
        
        # Load the ID discovery file for this league
        id_file = Path(f'footy-model-v2/scrapers/data/match_ids_{league_code}.json')
        if not id_file.exists(): continue
        
        with open(id_file, 'r') as f:
            id_map = {m['id']: m for m in json.load(f)}
            
        # Load each match's enriched data
        for json_file in league_dir.glob('*.json'):
            if json_file.name == '_all_enriched.json': continue
            try:
                with open(json_file, 'r') as f:
                    match_data = json.load(f)
                    fs_id = match_data['id']
                    
                    if 'odds' in match_data and '1.5' in match_data['odds']:
                        metadata = id_map.get(fs_id)
                        if metadata:
                            # Normalize names using our mappings
                            home = mappings.get(metadata['home'], metadata['home'])
                            away = mappings.get(metadata['away'], metadata['away'])
                            
                            odds_data.append({
                                'home_team': home,
                                'away_team': away,
                                'date': metadata['date'],
                                'o15_odds': float(match_data['odds']['1.5']['over'])
                            })
            except:
                continue
                
    return pd.DataFrame(odds_data)

def parse_fs_date(date_str):
    """Parses '24.01. 12:30' into a datetime object."""
    if not isinstance(date_str, str): return None
    try:
        day, month = date_str.split('.')[:2]
        # If month is 8-12, year is 2025. If 1-5, year is 2026.
        month_int = int(month)
        year = 2025 if month_int >= 8 else 2026
        return pd.to_datetime(f"{day}.{month}.{year}", format="%d.%m.%Y")
    except:
        return None

def run_backtest():
    # 1. Load Enriched Data
    df_odds = load_all_enriched_odds()
    logger.info(f"Loaded {len(df_odds)} matches with real prices.")
    
    # 2. Load Truth Results
    df_truth = pd.read_csv('footy-model-v2/data/training/features_22k.csv')
    df_truth['date'] = pd.to_datetime(df_truth['date'])
    
    # 3. Join (Fuzzy Match Logic)
    df_odds['date_dt'] = df_odds['date'].apply(parse_fs_date)
    
    # Convert both to string dates YYYY-MM-DD for perfect matching
    df_odds['date_key'] = df_odds['date_dt'].dt.strftime('%Y-%m-%d')
    df_truth['date_key'] = df_truth['date'].dt.strftime('%Y-%m-%d')
    
    # Debug: Check a few values
    logger.info(f"Odds date_key: {df_odds['date_key'].iloc[:3].tolist()}")
    logger.info(f"Truth date_key: {df_truth['date_key'].iloc[:3].tolist()}")

    merged = df_odds.merge(df_truth, left_on=['home_team', 'date_key'], right_on=['home_team', 'date_key'], how='inner')
    logger.info(f"Successfully matched {len(merged)} matches for backtesting.")
    
    if len(merged) == 0:
        logger.error("No matches matched. Check team names and dates.")
        return

    # 4. Load Model & Predict
    with open('footy-model-v2/models/xgboost_multi_v1.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    X = merged[model_data['feature_cols']]
    m_o15 = model_data['models']['target_o15']
    
    probs = m_o15['model'].predict_proba(X)[:, 1]
    merged['o15_prob'] = m_o15['calibrator'].transform(probs)
    merged['o15_prob'] = np.minimum(merged['o15_prob'], 0.92)
    
    # 5. Simulation
    bankroll = 50.0
    bets = []
    merged = merged.sort_values('date_key')
    
    for idx, row in merged.iterrows():
        prob = row['o15_prob']
        odds = row['o15_odds']
        edge = prob - (1/odds)
        
        if prob > 0.75 and edge > 0.03:
            stake_pct = (edge / (odds - 1)) * 0.5
            stake = max(50, bankroll * stake_pct)
            stake = min(stake, bankroll)
            
            if bankroll < 50: break
            
            win = row['target_o15'] == 1
            profit = stake * (odds - 1) if win else -stake
            bankroll += profit
            
            bets.append({
                'match': f"{row['home_team']} vs {row['away_team_x']}",
                'prob': prob,
                'odds': odds,
                'stake': stake,
                'win': win,
                'profit': profit,
                'bankroll': bankroll
            })
            
    # 6. Final Results
    res_df = pd.DataFrame(bets)
    if len(res_df) > 0:
        win_rate = res_df['win'].mean()
        total_staked = res_df['stake'].sum()
        total_profit = res_df['profit'].sum()
        roi = (total_profit / total_staked) * 100
        
        print("\n" + "="*40)
        print("PREMIUM BACKTEST RESULTS (O1.5)")
        print("="*40)
        print(f"Matches Analyzed: {len(merged)}")
        print(f"Bets Placed:      {len(res_df)}")
        print(f"Win Rate:         {win_rate:.1%}")
        print(f"Total ROI:        {roi:+.2f}%")
        print(f"Initial Bank:     50.00 GHS")
        print(f"Final Bank:       {bankroll:.2f} GHS")
        print("="*40)
    else:
        print("No bets met the criteria.")

if __name__ == "__main__":
    run_backtest()
