"""
Generate predictions for upcoming fixtures using trained Elite multi-output model.
Calculates elite features (Volatility, Luck, Congestion, Supremacy) for prediction.
"""

import pandas as pd
import numpy as np
import psycopg2
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = 'postgresql://neondb_owner:npg_csxeyQ6fNXF7@ep-long-salad-abo7gpdq-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require'

def load_model():
    """Load trained Elite multi-output model."""
    model_path = Path(__file__).parent / 'xgboost_multi_v1.pkl'
    if not model_path.exists():
        model_path = Path('footy-model-v2/models/xgboost_multi_v1.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def get_upcoming_fixtures():
    """Fetch upcoming fixtures from database."""
    conn = psycopg2.connect(DB_URL)
    query = """
    SELECT 
        uf.id, uf.match_date, uf.match_time, uf.home_team, uf.away_team, uf.league_code, uf.league_name
    FROM upcoming_fixtures uf
    WHERE uf.match_date >= CURRENT_DATE
      AND uf.match_date <= CURRENT_DATE + INTERVAL '3 days'
    ORDER BY uf.match_date, uf.match_time;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_team_history(team_name, match_date, conn):
    """Get raw history for elite feature calculation."""
    query = """
    SELECT 
        m.date,
        CASE WHEN m.home_team = %s THEN m.fthg ELSE m.ftag END as gf,
        CASE WHEN m.home_team = %s THEN m.ftag ELSE m.fthg END as ga,
        CASE WHEN m.home_team = %s THEN m.hst ELSE m.ast END as sot_for,
        CASE WHEN m.home_team = %s THEN tms_away.defense_strength ELSE tms_home.defense_strength END as opp_def
    FROM matches m
    LEFT JOIN team_match_snapshots tms_home ON m.id = tms_home.match_id AND tms_home.is_home = true
    LEFT JOIN team_match_snapshots tms_away ON m.id = tms_away.match_id AND tms_away.is_home = false
    WHERE (m.home_team = %s OR m.away_team = %s)
      AND m.date < %s
      AND m.fthg IS NOT NULL
    ORDER BY m.date DESC
    LIMIT 20;
    """
    cur = conn.cursor()
    cur.execute(query, (team_name, team_name, team_name, team_name, team_name, team_name, match_date))
    rows = cur.fetchall()
    cur.close()
    
    if not rows: return None
    
    df = pd.DataFrame(rows, columns=['date', 'gf', 'ga', 'sot_for', 'opp_def'])
    df = df.fillna(1.0)
    
    # Calculate Elite Stats
    l5 = df.head(5)
    
    stats = {
        'rgs5': float(l5['gf'].mean()),
        'rgc5': float(l5['ga'].mean()),
        'volatility': float(l5['gf'].std()) if len(l5) > 1 else 0.0,
        'adj_gf': float((l5['gf'] * l5['opp_def']).mean()),
        'luck': float(l5['gf'].sum() - (l5['sot_for'] * 0.33).sum()),
        'rgd5': float(l5['gf'].mean() - l5['ga'].mean()),
        'congestion': len(df[df['date'] > (match_date - timedelta(days=21))]),
        'rest': (match_date - df.iloc[0]['date']).days if len(df) > 0 else 7
    }
    return stats

def get_league_stats(league_code, conn):
    query = "SELECT AVG(fthg + ftag) FROM matches WHERE div = %s AND fthg IS NOT NULL;"
    cur = conn.cursor()
    cur.execute(query, (league_code,))
    row = cur.fetchone()
    cur.close()
    return float(row[0]) if row and row[0] is not None else 2.5

def engineer_features(home_stats, away_stats, league_avg):
    # Default stats if missing
    def_stats = {'rgs5': 1.2, 'rgc5': 1.2, 'volatility': 0.8, 'adj_gf': 1.2, 'luck': 0, 'rgd5': 0, 'congestion': 3, 'rest': 7}
    h = home_stats or def_stats
    a = away_stats or def_stats
    
    features = {
        'home_rgs5': h['rgs5'], 'home_rgc5': h['rgc5'],
        'home_attack': h['rgs5'] / (league_avg/2) if league_avg > 0 else 1,
        'home_defense': (league_avg/2) / h['rgc5'] if h['rgc5'] > 0 else 1,
        'home_rest': h['rest'],
        'away_rgs5': a['rgs5'], 'away_rgc5': a['rgc5'],
        'away_attack': a['rgs5'] / (league_avg/2) if league_avg > 0 else 1,
        'away_defense': (league_avg/2) / a['rgc5'] if a['rgc5'] > 0 else 1,
        'away_rest': a['rest'],
        'home_vol': h['volatility'], 'away_vol': a['volatility'],
        'home_adj_gf': h['adj_gf'], 'away_adj_gf': a['adj_gf'],
        'home_congestion': h['congestion'], 'away_congestion': a['congestion'],
        'home_luck': h['luck'], 'away_luck': a['luck'],
        'home_rgd5': h['rgd5'], 'away_rgd5': a['rgd5'],
        'league_avg_goals': league_avg,
        'league_tier': 1 if league_avg < 2.3 else (2 if league_avg > 2.8 else 1),
        'total_volatility': h['volatility'] + a['volatility'],
        'congestion_diff': h['congestion'] - a['congestion'],
        'adj_attack_diff': h['adj_gf'] - a['adj_gf'],
        'luck_diff': h['luck'] - a['luck'],
        'supremacy_form': h['rgd5'] - a['rgd5'],
        'supremacy_strength': (h['rgs5'] / (league_avg/2) - (league_avg/2) / a['rgc5']) - (a['rgs5'] / (league_avg/2) - (league_avg/2) / h['rgc5']) if league_avg > 0 else 0,
        'implied_home': 0.33, 'implied_draw': 0.33, 'implied_away': 0.33, 'implied_over25': 0.5
    }
    return features

def load_team_mappings():
    mapping_path = Path(__file__).parent.parent / 'data' / 'team_mappings.json'
    if not mapping_path.exists():
        mapping_path = Path('footy-model-v2/data/team_mappings.json')
    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def generate_daily_slips():
    logger.info("Generating Elite Predictions...")
    model_data = load_model()
    team_mappings = load_team_mappings()
    fixtures = get_upcoming_fixtures()
    if len(fixtures) == 0: return
    
    conn = psycopg2.connect(DB_URL)
    results = []
    
    for _, f in fixtures.iterrows():
        h_name = team_mappings.get(f['home_team'], f['home_team'])
        a_name = team_mappings.get(f['away_team'], f['away_team'])
        
        logger.info(f"  Predicting: {h_name} vs {a_name}")
        h_stats = get_team_history(h_name, f['match_date'], conn)
        a_stats = get_team_history(a_name, f['match_date'], conn)
        l_avg = get_league_stats(f['league_code'], conn)
        
        feats = engineer_features(h_stats, a_stats, l_avg)
        X = pd.DataFrame([feats])[model_data['feature_cols']]
        
        preds = {}
        for target, m_info in model_data['models'].items():
            prob = m_info['model'].predict_proba(X)[0, 1]
            prob = m_info['calibrator'].transform([prob])[0]
            preds[target] = min(prob, 0.92)
            
        results.append({
            'date': f['match_date'], 'time': f['match_time'], 'league': f['league_name'], 'match': f"{f['home_team']} vs {f['away_team']}",
            'O1.5': preds['target_o15'], 'O2.5': preds['target_o25'], 'BTTS': preds['target_btts_yes'],
            'C8.5': preds['target_c85'], 'U3.5': preds['target_u35'],
            'H-0.5': preds['target_h_win_by_1'], 'H-1.5': preds['target_h_win_by_2'],
            'A-0.5': preds['target_a_win_by_1'], 'A-1.5': preds['target_a_win_by_2']
        })
    
    conn.close()
    pred_df = pd.DataFrame(results).sort_values('O1.5', ascending=False)
    
    # Save Report
    report_path = Path(__file__).parent.parent / 'data' / 'daily_slips.md'
    if not report_path.parent.exists():
        report_path = Path('footy-model-v2/data/daily_slips.md')
        
    with open(report_path, 'w') as f:
        f.write("# Elite Multi-Market Predictions\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Top O1.5 Rollover Picks\n\n")
        f.write("| Date | Time | League | Match | O1.5 Prob | Confidence |\n")
        f.write("|------|------|--------|-------|-----------|------------|\n")
        for _, r in pred_df.head(15).iterrows():
            conf = "ELITE" if r['O1.5'] > 0.85 else "HIGH"
            f.write(f"| {r['date']} | {r['time']} | {r['league']} | {r['match']} | {r['O1.5']:.1%} | {conf} |\n")
        
        f.write("\n## High Value O2.5 & BTTS Picks\n\n")
        f.write("| Date | Time | Match | O2.5 Prob | BTTS Prob |\n")
        f.write("|------|------|-------|-----------|-----------|\n")
        o25_picks = pred_df[pred_df['O2.5'] > 0.60].sort_values('O2.5', ascending=False).head(15)
        for _, r in o25_picks.iterrows():
            f.write(f"| {r['date']} | {r['time']} | {r['match']} | {r['O2.5']:.1%} | {r['BTTS']:.1%} |\n")

        f.write("\n## Asian Handicap & Value Singles\n\n")
        f.write("| Date | Time | Match | Home -0.5 | Home -1.5 | Away -0.5 | Away -1.5 |\n")
        f.write("|------|------|-------|-----------|-----------|-----------|-----------|\n")
        # Find matches with high supremacy (AH Prob > 65%)
        ah_picks = pred_df[(pred_df['H-0.5'] > 0.65) | (pred_df['A-0.5'] > 0.65)].sort_values(['H-0.5', 'A-0.5'], ascending=False).head(15)
        for _, r in ah_picks.iterrows():
            f.write(f"| {r['date']} | {r['time']} | {r['match']} | {r['H-0.5']:.1%} | {r['H-1.5']:.1%} | {r['A-0.5']:.1%} | {r['A-1.5']:.1%} |\n")
            
    logger.info(f"Elite Report saved to: {report_path}")

if __name__ == '__main__':
    generate_daily_slips()
