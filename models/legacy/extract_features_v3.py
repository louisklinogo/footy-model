
"""
Elite V3 Feature Engineering Pipeline.
Integrates Premium Tactical Data (xG, Box Touches, Crosses).
Output: footy-model-v2/data/training/features_v3.csv
"""

import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DB_URL = 'postgresql://neondb_owner:npg_csxeyQ6fNXF7@ep-long-salad-abo7gpdq-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require'

def create_v3_dataset():
    conn = psycopg2.connect(DB_URL)
    
    logger.info("Extracting Enriched Match Data (Base + Premium Tactical)...")
    
    query = """
    SELECT 
        m.id as match_id,
        m.div as league_code,
        m.season,
        m.date,
        m.home_team,
        m.away_team,
        m.fthg, m.ftag,
        m.hs, m.as_ as away_shots,
        m.hst, m.ast as away_sot,
        m.hc, m.ac as away_corners,
        m.b365h, m.b365d, m.b365a,
        -- Premium Tactical Metrics
        p.h_xg, p.a_xg,
        p.h_xgot, p.a_xgot,
        p.h_box_touches, p.a_box_touches,
        p.h_crosses, p.a_crosses,
        p.h_blocked_shots, p.a_blocked_shots,
        p.h_through_passes, p.a_through_passes,
        p.h_big_chances, p.a_big_chances,
        p.ah_line, p.fidelity_score,
        -- Snapshot metrics
        tms_h.attack_strength as h_strength,
        tms_h.defense_strength as h_def_strength,
        tms_a.attack_strength as a_strength,
        tms_a.defense_strength as a_def_strength
    FROM matches m
    LEFT JOIN matches_premium p ON m.id = p.match_id
    LEFT JOIN team_match_snapshots tms_h ON m.id = tms_h.match_id AND tms_h.is_home = true
    LEFT JOIN team_match_snapshots tms_a ON m.id = tms_a.match_id AND tms_a.is_home = false
    WHERE m.fthg IS NOT NULL
    ORDER BY m.date, m.id;
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df):,} matches for engineering.")

    # 1. Hybrid xG Engine (Imputation)
    df['h_xg_proxy'] = df['hst'] * 0.33
    df['a_xg_proxy'] = df['away_sot'] * 0.33
    df['final_h_xg'] = df['h_xg'].fillna(df['h_xg_proxy'])
    df['final_a_xg'] = df['a_xg'].fillna(df['a_xg_proxy'])

    # 3. ROLLING AVERAGES (The "No-Leak" Engine)
    # We must calculate stats achieved BEFORE the match starts.
    logger.info("Calculating Rolling Premium Stats (No-Leak)...")
    
    # Flatten to Team-Date-Stat format
    home_stats = df[['date', 'home_team', 'final_h_xg', 'h_xgot', 'h_box_touches', 'h_crosses', 'h_blocked_shots', 'h_big_chances']].rename(
        columns={'home_team': 'team', 'final_h_xg': 'xg', 'h_xgot': 'xgot', 'h_box_touches': 'touches', 'h_crosses': 'crosses', 'h_blocked_shots': 'blocked', 'h_big_chances': 'big_chances'}
    )
    away_stats = df[['date', 'away_team', 'final_a_xg', 'a_xgot', 'a_box_touches', 'a_crosses', 'a_blocked_shots', 'a_big_chances']].rename(
        columns={'away_team': 'team', 'final_a_xg': 'xg', 'a_xgot': 'xgot', 'a_box_touches': 'touches', 'a_crosses': 'crosses', 'a_blocked_shots': 'blocked', 'a_big_chances': 'big_chances'}
    )
    
    team_stats = pd.concat([home_stats, away_stats]).sort_values(['team', 'date']).fillna(0)
    
    # Calculate Rolling 5-Game Averages (Closed Window - Shifted by 1 to exclude current match)
    # Group by team, compute rolling mean, then SHIFT(1) so row N uses stats from N-1, N-2...
    grouped = team_stats.groupby('team')
    
    metrics = ['xg', 'xgot', 'touches', 'crosses', 'blocked', 'big_chances']
    for m in metrics:
        team_stats[f'rolling_{m}'] = grouped[m].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())

    # Merge Rolling Stats back to Home/Away
    # Home Team Rolling
    df = df.merge(team_stats[['date', 'team'] + [f'rolling_{m}' for m in metrics]], 
                  left_on=['date', 'home_team'], right_on=['date', 'team'], how='left')
    df = df.rename(columns={f'rolling_{m}': f'home_roll_{m}' for m in metrics}).drop('team', axis=1)
    
    # Away Team Rolling
    df = df.merge(team_stats[['date', 'team'] + [f'rolling_{m}' for m in metrics]], 
                  left_on=['date', 'away_team'], right_on=['date', 'team'], how='left')
    df = df.rename(columns={f'rolling_{m}': f'away_roll_{m}' for m in metrics}).drop('team', axis=1)

    # Fill NaNs for early season matches
    df = df.fillna(0)

    # 4. Tactical Features (Using Pre-Match Rolling Stats)
    df['total_exp_goals'] = df['home_roll_xg'] + df['away_roll_xg']
    df['total_exp_touches'] = df['home_roll_touches'] + df['away_roll_touches']
    df['total_exp_crosses'] = df['home_roll_crosses'] + df['away_roll_crosses']
    
    # 5. Luck Delta
    df['h_luck'] = df['fthg'] - df['final_h_xg']
    df['a_luck'] = df['ftag'] - df['final_a_xg']
    
    # --- RE-CALCULATE TARGETS HERE TO ENSURE PERSISTENCE ---
    df['target_o15'] = (df['fthg'] + df['ftag'] >= 2).astype(int)
    df['target_o25'] = (df['fthg'] + df['ftag'] >= 3).astype(int)
    df['target_u35'] = (df['fthg'] + df['ftag'] <= 3).astype(int)
    df['target_u45'] = (df['fthg'] + df['ftag'] <= 4).astype(int)
    df['target_btts_yes'] = ((df['fthg'] > 0) & (df['ftag'] > 0)).astype(int)
    df['target_btts_no'] = ((df['fthg'] == 0) | (df['ftag'] == 0)).astype(int)
    df['target_c85'] = ((df['hc'] + df['away_corners']) >= 9).astype(int)
    df['target_c95'] = ((df['hc'] + df['away_corners']) >= 10).astype(int)
    df['target_h_win'] = (df['fthg'] > df['ftag']).astype(int)
    df['target_a_win'] = (df['ftag'] > df['fthg']).astype(int)
    df['target_h_win_by_2'] = ((df['fthg'] - df['ftag']) >= 2).astype(int)
    df['target_a_win_by_2'] = ((df['ftag'] - df['fthg']) >= 2).astype(int)

    # 6. Final Clean & Save
    df = df.fillna(0)
    
    output_path = Path('footy-model-v2/data/training/features_v3.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"V3 Training Set Ready: {output_path}")
    logger.info(f"Columns: {list(df.columns)}")
    return df

if __name__ == "__main__":
    create_v3_dataset()
