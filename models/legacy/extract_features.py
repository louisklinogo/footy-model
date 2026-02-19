"""
Elite Feature Engineering Pipeline.
Implements:
1. Opponent-Adjusted Form (Quality filter)
2. Goal Volatility (Consistency metric)
3. Schedule Congestion (Fatigue factor)
4. xG Efficiency Delta (Luck factor)
5. League Context & Clustering (Domain awareness)
Output: data/training/features_22k.csv
"""

import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = 'postgresql://neondb_owner:npg_csxeyQ6fNXF7@ep-long-salad-abo7gpdq-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require'

def create_training_dataset():
    """Extract features and targets for all 8 markets with Elite features."""
    
    conn = psycopg2.connect(DB_URL)
    
    logger.info("Extracting base match data with league stats...")
    
    # Main query
    query = """
    SELECT 
        m.id as match_id,
        m.div as league_code,
        m.season,
        m.date,
        m.home_team,
        m.away_team,
        m.fthg, m.ftag,
        m.hthg, m.htag,
        m.hs, m.as_ as away_shots,
        m.hst, m.ast as away_sot,
        m.hc, m.ac as away_corners,
        m.b365h, m.b365d, m.b365a,
        m.b365_over25, m.b365_under25,
        -- Team features (home)
        tms_home.rolling_goals_scored_5 as home_rgs5,
        tms_home.rolling_goals_conceded_5 as home_rgc5,
        tms_home.attack_strength as home_attack,
        tms_home.defense_strength as home_defense,
        tms_home.days_since_last_match as home_rest,
        -- Team features (away)
        tms_away.rolling_goals_scored_5 as away_rgs5,
        tms_away.rolling_goals_conceded_5 as away_rgc5,
        tms_away.attack_strength as away_attack,
        tms_away.defense_strength as away_defense,
        tms_away.days_since_last_match as away_rest,
        -- Market features
        mf.b365_implied_h,
        mf.overround_b365
    FROM matches m
    LEFT JOIN team_match_snapshots tms_home 
        ON m.id = tms_home.match_id AND tms_home.is_home = true
    LEFT JOIN team_match_snapshots tms_away 
        ON m.id = tms_away.match_id AND tms_away.is_home = false
    LEFT JOIN market_features mf ON m.id = mf.match_id
    WHERE m.fthg IS NOT NULL
    ORDER BY m.date, m.id;
    """
    
    df = pd.read_sql(query, conn)
    logger.info(f"Loaded {len(df):,} matches")
    
    conn.close()
    
    # Pre-processing
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'match_id'])
    
    # 1. Target Variables
    logger.info("Creating target variables...")
    df['target_o15'] = (df['fthg'] + df['ftag'] >= 2).astype(int)
    df['target_u35'] = (df['fthg'] + df['ftag'] <= 3).astype(int)
    df['target_btts_yes'] = ((df['fthg'] > 0) & (df['ftag'] > 0)).astype(int)
    df['target_c85'] = (df['hc'] + df['away_corners'] >= 9).astype(int)
    df['target_o25'] = (df['fthg'] + df['ftag'] >= 3).astype(int)
    df['target_u45'] = (df['fthg'] + df['ftag'] <= 4).astype(int)
    df['target_c95'] = (df['hc'] + df['away_corners'] >= 10).astype(int)
    df['target_btts_no'] = ((df['fthg'] == 0) | (df['ftag'] == 0)).astype(int)
    
    # Handicap Targets (Supremacy)
    df['target_goal_diff'] = df['fthg'] - df['ftag']
    df['target_h_win_by_1'] = (df['fthg'] - df['ftag'] >= 1).astype(int)
    df['target_h_win_by_2'] = (df['fthg'] - df['ftag'] >= 2).astype(int)
    df['target_a_win_by_1'] = (df['ftag'] - df['fthg'] >= 1).astype(int)
    df['target_a_win_by_2'] = (df['ftag'] - df['fthg'] >= 2).astype(int)
    
    # 2. xG Proxies
    df['home_xg_proxy'] = df['hst'] * 0.33
    df['away_xg_proxy'] = df['away_sot'] * 0.33
    df['match_goals'] = df['fthg'] + df['ftag']
    
    # 3. Elite Rolling Features
    logger.info("Calculating Elite Rolling features (Volatility, Adj Form, Congestion, Luck, Supremacy)...")
    
    # Flatten matches to team-date-stats
    home = df[['date', 'home_team', 'fthg', 'ftag', 'home_xg_proxy', 'away_defense']].rename(columns={
        'home_team': 'team', 'fthg': 'gf', 'ftag': 'ga', 'home_xg_proxy': 'xg', 'away_defense': 'opp_def'
    })
    away = df[['date', 'away_team', 'ftag', 'fthg', 'away_xg_proxy', 'home_defense']].rename(columns={
        'away_team': 'team', 'ftag': 'gf', 'fthg': 'ga', 'away_xg_proxy': 'xg', 'home_defense': 'opp_def'
    })
    team_stats = pd.concat([home, away]).sort_values(['team', 'date'])
    
    # Grouped operations
    grouped = team_stats.groupby('team')
    
    # Goal Volatility (Std Dev of GF in last 5)
    team_stats['volatility'] = grouped['gf'].transform(lambda x: x.rolling(5).std())
    
    # Goal Difference (For Supremacy)
    team_stats['gd'] = team_stats['gf'] - team_stats['ga']
    team_stats['rolling_gd_5'] = grouped['gd'].transform(lambda x: x.rolling(5).mean())
    
    # Luck Factor (Sum GF - Sum xG in last 5)
    team_stats['luck'] = grouped.apply(lambda x: x['gf'].rolling(5).sum() - x['xg'].rolling(5).sum()).reset_index(level=0, drop=True)
    
    # Opponent-Adjusted Form (Mean of (GF * Opponent Defense) in last 5)
    team_stats['adj_gf'] = team_stats['gf'] * team_stats['opp_def']
    team_stats['rolling_adj_gf'] = grouped['adj_gf'].transform(lambda x: x.rolling(5).mean())
    
    # Schedule Congestion (Matches in last 21 days)
    # Using a simpler method: count matches in the last 21 days
    def count_matches_21d(group):
        group = group.sort_values('date')
        counts = []
        for i in range(len(group)):
            current_date = group.iloc[i]['date']
            start_date = current_date - pd.Timedelta(days=21)
            count = len(group[(group['date'] > start_date) & (group['date'] <= current_date)])
            counts.append(count)
        return pd.Series(counts, index=group.index)
    
    logger.info("  Calculating congestion (this may take a minute)...")
    team_stats['congestion'] = grouped.apply(count_matches_21d).reset_index(level=0, drop=True)
    
    # Join back to main df
    df = df.merge(team_stats[['date', 'team', 'volatility', 'rolling_adj_gf', 'congestion', 'luck', 'rolling_gd_5']], 
                  left_on=['date', 'home_team'], right_on=['date', 'team'], how='left')
    df = df.rename(columns={'volatility': 'home_vol', 'rolling_adj_gf': 'home_adj_gf', 'congestion': 'home_congestion', 'luck': 'home_luck', 'rolling_gd_5': 'home_rgd5'}).drop('team', axis=1)
    
    df = df.merge(team_stats[['date', 'team', 'volatility', 'rolling_adj_gf', 'congestion', 'luck', 'rolling_gd_5']], 
                  left_on=['date', 'away_team'], right_on=['date', 'team'], how='left')
    df = df.rename(columns={'volatility': 'away_vol', 'rolling_adj_gf': 'away_adj_gf', 'congestion': 'away_congestion', 'luck': 'away_luck', 'rolling_gd_5': 'away_rgd5'}).drop('team', axis=1)

    # 4. League Context & Clustering
    logger.info("Adding League Context & Clustering...")
    league_avgs = df.groupby('league_code')['match_goals'].transform('mean')
    df['league_avg_goals'] = league_avgs
    
    # Clustering
    df['league_tier'] = pd.qcut(df['league_avg_goals'], 3, labels=[0, 1, 2]).astype(int)
    
    # 5. Interaction Features
    logger.info("Creating Interaction features...")
    df['total_volatility'] = df['home_vol'] + df['away_vol']
    df['congestion_diff'] = df['home_congestion'] - df['away_congestion']
    df['adj_attack_diff'] = df['home_adj_gf'] - df['away_adj_gf']
    df['luck_diff'] = df['home_luck'] - df['away_luck']
    df['supremacy_form'] = df['home_rgd5'] - df['away_rgd5']
    df['supremacy_strength'] = (df['home_attack'] - df['away_defense']) - (df['away_attack'] - df['home_defense'])
    
    # Fill NaNs
    df = df.fillna(df.median(numeric_only=True))
    
    # Odds Implied Probs
    df['implied_home'] = 1 / df['b365h']
    df['implied_draw'] = 1 / df['b365d']
    df['implied_away'] = 1 / df['b365a']
    df['implied_over25'] = 1 / df['b365_over25']
    
    # Save to CSV
    output_path = Path('data/training/features_22k.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Final training set saved with {len(df.columns)} features.")
    logger.info(f"Elite features added: Volatility, Adj Form, Congestion, Luck Factor, League Tier.")
    
    return df

if __name__ == '__main__':
    create_training_dataset()
