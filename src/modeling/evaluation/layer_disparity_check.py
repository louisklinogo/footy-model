import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.pricing.poisson import PoissonPricer
import math

def calculate_brier_score(p, actual):
    return (p - actual)**2

def main():
    conn = connect_db()
    pricer = PoissonPricer()
    
    query = """
    WITH l1 AS (
        SELECT fixture_id, 
               MAX(CASE WHEN market_code = 'lambda_home' THEN (metadata_json->>'lambda')::float END) as l1_h,
               MAX(CASE WHEN market_code = 'lambda_away' THEN (metadata_json->>'lambda')::float END) as l1_a
        FROM predictions WHERE model_name = 'lambda_xgb'
        GROUP BY fixture_id
    ),
    l15 AS (
        SELECT fixture_id, 
               MAX(CASE WHEN market_code = 'adj_lambda_home' THEN (metadata_json->>'lambda')::float END) as l15_h,
               MAX(CASE WHEN market_code = 'adj_lambda_away' THEN (metadata_json->>'lambda')::float END) as l15_a
        FROM predictions WHERE model_name = 'situational_xgb'
        GROUP BY fixture_id
    ),
    l2 AS (
        SELECT fixture_id, 
               MAX(CASE WHEN market_code = '1x2_h' THEN p_model END) as l2_1x2_h,
               MAX(CASE WHEN market_code = 'o25' THEN p_model END) as l2_o25
        FROM predictions WHERE model_name = 'market_outcome_gbm'
        GROUP BY fixture_id
    )
    SELECT 
        f.fixture_id,
        fr.home_goals,
        fr.away_goals,
        l1.l1_h, l1.l1_a,
        l15.l15_h, l15.l15_a,
        l2.l2_1x2_h, l2.l2_o25
    FROM fixtures f
    JOIN fixture_results fr ON f.fixture_id = fr.fixture_id
    JOIN l1 ON f.fixture_id = l1.fixture_id
    JOIN l15 ON f.fixture_id = l15.fixture_id
    JOIN l2 ON f.fixture_id = l2.fixture_id
    WHERE f.status = 'ft'
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    results = []
    for _, row in df.iterrows():
        h_goals = int(row['home_goals'])
        a_goals = int(row['away_goals'])
        tot = h_goals + a_goals
        t_1x2_h = 1.0 if h_goals > a_goals else 0.0
        t_o25 = 1.0 if tot >= 3 else 0.0
        
        m_l1 = pricer.generate_matrix(row['l1_h'], row['l1_a'])
        m_l15 = pricer.generate_matrix(row['l15_h'], row['l15_a'])
        
        results.append({
            "l1_1x2_h": calculate_brier_score(pricer.get_1x2(m_l1)['home'], t_1x2_h),
            "l15_1x2_h": calculate_brier_score(pricer.get_1x2(m_l15)['home'], t_1x2_h),
            "l2_1x2_h": calculate_brier_score(row['l2_1x2_h'], t_1x2_h),
            "l1_o25": calculate_brier_score(pricer.get_over_under(m_l1, 2.5), t_o25),
            "l15_o25": calculate_brier_score(pricer.get_over_under(m_l15, 2.5), t_o25),
            "l2_o25": calculate_brier_score(row['l2_o25'], t_o25)
        })

    res = pd.DataFrame(results)
    print("Market,L1,L1.5,L2")
    print(f"1X2 Home Win,{res['l1_1x2_h'].mean():.6f},{res['l15_1x2_h'].mean():.6f},{res['l2_1x2_h'].mean():.6f}")
    print(f"Over 2.5 Goals,{res['l1_o25'].mean():.6f},{res['l15_o25'].mean():.6f},{res['l2_o25'].mean():.6f}")

if __name__ == "__main__":
    main()
