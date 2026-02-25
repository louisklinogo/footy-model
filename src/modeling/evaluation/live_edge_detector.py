import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import csv
from src.db.db_utils import connect_db
from src.features.build_features import build_point_in_time_features
from src.pricing.poisson import PoissonPricer

def load_prematch_jsons(league_code: str) -> pd.DataFrame:
    """Read all scraped pre-match odds for a league."""
    json_dir = ROOT_DIR / 'data' / 'v1' / 'prematch_json' / league_code
    if not json_dir.exists():
        return pd.DataFrame()
    
    data = []
    for p in json_dir.glob("*.json"):
        with open(p, 'r') as f:
            data.append(json.load(f))
    return pd.DataFrame(data)

def detect_edges(min_kickoff: datetime = None):
    # 1. Load Model & Artifacts
    artifacts_dir = ROOT_DIR / 'model_artifacts' / 'poisson_model'
    with open(artifacts_dir / 'calibration_params.json', 'r') as f:
        params = json.load(f)
    
    rho = params['rho']
    feature_cols = params['features']
    
    model_h = xgb.Booster()
    model_h.load_model(artifacts_dir / 'xgb_lambda_home_v1.json')
    model_a = xgb.Booster()
    model_a.load_model(artifacts_dir / 'xgb_lambda_away_v1.json')

    # 2. Iterate through scaped leagues
    json_root = ROOT_DIR / 'data' / 'v1' / 'prematch_json'
    if not json_root.exists():
        print("No pre-match JSONs found. Run scrapers first.")
        return

    pricer = PoissonPricer()
    all_edges = []
    all_matches = [] # Output the match-sheets regardless of edges

    for league_dir in json_root.iterdir():
        if not league_dir.is_dir(): continue
        league_code = league_dir.name
        
        prematch_df = load_prematch_jsons(league_code)
        if prematch_df.empty: continue
        
        # --- ID MAPPING: String FS_ID -> Numeric DB_ID & Team Names ---
        fs_ids = prematch_df['id'].tolist()
        conn = connect_db()
        with conn.cursor() as cur:
            # We join to the teams table to get human-readable names
            cur.execute("""
                SELECT f.flashscore_id, f.fixture_id, th.team_name, ta.team_name 
                FROM fixtures f
                JOIN teams th ON f.home_team_id = th.team_id
                JOIN teams ta ON f.away_team_id = ta.team_id
                WHERE f.flashscore_id = ANY(%s)
            """, (fs_ids,))
            
            id_map = {}
            for row in cur.fetchall():
                id_map[row[0]] = {
                    "fixture_id": row[1],
                    "home_team": row[2],
                    "away_team": row[3]
                }
        conn.close()
        
        prematch_df['fixture_id'] = prematch_df['id'].apply(lambda x: id_map.get(x, {}).get('fixture_id'))
        prematch_df['home_team'] = prematch_df['id'].apply(lambda x: id_map.get(x, {}).get('home_team', 'Unknown'))
        prematch_df['away_team'] = prematch_df['id'].apply(lambda x: id_map.get(x, {}).get('away_team', 'Unknown'))
        valid_ids = [vid for vid in prematch_df['fixture_id'].tolist() if vid is not None]
        
        if not valid_ids:
            continue
            
        # 3. Build Model Features (Snapshots)
        feat_df = build_point_in_time_features(valid_ids, target_window_mins=0)
        if feat_df.empty: continue
        
        # --- Time Filtering ---
        if min_kickoff:
            feat_df = feat_df[feat_df['kickoff_time'] >= min_kickoff]
            if feat_df.empty: continue

        # Join snapshots with odds
        df = feat_df.merge(prematch_df, left_on='fixture_id', right_on='fixture_id')
        
        # 4. Predict Lambdas
        d_matrix = xgb.DMatrix(df[feature_cols])
        df['pred_lambda_h'] = model_h.predict(d_matrix)
        df['pred_lambda_a'] = model_a.predict(d_matrix)
        
        # 5. Scan for edges across 1X2 and OU
        for _, row in df.iterrows():
            lh, la = row['pred_lambda_h'], row['pred_lambda_a']
            matrix = pricer.generate_matrix(lh, la, rho=rho)
            
            h_rest = "N/A" if pd.isna(row['h_rest_days']) else f"{int(row['h_rest_days'])}d"
            a_rest = "N/A" if pd.isna(row['a_rest_days']) else f"{int(row['a_rest_days'])}d"
            insight = f"H_xG: {row['h_xg']:.2f} | A_xG: {row['a_xg']:.2f} | Rest: H={h_rest}, A={a_rest}"

            def get_kelly(prob, odds):
                if odds <= 1: return 0
                b = odds - 1
                q = 1 - prob
                k = (b * prob - q) / b
                return max(0, k * 0.1)

            match_info = {
                "league": league_code,
                "home_team": row['home_team'],
                "away_team": row['away_team'],
                "kickoff": row['kickoff_time'].strftime("%Y-%m-%d %H:%M UTC"),
                "lh": lh,
                "la": la,
                "1x2_probs": pricer.get_1x2(matrix),
                "1x2_odds": row['odds'].get('1x2', {}),
                "btts_prob": pricer.get_btts(matrix),
                "ou_probs": {},
                "ou_odds": row['odds'].get('ou', {}),
                "team_ou_probs": {
                    "home_1.5": pricer.get_team_over_under(matrix, 1.5, True),
                    "away_1.5": pricer.get_team_over_under(matrix, 1.5, False)
                },
                "edges": [],
            }

            # Map the 1X2 markets
            mapping = {"1": "home", "X": "draw", "2": "away"}
            for b_key, m_key in mapping.items():
                if b_key in match_info['1x2_odds']:
                    try:
                        odds = float(match_info['1x2_odds'][b_key])
                        prob = match_info['1x2_probs'][m_key]
                        ev = (prob * odds) - 1
                        if ev > 0.03:
                            edge = {
                                "league": league_code,
                                "match": f"{row['home_team']} vs {row['away_team']}",
                                "kickoff": row['kickoff_time'].isoformat(),
                                "market": "1X2",
                                "selection": m_key.upper(),
                                "ev": ev,
                                "odds": odds,
                                "prob": prob,
                                "stake": get_kelly(prob, odds),
                                "insight": insight
                            }
                            match_info["edges"].append(edge)
                            all_edges.append(edge)
                    except ValueError:
                        pass
            
            # O/U MARKETS 
            for ou_line in [1.5, 2.5, 3.5]:
                prob_over = pricer.get_over_under(matrix, ou_line)
                prob_ou = {"over": prob_over, "under": 1.0 - prob_over}
                match_info["ou_probs"][str(ou_line)] = prob_ou
                
                line_odds = match_info["ou_odds"].get(str(ou_line), {})
                for b_key, m_key in [('over', 'over'), ('under', 'under')]:
                    if b_key in line_odds:
                        try:
                            odds = float(line_odds[b_key])
                            prob = prob_ou[m_key]
                            ev = (prob * odds) - 1
                            if ev > 0.03:
                                edge = {
                                    "league": league_code,
                                    "match": f"{row['home_team']} vs {row['away_team']}",
                                    "kickoff": row['kickoff_time'].isoformat(),
                                    "market": f"O/U {ou_line}",
                                    "selection": m_key.upper(),
                                    "ev": ev,
                                    "odds": odds,
                                    "prob": prob,
                                    "stake": get_kelly(prob, odds),
                                    "insight": insight
                                }
                                match_info["edges"].append(edge)
                                all_edges.append(edge)
                        except ValueError:
                            pass
                            
            all_matches.append(match_info)

    # Output detailed report and CSV
    if not all_matches:
        print("\nNo upcoming matches found.")
        return

    md_lines = []
    md_lines.append(f"# FULL MATCH SHEETS | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md_lines.append(f"**Time Filter:** {min_kickoff if min_kickoff else 'None'}\n")

    for m in all_matches:
        md_lines.append("---")
        md_lines.append(f"## {m['league'].upper()}: {m['home_team']} vs {m['away_team']} ({m['kickoff']})")
        md_lines.append("\n### [PREDICTIONS]")
        md_lines.append(f"- **Home Intensity:** {m['lh']:.2f} goals | **Away Intensity:** {m['la']:.2f} goals")
        md_lines.append(f"- **True Probabilities:** Home: {m['1x2_probs']['home']:.1%} | Draw: {m['1x2_probs']['draw']:.1%} | Away: {m['1x2_probs']['away']:.1%}")
        o_probs = []
        for line in ['1.5', '2.5', '3.5']:
            if line in m['ou_probs']:
                o_probs.append(f"O{line}: {m['ou_probs'][line]['over']:.1%}")
        md_lines.append(f"- **Over Goals:** {' | '.join(o_probs)}")
        md_lines.append(f"- **BTTS:** {m['btts_prob']:.1%} | **Home 1UP:** {m['team_ou_probs']['home_1.5']:.1%} | **Away 1UP:** {m['team_ou_probs']['away_1.5']:.1%}")

        md_lines.append("\n### [MARKET COMPARISON]")
        # 1X2 formatting
        odds_str = "N/A"
        if m['1x2_odds']:
            h_odd = float(m['1x2_odds'].get('1', 0.0))
            d_odd = float(m['1x2_odds'].get('X', 0.0))
            a_odd = float(m['1x2_odds'].get('2', 0.0))
            
            h_imp = f"{1/h_odd:.1%}" if h_odd > 0 else "N/A"
            d_imp = f"{1/d_odd:.1%}" if d_odd > 0 else "N/A"
            a_imp = f"{1/a_odd:.1%}" if a_odd > 0 else "N/A"
            
            odds_str = f"{h_odd:.2f} ({h_imp}) / {d_odd:.2f} ({d_imp}) / {a_odd:.2f} ({a_imp})"
            
        edge_1x2_strs = []
        for e in m['edges']:
            if e['market'] == '1X2':
                edge_1x2_strs.append(f"**{e['selection']} (+{e['ev']:.1%})**")
        edge_1x2_format = ", ".join(edge_1x2_strs) if edge_1x2_strs else "None found"
        md_lines.append(f"- **1X2:** Bookie: `{odds_str}` | Edges: {edge_1x2_format}")
        
        for ou_line in ['1.5', '2.5', '3.5']:
            if m['ou_odds'].get(ou_line):
                o_odd = float(m['ou_odds'][ou_line].get('over', 0.0))
                u_odd = float(m['ou_odds'][ou_line].get('under', 0.0))
                
                o_imp = f"{1/o_odd:.1%}" if o_odd > 0 else "N/A"
                u_imp = f"{1/u_odd:.1%}" if u_odd > 0 else "N/A"
                
                mkt_str = f"Bookie: `{o_odd:.2f} ({o_imp} O) / {u_odd:.2f} ({u_imp} U)`"
                
                edge_ou_strs = []
                for e in m['edges']:
                    if e['market'] == f'O/U {ou_line}':
                        edge_ou_strs.append(f"**{e['selection']} (+{e['ev']:.1%})**")
                edge_ou_format = ", ".join(edge_ou_strs) if edge_ou_strs else "None found"
                md_lines.append(f"- **O/U {ou_line}:** {mkt_str} | Edges: {edge_ou_format}")

        md_lines.append("\n### [RECOMMENDATIONS]")
        if m['edges']:
            for e in m['edges']:
                md_lines.append(f"- > **SELECTION:** {e['market']} **{e['selection']}** @ {e['odds']:.2f} (Recommended Stake: {e['stake']:.1%})")
        else:
            md_lines.append("- No valuable edges identified.")
            
    md_path = ROOT_DIR / "data" / "v1" / f"daily_slip_{datetime.now().strftime('%Y%m%d')}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "="*85)
    print(f"       ANALYSIS COMPLETE | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*85)
    print(f"Full Match Sheets saved to: {md_path}")

    if all_edges:
        output_path = ROOT_DIR / "data" / "v1" / f"daily_slip_{datetime.now().strftime('%Y%m%d')}.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_edges[0].keys())
            writer.writeheader()
            writer.writerows(all_edges)
        print(f"Edges CSV saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-kickoff", type=str, help="ISO format min kickoff time (UTC)")
    args = parser.parse_args()
    
    min_k = None
    if args.min_kickoff:
        min_k = datetime.fromisoformat(args.min_kickoff).replace(tzinfo=timezone.utc)
    
    detect_edges(min_kickoff=min_k)
