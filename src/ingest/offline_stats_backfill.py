import json

from src.db.db_utils import connect_db

NAME_MAP = {
    "Expected goals": "xg",
    "Expected goals on target": "xgot",
    "Expected assists": "xa",
    "Big chances": "big_chances",
    "Ball possession": "possession",
    "Touches in penalty area": "box_touches",
    "Crosses": "crosses",
    "Blocked shots": "blocked_shots",
    "Through balls": "through_passes",
    "Shots on target": "sot",
    "Shots inside box": "shots_inside_box",
    "Corner kicks": "corners",
    "Goals prevented": "goals_prevented",
    "Tackles won": "tackles_pct",
    "Interceptions": "interceptions",
    "Error led to shot": "errors_lead_to_shot",
    "Offsides": "offsides",
    "Fouls": "fouls",
    "Dispossessed": "dispossessed",
    "Hit woodwork": "hit_woodwork",
    "Dribbles": "dribbles",
    "Recovered balls": "ball_recoveries",
    "Yellow cards": "yellow_cards",
    "Red cards": "red_cards",
    "Duels": "duels",
    "Ground duels": "ground_duels",
    "Aerial duels": "aerial_duels",
    "Final third entries": "final_third_entries",
    "Long balls": "long_balls",
}

def parse_sofa_value(val):
    if val is None: return 0
    if isinstance(val, str):
        if "%" in val:
            return float(val.replace("%", ""))
        if "/" in val:
            parts = val.split('/')
            return float(parts[0])
    return float(val)

def parse_sofa_total(val):
    if isinstance(val, str) and "/" in val:
        return float(val.split('/')[1])
    return 0

def process_batch():
    conn = connect_db()
    
    # Grab all fixtures with valid JSON but missing the new p1 columns
    with conn.cursor() as cur:
        cur.execute("""
            SELECT fixture_id, raw_json 
            FROM fixture_stats_premium 
            WHERE fidelity_score > 0 
              AND raw_json IS NOT NULL 
              AND (
                    h_shots_inside_box_p1 IS NULL
                    OR h_yellow_cards IS NULL
                    OR h_red_cards IS NULL
                  )
        """)
        rows = cur.fetchall()
        
    print(f"Found {len(rows)} fixtures to backfill offline.")
    
    updated_count = 0
    with conn.cursor() as cur:
        for fixture_id, raw_json in rows:
            if not raw_json or 'statistics' not in raw_json:
                continue
                
            update_data = {}
            for period_data in raw_json['statistics']:
                period = period_data['period']
                suffix = ""
                if period == "1ST": suffix = "_p1"
                elif period == "2ND": suffix = "_p2"
                elif period != "ALL": continue
                
                for group in period_data.get('groups', []):
                    for item in group.get('statisticsItems', []):
                        name = item['name']
                        if name in NAME_MAP:
                            col_base = NAME_MAP[name]
                            home_val = parse_sofa_value(item.get('homeValue'))
                            away_val = parse_sofa_value(item.get('awayValue'))
                            
                            if col_base == "dribbles":
                                update_data[f"h_dribbles_success{suffix}"] = home_val
                                update_data[f"a_dribbles_success{suffix}"] = away_val
                                update_data[f"h_dribbles_total{suffix}"] = parse_sofa_total(item.get('homeValue'))
                                update_data[f"a_dribbles_total{suffix}"] = parse_sofa_total(item.get('awayValue'))
                            elif col_base == "long_balls":
                                update_data[f"h_accurate_long_balls{suffix}"] = home_val
                                update_data[f"a_accurate_long_balls{suffix}"] = away_val
                                update_data[f"h_total_long_balls{suffix}"] = parse_sofa_total(item.get('homeValue'))
                                update_data[f"a_total_long_balls{suffix}"] = parse_sofa_total(item.get('awayValue'))
                            elif col_base in ["duels", "ground_duels", "aerial_duels"]:
                                if suffix == "":
                                    update_data[f"h_{col_base}_won"] = home_val
                                    update_data[f"a_{col_base}_won"] = away_val
                            else:
                                update_data[f"h_{col_base}{suffix}"] = home_val
                                update_data[f"a_{col_base}{suffix}"] = away_val
                                
            if not update_data:
                continue
                
            columns = list(update_data.keys())
            values = [update_data[c] for c in columns]
            
            set_clause = ", ".join([f"{c} = %s" for c in columns])
            
            cur.execute(f"""
                UPDATE fixture_stats_premium 
                SET {set_clause}
                WHERE fixture_id = %s
            """, values + [fixture_id])
            
            updated_count += 1
            if updated_count % 500 == 0:
                print(f"Updated {updated_count} rows...")
                conn.commit()
                
        conn.commit()
    conn.close()
    print(f"Offline backfill complete: {updated_count} rows updated.")

if __name__ == "__main__":
    process_batch()
