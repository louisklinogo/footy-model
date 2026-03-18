"""
Audit accumulator selections using DB stats.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from src.db.db_utils import connect_db

def get_team_stats(team_names):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT ON (t.team_name)
            t.team_name,
            tps.rolling_xg,
            tps.rolling_xg_against,
            tps.rolling_sot,
            tps.rolling_possession,
            tps.sample_size
        FROM team_premium_snapshots tps
        JOIN teams t ON tps.team_id = t.team_id
        WHERE t.team_name = ANY(%s)
        ORDER BY t.team_name, tps.built_at DESC
    ''', (team_names,))
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=cols)

def main():
    teams = [
        'Frosinone', 'Bari', 'Flamengo RJ', 'Remo', 'Palmeiras', 'Botafogo RJ',
        'Besiktas', 'Kasimpasa', 'Fenerbahce', 'Gaziantep', 'Basaksehir', 'Antalyaspor',
        'Athletico-PR', 'Cruzeiro', 'Santos', 'Internacional', 'Bahia', 'Bragantino',
        'Mirassol', 'Coritiba', 'Guadalajara Chivas', 'Club Leon', 'Vasco', 'Fluminense'
    ]
    
    print("Fetching team stats from DB...")
    df = get_team_stats(teams)
    
    if len(df) == 0:
        print("No team stats found. Checking team names...")
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT team_name FROM teams WHERE team_name ILIKE '%Besiktas%' OR team_name ILIKE '%Fener%' LIMIT 5")
        for r in cur.fetchall():
            print(f"  Found: {r[0]}")
        cur.close()
        conn.close()
        return
    
    print(f"\n{'='*70}")
    print("TEAM FORM ANALYSIS (Rolling Stats)")
    print('='*70)
    
    for _, r in df.iterrows():
        print(f"\n{r['team_name']}")
        print(f"  xG: {r['rolling_xg']:.2f} | xGA: {r['rolling_xg_against']:.2f}")
        print(f"  SOT: {r['rolling_sot']:.1f} | Poss: {r['rolling_possession']:.0f}%")
        print(f"  Sample: {r['sample_size']} games")

if __name__ == '__main__':
    main()