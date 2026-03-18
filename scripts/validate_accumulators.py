"""
Comprehensive accumulator validation with team context, H2H, risk scores, AH alternatives.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from src.db.db_utils import connect_db

def get_team_context(team_names):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT ON (t.team_name)
            t.team_name, tec.overall_rank, tec.overall_points, tec.overall_played,
            tec.overall_wins, tec.overall_draws, tec.overall_losses,
            tec.overall_goals_for, tec.overall_goals_against,
            tec.form_sequence, tec.form_points_last5
        FROM team_external_context tec
        JOIN teams t ON tec.team_id = t.team_id
        WHERE t.team_name = ANY(%s)
        ORDER BY t.team_name, tec.snapshot_time_utc DESC
    ''', (team_names,))
    result = {}
    for r in cur.fetchall():
        result[r[0]] = {'rank': r[1], 'pts': r[2], 'gp': r[3], 'w': r[4], 'd': r[5], 'l': r[6], 'gf': r[7], 'ga': r[8], 'form': r[9], 'form_pts': r[10]}
    cur.close()
    conn.close()
    return result

def get_h2h(team_a, team_b, limit=5):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('''
        SELECT f.match_datetime_utc, ht.team_name, at.team_name, fr.home_goals, fr.away_goals
        FROM fixtures f
        JOIN teams ht ON f.home_team_id = ht.team_id
        JOIN teams at ON f.away_team_id = at.team_id
        JOIN fixture_results fr ON f.fixture_id = fr.fixture_id
        WHERE ((ht.team_name = %s AND at.team_name = %s) OR (ht.team_name = %s AND at.team_name = %s))
        ORDER BY f.match_datetime_utc DESC LIMIT %s
    ''', (team_a, team_b, team_b, team_a, limit))
    results = [{'date': str(r[0])[:10], 'home': r[1], 'away': r[2], 'score': f"{r[3]}-{r[4]}"} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return results

def main():
    print("=" * 80)
    print("COMPREHENSIVE ACCUMULATOR VALIDATION")
    print("=" * 80)
    
    df = pd.read_csv('storage/reports/edges_today.csv')
    df = df[df['book_odds'].notna() & df['p_model'].notna()]
    
    # Get our accumulator teams (top 12 picks)
    best = df.sort_values('p_model', ascending=False).head(12)
    teams = list(set(best['home_team'].tolist() + best['away_team'].tolist()))[:15]
    
    print(f"\n{'='*80}")
    print("1. TEAM STANDINGS & FORM")
    print("="*80)
    
    ctx = get_team_context(teams)
    for team, c in sorted(ctx.items(), key=lambda x: x[1].get('rank', 99) or 99):
        if c['rank']:
            form = str(c['form'])[:10] if c['form'] else '-'
            print(f"\n{team}")
            print(f"  #{c['rank']} | {c['pts']}pts ({c['gp']}gp) | {c['w']}W-{c['d']}D-{c['l']}L | GF:{c['gf']} GA:{c['ga']}")
            print(f"  Form: {form} ({c['form_pts'] or '-'} pts)")
    
    print(f"\n{'='*80}")
    print("2. HEAD-TO-HEAD HISTORY")
    print("="*80)
    
    fixtures = best.drop_duplicates(subset=['fixture_id']).head(6)
    for _, f in fixtures.iterrows():
        h2h = get_h2h(f['home_team'], f['away_team'], limit=3)
        print(f"\n{f['home_team']} vs {f['away_team']}:")
        if h2h:
            for m in h2h:
                print(f"  {m['date']}: {m['home']} {m['score']} {m['away']}")
        else:
            print("  No recent H2H matches found")
    
    print(f"\n{'='*80}")
    print("3. MODEL PROBABILITY CONTEXT")
    print("="*80)
    
    print("\nTop 10 picks with validation:")
    for i, (_, r) in enumerate(best.head(10).iterrows(), 1):
        home_ctx = ctx.get(r['home_team'], {})
        away_ctx = ctx.get(r['away_team'], {})
        
        home_rank = home_ctx.get('rank', '-')
        away_rank = away_ctx.get('rank', '-')
        
        print(f"\n{i}. {r['home_team']} vs {r['away_team']}")
        print(f"   {r['market_code']} @ {r['book_odds']:.2f} | Model: {r['p_model']*100:.1f}%")
        print(f"   Ranks: {r['home_team'][:10]} #{home_rank} vs {r['away_team'][:10]} #{away_rank}")

if __name__ == '__main__':
    main()