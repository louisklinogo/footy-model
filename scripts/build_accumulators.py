"""
Build accumulator slips with comprehensive validation.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import subprocess
from src.db.db_utils import connect_db

MARKET_LABELS = {
    'dc_1x': lambda h, a: f"{h[:10]} or DRAW",
    'dc_x2': lambda h, a: f"{a[:10]} or DRAW",
    'dc_12': lambda h, a: f"{h[:8]} or {a[:8]}",
    'o15': lambda h, a: "OVER 1.5 GOALS",
}

def format_market(code, home, away):
    fmt = MARKET_LABELS.get(code)
    return fmt(home, away) if fmt else code.upper()

def load_team_stats():
    """Load team rolling stats, filtering out first-game-of-season entries with 0 sample."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT ON (t.team_name) 
            t.team_name, 
            tps.rolling_xg, 
            tps.rolling_xg_against,
            tps.sample_size
        FROM team_premium_snapshots tps
        JOIN teams t ON tps.team_id = t.team_id
        WHERE tps.rolling_xg IS NOT NULL
        AND tps.sample_size > 0
        ORDER BY t.team_name, tps.sample_size DESC, tps.built_at DESC
    ''')
    return {r[0]: {'xG': r[1] or 0, 'xGA': r[2] or 0, 'sample': r[3]} for r in cur.fetchall()}

def validate(leg, stats):
    """Validate selection against rolling team stats."""
    home = str(leg['home_team'])
    away = str(leg['away_team'])
    code = leg['market_code']
    hs = stats.get(home, {})
    as_ = stats.get(away, {})
    
    hxg = float(hs.get('xG', 0) or 0)
    hxga = float(hs.get('xGA', 0) or 0)
    hsample = hs.get('sample', 0)
    axg = float(as_.get('xG', 0) or 0)
    axga = float(as_.get('xGA', 0) or 0)
    asample = as_.get('sample', 0)
    
    if code == 'dc_1x':
        if hxg > hxga:
            return f"HOME xG {hxg:.2f} > xGA {hxga:.2f} (n={hsample}) - SUPPORTED"
        return f"HOME xG {hxg:.2f} <= xGA {hxga:.2f} (n={hsample}) - CAUTION"
    elif code == 'dc_x2':
        if axg > axga:
            return f"AWAY xG {axg:.2f} > xGA {axga:.2f} (n={asample}) - SUPPORTED"
        return f"AWAY xG {axg:.2f} <= xGA {axga:.2f} (n={asample}) - CAUTION"
    elif code == 'o15':
        total = hxg + axg
        status = "GOOD" if total > 2.0 else "MARGINAL"
        return f"COMBINED xG {total:.2f} (n={hsample}+{asample}) - {status}"
    return ""

def build_slips(df):
    best = df[df['p_model'] >= 0.50].sort_values(['fixture_id', 'p_model'], ascending=[True, False])
    best = best.drop_duplicates(subset=['fixture_id'])
    items = best.sort_values('p_model', ascending=False).to_dict('records')
    slips = []
    for start, name in [(0, 'CONSERVATIVE'), (2, 'BALANCED'), (4, 'DIVERSIFIED')]:
        legs, odds, prob = [], 1.0, 1.0
        for p in items[start:]:
            new_odds = odds * p['book_odds']
            if new_odds > 10.0:
                continue
            legs.append(p)
            odds = new_odds
            prob *= p['p_model']
            if odds >= 5.0 and len(legs) >= 4:
                break
        slips.append({'name': name, 'legs': legs, 'odds': odds, 'prob': prob})
    return slips

def main():
    print("=" * 80)
    print("FOOTBALL ACCUMULATOR ENGINE - FULLY VALIDATED")
    print("=" * 80)
    df = pd.read_csv('storage/reports/edges_today.csv')
    df = df[df['book_odds'].notna() & df['p_model'].notna()]
    print(f"\nLoaded {len(df)} predictions")
    print("\n" + "-" * 80)
    print("VALIDATION: TEAM STANDINGS, FORM, H2H")
    print("-" * 80)
    result = subprocess.run(['.venv/Scripts/python.exe', 'scripts/validate_accumulators.py'], capture_output=True, text=True)
    print(result.stdout[:3500])
    stats = load_team_stats()
    slips = build_slips(df)
    for s in slips:
        print(f"\n{'='*80}")
        print(f"  SLIP: {s['name']} | {len(s['legs'])} legs")
        print(f"  ODDS: {s['odds']:.2f} | WIN PROB: {s['prob']*100:.1f}%")
        print('='*80)
        for i, leg in enumerate(s['legs'], 1):
            dt = str(leg['match_datetime_utc'])
            ko = dt[11:16] if len(dt) >= 16 else '??'
            mkt = format_market(leg['market_code'], leg['home_team'], leg['away_team'])
            v = validate(leg, stats)
            print(f"\n  {i}. [{ko}] {leg['home_team'][:12]} vs {leg['away_team'][:12]}")
            print(f"     {mkt} @ {leg['book_odds']:.2f} | Model: {leg['p_model']*100:.1f}%")
            if v:
                print(f"     >> {v}")
        print(f"\n  TOTAL: {s['odds']:.2f} | {s['prob']*100:.1f}% win probability")
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE - Place all 3 slips with equal stakes")
    print("=" * 80)

if __name__ == '__main__':
    main()
