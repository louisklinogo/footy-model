"""
Check freshness of team_premium_snapshots data.
"""
import sys
sys.path.insert(0, '.')
from src.db.db_utils import connect_db
from datetime import datetime, timezone

conn = connect_db()
cur = conn.cursor()

print("TEAM PREMIUM SNAPSHOTS FRESHNESS CHECK")
print("=" * 90)
print(f"Current time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)

# Check when snapshots were built for our accumulator teams
teams = ['Frosinone', 'Besiktas', 'Fenerbahce', 'Flamengo RJ', 'Palmeiras', 
         'Bahia', 'Bragantino', 'Basaksehir', 'Antalyaspor', 'Santos']

cur.execute('''
    SELECT 
        t.team_name,
        tps.built_at,
        tps.sample_size,
        tps.rolling_xg,
        tps.rolling_xg_against,
        tps.fidelity_score
    FROM team_premium_snapshots tps
    JOIN teams t ON tps.team_id = t.team_id
    WHERE t.team_name = ANY(%s)
    ORDER BY t.team_name, tps.built_at DESC
''', (teams,))

print("\nSNAPSHOT AGE FOR ACCUMULATOR TEAMS:")
print("-" * 90)

seen_teams = set()
for r in cur.fetchall():
    name = r[0]
    if name in seen_teams:
        continue
    seen_teams.add(name)
    
    built = r[1]
    sample = r[2]
    xg = r[3] or 0
    xga = r[4] or 0
    fidelity = r[5] or 0
    
    if built:
        age = datetime.now(timezone.utc) - built
        age_str = f"{age.days}d {age.seconds//3600}h ago"
    else:
        age_str = "N/A"
    
    print(f"{name[:15]:15} | Built: {str(built)[:19]:19} | Age: {age_str:12} | Sample: {sample:3} | xG: {xg:.2f}")

# Check build frequency
print("\n" + "=" * 90)
print("SNAPSHOT BUILD FREQUENCY (last 7 days):")
print("-" * 90)

cur.execute('''
    SELECT DATE(built_at) as build_date, COUNT(*) as snapshots_built
    FROM team_premium_snapshots
    WHERE built_at > NOW() - INTERVAL '7 days'
    GROUP BY DATE(built_at)
    ORDER BY build_date DESC
''')

rows = cur.fetchall()
if rows:
    for r in rows:
        print(f"  {r[0]} | {r[1]:5} snapshots")
else:
    print("  NO SNAPSHOTS BUILT IN LAST 7 DAYS - DATA IS STALE!")

# Check most recent snapshot overall
print("\n" + "=" * 90)
print("MOST RECENT SNAPSHOTS IN DB:")
print("-" * 90)

cur.execute('''
    SELECT t.team_name, tps.built_at, tps.sample_size
    FROM team_premium_snapshots tps
    JOIN teams t ON tps.team_id = t.team_id
    ORDER BY tps.built_at DESC
    LIMIT 10
''')

for r in cur.fetchall():
    print(f"  {r[0][:20]:20} | {r[1]} | sample: {r[2]}")

cur.close()
conn.close()