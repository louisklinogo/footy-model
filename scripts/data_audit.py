import sys
sys.path.insert(0, '.')
from src.db.db_utils import connect_db

conn = connect_db()
cur = conn.cursor()

print('=== COMPREHENSIVE DATA AUDIT ===')
print()

# 1. fixture_stats_premium xG coverage
cur.execute('SELECT COUNT(*), COUNT(h_xg) FROM fixture_stats_premium')
row = cur.fetchone()
print(f'fixture_stats_premium: {row[1]}/{row[0]} have xG ({round(100*row[1]/row[0], 1)}%)')

# 2. Player xG by recent months
cur.execute('''
    SELECT 
        EXTRACT(YEAR FROM f.match_datetime_utc) as yr,
        EXTRACT(MONTH FROM f.match_datetime_utc) as mth,
        COUNT(fps.*) as total_rows,
        COUNT(fps.expected_goals) as xg_rows
    FROM fixture_player_stats fps
    JOIN fixtures f ON fps.fixture_id = f.fixture_id
    GROUP BY 1, 2
    ORDER BY 1 DESC, 2 DESC
    LIMIT 6
''')
print()
print('Player xG by month:')
for row in cur.fetchall():
    pct = round(100*row[3]/row[2], 1) if row[2] > 0 else 0
    print(f'  {int(row[0])}-{int(row[1]):02d}: {pct}%')

# 3. Scheduled fixtures with availability
cur.execute('''
    SELECT COUNT(DISTINCT pa.fixture_id)
    FROM player_availability pa
    JOIN fixtures f ON pa.fixture_id = f.fixture_id
    WHERE f.status = 'scheduled'
''')
sched_with_avail = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM fixtures WHERE status = 'scheduled'")
total_sched = cur.fetchone()[0]
print()
print(f'Scheduled fixtures with player availability: {sched_with_avail}/{total_sched}')
print(f'Scheduled fixtures MISSING player availability: {total_sched - sched_with_avail}')

cur.close()
conn.close()