import sys
sys.path.insert(0, '.')
from src.db.db_utils import connect_db

conn = connect_db()
cur = conn.cursor()

# Check fixture_stats_premium for corners-related data
cur.execute('''
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'fixture_stats_premium'
    ORDER BY ordinal_position
''')
premium_cols = [r[0] for r in cur.fetchall()]

print('=== FIXTURE_STATS_PREMIUM COLUMNS ===')
print()
print('Corner-related:', [c for c in premium_cols if 'corner' in c.lower()])
print('Shot-related:', [c for c in premium_cols if 'shot' in c.lower() or 'sot' in c.lower()])
print('Cross-related:', [c for c in premium_cols if 'cross' in c.lower()])
print('Possession-related:', [c for c in premium_cols if 'possess' in c.lower()])
print('xG-related:', [c for c in premium_cols if 'xg' in c.lower()])
print()

# Check actual data available
cur.execute('''
    SELECT 
        COUNT(*) as total,
        COUNT(h_corners) as h_corners_count,
        COUNT(a_corners) as a_corners_count,
        COUNT(h_shots) as h_shots_count,
        COUNT(a_shots) as a_shots_count,
        COUNT(h_sot) as h_sot_count,
        COUNT(a_sot) as a_sot_count
    FROM fixture_stats_premium
''')
row = cur.fetchone()
print('=== DATA COVERAGE ===')
print(f'Total rows: {row[0]}')
print(f'h_corners: {row[1]} ({round(100*row[1]/row[0], 1)}%)')
print(f'a_corners: {row[2]} ({round(100*row[2]/row[0], 1)}%)')
print(f'h_shots: {row[3]} ({round(100*row[3]/row[0], 1) if row[0] > 0 else 0}%)')
print(f'a_shots: {row[4]} ({round(100*row[4]/row[0], 1) if row[0] > 0 else 0}%)')
print(f'h_sot: {row[5]} ({round(100*row[5]/row[0], 1) if row[0] > 0 else 0}%)')
print(f'a_sot: {row[6]} ({round(100*row[6]/row[0], 1) if row[0] > 0 else 0}%)')

cur.close()
conn.close()