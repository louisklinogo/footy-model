"""
Seed script: populate team_rivalries from a hardcoded list of major derbies.
Matches team names in our database using fuzzy search.
Run once, before training the situational model.
"""

import sys
from pathlib import Path
from difflib import get_close_matches

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

# -----------------------------------------------------------------------
# Canonical rivalry list (team_a, team_b, rivalry_name, league_code)
# -----------------------------------------------------------------------
RIVALRIES = [
    # England
    ("Arsenal", "Tottenham", "North London Derby", "E0"),
    ("Manchester United", "Manchester City", "Manchester Derby", "E0"),
    ("Liverpool", "Everton", "Merseyside Derby", "E0"),
    ("Chelsea", "Arsenal", "London Derby", "E0"),
    ("West Ham", "Millwall", "South London / East End Derby", "E0"),
    ("Newcastle", "Sunderland", "Tyne-Wear Derby", "E1"),
    ("Leeds", "Manchester United", "Yorkshire Derby", "E0"),
    ("Aston Villa", "Birmingham", "Second City Derby", "E0"),
    # Spain
    ("Real Madrid", "Barcelona", "El Clasico", "SP1"),
    ("Real Madrid", "Atletico Madrid", "Madrid Derby", "SP1"),
    ("Sevilla", "Real Betis", "Seville Derby", "SP1"),
    ("Athletic Club", "Real Sociedad", "Basque Derby", "SP1"),
    ("Valencia", "Villarreal", "Valencian Community Derby", "SP1"),
    # Italy
    ("AC Milan", "Inter", "Derby della Madonnina", "I1"),
    ("Roma", "Lazio", "Derby della Capitale", "I1"),
    ("Juventus", "Torino", "Derby della Mole", "I1"),
    ("Napoli", "Roma", "South-North Derby", "I1"),
    ("Fiorentina", "Juventus", "Derby d'Italia Florence variant", "I1"),
    # Germany
    ("Borussia Dortmund", "Schalke", "Revierderby", "D1"),
    ("Bayern Munich", "Borussia Dortmund", "Klassiker", "D1"),
    ("Hamburg", "Werder Bremen", "Nordderby", "D1"),
    ("Stuttgart", "Karlsruhe", "Baden-Wuerttemberg Derby", "D1"),
    ("FC Koln", "Bayer Leverkusen", "Rhine Derby", "D1"),
    # France
    ("Paris Saint-Germain", "Marseille", "Le Classique", "F1"),
    ("Marseille", "Lyon", "Riviera Derby", "F1"),
    ("Nice", "Monaco", "Cote d'Azur Derby", "F1"),
    # Netherlands
    ("Ajax", "Feyenoord", "De Klassieker", "N1"),
    ("Ajax", "PSV", "Dutch Championship Derby", "N1"),
    ("Feyenoord", "Sparta Rotterdam", "Rotterdam Derby", "N1"),
    # Portugal
    ("Benfica", "Sporting CP", "O Derby de Lisboa", "P1"),
    ("Benfica", "Porto", "O Classico", "P1"),
    ("Porto", "Sporting CP", "Derby do Norte", "P1"),
    # Scotland
    ("Celtic", "Rangers", "Old Firm Derby", "SC0"),
    ("Hearts", "Hibernian", "Edinburgh Derby", "SC0"),
    # Belgium
    ("Anderlecht", "Club Brugge", "Belgian Clasico", "B1"),
    # Turkey
    ("Galatasaray", "Fenerbahce", "Intercontinental Derby", "T1"),
    ("Galatasaray", "Besiktas", "Istanbul Derby", "T1"),
    ("Fenerbahce", "Besiktas", "Istanbul Derby 2", "T1"),
    # Brazil
    ("Flamengo", "Fluminense", "Fla-Flu", "BSA"),
    ("Corinthians", "Palmeiras", "Derby Paulista", "BSA"),
    # Argentina
    ("River Plate", "Boca Juniors", "Superclasico", "ARG"),
]


def seed_rivalries():
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT team_id, team_name FROM teams")
            teams_db = {row[1]: row[0] for row in cur.fetchall()}
        
        team_names = list(teams_db.keys())
        inserted = 0
        skipped = 0
        
        with conn.cursor() as cur:
            for team_a_name, team_b_name, rivalry_name, league in RIVALRIES:
                # Fuzzy match
                match_a = get_close_matches(team_a_name, team_names, n=1, cutoff=0.7)
                match_b = get_close_matches(team_b_name, team_names, n=1, cutoff=0.7)
                
                if not match_a or not match_b:
                    print(f"  SKIP (no match): {team_a_name} vs {team_b_name}")
                    skipped += 1
                    continue
                
                id_a = teams_db[match_a[0]]
                id_b = teams_db[match_b[0]]
                
                # Always store with lower id first to ensure UNIQUE constraint works
                if id_a > id_b:
                    id_a, id_b = id_b, id_a
                
                cur.execute("""
                    INSERT INTO team_rivalries (team_id_a, team_id_b, rivalry_name, league_code)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (team_id_a, team_id_b) DO UPDATE SET rivalry_name = EXCLUDED.rivalry_name
                """, (id_a, id_b, rivalry_name, league))
                
                print(f"  OK: {match_a[0]} vs {match_b[0]} → '{rivalry_name}'")
                inserted += 1
            
            conn.commit()
        
        print(f"\nDone. Inserted/updated: {inserted}, Skipped: {skipped}")
    
    finally:
        conn.close()


if __name__ == "__main__":
    seed_rivalries()
