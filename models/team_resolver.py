import psycopg2
import json
import sys
from difflib import get_close_matches
from pathlib import Path
import logging


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_utils import connect_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAPPING_FILE = Path('data/team_mappings.json')

def get_unmatched_teams():
    conn = connect_db()
    cur = conn.cursor()
    
    # Get all unique teams from upcoming fixtures
    cur.execute("SELECT DISTINCT home_team FROM upcoming_fixtures UNION SELECT DISTINCT away_team FROM upcoming_fixtures")
    upcoming_teams = [r[0] for r in cur.fetchall()]
    
    # Get all unique teams from historical matches
    cur.execute("SELECT DISTINCT home_team FROM matches UNION SELECT DISTINCT away_team FROM matches")
    historical_teams = [r[0] for r in cur.fetchall()]
    
    conn.close()
    return upcoming_teams, historical_teams

def build_mapping():
    upcoming, historical = get_unmatched_teams()
    
    # Load existing mapping if it exists
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, 'r') as f:
            mapping = json.load(f)
    else:
        mapping = {}
        
    new_matches = 0
    unresolved = []
    
    # Common words to ignore for matching
    ignore_words = {'FC', 'City', 'Town', 'United', 'Utd', 'Rovers', 'Wanderers', 'Athletic', 'Ath.', 'St', 'St.', 'Real', 'De', 'La'}
    
    def simplify(name):
        parts = name.replace('.', ' ').replace('-', ' ').split()
        return {p for p in parts if p not in ignore_words and len(p) > 2}

    for team in upcoming:
        if team in historical or team in mapping:
            continue
            
        # Strategy 1: Fuzzy match
        matches = get_close_matches(team, historical, n=3, cutoff=0.5)
        
        # Strategy 2: Check if any historical team is a substring or share core words
        team_simple = simplify(team)
        
        found = False
        if matches:
            best_match = matches[0]
            # If high confidence or very similar simplified forms
            if simplify(best_match) == team_simple and len(team_simple) > 0:
                mapping[team] = best_match
                new_matches += 1
                found = True
        
        if not found:
            for hist_team in historical:
                hist_simple = simplify(hist_team)
                if team_simple == hist_simple and len(team_simple) > 0:
                    mapping[team] = hist_team
                    new_matches += 1
                    found = True
                    break
                # Substring check
                if len(team) > 5 and len(hist_team) > 3:
                    if hist_team in team or team in hist_team:
                        mapping[team] = hist_team
                        new_matches += 1
                        found = True
                        break
        
        if not found:
            unresolved.append((team, matches))
            
    # Save the mapping
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=4)
        
    logger.info(f"Mapping updated. {new_matches} new teams auto-mapped.")
    logger.info(f"Total mapped teams: {len(mapping)}")
    
    if unresolved:
        logger.warning(f"{len(unresolved)} teams still need manual resolution.")
        # Print first 10 unresolved for the user to see
        for team, suggestions in unresolved[:15]:
            logger.info(f"  ? {team} -> Suggestions: {suggestions}")

if __name__ == "__main__":
    build_mapping()
