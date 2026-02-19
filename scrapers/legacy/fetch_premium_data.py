"""
Direct Feed Ingester - v2.
Reverse-engineered for specific market sub-feeds.
"""

import requests
import re
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://global.flashscore.ninja/2/x/feed/"
# This signature seems to work for now
FS_SIGN = "SW9D1eZo"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'x-fsign': FS_SIGN,
    'Origin': 'https://www.flashscore.com',
    'Referer': 'https://www.flashscore.com/',
    'X-Referer': 'https://www.flashscore.com/'
}

def parse_fs_stats(text):
    """
    Parses stats from the SE÷...SG÷...SH÷ format.
    Correct mapping: SE is the name, SG is home value, SH is away value.
    """
    stats = {'home': {}, 'away': {}}
    # Each stat block starts with SE÷
    blocks = text.split('SE÷')
    for block in blocks[1:]:
        lines = block.split('¬')
        name = lines[0]
        
        home_val = None
        away_val = None
        
        for line in lines:
            if line.startswith('SG÷'):
                home_val = line[3:]
            elif line.startswith('SH÷'):
                away_val = line[3:]
        
        if name and home_val is not None and away_val is not None:
            stats['home'][name] = home_val
            stats['away'][name] = away_val
            
    return stats

def parse_fs_odds(text, target_line="1.5"):
    """
    Parses odds from the OU feed.
    Look for MI÷[Line] and then the OD÷[Odds] following it.
    """
    # Sample format: ...MI÷1.5¬...OD÷1.26¬...
    # We look for the segment matching the target line
    pattern = rf"MI÷{target_line}¬.*?OD÷([^¬÷]*)"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None

def fetch_match_details(match_id):
    data = {'id': match_id, 'stats': {}, 'odds': {}}
    
    # 1. Stats Feed
    r_stats = requests.get(f"{BASE_URL}df_st_1_{match_id}", headers=HEADERS)
    if r_stats.status_code == 200:
        data['stats'] = parse_fs_stats(r_stats.text)
        
    # 2. Over/Under Odds Feed (Market Code 2)
    r_odds_goals = requests.get(f"{BASE_URL}df_od_1_{match_id}_2", headers=HEADERS)
    if r_odds_goals.status_code == 200:
        data['odds']['o15'] = parse_fs_odds(r_odds_goals.text, "1.5")
        data['odds']['o25'] = parse_fs_odds(r_odds_goals.text, "2.5")
        
    # 3. Corners Odds Feed (Market Code 13)
    r_odds_corners = requests.get(f"{BASE_URL}df_od_1_{match_id}_13", headers=HEADERS)
    if r_odds_corners.status_code == 200:
        data['odds']['c85'] = parse_fs_odds(r_odds_corners.text, "8.5")
        
    return data

if __name__ == "__main__":
    # Test Match: Brentford vs Arsenal (lKNJm8ak)
    match_id = "lKNJm8ak"
    result = fetch_match_details(match_id)
    print(json.dumps(result, indent=4))
