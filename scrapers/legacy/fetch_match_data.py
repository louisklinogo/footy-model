"""
Prototype for direct Flashscore Feed Ingestion.
Targeting O1.5 Odds and Advanced Stats (xG, Crosses).
"""

import requests
import re
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'x-fsign': 'SW9D1eZo',
    'Origin': 'https://www.flashscore.com',
    'Referer': 'https://www.flashscore.com/',
    'X-Referer': 'https://www.flashscore.com/'
}

def get_match_stats(match_id):
    """Fetch advanced stats from direct stats feed."""
    url = f"https://global.flashscore.ninja/2/x/feed/df_st_1_{match_id}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            text = response.text
            stats = {'home': {}, 'away': {}}
            
            # Use regex to find all stat rows
            # Format: SE÷StatName¬SG÷HomeVal¬SH÷AwayVal¬
            # We need to handle the encoding. The character in logs was 
            # Let's try splitting by the 'SE÷' string and cleaning delimiters.
            parts = text.split('SE÷')
            for part in parts[1:]:
                # Extract name (until next delimiter)
                name = re.split(r'[¬÷]', part)[0]
                
                # Extract values using specific markers SG (Home) and SH (Away)
                home_val = re.search(r"SG÷([^¬÷]*)", part)
                away_val = re.search(r"SH÷([^¬÷]*)", part)
                
                if home_val and away_val:
                    stats['home'][name] = home_val.group(1)
                    stats['away'][name] = away_val.group(1)
            return stats
    except Exception as e:
        logger.error(f"Stats fetch failed: {e}")
    return None

def test_enrichment(match_id):
    logger.info(f"Testing Enrichment for ID: {match_id}")
    stats = get_match_stats(match_id)
    if stats and stats['home']:
        logger.info("✅ Stats Found:")
        for k, v in stats['home'].items():
            logger.info(f"  {k}: {v} vs {stats['away'].get(k)}")
    else:
        logger.warning("❌ No stats found in response")

if __name__ == "__main__":
    # Test with Brentford vs Arsenal (ID: lKNJm8ak)
    test_enrichment('lKNJm8ak')
