import sys
import asyncio
from pathlib import Path
import json

# Add root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Add vendor to sys.path
VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match

async def test_lineups(sofascore_id):
    api = SofascoreAPI()
    try:
        match = Match(api, sofascore_id)
        
        print(f"--- Lineups for {sofascore_id} ---")
        lineups_home = await match.lineups_home()
        
        # Save a sample to inspect
        with open("player_lineup_sample.json", "w") as f:
            json.dump(lineups_home, f, indent=2)
        
        print("Saved sample to player_lineup_sample.json")
        
        if "players" in lineups_home:
            p = lineups_home["players"][0]
            print(f"Sample Player: {p.get('player', {}).get('name')}")
            stats = p.get('statistics', {})
            print(f"Stats Keys: {list(stats.keys())}")
            # Map key stats to check coverage
            print(f"Minutes: {stats.get('minutesPlayed')}")
            print(f"Rating: {stats.get('rating')}")
            print(f"Goals: {stats.get('goals')}")
            
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(test_lineups(14317553))
