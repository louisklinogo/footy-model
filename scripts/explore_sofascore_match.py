import asyncio
import json
from sofascore_wrapper.api import SofascoreAPI

async def main():
    api = SofascoreAPI()
    try:
        # Fetching a finished high-profile match to see full stats (e.g., PL match)
        # Using a recent match ID (e.g., from the previous test 12437038)
        match_id = 12437038 # Tottenham vs West Ham
        print(f"Fetching details for match ID: {match_id}...")
        
        # Main match info
        match_info = await api._get(f"/event/{match_id}")
        
        # Statistics
        stats = await api._get(f"/event/{match_id}/statistics")
        
        # Lineups
        lineups = await api._get(f"/event/{match_id}/lineups")
        
        print("\n--- Match Info Keys ---")
        print(list(match_info.keys()))
        
        print("\n--- Statistics Structure ---")
        # Sofascore stats are usually per-period or overall
        if stats.get('statistics'):
            for s in stats['statistics']:
                print(f"Period: {s['period']}")
                for group in s.get('groups', []):
                    print(f"  Group: {group['groupName']}")
                    for item in group.get('statisticsItems', []):
                        print(f"    - {item['name']}: {item['homeValue']} / {item['awayValue']}")
        
        # Save to a file for deeper inspection if needed
        full_sample = {
            "info": match_info,
            "stats": stats,
            "lineups": lineups
        }
        with open("data/sofascore_match_sample.json", "w") as f:
            json.dump(full_sample, f, indent=2)
        print("\nFull sample saved to data/sofascore_match_sample.json")
            
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(main())
