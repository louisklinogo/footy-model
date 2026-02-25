import asyncio
import json
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.search import Search
from sofascore_wrapper.league import League

async def main():
    api = SofascoreAPI()
    try:
        search = Search(api, "Europa League")
        results = await search.search_leagues(sport="football")
        print("Search results for 'Europa League':")
        for item in results.get("results", []):
            if item.get("type") == "uniqueTournament":
                entity = item.get("entity", {})
                print(f"  Name: {entity.get('name')} (ID: {entity.get('id')})")
        
        # Also check CL and ECL historical seasons
        cups = {"CL": 7, "ECL": 17015}
        for name, cup_id in cups.items():
            print(f"\nChecking historical seasons for {name} (ID: {cup_id}):")
            league = League(api, cup_id)
            seasons = await league.get_seasons()
            for s in seasons[:3]: # Show last 3 seasons
                print(f"  Season: {s['name']} (ID: {s['id']})")
                
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(main())
