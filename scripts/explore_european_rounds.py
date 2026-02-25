import asyncio
import json
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.league import League

async def main():
    api = SofascoreAPI()
    try:
        # CL 25/26 Season ID: 76953
        league = League(api, 7)
        rounds_data = await league.rounds(76953)
        print("--- CL 25/26 Rounds ---")
        print(json.dumps(rounds_data, indent=2))
        
        # EL 25/26 Season ID: 76984
        league_el = League(api, 679)
        rounds_el = await league_el.rounds(76984)
        print("\n--- EL 25/26 Rounds ---")
        print(json.dumps(rounds_el, indent=2))
        
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(main())
