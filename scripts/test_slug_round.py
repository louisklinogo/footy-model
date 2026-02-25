import asyncio
import json
from sofascore_wrapper.api import SofascoreAPI

async def main():
    api = SofascoreAPI()
    try:
        # Testing CL 25/26 Playoff round slug
        # ID: 7, Season: 76953, Round: 636, Slug: playoff-round
        url = "/unique-tournament/7/season/76953/events/round/636/slug/playoff-round"
        print(f"Testing URL: {url}")
        data = await api._get(url)
        print("Success! Found events:")
        for event in data.get('events', []):
            print(f"  {event['homeTeam']['name']} vs {event['awayTeam']['name']} ({event['startTimestamp']})")
            
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(main())
