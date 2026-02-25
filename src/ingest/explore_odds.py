import sys
import asyncio
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from sofascore_wrapper.api import SofascoreAPI

async def explore(match_id):
    api = SofascoreAPI()
    try:
        endpoints = [
            f"/event/{match_id}/odds/1/all",
            f"/event/{match_id}/odds/1/1",
            f"/event/{match_id}/odds/providers"
        ]
        
        for ep in endpoints:
            print(f"\n--- Testing: {ep} ---")
            try:
                data = await api._get(ep)
                if data and "markets" in data:
                    markets = data["markets"]
                    market_names = [m.get("marketName") for m in markets]
                    print(f"Total Markets Found: {len(markets)}")
                    print(f"Markets list (first 15): {market_names[:15]}")
                    
                    # Print full 1X2 data
                    for m in markets:
                        if m.get("marketName") == "Full time":
                            print("\n[Full Time 1X2 Market Data]")
                            print(json.dumps(m, indent=2))
                            break
                else:
                    print("Status: 200 OK, but no data.")
            except Exception as e:
                print(f"FAILED: {e}")
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(explore(14025212))
