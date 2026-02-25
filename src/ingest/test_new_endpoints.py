import sys
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Add vendor to sys.path
VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from sofascore_wrapper.api import SofascoreAPI

async def test_endpoints(match_id, team_id):
    api = SofascoreAPI()
    try:
        endpoints = [
            f"/event/{match_id}/missing-players",
            f"/event/{match_id}/injuries",
            f"/team/{team_id}/missing-players",
            f"/team/{team_id}/injuries"
        ]
        
        for ep in endpoints:
            print(f"\nTesting endpoint: {ep}")
            try:
                data = await api._get(ep)
                if data:
                    print(f"  SUCCESS! Data found: {str(data)[:200]}...")
                else:
                    print("  Success, but no data.")
            except Exception as e:
                print(f"  FAILED: {e}")
                
    finally:
        await api.close()

if __name__ == "__main__":
    # Man City vs Fulham (match 14025212, City team 17)
    asyncio.run(test_endpoints(14025212, 17))
