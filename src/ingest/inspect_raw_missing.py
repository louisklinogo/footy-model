import sys
import asyncio
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Add vendor to sys.path
VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from sofascore_wrapper.api import SofascoreAPI

async def test_multiple_fixtures():
    api = SofascoreAPI()
    # 14025212: Man City vs Fulham
    # 14023969: West Ham vs Palace 
    # 13146479: Random older one
    ids = [14024007, 14025212, 14023969]
    try:
        for ss_id in ids:
            print(f"\n--- Fixture {ss_id} ---")
            url = f"/event/{ss_id}/lineups"
            try:
                data = await api._get(url)
            except Exception as e:
                print(f"Error fetching {ss_id}: {e}")
                continue

            for side in ["home", "away"]:
                missing = data.get(side, {}).get("missingPlayers", [])
                if not missing:
                    continue
                print(f"  {side.upper()} Missing Players:")
                for m in missing:
                    name = m.get("player", {}).get("name")
                    m_type = m.get("type", "N/A")
                    m_reason = m.get("reason", "N/A")
                    m_desc = m.get("description", "N/A")
                    m_class = m.get("playerClass", "N/A")
                    m_return = m.get("expectedEndDate", "N/A")
                    
                    print(f"    - {str(name):20} | Type: {m_type:8} | Reason: {m_reason} | Desc: {m_desc:15} | Return: {m_return}")
                    
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(test_multiple_fixtures())
