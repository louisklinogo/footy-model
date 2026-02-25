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

async def debug_stats(sofa_id):
    api = SofascoreAPI()
    try:
        data = await api._get(f"/event/{sofa_id}/statistics")
        if data and 'statistics' in data:
            print(f"Stats found for {sofa_id}:")
            periods = [p['period'] for p in data['statistics']]
            print(f"  Periods available: {periods}")
            
            for p in data['statistics']:
                if p['period'] == 'ALL':
                    groups = [g['groupName'] for g in p['groups']]
                    print(f"  'ALL' Groups: {groups}")
        else:
            print(f"No stats found for {sofa_id}")
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(debug_stats(14391225))
