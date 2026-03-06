import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "sofascore-wrapper"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from src.db.db_utils import connect_db
from src.ingest.backfill_sofascore_odds_markets_v1 import extract_markets
from sofascore_wrapper.api import SofascoreAPI


def fetch_examples() -> dict[str, dict[str, object] | None]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fixture_id, sofascore_id, league_code, match_datetime_utc
                FROM fixtures
                WHERE status = 'ft' AND sofascore_id IS NOT NULL
                ORDER BY match_datetime_utc DESC
                LIMIT 1
                """
            )
            ft = cur.fetchone()
            cur.execute(
                """
                SELECT fixture_id, sofascore_id, league_code, match_datetime_utc
                FROM fixtures
                WHERE status = 'scheduled' AND sofascore_id IS NOT NULL
                ORDER BY match_datetime_utc ASC
                LIMIT 1
                """
            )
            sched = cur.fetchone()
        return {
            "finished": None if not ft else {"fixture_id": ft[0], "sofascore_id": ft[1], "league_code": ft[2], "match_datetime_utc": str(ft[3])},
            "scheduled": None if not sched else {"fixture_id": sched[0], "sofascore_id": sched[1], "league_code": sched[2], "match_datetime_utc": str(sched[3])},
        }
    finally:
        conn.close()


async def main() -> None:
    examples = fetch_examples()
    out: dict[str, object] = {"examples": examples}
    api = SofascoreAPI()
    try:
        finished = examples.get("finished")
        if finished:
            ss = finished["sofascore_id"]
            stats = await api._get(f"/event/{ss}/statistics")
            incidents = await api._get(f"/event/{ss}/incidents")
            lineups = await api._get(f"/event/{ss}/lineups")
            out["finished_probe"] = {
                "statistics_has_root": isinstance(stats, dict) and "statistics" in stats,
                "statistics_periods": [p.get("period") for p in (stats.get("statistics") or [])[:5]] if isinstance(stats, dict) else [],
                "incidents_count": len((incidents or {}).get("incidents", [])) if isinstance(incidents, dict) else None,
                "lineups_home_keys": sorted(list((lineups.get("home") or {}).keys()))[:20] if isinstance(lineups, dict) else [],
                "lineups_home_missing_players": len((lineups.get("home") or {}).get("missingPlayers", [])) if isinstance(lineups, dict) else None,
                "lineups_away_missing_players": len((lineups.get("away") or {}).get("missingPlayers", [])) if isinstance(lineups, dict) else None,
            }
        scheduled = examples.get("scheduled")
        if scheduled:
            ss = scheduled["sofascore_id"]
            odds = await api._get(f"/event/{ss}/odds/1/all")
            mapped = extract_markets(odds or {}, 1)
            sched_probe = {
                "odds_root_keys": sorted(list((odds or {}).keys()))[:20] if isinstance(odds, dict) else [],
                "raw_market_count": len((odds or {}).get("markets", [])) if isinstance(odds, dict) else None,
                "mapped_market_count": len(mapped),
                "mapped_market_codes": sorted({m["market_code"] for m in mapped}),
                "mapped_corner_lines": sorted({str(m["line_num"]) for m in mapped if m["market_code"] == "corners_ou" and m["line_num"] is not None}),
            }
            try:
                lineups = await api._get(f"/event/{ss}/lineups")
                sched_probe["lineups_home_missing_players"] = len((lineups.get("home") or {}).get("missingPlayers", [])) if isinstance(lineups, dict) else None
                sched_probe["lineups_away_missing_players"] = len((lineups.get("away") or {}).get("missingPlayers", [])) if isinstance(lineups, dict) else None
            except Exception as exc:
                sched_probe["lineups_error"] = str(exc)
            out["scheduled_probe"] = sched_probe
    finally:
        await api.close()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())

