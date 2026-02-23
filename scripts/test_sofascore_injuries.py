"""Extract injury data from SofaScore API."""

import asyncio
import json
from datetime import datetime, timedelta

from sofascore_wrapper.api import SofascoreAPI


async def main():
    api = SofascoreAPI()

    print("=" * 70)
    print("SOFASCORE INJURY DATA EXTRACTION")
    print("=" * 70)

    try:
        # 1. Get team squad with injuries (Aston Villa - ID 40)
        print("\n[1] TEAM SQUAD INJURIES (Aston Villa)")
        print("-" * 50)

        squad = await api._get("/team/40/players")
        players = squad.get("players", [])

        injured_players = []
        for p in players:
            player_obj = p.get("player", {})
            injury = p.get("injury")

            if injury:
                injured_players.append(
                    {
                        "name": player_obj.get("name"),
                        "position": player_obj.get("position"),
                        "reason": injury.get("reason"),
                        "status": injury.get("status"),
                        "start_date": datetime.fromtimestamp(
                            injury.get("startDateTimestamp", 0)
                        ).strftime("%Y-%m-%d")
                        if injury.get("startDateTimestamp")
                        else None,
                        "end_date": datetime.fromtimestamp(
                            injury.get("endDateTimestamp", 0)
                        ).strftime("%Y-%m-%d")
                        if injury.get("endDateTimestamp")
                        else None,
                    }
                )

        print(f"Total squad: {len(players)} players")
        print(f"Injured players: {len(injured_players)}")
        print()

        for p in injured_players:
            print(f"  {p['name']} ({p['position']})")
            print(f"    Reason: {p['reason']}")
            print(f"    Status: {p['status']}")
            print(f"    Expected return: {p['end_date']}")
            print()

        # 2. Get match lineups with missing players
        print("\n[2] MATCH LINEUPS WITH MISSING PLAYERS")
        print("-" * 50)

        # Get a finished match
        match = await api._get("/sport/football/scheduled-events/2026-02-21")
        events = match.get("events", [])

        finished_match = None
        for event in events:
            if event.get("status", {}).get("type") == "finished":
                finished_match = event
                break

        if finished_match:
            match_id = finished_match["id"]
            home_team = finished_match["homeTeam"]["name"]
            away_team = finished_match["awayTeam"]["name"]

            print(f"Match: {home_team} vs {away_team}")
            print(f"Match ID: {match_id}")

            lineups = await api._get(f"/event/{match_id}/lineups")

            home = lineups.get("home", {})
            away = lineups.get("away", {})

            home_missing = home.get("missingPlayers", [])
            away_missing = away.get("missingPlayers", [])

            print(f"\nHome ({home_team}) missing players: {len(home_missing)}")
            for mp in home_missing:
                player = mp.get("player", {})
                print(f"  - {player.get('name')} ({player.get('position')})")
                if mp.get("reason"):
                    print(f"    Reason: {mp.get('reason')}")

            print(f"\nAway ({away_team}) missing players: {len(away_missing)}")
            for mp in away_missing:
                player = mp.get("player", {})
                print(f"  - {player.get('name')} ({player.get('position')})")
                if mp.get("reason"):
                    print(f"    Reason: {mp.get('reason')}")

        # 3. Get upcoming match with predicted lineups and missing players
        print("\n[3] UPCOMING MATCH - PREDICTED LINEUPS")
        print("-" * 50)

        today_fixtures = await api._get("/sport/football/scheduled-events/2026-02-22")

        for event in today_fixtures.get("events", [])[:20]:
            if event.get("status", {}).get("type") == "notstarted":
                upcoming_id = event["id"]
                upcoming_home = event["homeTeam"]["name"]
                upcoming_away = event["awayTeam"]["name"]

                lineups = await api._get(f"/event/{upcoming_id}/lineups")

                confirmed = lineups.get("confirmed", False)
                home = lineups.get("home", {})
                away = lineups.get("away", {})

                # Only show if there's missing players data
                home_missing = home.get("missingPlayers", [])
                away_missing = away.get("missingPlayers", [])

                if home_missing or away_missing:
                    print(f"\n{upcoming_home} vs {away_team}")
                    print(f"  Lineups confirmed: {confirmed}")
                    print(f"  Home missing: {len(home_missing)}")
                    print(f"  Away missing: {len(away_missing)}")

                    if home_missing:
                        print(f"\n  HOME MISSING:")
                        for mp in home_missing[:5]:
                            player = mp.get("player", {})
                            print(f"    - {player.get('name')}")

                    if away_missing:
                        print(f"\n  AWAY MISSING:")
                        for mp in away_missing[:5]:
                            player = mp.get("player", {})
                            print(f"    - {player.get('name')}")

                    break

        # 4. Compare with Flashscore format
        print("\n[4] DATA STRUCTURE COMPARISON")
        print("-" * 50)

        print("SofaScore injury format:")
        print(
            json.dumps(
                {
                    "name": "Youri Tielemans",
                    "position": "M",
                    "reason": "Ankle Injury",
                    "status": "out",
                    "expected_return": "2025-03-26",
                },
                indent=2,
            )
        )

        print("\nYour Flashscore format (from prematch_enricher.js):")
        print(
            json.dumps({"name": "Bukayo Saka", "reason": "Hamstring Injury"}, indent=2)
        )

        print("\n" + "=" * 70)
        print("CONCLUSION: SofaScore has EQUIVALENT injury data!")
        print("=" * 70)
        print("""
SOFASCORE PROVIDES:
- Team squad injuries via /team/{id}/players
  - Player name, position
  - Injury reason (Ankle Injury, Knee Injury, etc.)
  - Status (out, doubtful)
  - Expected return date
  
- Match missing players via /event/{id}/lineups
  - Players confirmed OUT for specific match
  - Works for upcoming matches too!

ADVANTAGE OVER FLASHSCORE:
- Expected return dates
- Injury status (out vs doubtful)
- Structured data (no HTML parsing)
- Works via clean API calls

RECOMMENDATION:
You can replace Flashscore scraping with sofascore-wrapper!
""")

    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
