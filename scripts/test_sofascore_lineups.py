"""Deeper exploration of sofascore-wrapper - focus on lineups and injury data."""

import asyncio
import json
from datetime import datetime, timedelta

from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match
from sofascore_wrapper.team import Team


async def main():
    api = SofascoreAPI()

    print("=" * 70)
    print("SOFASCORE LINEUPS & INJURY EXPLORATION")
    print("=" * 70)

    try:
        match = Match(api)

        # 1. Find a RECENT finished match to get actual lineups
        print("\n[1] FINDING RECENT FINISHED MATCH")
        print("-" * 40)

        # Check yesterday for finished matches
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        fixtures = await match.games_by_date(sport="football", date=yesterday)
        events = fixtures.get("events", [])

        finished_match = None
        for event in events:
            status = event.get("status", {})
            if status.get("type") == "finished":
                tournament = event.get("tournament", {})
                ut = tournament.get("uniqueTournament", {})
                # Prefer PL or other major leagues
                if ut.get("name") in [
                    "Premier League",
                    "LaLiga",
                    "Bundesliga",
                    "Serie A",
                    "Ligue 1",
                ]:
                    finished_match = event
                    break

        if not finished_match:
            # Use any finished match
            for event in events:
                if event.get("status", {}).get("type") == "finished":
                    finished_match = event
                    break

        if finished_match:
            match_id = finished_match["id"]
            home_team = finished_match["homeTeam"]["name"]
            away_team = finished_match["awayTeam"]["name"]
            print(f"Found finished match: {home_team} vs {away_team}")
            print(f"  Match ID: {match_id}")
            print(
                f"  Score: {finished_match.get('homeScore', {}).get('current')} - {finished_match.get('awayScore', {}).get('current')}"
            )
        else:
            print("No finished matches found, using sample ID")
            match_id = 12436472  # Arsenal vs Liverpool

        match_with_id = Match(api, match_id=match_id)

        # 2. Get FULL lineups data
        print(f"\n[2] FULL LINEUPS DATA")
        print("-" * 40)
        try:
            lineups = await api._get(f"/event/{match_id}/lineups")
            print(f"Lineups keys: {list(lineups.keys())}")

            # Check for injury-related fields
            home = lineups.get("home", {})
            away = lineups.get("away", {})

            print(f"\nHome lineup keys: {list(home.keys())}")
            print(f"Away lineup keys: {list(away.keys())}")

            # Check all fields for injury data
            for side_name, side_data in [("HOME", home), ("AWAY", away)]:
                print(f"\n{side_name} TEAM:")

                # Players
                players = side_data.get("players", [])
                print(f"  Players: {len(players)}")
                if players:
                    print(f"  Sample player keys: {list(players[0].keys())}")
                    for p in players[:2]:
                        print(
                            f"    - {p.get('player', {}).get('name')} (#{p.get('shirtNumber')})"
                        )

                # Look for missing/injured
                for key in side_data.keys():
                    if any(
                        x in key.lower()
                        for x in [
                            "miss",
                            "injur",
                            "absent",
                            "unavailable",
                            "bench",
                            "substitute",
                        ]
                    ):
                        print(f"  FOUND KEY: {key}")
                        val = side_data[key]
                        if isinstance(val, list):
                            print(f"    Count: {len(val)}")
                            if val:
                                print(f"    Sample: {val[0]}")
                        else:
                            print(f"    Value: {val}")

                # Print ALL keys for inspection
                print(f"  ALL keys: {list(side_data.keys())}")

        except Exception as e:
            print(f"Lineups error: {e}")

        # 3. Check pre_match_form (might have availability info)
        print(f"\n[3] PRE-MATCH FORM")
        print("-" * 40)
        try:
            form = await match_with_id.pre_match_form()
            print(
                f"Form keys: {list(form.keys()) if isinstance(form, dict) else type(form)}"
            )
            if isinstance(form, dict):
                for k, v in form.items():
                    if isinstance(v, dict):
                        print(f"  {k}: {list(v.keys())[:10]}")
                    elif isinstance(v, list):
                        print(f"  {k}: list of {len(v)}")
                    else:
                        print(f"  {k}: {v}")
        except Exception as e:
            print(f"Pre-match form error: {e}")

        # 4. Check team near_events (might show injury status)
        home_team_id = finished_match["homeTeam"]["id"] if finished_match else 42
        team = Team(api, team_id=home_team_id)

        print(f"\n[4] TEAM NEAR EVENTS (Team ID: {home_team_id})")
        print("-" * 40)
        try:
            near_events = await team.near_events()
            print(
                f"Near events keys: {list(near_events.keys()) if isinstance(near_events, dict) else type(near_events)}"
            )
            if isinstance(near_events, dict):
                for k in near_events.keys():
                    if (
                        "injur" in k.lower()
                        or "miss" in k.lower()
                        or "avail" in k.lower()
                    ):
                        print(f"  FOUND: {k}")
        except Exception as e:
            print(f"Near events error: {e}")

        # 5. Try raw team squad with injury status
        print(f"\n[5] TEAM SQUAD (check for injury status)")
        print("-" * 40)
        try:
            squad = await api._get(f"/team/{home_team_id}/players")
            print(f"Squad keys: {list(squad.keys())}")

            players = squad.get("players", [])
            print(f"Total players: {len(players)}")

            # Check if any player has injury status
            for p in players:
                player_obj = p.get("player", {})
                # Check for injury-related fields
                for key in p.keys():
                    if any(
                        x in key.lower() for x in ["injur", "miss", "status", "avail"]
                    ):
                        print(f"  {player_obj.get('name')}: {key} = {p[key]}")

                # Also check player object
                for key in player_obj.keys():
                    if any(
                        x in key.lower() for x in ["injur", "miss", "status", "avail"]
                    ):
                        print(f"  {player_obj.get('name')}: {key} = {player_obj[key]}")
        except Exception as e:
            print(f"Squad error: {e}")

        # 6. Try direct API endpoints that might have injury data
        print(f"\n[6] EXPLORATORY ENDPOINTS")
        print("-" * 40)

        endpoints_to_try = [
            f"/event/{match_id}/lineups/incidents",
            f"/event/{match_id}/player-performance",
            f"/team/{home_team_id}/current-squad",
            f"/team/{home_team_id}/injured-players",
            f"/team/{home_team_id}/unavailable-players",
            f"/team/{home_team_id}/squad/confirmed",
        ]

        for endpoint in endpoints_to_try:
            try:
                result = await api._get(endpoint)
                print(f"  {endpoint}: SUCCESS!")
                if isinstance(result, dict):
                    print(f"    Keys: {list(result.keys())}")
            except Exception as e:
                err = str(e)[:60]
                print(f"  {endpoint}: {err}")

        # 7. Check what the confirmed lineups contain for TODAY's upcoming match
        print(f"\n[7] TODAY'S UPCOMING MATCH LINEUPS (check for predicted/confirmed)")
        print("-" * 40)
        today = datetime.now().strftime("%Y-%m-%d")
        today_fixtures = await match.games_by_date(sport="football", date=today)

        for event in today_fixtures.get("events", [])[:10]:
            status = event.get("status", {})
            if status.get("type") == "notstarted":
                upcoming_id = event["id"]
                upcoming_home = event["homeTeam"]["name"]
                upcoming_away = event["awayTeam"]["name"]

                print(f"Checking: {upcoming_home} vs {upcoming_away}")

                try:
                    lineups = await api._get(f"/event/{upcoming_id}/lineups")

                    confirmed = lineups.get("confirmed", False)
                    home = lineups.get("home", {})
                    away = lineups.get("away", {})

                    print(f"  Confirmed: {confirmed}")
                    print(f"  Home players: {len(home.get('players', []))}")
                    print(f"  Home keys: {list(home.keys())}")

                    # Check for missing players
                    if home.get("missingPlayers"):
                        print(f"  HOME MISSING: {home.get('missingPlayers')}")
                    if away.get("missingPlayers"):
                        print(f"  AWAY MISSING: {away.get('missingPlayers')}")

                    # Print full structure for first match
                    print(f"\n  FULL HOME STRUCTURE:")
                    for k, v in home.items():
                        if isinstance(v, list):
                            print(f"    {k}: {len(v)} items")
                        elif isinstance(v, dict):
                            print(f"    {k}: dict with keys {list(v.keys())[:5]}")
                        else:
                            print(f"    {k}: {v}")

                except Exception as e:
                    print(f"  Error: {e}")

                break  # Just check first upcoming match

    finally:
        await api.close()

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
