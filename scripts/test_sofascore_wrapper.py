"""Test script to explore sofascore-wrapper capabilities.

Tests:
1. Fixtures by date (today's matches)
2. Match odds
3. Team squad/players
4. Lineups for a match
5. Search functionality
6. Try to find injury/availability data
"""

import asyncio
import json
from datetime import datetime

from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match
from sofascore_wrapper.team import Team
from sofascore_wrapper.search import Search
from sofascore_wrapper.player import Player


async def main():
    api = SofascoreAPI()

    print("=" * 70)
    print("SOFASCORE WRAPPER TEST")
    print("=" * 70)

    try:
        # 1. Get today's fixtures
        print("\n[1] TODAY'S FIXTURES")
        print("-" * 40)
        match = Match(api)
        today = datetime.now().strftime("%Y-%m-%d")

        fixtures = await match.games_by_date(sport="football", date=today)
        events = fixtures.get("events", [])
        print(f"Found {len(events)} fixtures for {today}")

        # Pick a Premier League match if available
        pl_match = None
        for event in events[:20]:
            tournament = event.get("tournament", {})
            ut = tournament.get("uniqueTournament", {})
            if ut.get("name") == "Premier League":
                pl_match = event
                break

        if pl_match:
            print(
                f"\nFound PL match: {pl_match['homeTeam']['name']} vs {pl_match['awayTeam']['name']}"
            )
            match_id = pl_match["id"]
            home_team_id = pl_match["homeTeam"]["id"]
            away_team_id = pl_match["awayTeam"]["id"]
            print(f"  Match ID: {match_id}")
            print(f"  Home Team ID: {home_team_id}")
            print(f"  Away Team ID: {away_team_id}")
        else:
            # Use first available match
            if events:
                pl_match = events[0]
                match_id = pl_match["id"]
                home_team_id = pl_match["homeTeam"]["id"]
                away_team_id = pl_match["awayTeam"]["id"]
                print(
                    f"\nUsing match: {pl_match['homeTeam']['name']} vs {pl_match['awayTeam']['name']}"
                )
            else:
                print("No matches found today, using hardcoded IDs")
                match_id = 12436472  # Arsenal vs Liverpool example
                home_team_id = 42  # Arsenal
                away_team_id = 44  # Liverpool

        # 2. Get match odds
        print(f"\n[2] MATCH ODDS (match_id: {match_id})")
        print("-" * 40)
        match_with_id = Match(api, match_id=match_id)

        try:
            odds = await match_with_id.match_odds()
            markets = odds.get("markets", [])
            print(f"Found {len(markets)} betting markets")
            for m in markets[:3]:
                print(f"  - {m.get('marketName')}: {len(m.get('choices', []))} choices")
                for c in m.get("choices", [])[:3]:
                    print(f"      {c.get('name')}: {c.get('fractionalValue')}")
        except Exception as e:
            print(f"Odds error: {e}")

        # 3. Get featured odds (includes Asian handicap, O/U)
        print(f"\n[3] FEATURED ODDS")
        print("-" * 40)
        try:
            featured = await match_with_id.featured_odds()
            featured_markets = featured.get("featured", {})
            for market_type, data in featured_markets.items():
                print(f"  {market_type}: {data.get('marketName', 'N/A')}")
        except Exception as e:
            print(f"Featured odds error: {e}")

        # 4. Get team squad (Arsenal)
        print(f"\n[4] TEAM SQUAD (Arsenal - ID: {home_team_id})")
        print("-" * 40)
        team = Team(api, team_id=home_team_id)

        try:
            squad = await team.squad()
            players = squad.get("players", [])
            print(f"Found {len(players)} players in squad")
            for p in players[:5]:
                print(
                    f"  - {p.get('player', {}).get('name')} ({p.get('player', {}).get('position')}) - #{p.get('player', {}).get('jerseyNumber')}"
                )
        except Exception as e:
            print(f"Squad error: {e}")

        # 5. Get lineups
        print(f"\n[5] MATCH LINEUPS (match_id: {match_id})")
        print("-" * 40)
        try:
            lineups_home = await match_with_id.lineups_home()
            lineups_away = await match_with_id.lineups_away()
            print(f"Home lineup: {len(lineups_home.get('players', []))} players")
            print(f"Away lineup: {len(lineups_away.get('players', []))} players")

            # Check for injury data in lineup response
            if "missing" in lineups_home or "injured" in lineups_home:
                print(f"  Found injury data in home lineup!")
            if "missing" in lineups_away or "injured" in lineups_away:
                print(f"  Found injury data in away lineup!")
        except Exception as e:
            print(f"Lineups error: {e}")

        # 6. Search for a player
        print(f"\n[6] PLAYER SEARCH (Saka)")
        print("-" * 40)
        search = Search(api, search_string="saka")
        try:
            results = await search.search_all()
            for r in results.get("results", [])[:3]:
                entity = r.get("entity", {})
                print(
                    f"  - {entity.get('name')} ({entity.get('position')}) - {entity.get('team', {}).get('name')}"
                )
        except Exception as e:
            print(f"Search error: {e}")

        # 7. Check available methods on Match class
        print(f"\n[7] AVAILABLE METHODS")
        print("-" * 40)
        print("Match methods:")
        for method in dir(match_with_id):
            if not method.startswith("_"):
                print(f"  - {method}")

        print("\nTeam methods:")
        for method in dir(team):
            if not method.startswith("_"):
                print(f"  - {method}")

        # 8. Try raw API call for injuries (exploratory)
        print(f"\n[8] EXPLORATORY: Try injury endpoints")
        print("-" * 40)
        try:
            # Try common injury endpoint patterns
            endpoints_to_try = [
                f"/event/{match_id}/injuries",
                f"/event/{match_id}/missing-players",
                f"/event/{match_id}/lineups",  # Full lineups might have injury info
                f"/team/{home_team_id}/injuries",
                f"/team/{home_team_id}/missing-players",
            ]

            for endpoint in endpoints_to_try:
                try:
                    result = await api._get(endpoint)
                    print(f"  {endpoint}: SUCCESS!")
                    if isinstance(result, dict):
                        print(f"    Keys: {list(result.keys())[:10]}")
                    break
                except Exception as e:
                    print(f"  {endpoint}: {str(e)[:50]}")
        except Exception as e:
            print(f"Exploratory error: {e}")

        # 9. Get full match info
        print(f"\n[9] FULL MATCH INFO")
        print("-" * 40)
        try:
            match_info = await match_with_id.get_match()
            print(f"Match keys: {list(match_info.keys())}")
            event = match_info.get("event", {})
            print(f"Event keys: {list(event.keys())}")
        except Exception as e:
            print(f"Match info error: {e}")

        # 10. H2H data
        print(f"\n[10] HEAD-TO-HEAD")
        print("-" * 40)
        try:
            h2h = await match_with_id.h2h()
            print(f"H2H: {json.dumps(h2h, indent=2)[:500]}")
        except Exception as e:
            print(f"H2H error: {e}")

    finally:
        await api.close()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
