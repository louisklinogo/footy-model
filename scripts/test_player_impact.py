"""Test player valuation and team performance metrics."""

import asyncio
import json
from datetime import datetime

from sofascore_wrapper.api import SofascoreAPI


async def main():
    api = SofascoreAPI()

    print("=" * 70)
    print("PLAYER VALUATION & TEAM PERFORMANCE METRICS")
    print("=" * 70)

    try:
        # Arsenal team ID = 42
        team_id = 42

        # 1. Team Performance Graph
        print("\n[1] TEAM PERFORMANCE GRAPH")
        print("-" * 50)
        try:
            perf = await api._get(f"/team/{team_id}/performance-graph")
            # This might show rating over time
            print(
                f"Keys: {list(perf.keys()) if isinstance(perf, dict) else type(perf)}"
            )
            if isinstance(perf, dict):
                for k, v in list(perf.items())[:5]:
                    if isinstance(v, list):
                        print(f"  {k}: {len(v)} items")
                    else:
                        print(f"  {k}: {v}")
        except Exception as e:
            print(f"  Error: {e}")

        # 2. Team Statistics (season)
        print("\n[2] TEAM SEASON STATISTICS")
        print("-" * 50)
        try:
            # Try getting team stats for Premier League (tournament 17)
            team_stats = await api._get(
                f"/team/{team_id}/statistics/17/61627"
            )  # PL, current season
            print(
                f"Keys: {list(team_stats.keys()) if isinstance(team_stats, dict) else type(team_stats)}"
            )

            if isinstance(team_stats, dict):
                stats = team_stats.get("statistics", {})
                for group_name, items in list(stats.items())[:5]:
                    print(f"\n  {group_name}:")
                    if isinstance(items, list):
                        for item in items[:5]:
                            print(
                                f"    {item.get('name', '?')}: {item.get('value', '?')}"
                            )
        except Exception as e:
            print(f"  Error: {e}")

        # 3. Team Top Players
        print("\n[3] TEAM TOP PLAYERS (season)")
        print("-" * 50)
        try:
            top = await api._get(f"/team/{team_id}/top-players/17/61627")
            print(f"Keys: {list(top.keys()) if isinstance(top, dict) else type(top)}")

            if isinstance(top, dict):
                for category, players in list(top.items())[:5]:
                    print(f"\n  {category}:")
                    if isinstance(players, list):
                        for p in players[:3]:
                            player = p.get("player", {})
                            print(f"    - {player.get('name')}: {p.get('value', '?')}")
        except Exception as e:
            print(f"  Error: {e}")

        # 4. Player Details with Market Value
        print("\n[4] PLAYER DETAILS & MARKET VALUE")
        print("-" * 50)
        try:
            # Bukayo Saka ID = 934235
            player_id = 934235
            player = await api._get(f"/player/{player_id}")

            print(f"Keys: {list(player.keys())}")

            player_data = player.get("player", {})

            print(f"\n  Name: {player_data.get('name')}")
            print(f"  Position: {player_data.get('position')}")
            print(f"  Team: {player_data.get('team', {}).get('name')}")

            # Check for market value
            if "proposedMarketValueRaw" in player_data:
                mv = player_data["proposedMarketValueRaw"]
                print(f"  Market Value: {mv.get('value')} {mv.get('currency')}")

            if "marketValueCurrency" in player_data:
                print(f"  Market Value Currency: {player_data['marketValueCurrency']}")

            # Print all keys to find value fields
            print(f"\n  All player keys: {list(player_data.keys())}")

        except Exception as e:
            print(f"  Error: {e}")

        # 5. Player Statistics (season)
        print("\n[5] PLAYER SEASON STATISTICS")
        print("-" * 50)
        try:
            # Saka season stats
            player_stats = await api._get(f"/player/{player_id}/statistics/17/61627")

            print(f"Keys: {list(player_stats.keys())}")

            stats = player_stats.get("statistics", {})
            if stats:
                print(f"\n  Goals: {stats.get('goals', '?')}")
                print(f"  Assists: {stats.get('assists', '?')}")
                print(f"  Rating: {stats.get('rating', '?')}")
                print(f"  Appearances: {stats.get('appearances', '?')}")
                print(f"  Minutes: {stats.get('minutesPlayed', '?')}")

                print(f"\n  All stat keys: {list(stats.keys())}")
        except Exception as e:
            print(f"  Error: {e}")

        # 6. Player Ratings History
        print("\n[6] PLAYER RATINGS (recent matches)")
        print("-" * 50)
        try:
            ratings = await api._get(f"/player/{player_id}/rating/17/61627")
            print(
                f"Keys: {list(ratings.keys()) if isinstance(ratings, dict) else type(ratings)}"
            )

            if isinstance(ratings, dict) and "ratings" in ratings:
                for r in ratings["ratings"][:5]:
                    match = r.get("event", {})
                    print(
                        f"  vs {match.get('awayTeam', {}).get('name') if r.get('isHome') else match.get('homeTeam', {}).get('name')}: {r.get('rating')}"
                    )
        except Exception as e:
            print(f"  Error: {e}")

        # 7. League Standings
        print("\n[7] LEAGUE STANDINGS (team position)")
        print("-" * 50)
        try:
            standings = await api._get(
                f"/unique-tournament/17/season/61627/standings/total"
            )

            print(f"Keys: {list(standings.keys())}")

            rows = standings.get("standings", [{}])[0].get("rows", [])

            for row in rows:
                team = row.get("team", {})
                if team.get("id") == team_id:
                    print(f"\n  Arsenal Position: {row.get('position')}")
                    print(f"  Points: {row.get('points')}")
                    print(f"  Played: {row.get('played')}")
                    print(f"  Wins: {row.get('wins')}")
                    print(f"  Draws: {row.get('draws')}")
                    print(f"  Losses: {row.get('losses')}")
                    print(f"  Goals For: {row.get('scoresFor')}")
                    print(f"  Goals Against: {row.get('scoresAgainst')}")
                    break

        except Exception as e:
            print(f"  Error: {e}")

        # 8. Check for "player importance" type data
        print("\n[8] TEAM SQUAD WITH DETAILED INFO")
        print("-" * 50)
        try:
            squad = await api._get(f"/team/{team_id}/players")

            players = squad.get("players", [])
            print(f"Total players: {len(players)}")

            # Show top players by position with market value
            print("\n  Key players with market values:")
            for p in players[:10]:
                player = p.get("player", {})
                name = player.get("name", "?")
                position = player.get("position", "?")
                mv = player.get("proposedMarketValueRaw", {})
                mv_value = mv.get("value", "?")
                mv_currency = mv.get("currency", "EUR")

                # Format market value
                if isinstance(mv_value, (int, float)) and mv_value > 0:
                    mv_str = (
                        f"EUR {mv_value / 1000000:.1f}M"
                        if mv_value > 1000000
                        else f"EUR {mv_value / 1000:.0f}K"
                    )
                else:
                    mv_str = "N/A"

                print(f"    {name} ({position}): {mv_str}")

        except Exception as e:
            print(f"  Error: {e}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: PLAYER IMPACT DATA AVAILABLE")
        print("=" * 70)

        print("""
KEY DATA FOR ASSESSING MISSING PLAYER IMPACT:

1. MARKET VALUE
   - Endpoint: /player/{id}
   - Field: proposedMarketValueRaw (value + currency)
   - Use: Higher value = more important player

2. PLAYER RATINGS
   - Endpoint: /player/{id}/statistics/{tournament}/{season}
   - Fields: rating, goals, assists, minutesPlayed
   - Use: High rating + high minutes = key player

3. PLAYER SEASON STATS
   - Goals, assists, appearances, minutes
   - Can calculate: goals/90, assists/90

4. TEAM PERFORMANCE
   - Standings, points, goals for/against
   - Form (recent results)

5. TEAM TOP PLAYERS
   - Endpoint: /team/{id}/top-players/{tournament}/{season}
   - Rankings: top scorers, assisters, etc.

CALCULATED IMPACT APPROACH:
Instead of AI research, we could:
1. Get missing player's market value (proxy for importance)
2. Get their season stats (goals, assists, rating)
3. Get their minutes played (starter vs bench)
4. Calculate "impact score" = market_value * avg_rating * minutes_factor

LIMITATION: SofaScore doesn't provide:
- "Team record with/without player" (would need to calculate)
- "Player is captain/vice-captain" (leadership impact)
- Pre-calculated "importance score"
""")

    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
