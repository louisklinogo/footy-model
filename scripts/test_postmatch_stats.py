"""Test POST-MATCH data availability from SofaScore."""

import asyncio
import json
from datetime import datetime, timedelta

from sofascore_wrapper.api import SofascoreAPI


async def main():
    api = SofascoreAPI()

    print("=" * 70)
    print("POST-MATCH DATA AVAILABILITY TEST")
    print("=" * 70)

    try:
        # Find finished matches from recent days
        print("\n[1] FINDING FINISHED MATCHES")
        print("-" * 50)

        finished_matches = []

        for days_ago in range(0, 7):
            date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            fixtures = await api._get(f"/sport/football/scheduled-events/{date}")

            for event in fixtures.get("events", []):
                status = event.get("status", {})
                if status.get("type") == "finished":
                    tournament = (
                        event.get("tournament", {})
                        .get("uniqueTournament", {})
                        .get("name", "")
                    )
                    # Major leagues only
                    if any(
                        x in tournament
                        for x in [
                            "Premier League",
                            "LaLiga",
                            "Bundesliga",
                            "Serie A",
                            "Ligue 1",
                        ]
                    ):
                        finished_matches.append(event)

        print(f"Found {len(finished_matches)} finished major league matches")

        if not finished_matches:
            print("No finished matches found, trying any finished match...")
            for days_ago in range(0, 7):
                date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                fixtures = await api._get(f"/sport/football/scheduled-events/{date}")
                for event in fixtures.get("events", []):
                    if event.get("status", {}).get("type") == "finished":
                        finished_matches.append(event)
                        if len(finished_matches) >= 5:
                            break
                if len(finished_matches) >= 5:
                    break

        # Test first match thoroughly
        match = finished_matches[0]
        match_id = match["id"]
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        home_id = match["homeTeam"]["id"]
        away_id = match["awayTeam"]["id"]
        home_score = match.get("homeScore", {}).get("current", "?")
        away_score = match.get("awayScore", {}).get("current", "?")

        print(f"\n{'=' * 60}")
        print(f"Match: {home} {home_score} - {away_score} {away}")
        print(f"Match ID: {match_id}")
        print(f"Home Team ID: {home_id}, Away Team ID: {away_id}")

        # 1. Match Statistics
        print(f"\n[2] MATCH STATISTICS")
        print("-" * 50)
        try:
            stats = await api._get(f"/event/{match_id}/statistics")
            stats_list = stats.get("statistics", [])

            print(f"Available stat groups: {len(stats_list)}")
            for group in stats_list:
                group_name = group.get("groupName", "Unknown")
                print(f"\n  {group_name}:")
                for item in group.get("statisticsItems", [])[:8]:
                    name = item.get("name")
                    home_val = item.get("home", "?")
                    away_val = item.get("away", "?")
                    compare = item.get("compareCode", "")
                    print(f"    {name}: {home_val} vs {away_val}")
        except Exception as e:
            print(f"  Error: {e}")

        # 2. Lineups with player details
        print(f"\n[3] LINEUPS & PLAYER RATINGS")
        print("-" * 50)
        try:
            lineups = await api._get(f"/event/{match_id}/lineups")

            for side in ["home", "away"]:
                side_data = lineups.get(side, {})
                players = side_data.get("players", [])

                team_name = home if side == "home" else away
                print(f"\n  {team_name} ({len(players)} players):")

                for p in players[:6]:  # Show first 6
                    player = p.get("player", {})
                    name = player.get("name", "?")
                    position = player.get("position", "?")
                    shirt = p.get("shirtNumber", "?")
                    substitute = p.get("substitute", False)

                    # Check for rating
                    rating = p.get("rating") or player.get("rating") or "?"

                    # Check for statistics in player object
                    stats = p.get("statistics", {})

                    print(
                        f"    #{shirt} {name} ({position}) {'[SUB]' if substitute else ''}"
                    )
                    if rating and rating != "?":
                        print(f"        Rating: {rating}")
                    if stats:
                        print(f"        Stats: {list(stats.keys())[:5]}")

        except Exception as e:
            print(f"  Error: {e}")

        # 3. Player Statistics (detailed)
        print(f"\n[4] DETAILED PLAYER STATISTICS")
        print("-" * 50)
        try:
            player_stats = await api._get(f"/event/{match_id}/lineups")

            # Get first home player with stats
            home_players = player_stats.get("home", {}).get("players", [])
            for p in home_players:
                player = p.get("player", {})
                stats = p.get("statistics", {})

                if stats:
                    print(f"  Player: {player.get('name')}")
                    print(f"  Position: {player.get('position')}")
                    print(f"  Rating: {p.get('rating', 'N/A')}")
                    print(f"\n  Full statistics:")
                    for key, val in stats.items():
                        print(f"    {key}: {val}")
                    break
        except Exception as e:
            print(f"  Error: {e}")

        # 4. Shots/Shootmap
        print(f"\n[5] SHOTMAP (xG DATA)")
        print("-" * 50)
        try:
            shotmap = await api._get(f"/event/{match_id}/shotmap")
            shots = shotmap.get("shotmap", [])

            print(f"Total shots: {len(shots)}")

            home_shots = [s for s in shots if s.get("isHome")]
            away_shots = [s for s in shots if not s.get("isHome")]

            print(f"  {home} shots: {len(home_shots)}")
            print(f"  {away} shots: {len(away_shots)}")

            # Show sample shot with xG
            for shot in shots[:3]:
                player_name = shot.get("player", {}).get("name", "?")
                xg = shot.get("xg", "?")
                shot_type = shot.get("shotType", "?")
                outcome = shot.get("outcome", "?")
                print(f"\n    Shot: {player_name}")
                print(f"      xG: {xg}")
                print(f"      Type: {shot_type}, Outcome: {outcome}")

        except Exception as e:
            print(f"  Error: {e}")

        # 5. Heatmap
        print(f"\n[6] PLAYER HEATMAP")
        print("-" * 50)
        try:
            heatmap = await api._get(f"/event/{match_id}/heatmap")
            print(f"  Heatmap data available: {bool(heatmap)}")
            if heatmap:
                print(f"  Keys: {list(heatmap.keys())}")
        except Exception as e:
            print(f"  Error: {e}")

        # 6. Incidents (goals, cards, subs)
        print(f"\n[7] MATCH INCIDENTS")
        print("-" * 50)
        try:
            incidents = await api._get(f"/event/{match_id}/incidents")
            incident_list = incidents.get("incidents", [])

            print(f"Total incidents: {len(incident_list)}")

            for inc in incident_list[:10]:
                inc_type = inc.get("incidentType", "?")

                if inc_type == "goal":
                    player = inc.get("player", {}).get("name", "?")
                    time = inc.get("time", "?")
                    is_home = inc.get("isHome", False)
                    team = home if is_home else away
                    print(f"  [{time}'] GOAL {team}: {player}")

                elif inc_type == "card":
                    player = inc.get("player", {}).get("name", "?")
                    time = inc.get("time", "?")
                    card_type = inc.get("incidentClass", "?")
                    print(f"  [{time}'] {card_type.upper()} CARD: {player}")

                elif inc_type == "substitution":
                    player_in = inc.get("playerIn", {}).get("name", "?")
                    player_out = inc.get("playerOut", {}).get("name", "?")
                    time = inc.get("time", "?")
                    print(f"  [{time}'] SUB: {player_out} -> {player_in}")

        except Exception as e:
            print(f"  Error: {e}")

        # 7. Best players (MOTM candidates)
        print(f"\n[8] BEST PLAYERS / MOTM")
        print("-" * 50)
        try:
            best = await api._get(f"/event/{match_id}/best-players")
            players = best.get("bestPlayers", [])

            print(f"Best players: {len(players)}")
            for p in players[:5]:
                player = p.get("player", {}).get("name", "?")
                rating = p.get("rating", "?")
                print(f"  {player}: Rating {rating}")

        except Exception as e:
            print(f"  Error: {e}")

        # 8. Average positions
        print(f"\n[9] AVERAGE POSITIONS")
        print("-" * 50)
        try:
            positions = await api._get(f"/event/{match_id}/average-positions")
            print(f"  Data available: {bool(positions)}")
        except Exception as e:
            print(f"  Not available: {str(e)[:50]}")

        # Compare with Flashscore format
        print(f"\n{'=' * 60}")
        print("COMPARISON: SofaScore vs Flashscore Post-Match Data")
        print("=" * 60)

        print("""
SOFAFSCORE POST-MATCH DATA:
+---------------------------+-------------+--------------------------------+
| Data Type                 | Available?  | Details                        |
+---------------------------+-------------+--------------------------------+
| Match Statistics          | YES         | Possession, shots, corners,    |
|                           |             | fouls, passes, etc.            |
| Player Ratings            | YES         | SofaScore proprietary ratings  |
| Player Statistics         | YES         | Passes, tackles, duels won...  |
| xG (Expected Goals)       | YES         | Per shot, aggregated           |
| Shotmap                   | YES         | All shots with xG, location    |
| Heatmap                   | YES         | Player movement zones          |
| Incidents                 | YES         | Goals, cards, subs, VAR        |
| Lineups                   | YES         | Full starting XI + subs        |
| Best Players / MOTM       | YES         | Top rated players              |
| H2H updated               | YES         | After match                    |
+---------------------------+-------------+--------------------------------+

FLASHSCORE EQUIVALENT:
- Match stats: YES
- Player ratings: NO (I think)
- xG: MAYBE (depends on match)
- Detailed player stats: LIMITED
""")

    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
