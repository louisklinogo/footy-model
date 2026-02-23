"""Test DEFENSIVE player impact data from SofaScore."""

import asyncio
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.league import League


async def main():
    api = SofascoreAPI()

    print("=" * 70)
    print("DEFENSIVE PLAYER IMPACT - DATA EXPLORATION")
    print("=" * 70)

    try:
        league = League(api, league_id=17)  # Premier League
        current_season = await league.current_season()
        season_id = current_season.get("id")

        # 1. Get top players - look for DEFENDERS
        print("\n[1] TOP DEFENDERS BY RATING")
        print("-" * 50)
        top_players = await league.top_players(season_id)
        top_players_dict = top_players.get("topPlayers", {})

        # Find defenders (position = D)
        all_players = top_players_dict.get("rating", [])
        defenders = [
            p for p in all_players if p.get("player", {}).get("position") == "D"
        ]

        print(f"Found {len(defenders)} defenders in top players list\n")

        for p in defenders[:10]:
            player = p.get("player", {})
            team = p.get("team", {})
            stats = p.get("statistics", {})

            print(f"  {player.get('name')} ({team.get('name')})")
            print(f"    Position: {player.get('position')}")
            print(f"    Rating: {stats.get('rating'):.2f}")
            print(f"    Appearances: {stats.get('appearances')}")
            print()

        # 2. Check what other stat categories exist for defenders
        print("\n[2] ALL AVAILABLE STAT CATEGORIES")
        print("-" * 50)
        print(f"Categories: {list(top_players_dict.keys())}")

        # Check for defensive stats
        for category in [
            "tackles",
            "interceptions",
            "clearances",
            "blocks",
            "cleanSheets",
            "passes",
        ]:
            if category in top_players_dict:
                print(f"\nTop 3 by {category}:")
                for p in top_players_dict[category][:3]:
                    player = p.get("player", {})
                    team = p.get("team", {})
                    stats = p.get("statistics", {})
                    print(
                        f"  {player.get('name')} ({team.get('name')}): {stats.get(category, stats.get('value', '?'))}"
                    )

        # 3. Get standings for defensive analysis
        print("\n[3] TEAM DEFENSIVE RECORDS (Goals Against)")
        print("-" * 50)

        standings = await league.standings(season_id)
        rows = standings.get("standings", [{}])[0].get("rows", [])

        # Sort by GA (best defense first)
        sorted_by_ga = sorted(rows, key=lambda x: x.get("scoresAgainst", 999))

        print(f"{'Team':<20} {'GA':<4} {'GA/Game':<8} {'Position':<8}")
        print("-" * 45)
        for row in sorted_by_ga[:10]:
            team = row.get("team", {})
            ga = row.get("scoresAgainst", "?")
            matches = row.get("matches", 1) or 1
            ga_per_game = (ga or 0) / matches
            print(
                f"{team.get('name')[:20]:<20} {ga:<4} {ga_per_game:.2f}     {row.get('position')}"
            )

        # 4. Home vs Away defensive records
        print("\n[4] HOME vs AWAY DEFENSIVE RECORDS")
        print("-" * 50)

        home_standings = await league.standings_home(season_id)
        away_standings = await league.standings_away(season_id)

        home_rows = home_standings.get("standings", [{}])[0].get("rows", [])
        away_rows = away_standings.get("standings", [{}])[0].get("rows", [])

        # Pick Arsenal for example
        arsenal_id = 42
        home_row = next(
            (r for r in home_rows if r.get("team", {}).get("id") == arsenal_id), None
        )
        away_row = next(
            (r for r in away_rows if r.get("team", {}).get("id") == arsenal_id), None
        )

        if home_row and away_row:
            home_matches = home_row.get("matches", 1) or 1
            away_matches = away_row.get("matches", 1) or 1

            home_ga = home_row.get("scoresAgainst", 0) or 0
            away_ga = away_row.get("scoresAgainst", 0) or 0

            print(f"ARSENAL DEFENSIVE RECORD:")
            print(
                f"  HOME:  {home_ga} goals conceded in {home_matches} games ({home_ga / home_matches:.2f} GA/game)"
            )
            print(
                f"  AWAY:  {away_ga} goals conceded in {away_matches} games ({away_ga / away_matches:.2f} GA/game)"
            )

            # Defensive intensity ratio
            home_defense_strength = 1 - (home_ga / home_matches) / 2.0  # normalize
            away_defense_strength = 1 - (away_ga / away_matches) / 2.0

            print(f"\n  Home defensive strength: {home_defense_strength:.2f}")
            print(f"  Away defensive strength: {away_defense_strength:.2f}")

        # 5. Get a defender's detailed stats from a finished match
        print("\n[5] DEFENDER STATS FROM FINISHED MATCH")
        print("-" * 50)

        # Get a recent finished match
        fixtures = await api._get("/sport/football/scheduled-events/2026-02-21")
        finished = None
        for event in fixtures.get("events", []):
            if event.get("status", {}).get("type") == "finished":
                finished = event
                break

        if finished:
            match_id = finished["id"]
            home_team = finished["homeTeam"]["name"]
            away_team = finished["awayTeam"]["name"]

            print(f"Match: {home_team} vs {away_team}")

            # Get lineups with player stats
            lineups = await api._get(f"/event/{match_id}/lineups")

            # Find defenders and their stats
            for side in ["home", "away"]:
                side_data = lineups.get(side, {})
                players = side_data.get("players", [])

                team_name = home_team if side == "home" else away_team
                defenders_found = [
                    p for p in players if p.get("player", {}).get("position") == "D"
                ]

                print(f"\n  {team_name} defenders ({len(defenders_found)}):")

                for p in defenders_found[:3]:
                    player = p.get("player", {})
                    stats = p.get("statistics", {})
                    rating = p.get("rating", "?")

                    print(f"\n    {player.get('name')} (#{p.get('shirtNumber')})")
                    print(f"      Rating: {rating}")

                    # Show defensive stats if available
                    if stats:
                        defensive_keys = [
                            "totalClearance",
                            "interceptionWon",
                            "duelWon",
                            "duelLost",
                            "tackleWon",
                            "tackleLost",
                            "ballRecovery",
                            "blockedScoringAttempt",
                            "totalContest",
                            "wonContest",
                            "aerialWon",
                            "aerialLost",
                        ]

                        for key in defensive_keys:
                            if key in stats:
                                print(f"      {key}: {stats[key]}")

        # 6. Goalkeepers
        print("\n[6] GOALKEEPER STATS")
        print("-" * 50)

        goalkeepers = [
            p for p in all_players if p.get("player", {}).get("position") == "G"
        ]

        print(f"Found {len(goalkeepers)} goalkeepers in top players\n")

        for p in goalkeepers[:5]:
            player = p.get("player", {})
            team = p.get("team", {})
            stats = p.get("statistics", {})

            print(f"  {player.get('name')} ({team.get('name')})")
            print(
                f"    Rating: {stats.get('rating'):.2f}, Appearances: {stats.get('appearances')}"
            )

        # Summary
        print("\n" + "=" * 70)
        print("DEFENSIVE IMPACT CALCULATION")
        print("=" * 70)

        print("""
AVAILABLE FOR PRE-MATCH:

1. DEFENDER RATING
   - Higher rating = better defender
   - Compare to team average rating
   - Rating gap indicates importance

2. APPEARANCES
   - Regular starter: high appearances
   - Squad rotation: low appearances
   - Use to determine if "key defender"

3. TEAM DEFENSIVE RECORD (from standings)
   - Total GA (goals against)
   - Home GA vs Away GA
   - GA per game

4. DEFENSIVE STRENGTH RATIO
   - home_defense = 1 - (home_GA/game) / league_avg
   - away_defense = 1 - (away_GA/game) / league_avg

CALCULATION FOR MISSING DEFENDER:

Example: Gabriel Magalhães (Arsenal CB) out

1. His rating: 7.30
2. Team avg rating: 6.94
3. Rating above team: +0.36 (quality defender)
4. Team GA: 21 in 28 games = 0.75 GA/game (EXCELLENT)
5. Home GA: 8 in 13 games = 0.62 GA/game
6. Away GA: 13 in 15 games = 0.87 GA/game

IMPACT SCORE:
- Quality factor: (7.30 - 6.94) / 6.94 = 5.2% above team avg
- Position factor: CB = critical for defense
- Team reliance: Arsenal has BEST defense (21 GA)

If Gabriel is OUT:
- Team defensive rating drops
- Expected additional goals conceded: +0.2-0.4 goals
- More significant at HOME (where defense is key to wins)

LIMITATION:
- No pre-match "tackles/interceptions per game" stats
- Only post-match defensive stats available
- Must infer from rating + team GA record

POST-MATCH STATS AVAILABLE:
- totalClearance
- interceptionWon
- tackleWon/tackleLost
- duelWon/duelLost
- aerialWon/aerialLost
- blockedScoringAttempt
- ballRecovery
- goalsPrevented (for GK)
""")

    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
