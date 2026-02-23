"""Test correct SofaScore endpoints for player impact calculation."""

import asyncio
from datetime import datetime

from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.league import League


async def main():
    api = SofascoreAPI()

    print("=" * 70)
    print("CORRECT ENDPOINTS - PLAYER IMPACT DATA")
    print("=" * 70)

    try:
        # Premier League ID = 17
        league = League(api, league_id=17)

        # 1. Get current season
        print("\n[1] CURRENT SEASON")
        print("-" * 50)
        current_season = await league.current_season()
        season_id = current_season.get("id")
        print(f"Season: {current_season.get('name')} (ID: {season_id})")

        # 2. League Info (goals, home/away wins, etc.)
        print("\n[2] LEAGUE INFO (Attack/Defense Stats)")
        print("-" * 50)
        info = await league.get_info(season_id)
        league_info = info.get("info", {})

        print(f"Total Goals: {league_info.get('goals')}")
        print(f"Home Team Wins: {league_info.get('homeTeamWins')}")
        print(f"Away Team Wins: {league_info.get('awayTeamWins')}")
        print(f"Draws: {league_info.get('draws')}")

        # 3. Top Players (by rating, goals, assists)
        print("\n[3] TOP PLAYERS BY RATING")
        print("-" * 50)
        top_players = await league.top_players(season_id)
        top_players_dict = top_players.get("topPlayers", {})

        # By rating
        print("\nTop 10 by Rating:")
        for p in top_players_dict.get("rating", [])[:10]:
            player = p.get("player", {})
            team = p.get("team", {})
            stats = p.get("statistics", {})

            print(f"  {player.get('name')} ({team.get('name')})")
            print(
                f"    Rating: {stats.get('rating'):.2f}, Appearances: {stats.get('appearances')}"
            )

        # By goals
        print("\nTop 5 by Goals:")
        for p in top_players_dict.get("goals", [])[:5]:
            player = p.get("player", {})
            team = p.get("team", {})
            stats = p.get("statistics", {})

            print(
                f"  {player.get('name')} ({team.get('name')}): {stats.get('goals')} goals"
            )

        # By assists
        print("\nTop 5 by Assists:")
        for p in top_players_dict.get("assists", [])[:5]:
            player = p.get("player", {})
            team = p.get("team", {})
            stats = p.get("statistics", {})

            print(
                f"  {player.get('name')} ({team.get('name')}): {stats.get('assists')} assists"
            )

        # 4. Top Teams (by avg rating, goals scored, etc.)
        print("\n[4] TOP TEAMS BY AVERAGE RATING")
        print("-" * 50)
        top_teams = await league.top_teams(season_id)
        top_teams_dict = top_teams.get("topTeams", {})

        print("\nBy Average Rating:")
        for t in top_teams_dict.get("avgRating", [])[:10]:
            team = t.get("team", {})
            stats = t.get("statistics", {})

            print(
                f"  {team.get('name')}: {stats.get('avgRating'):.2f} (from {stats.get('matches')} matches)"
            )

        # 5. Standings - Total
        print("\n[5] STANDINGS - TOTAL")
        print("-" * 50)
        standings = await league.standings(season_id)
        rows = standings.get("standings", [{}])[0].get("rows", [])

        print(
            f"{'Pos':<4} {'Team':<20} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<4} {'GA':<4} {'Pts':<4}"
        )
        print("-" * 55)
        for row in rows[:10]:
            team = row.get("team", {})
            print(
                f"{row.get('position'):<4} {team.get('name')[:20]:<20} {row.get('matches', '?'):<3} {row.get('wins', '?'):<3} {row.get('draws', '?'):<3} {row.get('losses', '?'):<3} {row.get('scoresFor', '?'):<4} {row.get('scoresAgainst', '?'):<4} {row.get('points', '?'):<4}"
            )

        # 6. Standings - HOME (key for home impact!)
        print("\n[6] STANDINGS - HOME ONLY")
        print("-" * 50)
        home_standings = await league.standings_home(season_id)
        home_rows = home_standings.get("standings", [{}])[0].get("rows", [])

        print(
            f"{'Pos':<4} {'Team':<20} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<4} {'GA':<4} {'Pts':<4}"
        )
        print("-" * 55)
        for row in home_rows[:10]:
            team = row.get("team", {})
            print(
                f"{row.get('position'):<4} {team.get('name')[:20]:<20} {row.get('matches', '?'):<3} {row.get('wins', '?'):<3} {row.get('draws', '?'):<3} {row.get('losses', '?'):<3} {row.get('scoresFor', '?'):<4} {row.get('scoresAgainst', '?'):<4} {row.get('points', '?'):<4}"
            )

        # 7. Standings - AWAY (key for away impact!)
        print("\n[7] STANDINGS - AWAY ONLY")
        print("-" * 50)
        away_standings = await league.standings_away(season_id)
        away_rows = away_standings.get("standings", [{}])[0].get("rows", [])

        print(
            f"{'Pos':<4} {'Team':<20} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<4} {'GA':<4} {'Pts':<4}"
        )
        print("-" * 55)
        for row in away_rows[:10]:
            team = row.get("team", {})
            print(
                f"{row.get('position'):<4} {team.get('name')[:20]:<20} {row.get('matches', '?'):<3} {row.get('wins', '?'):<3} {row.get('draws', '?'):<3} {row.get('losses', '?'):<3} {row.get('scoresFor', '?'):<4} {row.get('scoresAgainst', '?'):<4} {row.get('points', '?'):<4}"
            )

        # 8. Calculate Arsenal's home vs away strength
        print("\n[8] ARSENAL HOME vs AWAY STRENGTH")
        print("-" * 50)

        # Find Arsenal in each table
        arsenal_id = 42

        total_row = next(
            (r for r in rows if r.get("team", {}).get("id") == arsenal_id), None
        )
        home_row = next(
            (r for r in home_rows if r.get("team", {}).get("id") == arsenal_id), None
        )
        away_row = next(
            (r for r in away_rows if r.get("team", {}).get("id") == arsenal_id), None
        )

        if total_row and home_row and away_row:
            print(
                f"TOTAL: {total_row.get('wins')}W {total_row.get('draws')}D {total_row.get('losses')}L | GF: {total_row.get('scoresFor')} GA: {total_row.get('scoresAgainst')}"
            )
            print(
                f"HOME:  {home_row.get('wins')}W {home_row.get('draws')}D {home_row.get('losses')}L | GF: {home_row.get('scoresFor')} GA: {home_row.get('scoresAgainst')}"
            )
            print(
                f"AWAY:  {away_row.get('wins')}W {away_row.get('draws')}D {away_row.get('losses')}L | GF: {away_row.get('scoresFor')} GA: {away_row.get('scoresAgainst')}"
            )

            # Calculate attack/defense strength
            home_matches = home_row.get("matches", 1) or 1
            away_matches = away_row.get("matches", 1) or 1

            home_gf_per_game = (home_row.get("scoresFor") or 0) / home_matches
            home_ga_per_game = (home_row.get("scoresAgainst") or 0) / home_matches
            away_gf_per_game = (away_row.get("scoresFor") or 0) / away_matches
            away_ga_per_game = (away_row.get("scoresAgainst") or 0) / away_matches

            print(f"\nHome Attack: {home_gf_per_game:.2f} goals/game")
            print(f"Home Defense: {home_ga_per_game:.2f} goals conceded/game")
            print(f"Away Attack: {away_gf_per_game:.2f} goals/game")
            print(f"Away Defense: {away_ga_per_game:.2f} goals conceded/game")

    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
