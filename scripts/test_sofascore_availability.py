import asyncio
import json
from datetime import datetime
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.league import League
from sofascore_wrapper.match import Match

async def test_european_cups(api):
    print("\n--- Testing European Cups ---")
    # Known Sofascore IDs for European Cups
    cups = {
        "CL": 7,    # Champions League
        "EL": 677,  # Europa League
        "ECL": 17015 # Conference League
    }
    
    for name, league_id in cups.items():
        print(f"\nChecking {name} (ID: {league_id})...")
        league = League(api, league_id)
        try:
            seasons = await league.get_seasons()
            if seasons:
                current_season = seasons[0]
                print(f"  Latest Season: {current_season['name']} (ID: {current_season['id']})")
                
                # Try to get standings as a proxy for team list
                standings = await league.standings(current_season['id'])
                if standings.get('standings'):
                    teams_count = len(standings['standings'][0].get('rows', []))
                    print(f"  Teams in standings: {teams_count}")
                else:
                    print("  No standings found/format different.")
            else:
                print("  No seasons found.")
        except Exception as e:
            print(f"  Error checking {name}: {e}")

async def test_historical_injuries(api):
    print("\n--- Testing Historical Injury Depth ---")
    # Use a high-profile match from last season (e.g., Man City vs Inter, CL Final 2023?)
    # Or just a recent PL match from earlier this season
    
    # Let's try to find a match from Oct 2024
    match_api = Match(api, 0)
    try:
        data = await match_api.games_by_date("football", "2024-10-19") # A Saturday in October
        events = data.get("events", [])
        if events:
            # Pick a finished match from a major league (e.g., E0)
            target_match = None
            for event in events:
                if event.get("tournament", {}).get("uniqueTournament", {}).get("slug") == "premier-league":
                    target_match = event
                    break
            
            if target_match:
                match_id = target_match['id']
                print(f"Testing historical match: {target_match['homeTeam']['name']} vs {target_match['awayTeam']['name']} (ID: {match_id})")
                
                # Check lineups/missing players
                lineups = await api._get(f"/event/{match_id}/lineups")
                home_missing = lineups.get("home", {}).get("missingPlayers", [])
                away_missing = lineups.get("away", {}).get("missingPlayers", [])
                
                print(f"  Home missing players: {len(home_missing)}")
                print(f"  Away missing players: {len(away_missing)}")
                
                if home_missing or away_missing:
                    print("  ✓ Historical injury data IS available for this match.")
                else:
                    print("  ✗ No historical injury data found for this match.")
            else:
                print("  No PL match found on that date.")
        else:
            print("  No events found on that date.")
    except Exception as e:
        print(f"  Error testing historical injuries: {e}")

async def main():
    api = SofascoreAPI()
    try:
        await test_european_cups(api)
        await test_historical_injuries(api)
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(main())
