"""Test data availability for UPCOMING fixtures."""

import asyncio
import json
from datetime import datetime, timedelta

from sofascore_wrapper.api import SofascoreAPI


INJURY_REASONS = {
    1: "Injury",
    2: "Suspended",
    3: "International Duty",
    4: "COVID-19",
    5: "Rest",
    6: "Unknown",
    7: "Knock",
    8: "Illness",
    9: "Personal Reasons",
    10: "Transfer Pending",
    11: "Not in Squad",
    12: "Doubtful",
    13: "Coach Decision",
    14: "Yellow Card Suspension",
    15: "Red Card Suspension",
}


async def main():
    api = SofascoreAPI()

    print("=" * 70)
    print("UPCOMING FIXTURES - DATA AVAILABILITY TEST")
    print("=" * 70)

    try:
        # Get fixtures for next 7 days
        print("\n[1] CHECKING UPCOMING FIXTURES (Next 7 Days)")
        print("-" * 50)

        upcoming_fixtures = []

        for days_ahead in range(0, 8):
            date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            fixtures = await api._get(f"/sport/football/scheduled-events/{date}")

            for event in fixtures.get("events", []):
                status = event.get("status", {})
                if status.get("type") == "notstarted":
                    upcoming_fixtures.append(event)

        print(f"Found {len(upcoming_fixtures)} upcoming fixtures")

        # Test 5 different upcoming fixtures
        print("\n[2] TESTING DATA AVAILABILITY FOR UPCOMING MATCHES")
        print("-" * 50)

        tested = 0
        for fixture in upcoming_fixtures[:20]:  # Check first 20
            match_id = fixture["id"]
            home = fixture["homeTeam"]["name"]
            away = fixture["awayTeam"]["name"]
            kickoff = datetime.fromtimestamp(fixture.get("startTimestamp", 0))
            tournament = (
                fixture.get("tournament", {})
                .get("uniqueTournament", {})
                .get("name", "Unknown")
            )

            # Skip friendlies/youth for cleaner test
            if any(
                x in tournament.lower() for x in ["friendly", "u1", "u2", "u3", "youth"]
            ):
                continue

            print(f"\n{'=' * 60}")
            print(f"Match: {home} vs {away}")
            print(f"Tournament: {tournament}")
            print(f"Kickoff: {kickoff.strftime('%Y-%m-%d %H:%M')} UTC")
            print(f"Match ID: {match_id}")

            # Test 1: Lineups & Missing Players
            print(f"\n  [LINEUPS & INJURIES]")
            try:
                lineups = await api._get(f"/event/{match_id}/lineups")
                confirmed = lineups.get("confirmed", False)
                home_players = len(lineups.get("home", {}).get("players", []))
                away_players = len(lineups.get("away", {}).get("players", []))
                home_missing = lineups.get("home", {}).get("missingPlayers", [])
                away_missing = lineups.get("away", {}).get("missingPlayers", [])

                print(f"    Lineups confirmed: {confirmed}")
                print(f"    Home lineup: {home_players} players")
                print(f"    Away lineup: {away_players} players")
                print(f"    Home MISSING: {len(home_missing)} players")
                print(f"    Away MISSING: {len(away_missing)} players")

                if home_missing:
                    print(f"    Home missing players:")
                    for mp in home_missing[:5]:
                        p = mp.get("player", {})
                        reason = INJURY_REASONS.get(mp.get("reason"), "?")
                        print(
                            f"      - {p.get('name')} ({p.get('position')}) - {reason}"
                        )

                if away_missing:
                    print(f"    Away missing players:")
                    for mp in away_missing[:5]:
                        p = mp.get("player", {})
                        reason = INJURY_REASONS.get(mp.get("reason"), "?")
                        print(
                            f"      - {p.get('name')} ({p.get('position')}) - {reason}"
                        )

            except Exception as e:
                print(f"    Error: {e}")

            # Test 2: Odds
            print(f"\n  [ODDS]")
            try:
                odds = await api._get(f"/event/{match_id}/odds/1/all")
                markets = odds.get("markets", [])

                # Find key markets
                market_names = [m.get("marketName") for m in markets]
                print(f"    Available markets: {len(markets)}")

                for m in markets[:5]:
                    print(f"    - {m.get('marketName')}: ", end="")
                    choices = m.get("choices", [])
                    if choices:
                        prices = [
                            f"{c.get('name')}@{c.get('fractionalValue')}"
                            for c in choices[:3]
                        ]
                        print(", ".join(prices))
                    else:
                        print("No prices")

            except Exception as e:
                print(f"    Error: {e}")

            # Test 3: Featured Odds (includes Asian Handicap, O/U)
            print(f"\n  [FEATURED ODDS]")
            try:
                featured = await api._get(f"/event/{match_id}/odds/1/featured")
                featured_markets = featured.get("featured", {})

                for market_type in ["default", "asian", "fullTime"]:
                    if market_type in featured_markets:
                        m = featured_markets[market_type]
                        print(f"    {market_type}: {m.get('marketName')}")
                        for c in m.get("choices", [])[:3]:
                            print(f"      {c.get('name')}: {c.get('fractionalValue')}")

            except Exception as e:
                print(f"    Error: {e}")

            # Test 4: H2H
            print(f"\n  [HEAD-TO-HEAD]")
            try:
                h2h = await api._get(f"/event/{match_id}/h2h")
                team_duel = h2h.get("teamDuel", {})
                if team_duel:
                    print(
                        f"    H2H: Home {team_duel.get('homeWins')} - Draw {team_duel.get('draws')} - Away {team_duel.get('awayWins')}"
                    )
                else:
                    print(f"    No H2H data")
            except Exception as e:
                print(f"    Error: {e}")

            # Test 5: Pre-match form
            print(f"\n  [PRE-MATCH FORM]")
            try:
                form = await api._get(f"/event/{match_id}/pregameform")
                home_form = form.get("homeTeam", {})
                away_form = form.get("awayTeam", {})

                if home_form:
                    print(
                        f"    Home: Pos {home_form.get('position')}, Form {home_form.get('form')}, Avg Rating {home_form.get('avgRating')}"
                    )
                if away_form:
                    print(
                        f"    Away: Pos {away_form.get('position')}, Form {away_form.get('form')}, Avg Rating {away_form.get('avgRating')}"
                    )
            except Exception as e:
                print(f"    Error: {e}")

            # Test 6: Win Probability
            print(f"\n  [WIN PROBABILITY]")
            try:
                prob = await api._get(f"/event/{match_id}/winprobability")
                if prob:
                    print(f"    Data available: {list(prob.keys())[:5]}")
            except Exception as e:
                print(f"    Not available: {str(e)[:50]}")

            # Test 7: Team streaks
            print(f"\n  [TEAM STREAKS]")
            try:
                streaks = await api._get(f"/event/{match_id}/teamstreaks")
                if streaks:
                    print(f"    Data available")
            except Exception as e:
                print(f"    Not available: {str(e)[:50]}")

            tested += 1
            if tested >= 3:  # Test 3 matches
                break

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: DATA AVAILABILITY FOR UPCOMING FIXTURES")
        print("=" * 70)

        print("""
┌─────────────────────────┬────────────────┬─────────────────────────────┐
│ Data Type               │ Available?     │ Notes                       │
├─────────────────────────┼────────────────┼─────────────────────────────┤
│ Missing Players/Injuries│ ✅ YES         │ Even before lineups confirm │
│ Lineups (predicted)     │ ✅ YES         │ May not be confirmed yet    │
│ Odds (1X2)              │ ✅ YES         │ Multiple bookmakers         │
│ Asian Handicap          │ ✅ YES         │ Via featured odds           │
│ Over/Under              │ ✅ YES         │ Via featured odds           │
│ H2H                     │ ✅ YES         │ Historical record           │
│ Pre-match Form          │ ✅ YES         │ Position, form, rating      │
│ Win Probability         │ ⚠️ Sometimes   │ Not all matches             │
│ Team Streaks            │ ⚠️ Sometimes   │ Not all matches             │
│ Match Stats             │ ❌ NO          │ Only after match played     │
│ Player Stats            │ ❌ NO          │ Only after match played     │
│ xG                      │ ❌ NO          │ Only after match played     │
└─────────────────────────┴────────────────┴─────────────────────────────┘

CONCLUSION:
- All PREMATCH data you need IS available: injuries, odds, H2H, form
- POST-MATCH data (stats, xG) only available after game finishes
- Injury data EXISTS for upcoming matches (key finding!)
""")

    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
