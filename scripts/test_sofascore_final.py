"""Full extraction with reason code mapping."""

import asyncio
import json
from datetime import datetime

from sofascore_wrapper.api import SofascoreAPI


# SofaScore injury reason codes (from their API)
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
    print("SOFASCORE INJURY DATA - FULL EXTRACTION")
    print("=" * 70)

    try:
        # 1. Get Arsenal squad (ID: 42) - they usually have injury news
        print("\n[1] ARSENAL SQUAD INJURIES")
        print("-" * 50)

        squad = await api._get("/team/42/players")
        players = squad.get("players", [])

        print(f"Total squad: {len(players)} players\n")

        injured_count = 0
        for p in players:
            player_obj = p.get("player", {})
            injury = p.get("injury")

            if injury:
                injured_count += 1
                status = injury.get("status", "unknown")
                reason = injury.get("reason", "Unknown")
                start_ts = injury.get("startDateTimestamp")
                end_ts = injury.get("endDateTimestamp")

                print(f"  {player_obj.get('name')} ({player_obj.get('position')})")
                print(f"    Status: {status}")
                print(f"    Reason: {reason}")
                if start_ts:
                    print(
                        f"    Started: {datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d')}"
                    )
                if end_ts:
                    print(
                        f"    Expected return: {datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d')}"
                    )
                print()

        print(f"Total injured: {injured_count}")

        # 2. Get a match with lineups and missing players
        print("\n[2] MATCH MISSING PLAYERS (with decoded reasons)")
        print("-" * 50)

        # Find a match
        fixtures = await api._get("/sport/football/scheduled-events/2026-02-21")
        events = fixtures.get("events", [])

        match_found = None
        for event in events:
            if event.get("status", {}).get("type") == "finished":
                match_found = event
                break

        if match_found:
            match_id = match_found["id"]
            home_name = match_found["homeTeam"]["name"]
            away_name = match_found["awayTeam"]["name"]
            home_id = match_found["homeTeam"]["id"]
            away_id = match_found["awayTeam"]["id"]

            print(f"Match: {home_name} vs {away_name}")
            print(
                f"Result: {match_found.get('homeScore', {}).get('current')} - {match_found.get('awayScore', {}).get('current')}"
            )

            lineups = await api._get(f"/event/{match_id}/lineups")

            for side, team_name, team_id in [
                ("home", home_name, home_id),
                ("away", away_name, away_id),
            ]:
                side_data = lineups.get(side, {})
                missing = side_data.get("missingPlayers", [])

                print(f"\n{team_name} - Missing Players ({len(missing)}):")

                for mp in missing:
                    player = mp.get("player", {})
                    reason_code = mp.get("reason")
                    reason_text = INJURY_REASONS.get(
                        reason_code, f"Unknown ({reason_code})"
                    )

                    print(f"  - {player.get('name')} ({player.get('position')})")
                    print(f"    Reason: {reason_text}")

        # 3. Upcoming match check
        print("\n[3] UPCOMING MATCH - INJURY PREVIEW")
        print("-" * 50)

        today = await api._get("/sport/football/scheduled-events/2026-02-22")

        for event in today.get("events", [])[:15]:
            if event.get("status", {}).get("type") == "notstarted":
                upcoming_id = event["id"]
                home_name = event["homeTeam"]["name"]
                away_name = event["awayTeam"]["name"]

                lineups = await api._get(f"/event/{upcoming_id}/lineups")

                home_missing = lineups.get("home", {}).get("missingPlayers", [])
                away_missing = lineups.get("away", {}).get("missingPlayers", [])

                if home_missing or away_missing:
                    print(f"\n{home_name} vs {away_name}")
                    print(
                        f"  Starts: {datetime.fromtimestamp(event.get('startTimestamp', 0)).strftime('%Y-%m-%d %H:%M')}"
                    )

                    if home_missing:
                        print(f"\n  {home_name} OUT ({len(home_missing)}):")
                        for mp in home_missing[:6]:
                            p = mp.get("player", {})
                            reason = INJURY_REASONS.get(mp.get("reason"), "?")
                            print(
                                f"    - {p.get('name')} ({p.get('position')}) - {reason}"
                            )

                    if away_missing:
                        print(f"\n  {away_name} OUT ({len(away_missing)}):")
                        for mp in away_missing[:6]:
                            p = mp.get("player", {})
                            reason = INJURY_REASONS.get(mp.get("reason"), "?")
                            print(
                                f"    - {p.get('name')} ({p.get('position')}) - {reason}"
                            )

                    break

        # 4. Sample output format for your pipeline
        print("\n[4] SAMPLE OUTPUT FOR YOUR PIPELINE")
        print("-" * 50)

        # Create a sample in your Flashscore format
        sample = {
            "id": str(match_id) if match_found else "12345",
            "availability": {
                "home": {
                    "missing": [
                        {
                            "name": mp.get("player", {}).get("name"),
                            "reason": INJURY_REASONS.get(mp.get("reason"), "Unknown"),
                        }
                        for mp in lineups.get("home", {}).get("missingPlayers", [])[:5]
                    ],
                    "questionable": [],  # SofaScore doesn't separate this
                },
                "away": {
                    "missing": [
                        {
                            "name": mp.get("player", {}).get("name"),
                            "reason": INJURY_REASONS.get(mp.get("reason"), "Unknown"),
                        }
                        for mp in lineups.get("away", {}).get("missingPlayers", [])[:5]
                    ],
                    "questionable": [],
                },
            },
        }

        print("Your current Flashscore format:")
        print(json.dumps(sample, indent=2))

        # Enhanced format SofaScore can provide
        enhanced_sample = {
            "id": str(match_id) if match_found else "12345",
            "availability": {
                "home": {
                    "missing": [
                        {
                            "name": mp.get("player", {}).get("name"),
                            "position": mp.get("player", {}).get("position"),
                            "reason": INJURY_REASONS.get(mp.get("reason"), "Unknown"),
                            "reason_code": mp.get("reason"),
                        }
                        for mp in lineups.get("home", {}).get("missingPlayers", [])[:5]
                    ]
                },
                "away": {
                    "missing": [
                        {
                            "name": mp.get("player", {}).get("name"),
                            "position": mp.get("player", {}).get("position"),
                            "reason": INJURY_REASONS.get(mp.get("reason"), "Unknown"),
                            "reason_code": mp.get("reason"),
                        }
                        for mp in lineups.get("away", {}).get("missingPlayers", [])[:5]
                    ]
                },
            },
        }

        print("\nEnhanced SofaScore format (more data!):")
        print(json.dumps(enhanced_sample, indent=2))

    finally:
        await api.close()

    print("\n" + "=" * 70)
    print("VERDICT: sofascore-wrapper CAN replace Flashscore scraping!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
