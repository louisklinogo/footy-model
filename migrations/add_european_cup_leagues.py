"""
Add European Cup Leagues to the database.

This adds Champions League, Europa League, and Conference League as leagues
so we can detect when teams have "higher priority" upcoming matches.

Usage:
    python migrations/add_european_cup_leagues.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

# European Cup League Codes (using Flashscore-style codes)
EUROPEAN_CUPS = [
    {
        "league_code": "CL",  # Champions League
        "league_name": "UEFA Champions League",
        "country": "International",
    },
    {
        "league_code": "EL",  # Europa League
        "league_name": "UEFA Europa League",
        "country": "International",
    },
    {
        "league_code": "ECL",  # Europa Conference League
        "league_name": "UEFA Europa Conference League",
        "country": "International",
    },
]


def add_european_cup_leagues() -> int:
    """Insert European cup leagues if they don't exist."""
    conn = connect_db()
    inserted = 0

    try:
        with conn.cursor() as cur:
            for cup in EUROPEAN_CUPS:
                # Check if exists
                cur.execute(
                    "SELECT 1 FROM leagues WHERE league_code = %s",
                    (cup["league_code"],),
                )
                exists = cur.fetchone() is not None

                if not exists:
                    cur.execute(
                        """
                        INSERT INTO leagues (league_code, league_name, country)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (league_code) DO NOTHING
                        """,
                        (cup["league_code"], cup["league_name"], cup["country"]),
                    )
                    inserted += 1
                    print(f"Added: {cup['league_code']} - {cup['league_name']}")
                else:
                    print(f"Already exists: {cup['league_code']}")

        conn.commit()
    finally:
        conn.close()

    return inserted


def get_european_cup_codes() -> list[str]:
    """Return list of European cup league codes."""
    return [cup["league_code"] for cup in EUROPEAN_CUPS]


if __name__ == "__main__":
    print("Adding European Cup Leagues...")
    count = add_european_cup_leagues()
    print(f"\nDone. Added {count} leagues.")
    print(f"League codes: {get_european_cup_codes()}")
