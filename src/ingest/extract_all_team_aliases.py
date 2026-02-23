"""
Comprehensive team alias extraction for ALL leagues.

This script:
1. Gets all leagues with SofaScore mappings from DB
2. Fetches teams from SofaScore standings for each league
3. Matches SofaScore teams to DB teams using multiple fuzzy strategies
4. Inserts ALL missing team aliases into team_aliases table
5. Also populates teams.sofascore_id

Usage:
    python src/ingest/extract_all_team_aliases.py --dry-run
    python src/ingest/extract_all_team_aliases.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import io
import re
import unicodedata
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import psycopg2

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import get_database_url


def normalize_for_matching(name: str) -> str:
    """Normalize team name for fuzzy matching."""
    name = name.lower().strip()

    # Normalize unicode (é -> e, ö -> o, etc.)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Remove common prefixes/suffixes
    prefixes = [
        "fc ",
        "afc ",
        "sc ",
        "1.",
        "1 ",
        "2.",
        "as ",
        "ss ",
        "ac ",
        "rc ",
        "cd ",
        "ca ",
    ]
    suffixes = [
        " fc",
        " afc",
        " sc",
        " cf",
        " ac",
        " rc",
        " cd",
        " ca",
        " a/s",
        " s.r.l.",
        " srl",
    ]

    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]

    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Replace common patterns
    replacements = [
        ("&", " and "),
        ("st.", "st "),
        ("saint", "st "),
        ("manchester", "man "),
        ("united", "utd"),
        ("wolverhampton", "wolves"),
        ("borussia", ""),
        ("red bull", ""),
        ("athletic", "ath"),
        ("athletico", "atletico"),
        ("atletico-mg", "atletico mg"),
    ]

    for old, new in replacements:
        name = name.replace(old, new)

    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def get_db_teams_by_league(conn) -> dict[str, dict[str, dict]]:
    """Get all teams from database grouped by league_code."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT t.team_id, t.team_name, t.league_code
            FROM teams t
            WHERE t.league_code IS NOT NULL
        """)
        teams_by_league: dict[str, dict[str, dict]] = {}
        for row in cur.fetchall():
            league_code = row[2]
            if league_code not in teams_by_league:
                teams_by_league[league_code] = {}
            teams_by_league[league_code][row[1].lower()] = {
                "team_id": row[0],
                "team_name": row[1],
            }
        return teams_by_league


def get_db_leagues_with_mapping(conn) -> list[dict]:
    """Get leagues that have SofaScore mappings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT league_id, league_code, league_name, country,
                   sofascore_league_id, sofascore_season_id
            FROM leagues
            WHERE sofascore_league_id IS NOT NULL
            ORDER BY league_code
        """)
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_existing_aliases(conn) -> set[str]:
    """Get existing SofaScore team aliases (lowercase)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT LOWER(alias_name)
            FROM team_aliases
            WHERE provider = 'sofascore'
        """)
        return {row[0] for row in cur.fetchall()}


async def get_sofascore_teams(api, league_id: int, season_id: int) -> list[dict]:
    """Get teams from SofaScore league standings."""
    from sofascore_wrapper.league import League

    league = League(api, league_id)
    teams = []

    try:
        data = await league.standings(season_id)
        if data:
            for table in data.get("standings", []):
                for row in table.get("rows", []):
                    team = row.get("team")
                    if team:
                        teams.append(
                            {
                                "id": team.get("id"),
                                "name": team.get("name"),
                                "short_name": team.get("shortName"),
                            }
                        )
    except Exception as e:
        print(f"    Error getting standings: {e}")

    return teams


# Special mappings for known mismatches
SPECIAL_MAPPINGS = {
    # England
    "queens park rangers": "qpr",
    "milton keynes dons": "mk dons",
    "wolverhampton wanderers": "wolves",
    "tottenham hotspur": "tottenham",
    "manchester united": "man utd",
    "manchester city": "man city",
    "newcastle united": "newcastle",
    "brighton and hove albion": "brighton",
    "west ham united": "west ham",
    "crystal palace": "c palace",
    "nottingham forest": "nottm forest",
    # Germany
    "fc bayern munchen": "bayern munich",
    "fc bayern muenchen": "bayern munich",
    "borussia m'gladbach": "b. monchengladbach",
    "bayer 04 leverkusen": "bayer leverkusen",
    "borussia monchengladbach": "b. monchengladbach",
    "borussia dortmund": "dortmund",
    "1. fc koln": "fc koln",
    "vfb stuttgart": "stuttgart",
    "tsg hoffenheim": "hoffenheim",
    "sc freiburg": "freiburg",
    "fc augsburg": "augsburg",
    "1. fc union berlin": "union berlin",
    "vfl wolfsburg": "wolfsburg",
    "eintracht frankfurt": "frankfurt",
    "1. fsv mainz 05": "mainz",
    "fc schalke 04": "schalke",
    "hamburger sv": "hamburg",
    "hertha bsc": "hertha",
    "preussen muenster": "preussen munster",
    "sv werder bremen": "werder bremen",
    "vfl bochum": "bochum",
    # France
    # Belgium
    "sint-truidense vv": "st truiden",
    "standard liege": "standard liege",
    # France
    "stade rennais": "rennes",
    "saint-etienne": "st etienne",
    "olympique de marseille": "marseille",
    "olympique lyonnais": "lyon",
    # Spain
    "athletic club": "ath bilbao",
    "atletico de madrid": "atl. madrid",
    "atletico madrid": "atl. madrid",
    "real betis balompie": "real betis",
    "deportivo alaves": "alaves",
    # Portugal
    "famalicao": "famalicao",
    "vitoria sc": "vitoria guimaraes",
    "avs - futebol sad": "afs",
    "sporting cp": "sporting",
    "sl benfica": "benfica",
    "fc porto": "porto",
    # Italy
    "hellas verona": "verona",
    "ac monza": "monza",
    "us lecce": "lecce",
    "us sassuolo calcio": "sassuolo",
    "frosinone calcio": "frosinone",
    "ac cesena": "cesena",
    "ssc bari": "bari",
    "ssc venezia": "venezia",
    # Netherlands
    "go ahead eagles": "g.a. eagles",
    "afc ajax": "ajax",
    "feijenoord": "feyenoord",
    # Belgium
    "club brugge kv": "club brugge",
    "cercle brugge ksv": "cercle brugge",
    "kvc westerlo": "westerlo",
    "oh leuven": "oh leuven",
    "ka gent": "gent",
    "rsc anderlecht": "anderlecht",
    "royal antwerp": "antwerp",
    "krc genk": "genk",
    "standard liege": "standard liege",
    # Turkey
    "fenerbahce": "fenerbahce",
    "galatasaray": "galatasaray",
    "besiktas": "besiktas",
    "basaksehir": "basaksehir",
    # Brazil
    "atletico mineiro": "atletico-mg",
    "atletico mineiro": "atletico-mg",
    "flamengo": "flamengo rj",
    "athletico paranaense": "athletico-pr",
    "red bull bragantino": "bragantino",
    "coritiba fc": "coritiba",
    "chapecoense": "chapecoense-sc",
    # Austria
    "austria wien": "austria vienna",
    "sk sturm graz": "sturm graz",
    "red bull salzburg": "salzburg",
    "sk rapid wien": "sk rapid",
    "grazer ak 1902": "grazer ak",
    "fc blau weiss linz": "bw linz",
    # Switzerland
    # Switzerland
    "fc zurich": "zurich",
    "bsc young boys": "young boys",
    "fc basel 1893": "basel",
    "fc lugano": "lugano",
    "fc st. gallen": "st gallen",
    "servette fc": "servette",
    "fc stade-lausanne-ouchy": "lausanne ouchy",
    "etoile carouge fc": "etoile-carouge",
    # Scotland
    # Scotland
    "celtic": "celtic",
    "rangers": "rangers",
    "heart of midlothian": "hearts",
    "hibernian": "hibernian",
    # Poland
    "legia warszawa": "legia warsaw",
    "lech poznan": "lech poznan",
    "wisa plock": "plock",
    "zaglebie lubin": "lubin",
    "widzew lodz": "lodz",
    # Greece
    # Greece
    "olympiakos piraeus": "olympiakos",
    "panathinaikos": "panathinaikos",
    "paok thessaloniki": "paok",
    "aek athens": "aek athens",
    "ael novibet": "ael larissa",
    "asteras aktor": "asteras tripolis",
    # Mexico
    # Mexico
    "club america": "america",
    "chivas guadalajara": "guadalajara",
    "cruz azul": "cruz azul",
    "tigres uanl": "tigres",
    "cf monterrey": "monterrey",
    # Romania
    "fcsb": "fcsb",
    "cfr cluj": "cfr cluj",
    "universitatea craiova": "u craiova",
    # Russia
    "zenit st. petersburg": "zenit",
    "cska moscow": "cska moscow",
    "fc lokomotiv moscow": "lokomotiv moscow",
    "fc dynamo moscow": "dynamo moscow",
    "fc krasnodar": "krasnodar",
    # Japan
    "yokohama f. marinos": "yokohama marinos",
    "kawasaki frontale": "kawasaki",
    "urawa red diamonds": "urawa red diamonds",
    "kashima antlers": "kashima",
    "kawasaki frontale": "kawasaki",
    "urawa red diamonds": "urawa",
    "kashima antlers": "kashima",
    # Denmark
    "fc copenhagen": "copenhagen",
    "fc copenhagen": "copenhagen",
    "fc kobenhavn": "copenhagen",
    "brondby if": "brondby",
    "sonderjyske fodbold": "sonderjyske",
    "fc nordsjaelland": "nordsjaelland",
    "midtjylland": "midtjylland",
    "agf aarhus": "aarhus",
    # Romania
    "universitatea craiova": "u craiova",
    "fc dinamo bucuresti": "din. bucuresti",
    "fc metaloglobus bucuresti": "metaloglobus bucharest",
    # Russia
    # Ireland
    "shamrock rovers": "shamrock",
    "bohemians": "bohemians",
    "st patrick's athletic": "st patricks",
    # Ireland
    "shamrock rovers": "shamrock",
    "bohemians": "bohemians",
    "st patrick's athletic": "st patricks",
    # More mappings for remaining teams
    "kasimpasa": "kasimpasa",
    "real racing club": "racing santander",
    "pari nizhny novgorod": "pari nn",
}


def fuzzy_match_team(
    sofa_name: str, sofa_id: int, db_teams: dict[str, dict], league_code: str
) -> tuple[int | None, str | None, str]:
    """Find best matching team using multiple fuzzy strategies."""
    sofa_lower = sofa_name.lower()
    sofa_normalized = normalize_for_matching(sofa_name)

    # Strategy 1: Exact match (case-insensitive)
    if sofa_lower in db_teams:
        t = db_teams[sofa_lower]
        return t["team_id"], t["team_name"], "exact"

    # Strategy 2: Normalized exact match
    for db_name_lower, t in db_teams.items():
        db_normalized = normalize_for_matching(t["team_name"])
        if db_normalized == sofa_normalized:
            return t["team_id"], t["team_name"], "normalized"

    # Strategy 3: Substring match
    for db_name_lower, t in db_teams.items():
        db_normalized = normalize_for_matching(t["team_name"])
        if db_normalized and sofa_normalized:
            if db_normalized in sofa_normalized or sofa_normalized in db_normalized:
                return t["team_id"], t["team_name"], "substring"

    # Strategy 4: Match by key words
    sofa_words = set(sofa_normalized.split())
    for db_name_lower, t in db_teams.items():
        db_normalized = normalize_for_matching(t["team_name"])
        db_words = set(db_normalized.split())

        if sofa_words and db_words:
            common = sofa_words & db_words
            if len(common) >= min(len(sofa_words), len(db_words)) * 0.5:
                if len(common) >= 2:
                    return t["team_id"], t["team_name"], "words"

    # Strategy 5: Special cases
    sofa_lower_clean = sofa_lower.strip()
    if sofa_lower_clean in SPECIAL_MAPPINGS:
        mapped_name = SPECIAL_MAPPINGS[sofa_lower_clean]
        if mapped_name.lower() in db_teams:
            t = db_teams[mapped_name.lower()]
            return t["team_id"], t["team_name"], "special"

    # Check if any special mapping key is contained in the sofa name
    for key, mapped_name in SPECIAL_MAPPINGS.items():
        if key in sofa_lower_clean or sofa_lower_clean in key:
            if mapped_name.lower() in db_teams:
                t = db_teams[mapped_name.lower()]
                return t["team_id"], t["team_name"], "special_reverse"

    return None, None, "none"


def insert_team_alias(
    conn, team_id: int, sofascore_name: str, sofascore_id: int
) -> bool:
    """Insert a team alias mapping and update team's sofascore_id."""
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                INSERT INTO team_aliases (provider, alias_name, team_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (provider, alias_name) DO UPDATE
                SET team_id = EXCLUDED.team_id
                """,
                ("sofascore", sofascore_name, team_id),
            )

            cur.execute(
                """
                UPDATE teams
                SET sofascore_id = %s, updated_at = NOW()
                WHERE team_id = %s AND (sofascore_id IS NULL OR sofascore_id != %s)
                """,
                (sofascore_id, team_id, sofascore_id),
            )

            return True
        except Exception as e:
            print(f"    Error inserting alias: {e}")
            return False


async def process_league(
    api,
    conn,
    league: dict,
    db_teams: dict[str, dict],
    existing_aliases: set[str],
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    """Process a single league and return match stats."""
    league_code = league["league_code"]
    league_name = league["league_name"]

    matched = 0
    already_exists = 0
    unmatched_teams = []

    print(f"\n[{league_code}] {league_name}")
    print(
        f"  SofaScore: league_id={league['sofascore_league_id']}, season_id={league['sofascore_season_id']}"
    )

    if league_code not in db_teams:
        print(f"  No teams in DB for this league")
        return 0, 0, []

    league_teams = db_teams[league_code]
    print(f"  DB teams: {len(league_teams)}")

    sofa_teams = await get_sofascore_teams(
        api, league["sofascore_league_id"], league["sofascore_season_id"]
    )

    if not sofa_teams:
        print(f"  No teams from SofaScore")
        return 0, 0, []

    print(f"  SofaScore teams: {len(sofa_teams)}")

    for sofa_team in sofa_teams:
        sofa_name = sofa_team["name"]
        sofa_id = sofa_team["id"]
        sofa_lower = sofa_name.lower()

        if sofa_lower in existing_aliases:
            already_exists += 1
            continue

        team_id, db_team_name, match_type = fuzzy_match_team(
            sofa_name, sofa_id, league_teams, league_code
        )

        if team_id:
            matched += 1
            status = "OK" if match_type == "exact" else f"OK [{match_type}]"
            print(f"    [{status}] {sofa_name} -> {db_team_name}")

            if not dry_run:
                insert_team_alias(conn, team_id, sofa_name, sofa_id)
        else:
            unmatched_teams.append(sofa_name)
            print(f"    [NO MATCH] {sofa_name} (ID: {sofa_id})")

    return matched, already_exists, unmatched_teams


async def extract_all_aliases(dry_run: bool = False) -> None:
    """Main function to extract all team aliases."""
    from sofascore_wrapper.api import SofascoreAPI

    database_url = get_database_url()
    conn = psycopg2.connect(database_url)

    try:
        db_leagues = get_db_leagues_with_mapping(conn)
        db_teams = get_db_teams_by_league(conn)
        existing_aliases = get_existing_aliases(conn)

        print(f"Found {len(db_leagues)} leagues with SofaScore mapping")
        print(
            f"Found {sum(len(t) for t in db_teams.values())} teams in DB across {len(db_teams)} leagues"
        )
        print(f"Found {len(existing_aliases)} existing SofaScore aliases")

        if dry_run:
            print("\n*** DRY RUN - No changes will be made ***")

        api = SofascoreAPI()
        try:
            total_matched = 0
            total_exists = 0
            all_unmatched = []

            for league in db_leagues:
                matched, exists, unmatched = await process_league(
                    api, conn, league, db_teams, existing_aliases, dry_run
                )
                total_matched += matched
                total_exists += exists
                all_unmatched.extend([(league["league_code"], t) for t in unmatched])

            print(f"\n{'=' * 60}")
            print(f"SUMMARY")
            print(f"{'=' * 60}")
            print(f"New aliases to insert: {total_matched}")
            print(f"Already existing: {total_exists}")
            print(f"Unmatched teams: {len(all_unmatched)}")

            if all_unmatched:
                print(f"\nUnmatched teams by league:")
                unmatched_by_league: dict[str, list[str]] = {}
                for league_code, team in all_unmatched:
                    if league_code not in unmatched_by_league:
                        unmatched_by_league[league_code] = []
                    unmatched_by_league[league_code].append(team)

                for league_code, teams in sorted(unmatched_by_league.items()):
                    print(f"  [{league_code}] {', '.join(teams)}")

            if not dry_run:
                conn.commit()
                print(f"\nCommitted {total_matched} aliases to database")
            else:
                print(f"\nDry run - no changes made")

        finally:
            await api.close()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all team aliases from SofaScore"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without inserting"
    )
    args = parser.parse_args()

    asyncio.run(extract_all_aliases(args.dry_run))


if __name__ == "__main__":
    main()
