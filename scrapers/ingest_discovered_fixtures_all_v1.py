import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scrapers.ingest_discovered_fixtures_v1 import ingest_league


REGISTRY_PATH = Path("scrapers/config/league_registry.json")


def load_enabled_leagues():
    leagues = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [l["league_code"] for l in leagues if l.get("enabled") and l.get("league_code")]


def main():
    parser = argparse.ArgumentParser(description="Ingest discovery fixtures for all enabled leagues")
    parser.add_argument(
        "--leagues",
        default=None,
        help="Comma-separated list of league codes to ingest (defaults to all enabled)",
    )
    args = parser.parse_args()

    if args.leagues:
        league_codes = [s.strip() for s in args.leagues.split(",") if s.strip()]
    else:
        league_codes = load_enabled_leagues()

    total = len(league_codes)
    results = []
    failed = []
    for i, code in enumerate(league_codes, 1):
        try:
            result = ingest_league(code, limit=None)
            results.append(result)
            print(f"[{i}/{total}] {code}: {result['inserted']} fixtures ingested, {result['skipped']} skipped")
        except Exception as exc:
            failed.append(code)
            print(f"[{i}/{total}] {code}: FAILED - {exc}", file=sys.stderr)

    print(f"\nDone: {len(results)}/{total} leagues succeeded.")
    if failed:
        print(f"Failed: {failed}", file=sys.stderr)


if __name__ == "__main__":
    if not (os.getenv("DATABASE_URL") or os.getenv("DEV_DATABASE_URL") or os.getenv("PROD_DATABASE_URL")):
        raise SystemExit(
            "Missing DATABASE_URL (or DEV_DATABASE_URL / PROD_DATABASE_URL). Set DATABASE_URL before running this script."
        )
    main()
