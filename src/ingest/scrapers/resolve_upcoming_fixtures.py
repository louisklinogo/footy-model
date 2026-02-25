"""
Resolve upcoming_fixtures rows to Flashscore IDs.

Workflow:
1) Ensures upcoming_fixtures has flashscore_id / flashscore_url / status columns.
2) First pass: loads upcoming IDs from data/scraper/upcoming_ids_*.json.
3) Second pass: for unresolved rows, tries historical IDs from data/scraper/match_ids_*.json.
4) Matches by league_code + date + normalized team names (exact + fuzzy fallback).
5) Updates upcoming_fixtures with flashscore_id and flashscore_url.
6) Marks unresolved past-dated rows as status='postponed'.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import psycopg2
import psycopg2.extras


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


IDS_DIR = Path("data/scraper")
TEAM_MAP_PATH = Path("data/team_mappings.json")


def norm(s: str) -> str:
    x = (s or "").strip().lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9 ]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def norm_team_match(s: str) -> str:
    x = norm(s)
    replacements = {
        "queens park rangers": "qpr",
        "west bromwich albion": "west brom",
        "sheffield united": "sheffield utd",
        "oxford united": "oxford utd",
        "newcastle united": "newcastle utd",
        "manchester united": "manchester utd",
        "leeds united": "leeds utd",
        "united": "utd",
        " city": "",
        " afc": "",
        " fc": "",
    }
    for a, b in replacements.items():
        x = x.replace(a, b)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_team_match(a), norm_team_match(b)).ratio()


def parse_fs_date(v: str) -> Optional[date]:
    # Handles values like: "18.02. 20:00"
    try:
        dpart = v.split(" ")[0]
        bits = [b for b in dpart.split(".") if b]
        day = int(bits[0])
        month = int(bits[1])
        year = 2025 if month >= 7 else 2026
        return date(year, month, day)
    except Exception:
        return None


def load_aliases() -> Dict[str, Set[str]]:
    aliases: Dict[str, Set[str]] = {}
    if not TEAM_MAP_PATH.exists():
        return aliases

    mapping = json.loads(TEAM_MAP_PATH.read_text(encoding="utf-8"))
    reverse = {v: k for k, v in mapping.items()}
    names = set(mapping.keys()) | set(mapping.values())

    for n in names:
        can = norm(n)
        opts = {can}
        if n in mapping:
            opts.add(norm(mapping[n]))
        if n in reverse:
            opts.add(norm(reverse[n]))
        aliases[can] = opts
    return aliases


def variants(name: str, aliases: Dict[str, Set[str]]) -> Set[str]:
    can = norm(name)
    return aliases.get(can, {can})


def load_ids(prefix: str, league_filter: Optional[str]) -> List[Tuple[str, date, str, str, str]]:
    rows: List[Tuple[str, date, str, str, str]] = []
    for fp in IDS_DIR.glob(f"{prefix}_*.json"):
        league_code = fp.stem.replace(f"{prefix}_", "")
        if league_filter and league_code != league_filter:
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for item in data:
            fsid = item.get("id")
            dt = parse_fs_date(item.get("date", ""))
            home = item.get("home")
            away = item.get("away")
            if fsid and dt and home and away:
                rows.append((league_code, dt, str(home), str(away), str(fsid)))
    return rows


def ensure_columns(cur) -> None:
    cur.execute("ALTER TABLE upcoming_fixtures ADD COLUMN IF NOT EXISTS flashscore_id TEXT")
    cur.execute("ALTER TABLE upcoming_fixtures ADD COLUMN IF NOT EXISTS flashscore_url TEXT")
    cur.execute("ALTER TABLE upcoming_fixtures ADD COLUMN IF NOT EXISTS status TEXT")
    cur.execute("ALTER TABLE upcoming_fixtures ADD COLUMN IF NOT EXISTS match_datetime_utc TIMESTAMPTZ")
    cur.execute(
        """
        UPDATE upcoming_fixtures
        SET status = CASE
            WHEN status IS NULL THEN 'scheduled'
            WHEN LOWER(status) = 'stale_needs_settlement' THEN 'postponed'
            WHEN LOWER(status) IN ('scheduled', 'live', 'ft', 'postponed', 'cancelled', 'abandoned') THEN LOWER(status)
            ELSE 'scheduled'
        END
        """
    )
    cur.execute(
        """
        UPDATE upcoming_fixtures
        SET match_datetime_utc = ((match_date::text || ' ' || match_time::text)::timestamp AT TIME ZONE 'UTC')
        WHERE match_datetime_utc IS NULL
          AND match_date IS NOT NULL
          AND match_time IS NOT NULL
          AND match_time::text ~ '^[0-2][0-9]:[0-5][0-9]$'
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_upcoming_flashscore_id ON upcoming_fixtures(flashscore_id)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_upcoming_status_match_datetime_utc ON upcoming_fixtures(status, match_datetime_utc)"
    )


def build_bucket(rows: List[Tuple[str, date, str, str, str]]) -> Dict[Tuple[str, date], List[Tuple[str, str, str]]]:
    bucket: Dict[Tuple[str, date], List[Tuple[str, str, str]]] = {}
    for lg, dt, h, a, fsid in rows:
        bucket.setdefault((lg, dt), []).append((h, a, fsid))
    return bucket


def match_from_candidates(
    home: str,
    away: str,
    candidates: List[Tuple[str, str, str]],
    aliases: Dict[str, Set[str]],
) -> Tuple[Optional[str], bool]:
    """Return (fsid, ambiguous)."""
    hv = variants(home, aliases)
    av = variants(away, aliases)

    exact_hits = []
    for ch, ca, fsid in candidates:
        if norm(ch) in hv and norm(ca) in av:
            exact_hits.append(fsid)

    if len(exact_hits) == 1:
        return exact_hits[0], False
    if len(exact_hits) > 1:
        return None, True

    best = None
    best_score = 0.0
    second_score = 0.0
    for ch, ca, fsid in candidates:
        score = (sim(home, ch) + sim(away, ca)) / 2.0
        if score > best_score:
            second_score = best_score
            best_score = score
            best = fsid
        elif score > second_score:
            second_score = score

    if best is not None and best_score >= 0.80 and (best_score - second_score) >= 0.05:
        return best, False

    return None, False


def resolve(cur, league_filter: Optional[str]) -> Dict[str, int]:
    aliases = load_aliases()
    upcoming_rows = load_ids("upcoming_ids", league_filter)
    historical_rows = load_ids("match_ids", league_filter)

    upcoming_bucket = build_bucket(upcoming_rows)
    historical_bucket = build_bucket(historical_rows)

    if league_filter:
        cur.execute(
            """
            SELECT id, league_code, match_date, home_team, away_team
            FROM upcoming_fixtures
            WHERE league_code = %s
            """,
            (league_filter,),
        )
    else:
        cur.execute(
            """
            SELECT id, league_code, match_date, home_team, away_team
            FROM upcoming_fixtures
            """
        )
    up_rows = cur.fetchall()

    updates: List[Tuple[str, str, int]] = []
    stale_updates: List[Tuple[int]] = []

    matched = 0
    matched_from_historical = 0
    unmatched = 0
    ambiguous = 0
    stale_marked = 0
    today = date.today()

    for uid, lg, dt, home, away in up_rows:
        # Pass 1: upcoming fixtures source
        cands = upcoming_bucket.get((lg, dt), [])
        if cands:
            fsid, amb = match_from_candidates(home, away, cands, aliases)
            if fsid:
                updates.append((fsid, f"https://www.flashscore.com/match/{fsid}/", uid))
                matched += 1
                continue
            if amb:
                ambiguous += 1
                continue

        # Pass 2: historical results source
        cands_hist = historical_bucket.get((lg, dt), [])
        if cands_hist:
            fsid, amb = match_from_candidates(home, away, cands_hist, aliases)
            if fsid:
                updates.append((fsid, f"https://www.flashscore.com/match/{fsid}/", uid))
                matched += 1
                matched_from_historical += 1
                continue
            if amb:
                ambiguous += 1
                continue

        unmatched += 1
        if dt <= today:
            stale_updates.append((uid,))
            stale_marked += 1

    if updates:
        psycopg2.extras.execute_batch(
            cur,
            """
            UPDATE upcoming_fixtures
            SET flashscore_id = %s,
                flashscore_url = %s,
                updated_at = NOW(),
                status = CASE
                    WHEN status IS NULL THEN 'scheduled'
                    WHEN LOWER(status) = 'postponed' THEN 'scheduled'
                    WHEN LOWER(status) IN ('scheduled', 'live', 'ft', 'cancelled', 'abandoned') THEN LOWER(status)
                    ELSE 'scheduled'
                END
            WHERE id = %s
            """,
            updates,
            page_size=500,
        )

    if stale_updates:
        psycopg2.extras.execute_batch(
            cur,
            """
            UPDATE upcoming_fixtures
            SET status = 'postponed',
                updated_at = NOW()
            WHERE id = %s
              AND (flashscore_id IS NULL OR flashscore_id = '')
            """,
            stale_updates,
            page_size=500,
        )

    return {
        "source_upcoming_rows": len(upcoming_rows),
        "source_historical_rows": len(historical_rows),
        "upcoming_rows": len(up_rows),
        "matched": matched,
        "matched_from_historical": matched_from_historical,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "stale_marked": stale_marked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve upcoming fixtures to flashscore_id")
    parser.add_argument("--league", help="Optional single league code (e.g. E0)")
    args = parser.parse_args()

    conn = connect_db()
    cur = conn.cursor()
    try:
        ensure_columns(cur)
        stats = resolve(cur, league_filter=args.league)
        conn.commit()
    finally:
        cur.close()
        conn.close()

    print("upcoming fixture resolution complete")
    for k, v in stats.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
