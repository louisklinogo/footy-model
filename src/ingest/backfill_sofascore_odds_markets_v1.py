"""
Backfill Sofascore odds markets into fixture_odds_markets.

Usage:
    python src/ingest/backfill_sofascore_odds_markets_v1.py --league E0 --limit 50
    python src/ingest/backfill_sofascore_odds_markets_v1.py --limit 10 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from src.db.db_utils import connect_db
from sofascore_wrapper.api import SofascoreAPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Sofascore 1X2 odds into fixture_odds_markets"
    )
    parser.add_argument(
        "--league", type=str, default=None, help="Optional league code filter"
    )
    parser.add_argument("--limit", type=int, default=50, help="Max fixtures to process")
    parser.add_argument(
        "--sleep-sec", type=float, default=1.0, help="Sleep between API calls"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Fetch data but do not write to DB"
    )
    parser.add_argument(
        "--resume-file", type=str, default=None, help="Path to resume log (JSONL)"
    )
    parser.add_argument(
        "--provider-id", type=int, default=1, help="Provider ID (default: Bet365 = 1)"
    )
    return parser.parse_args()


def parse_fractional(value: str) -> float | None:
    raw = value.strip()
    if not raw:
        return None
    if "/" in raw:
        num_str, den_str = raw.split("/", 1)
        try:
            num = float(num_str.strip())
            den = float(den_str.strip())
        except ValueError:
            return None
        if den <= 0:
            return None
        return 1.0 + (num / den)
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 1.0 else None


def parse_decimal(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def map_choice_name(raw_name: str) -> str | None:
    name = raw_name.strip().lower()
    if name in {"1", "home", "home team", "home win"}:
        return "home"
    if name in {"x", "draw", "tie"}:
        return "draw"
    if name in {"2", "away", "away team", "away win"}:
        return "away"
    if name in {"yes", "no"}:
        return name
    if name in {"1x", "1x2", "12", "x2"}:
        return {"1x": "home_draw", "12": "home_away", "x2": "draw_away"}.get(name)
    if "over" in name:
        return "over"
    if "under" in name:
        return "under"
    return None


def _normalize_label(raw_text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", raw_text.strip().lower())


def _market_team_names(market: dict[str, Any], side: str) -> list[str]:
    if side == "home":
        keys = (
            "homeTeamName",
            "homeTeam",
            "home",
            "homeCompetitor",
            "homeParticipant",
            "home_name",
        )
    else:
        keys = (
            "awayTeamName",
            "awayTeam",
            "away",
            "awayCompetitor",
            "awayParticipant",
            "away_name",
        )
    names: list[str] = []
    for key in keys:
        value = market.get(key)
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
            continue
        if isinstance(value, dict):
            for nested in ("name", "shortName", "displayName"):
                nested_value = value.get(nested)
                if isinstance(nested_value, str) and nested_value.strip():
                    names.append(nested_value.strip())
    return names


def _infer_ah_side_from_choice_name(
    choice_name: str, market: dict[str, Any]
) -> str | None:
    label = choice_name.strip().lower()
    if re.search(r"\b(home|home team|team 1|team1)\b", label):
        return "home"
    if re.search(r"\b(away|away team|team 2|team2)\b", label):
        return "away"

    normalized = _normalize_label(choice_name)
    if not normalized:
        return None

    for side in ("home", "away"):
        for team_name in _market_team_names(market, side):
            team_norm = _normalize_label(team_name)
            if team_norm and team_norm in normalized:
                return side
    return None


def _apply_ah_fallback_mapping(
    *,
    mapped: dict[str, dict[str, Any]],
    choices_filtered: list[dict[str, Any]],
    market: dict[str, Any],
) -> None:
    if {"home", "away"}.issubset(mapped.keys()):
        return

    for choice in choices_filtered:
        if choice in mapped.values():
            continue
        raw_name = str(choice.get("name", "")).strip()
        side = _infer_ah_side_from_choice_name(raw_name, market)
        if side and side not in mapped:
            mapped[side] = choice

    if {"home", "away"}.issubset(mapped.keys()):
        return

    if len(choices_filtered) != 2:
        return

    remaining = [choice for choice in choices_filtered if choice not in mapped.values()]
    if not remaining:
        return

    if "home" not in mapped:
        mapped["home"] = remaining[0]
        remaining = remaining[1:]
    if "away" not in mapped and remaining:
        mapped["away"] = remaining[0]


def extract_line(raw_text: str) -> float | None:
    if not raw_text:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", raw_text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _filter_choices(
    choices: list[Any], provider_id: int | None
) -> list[dict[str, Any]]:
    has_provider_id = any("providerId" in c for c in choices if isinstance(c, dict))
    filtered: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if (
            provider_id is not None
            and has_provider_id
            and choice.get("providerId") != provider_id
        ):
            continue
        filtered.append(choice)
    return filtered


def _build_price_map(
    choices: dict[str, dict[str, Any]], use_initial: bool
) -> dict[str, float] | None:
    prices: dict[str, float] = {}
    for key, choice in choices.items():
        raw_val = (
            choice.get("initialFractionalValue")
            if use_initial
            else choice.get("fractionalValue")
        )
        dec = None
        if raw_val is not None:
            if isinstance(raw_val, str):
                dec = parse_fractional(raw_val)
            else:
                dec = parse_decimal(raw_val)
        if dec is None:
            dec = parse_decimal(choice.get("decimalValue"))
        if dec is None or dec <= 1.0:
            return None
        prices[key] = float(dec)
    return prices


def normalize_prices(choices: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    latest = _build_price_map(choices, use_initial=False)
    if not latest:
        return None

    # Fallback to latest for opening if initial is missing/invalid
    opening = _build_price_map(choices, use_initial=True)
    if not opening:
        opening = latest

    inv_sum = sum(1.0 / p for p in latest.values())
    if inv_sum <= 0:
        return None
    implied = {k: (1.0 / v) / inv_sum for k, v in latest.items()}

    return {
        "opening": opening,
        "latest": latest,
        "implied_prob": implied,
    }


def map_market_code(market_name: str) -> str | None:
    name = market_name.strip().lower()

    def _team_side(label: str) -> str | None:
        if re.search(r"\b(home|home team|team 1|team1)\b", label):
            return "home"
        if re.search(r"\b(away|away team|team 2|team2)\b", label):
            return "away"
        return None

    if name in {"1x2", "match result", "full time"} or (
        "full time" in name and "1x2" not in name
    ):
        return "1x2"
    if "double chance" in name:
        return "dc"
    if "draw no bet" in name:
        return "dnb"
    if "both teams to score" in name:
        return "btts"
    if "corner" in name:
        side = _team_side(name)
        if side == "home":
            return "home_corners_ou"
        if side == "away":
            return "away_corners_ou"
        return "corners_ou"
    if "goal" in name and "total" in name:
        side = _team_side(name)
        if side == "home":
            return "home_ou"
        if side == "away":
            return "away_ou"
    if "match goals" in name or "total goals" in name:
        return "ou"
    if "asian handicap" in name:
        return "ah"
    if "card" in name:
        return "cards_ou"
    return None


def extract_markets(
    odds_payload: dict[str, Any], provider_id: int | None
) -> list[dict[str, Any]]:
    markets = odds_payload.get("markets") or []
    if not isinstance(markets, list):
        return []

    results: list[dict[str, Any]] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        market_name = str(market.get("marketName", "") or "").strip()
        if not market_name:
            continue
        market_code = map_market_code(market_name)
        if not market_code:
            continue

        choices_raw = market.get("choices") or []
        if not isinstance(choices_raw, list) or not choices_raw:
            continue
        choices_filtered = _filter_choices(choices_raw, provider_id)
        if not choices_filtered:
            continue

        mapped: dict[str, dict[str, Any]] = {}
        for choice in choices_filtered:
            name_raw = str(choice.get("name", "")).strip()
            key = map_choice_name(name_raw)
            if key is None and market_code == "ah":
                key = _infer_ah_side_from_choice_name(name_raw, market)
            if not key:
                continue
            mapped[key] = choice

        if market_code == "ah":
            _apply_ah_fallback_mapping(
                mapped=mapped, choices_filtered=choices_filtered, market=market
            )

        if market_code == "1x2" and not {"home", "draw", "away"}.issubset(
            mapped.keys()
        ):
            continue
        if market_code in {"dc"} and not {
            "home_draw",
            "home_away",
            "draw_away",
        }.issubset(mapped.keys()):
            continue
        if market_code in {"dnb"} and not {"home", "away"}.issubset(mapped.keys()):
            continue
        if market_code in {"btts"} and not {"yes", "no"}.issubset(mapped.keys()):
            continue
        if market_code in {
            "ou",
            "home_ou",
            "away_ou",
            "corners_ou",
            "home_corners_ou",
            "away_corners_ou",
            "cards_ou",
        } and not {"over", "under"}.issubset(mapped.keys()):
            continue
        if market_code == "ah" and not {"home", "away"}.issubset(mapped.keys()):
            continue

        line_num = None
        line_text = None
        if market_code in {
            "ou",
            "home_ou",
            "away_ou",
            "corners_ou",
            "home_corners_ou",
            "away_corners_ou",
            "cards_ou",
            "ah",
        }:
            choice_group = market.get("choiceGroup")
            if choice_group is not None:
                line_num = extract_line(str(choice_group))
            if line_num is None:
                for choice in mapped.values():
                    line_num = extract_line(str(choice.get("name", "")))
                    if line_num is not None:
                        break
            if line_num is None:
                line_num = extract_line(market_name)
            if line_num is None:
                line_text = market_name

        normalized = normalize_prices(mapped)
        if not normalized:
            continue

        results.append(
            {
                "market_code": market_code,
                "line_num": line_num,
                "line_text": line_text,
                "prices_opening": normalized["opening"],
                "prices_latest": normalized["latest"],
                "implied_prob": normalized["implied_prob"],
                "raw": market,
            }
        )

    return results


def load_resume_set(resume_file: str | None) -> set[int]:
    if not resume_file:
        return set()
    path = Path(resume_file)
    if not path.exists():
        return set()
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        fid = payload.get("fixture_id")
        if isinstance(fid, int):
            seen.add(fid)
    return seen


def append_resume(resume_file: str | None, fixture_id: int, status: str) -> None:
    if not resume_file:
        return
    path = Path(resume_file)
    payload = {
        "fixture_id": fixture_id,
        "status": status,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def fetch_targets(
    league: str | None, limit: int, resume_set: set[int], provider: str
) -> list[dict[str, Any]]:
    # Exclude fixtures that already have a closing row OR that are marked as
    # permanently failed (404) in odds_fetch_failures.
    query = """
        SELECT f.fixture_id, f.sofascore_id, f.match_datetime_utc
        FROM fixtures f
        WHERE f.status = 'ft'
          AND f.sofascore_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1
            FROM fixture_odds_markets fom
            WHERE fom.fixture_id = f.fixture_id
              AND fom.provider = %s
              AND fom.market_code = '1x2'
              AND fom.snapshot_type = 'closing'
          )
          AND NOT EXISTS (
            SELECT 1
            FROM odds_fetch_failures off2
            WHERE off2.fixture_id = f.fixture_id
              AND off2.provider = %s
          )
    """
    params: list[Any] = [provider, provider]
    if league:
        query += " AND f.league_code = %s"
        params.append(league)
    # Newest first: recent fixtures are more likely to have odds on Sofascore;
    # old fixtures with no odds will only be reached after fresh ones are exhausted.
    query += " ORDER BY f.match_datetime_utc DESC LIMIT %s"
    params.append(limit)

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            cols = [desc[0] for desc in (cur.description or [])]
            out = [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()

    if resume_set:
        out = [row for row in out if int(row["fixture_id"]) not in resume_set]
    return out


def _insert_not_available(fixture_id: int, provider: str, match_dt: Any) -> None:
    """Record a permanent 404 failure so fetch_targets never retries this fixture."""
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO odds_fetch_failures (fixture_id, provider, reason)
                VALUES (%s, %s, '404')
                ON CONFLICT (fixture_id, provider) DO NOTHING
                """,
                (fixture_id, provider),
            )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


async def run_backfill(args: argparse.Namespace) -> int:
    provider = "sofascore"
    tracked_team_codes = ["home_ou", "away_ou", "home_corners_ou", "away_corners_ou"]
    dry_seen_counts = {code: 0 for code in tracked_team_codes}
    resume_set = load_resume_set(args.resume_file)
    targets = fetch_targets(args.league, args.limit, resume_set, provider)

    if not targets:
        print("No fixtures found for odds backfill.")
        return 0

    api = SofascoreAPI()
    try:
        for row in targets:
            fixture_id = int(row["fixture_id"])
            ss_id = row["sofascore_id"]
            match_dt = row["match_datetime_utc"]
            if match_dt is None:
                append_resume(args.resume_file, fixture_id, "missing_match_datetime")
                continue

            print(f"Fixture {fixture_id} (SS: {ss_id})")
            try:
                odds_payload = await api._get(f"/event/{ss_id}/odds/1/all")
            except Exception as exc:
                err_str = str(exc)
                print(f"  Error fetching odds: {exc}")
                # On 404 the fixture will never have odds — write a sentinel row so
                # fetch_targets() permanently excludes it and we don't loop forever.
                is_404 = "404" in err_str
                if is_404:
                    _insert_not_available(fixture_id, provider, match_dt)
                    print(
                        f"  Marked as not_available (404 — will skip in future batches)."
                    )
                append_resume(
                    args.resume_file,
                    fixture_id,
                    "fetch_error_404" if is_404 else "fetch_error",
                )
                await asyncio.sleep(args.sleep_sec)
                continue

            markets = extract_markets(odds_payload or {}, args.provider_id)
            if not markets:
                print("  No supported markets found.")
                append_resume(args.resume_file, fixture_id, "no_market")
                await asyncio.sleep(args.sleep_sec)
                continue

            odds_rows = []
            for market in markets:
                odds_json = {
                    "provider": provider,
                    "provider_id": args.provider_id,
                    "market_code": market["market_code"],
                    "prices_opening": market["prices_opening"],
                    "prices_latest": market["prices_latest"],
                    "implied_prob": market["implied_prob"],
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "raw": market["raw"],
                }
                odds_rows.append(
                    (
                        fixture_id,
                        provider,
                        args.provider_id,
                        market["market_code"],
                        market["line_num"],
                        market["line_text"],
                        json.dumps(odds_json),
                        match_dt,
                        "closing",
                    )
                )

            if args.dry_run:
                sample = markets[0]
                market_codes = sorted({m["market_code"] for m in markets})
                market_code_set = set(market_codes)
                sample_lines = [
                    f"{m['market_code']}:{m['line_num']}"
                    for m in markets
                    if m["line_num"] is not None
                ][:6]
                for code in tracked_team_codes:
                    if code in market_code_set:
                        dry_seen_counts[code] += 1
                missing_team_codes = [
                    code for code in tracked_team_codes if code not in market_code_set
                ]
                print(
                    f"  [DRY RUN] markets={len(markets)} codes={market_codes} "
                    f"sample={sample['market_code']} odds={sample['prices_latest']} "
                    f"lines={sample_lines}"
                )
                if missing_team_codes:
                    print(f"  [DRY RUN] missing_team_codes={missing_team_codes}")
                append_resume(args.resume_file, fixture_id, "dry_run")
                await asyncio.sleep(args.sleep_sec)
                continue

            conn = connect_db()
            try:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO fixture_odds_markets (
                            fixture_id,
                            provider,
                            provider_id,
                            market_code,
                            line_num,
                            line_text,
                            odds_json,
                            snapshot_time_utc,
                            snapshot_type
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT ON CONSTRAINT uq_fixture_odds_markets DO UPDATE SET
                            odds_json = EXCLUDED.odds_json,
                            created_at = NOW()
                        """,
                        odds_rows,
                    )
                conn.commit()
            finally:
                conn.close()

            print(
                f"  Inserted {len(odds_rows)} closing market snapshot(s) at {match_dt.isoformat()}"
            )
            append_resume(args.resume_file, fixture_id, "inserted")
            await asyncio.sleep(args.sleep_sec)
    finally:
        await api.close()

    if args.dry_run:
        total = len(targets)
        extracted_any = [code for code, count in dry_seen_counts.items() if count > 0]
        not_found = [code for code, count in dry_seen_counts.items() if count == 0]
        print(
            f"[DRY RUN] extracted_team_market_codes fixtures={total} codes={extracted_any}"
        )
        if not_found:
            print(
                f"[DRY RUN] team_market_codes_not_found fixtures={total} codes={not_found}"
            )

    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_backfill(args))


if __name__ == "__main__":
    raise SystemExit(main())
