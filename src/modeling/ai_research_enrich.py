"""AI Research Enrichment for Fixtures.

Queries DB for fixtures with injury data, enriches with AI research:
- Impact analysis of missing players (categorical, not hallucinated)
- Market sentiment (odds movement, betting patterns)
- Recommendations grounded in web data

Usage:
    python src/modeling/ai_research_enrich.py --days 3
    python src/modeling/ai_research_enrich.py --fixture 12345
    python src/modeling/ai_research_enrich.py --dry-run
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.db_utils import connect_db


@dataclass(frozen=True)
class Options:
    days: int
    fixture_id: int | None
    limit: int
    dry_run: bool
    force: bool


def parse_args() -> Options:
    parser = argparse.ArgumentParser(description="AI research enrichment for fixtures")
    parser.add_argument("--days", type=int, default=3, help="Days ahead to research")
    parser.add_argument("--fixture", type=int, default=None, help="Specific fixture ID")
    parser.add_argument(
        "--limit", type=int, default=10, help="Max fixtures to research"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-research already researched"
    )
    ns = parser.parse_args()
    return Options(
        days=int(ns.days),
        fixture_id=int(ns.fixture) if ns.fixture else None,
        limit=int(ns.limit),
        dry_run=bool(ns.dry_run),
        force=bool(ns.force),
    )


@dataclass
class FixtureContext:
    fixture_id: int
    flashscore_id: str
    league_code: str
    home_team: str
    away_team: str
    match_datetime: datetime
    home_missing: list[dict]
    away_missing: list[dict]
    home_questionable: list[dict]
    away_questionable: list[dict]


def get_fixtures_to_research(options: Options) -> list[FixtureContext]:
    """Query DB for fixtures needing research."""
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            if options.fixture_id:
                cur.execute(
                    """
                    SELECT 
                        f.fixture_id, f.flashscore_id, f.league_code,
                        th.team_name as home_team, ta.team_name as away_team,
                        f.match_datetime_utc,
                        COALESCE(fa.home_missing, '[]'::jsonb) as home_missing,
                        COALESCE(fa.away_missing, '[]'::jsonb) as away_missing,
                        COALESCE(fa.home_questionable, '[]'::jsonb) as home_questionable,
                        COALESCE(fa.away_questionable, '[]'::jsonb) as away_questionable
                    FROM fixtures f
                    JOIN teams th ON f.home_team_id = th.team_id
                    JOIN teams ta ON f.away_team_id = ta.team_id
                    LEFT JOIN fixture_availability fa ON f.fixture_id = fa.fixture_id
                    LEFT JOIN fixture_research fr ON f.fixture_id = fr.fixture_id
                    WHERE f.fixture_id = %s
                    AND f.status = 'scheduled'
                    AND (fr.fixture_id IS NULL OR %s = TRUE)
                    """,
                    (options.fixture_id, options.force),
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        f.fixture_id, f.flashscore_id, f.league_code,
                        th.team_name as home_team, ta.team_name as away_team,
                        f.match_datetime_utc,
                        COALESCE(fa.home_missing, '[]'::jsonb) as home_missing,
                        COALESCE(fa.away_missing, '[]'::jsonb) as away_missing,
                        COALESCE(fa.home_questionable, '[]'::jsonb) as home_questionable,
                        COALESCE(fa.away_questionable, '[]'::jsonb) as away_questionable
                    FROM fixtures f
                    JOIN teams th ON f.home_team_id = th.team_id
                    JOIN teams ta ON f.away_team_id = ta.team_id
                    LEFT JOIN fixture_availability fa ON f.fixture_id = fa.fixture_id
                    LEFT JOIN fixture_research fr ON f.fixture_id = fr.fixture_id
                    WHERE f.status = 'scheduled'
                    AND f.match_datetime_utc > NOW()
                    AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
                    AND (
                        -- Has injury data to analyze
                        jsonb_array_length(COALESCE(fa.home_missing, '[]'::jsonb)) > 0
                        OR jsonb_array_length(COALESCE(fa.away_missing, '[]'::jsonb)) > 0
                    )
                    AND (fr.fixture_id IS NULL OR %s = TRUE)
                    ORDER BY f.match_datetime_utc ASC
                    LIMIT %s
                    """,
                    (options.days, options.force, options.limit),
                )
            rows = cur.fetchall()

        results = []
        for row in rows:
            results.append(
                FixtureContext(
                    fixture_id=row[0],
                    flashscore_id=row[1],
                    league_code=row[2],
                    home_team=row[3],
                    away_team=row[4],
                    match_datetime=row[5],
                    home_missing=row[6] if isinstance(row[6], list) else [],
                    away_missing=row[7] if isinstance(row[7], list) else [],
                    home_questionable=row[8] if isinstance(row[8], list) else [],
                    away_questionable=row[9] if isinstance(row[9], list) else [],
                )
            )
        return results
    finally:
        conn.close()


def build_prompt(ctx: FixtureContext) -> str:
    """Build the research prompt with pre-filled context."""
    home_missing_str = ""
    if ctx.home_missing:
        home_missing_str = "HOME TEAM MISSING PLAYERS:\n" + "\n".join(
            f"  - {p.get('name', 'Unknown')}: {p.get('reason', 'Unknown reason')}"
            for p in ctx.home_missing
        )

    away_missing_str = ""
    if ctx.away_missing:
        away_missing_str = "AWAY TEAM MISSING PLAYERS:\n" + "\n".join(
            f"  - {p.get('name', 'Unknown')}: {p.get('reason', 'Unknown reason')}"
            for p in ctx.away_missing
        )

    prompt = f"""
Analyze the impact of missing players for this upcoming football match.

MATCH: {ctx.home_team} vs {ctx.away_team}
LEAGUE: {ctx.league_code}
KICKOFF: {ctx.match_datetime.isoformat()}

{home_missing_str}
{away_missing_str}

TASK:
1. Research how important each missing player is to their team (starter vs squad, position, minutes)
2. Check for any news about these injuries or team reactions
3. Look for market reaction (odds movement, betting patterns)
4. Assess OVERALL impact using ONLY these categories: none, low, medium, high
5. Assess DEFENSIVE impact (goalkeepers, defenders) using: none, low, medium, high
6. Assess ATTACKING impact (forwards, creative midfielders) using: none, low, medium, high
7. Determine market sentiment: with_home, with_away, neutral, against_home, against_away
8. Provide a recommendation: CONFIRM (bet as planned), DOWNGRADE (reduce confidence), REJECT (avoid), NO_CHANGE

IMPORTANT:
- Use CATEGORIES only, not numbers - do not make up xG or goal impacts
- Base your assessment on ACTUAL web data you find
- If you cannot find reliable data, set research_confidence to "low"
- Be specific about WHY in reasoning field
"""
    return prompt


OUTPUT_SCHEMA = {
    "properties": {
        "injury_news": {
            "type": "string",
            "description": "Key news found about the injuries (quote or summary)",
        },
        "odds_movement": {
            "type": "string",
            "description": "Any odds movement related to team news",
        },
        "market_volume": {
            "type": "string",
            "description": "Betting volume patterns if found",
        },
        "overall_impact": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
            "description": "Overall impact of missing players",
        },
        "home_defensive_impact": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
            "description": "Impact on home team defense",
        },
        "home_attacking_impact": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
            "description": "Impact on home team attack",
        },
        "away_defensive_impact": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
            "description": "Impact on away team defense",
        },
        "away_attacking_impact": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
            "description": "Impact on away team attack",
        },
        "market_sentiment": {
            "type": "string",
            "enum": [
                "with_home",
                "with_away",
                "neutral",
                "against_home",
                "against_away",
            ],
            "description": "Where the market is leaning",
        },
        "sentiment_confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "How much market data was found",
        },
        "recommendation": {
            "type": "string",
            "enum": ["CONFIRM", "DOWNGRADE", "REJECT", "NO_CHANGE"],
            "description": "Final recommendation",
        },
        "reasoning": {
            "type": "string",
            "description": "Detailed reasoning for the recommendation",
        },
        "research_confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Overall confidence in the research (low if limited data found)",
        },
    },
    "required": [
        "overall_impact",
        "recommendation",
        "reasoning",
        "research_confidence",
    ],
}


def run_research(ctx: FixtureContext, dry_run: bool) -> dict[str, Any] | None:
    """Run Tavily research for a fixture."""
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("  [ERROR] TAVILY_API_KEY not set")
        return None

    client = TavilyClient(api_key=api_key)
    prompt = build_prompt(ctx)

    if dry_run:
        print(f"  DRY RUN - would research with prompt:\n{prompt[:200]}...")
        return {
            "overall_impact": "medium",
            "recommendation": "NO_CHANGE",
            "reasoning": "Dry run - no actual research",
            "research_confidence": "low",
        }

    try:
        print(f"  Initiating Tavily research...")
        research_task = client.research(
            input=prompt,
            model="mini",
            output_schema=OUTPUT_SCHEMA,
        )

        request_id = research_task.get("request_id")
        if not request_id:
            print(f"  [ERROR] No request_id returned")
            return None

        # Poll for completion
        max_wait = 120  # seconds
        waited = 0
        while waited < max_wait:
            time.sleep(10)
            waited += 10
            res = client.get_research(request_id)
            status = res.get("status")

            if status == "completed":
                content = res.get("content")
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        if "```json" in content:
                            content = json.loads(
                                content.split("```json")[-1].split("```")[0].strip()
                            )
                        else:
                            print(f"  [ERROR] Could not parse response")
                            return None
                content["tavily_request_id"] = request_id
                return content

            elif status == "failed":
                print(f"  [ERROR] Research failed: {res.get('error')}")
                return None

            else:
                print(f"  Status: {status} (waited {waited}s)...")

        print(f"  [ERROR] Timeout waiting for research")
        return None

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def save_research(ctx: FixtureContext, result: dict[str, Any], dry_run: bool) -> bool:
    """Save research results to DB."""
    if dry_run:
        print(f"  DRY RUN - would save research for fixture {ctx.fixture_id}")
        return True

    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO fixture_research (
                        fixture_id,
                        home_missing_json, away_missing_json,
                        home_questionable_json, away_questionable_json,
                        injury_news, odds_movement, market_volume,
                        overall_impact,
                        home_defensive_impact, home_attacking_impact,
                        away_defensive_impact, away_attacking_impact,
                        market_sentiment, sentiment_confidence,
                        recommendation, reasoning,
                        tavily_request_id, research_confidence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (fixture_id) DO UPDATE SET
                        injury_news = EXCLUDED.injury_news,
                        odds_movement = EXCLUDED.odds_movement,
                        market_volume = EXCLUDED.market_volume,
                        overall_impact = EXCLUDED.overall_impact,
                        home_defensive_impact = EXCLUDED.home_defensive_impact,
                        home_attacking_impact = EXCLUDED.home_attacking_impact,
                        away_defensive_impact = EXCLUDED.away_defensive_impact,
                        away_attacking_impact = EXCLUDED.away_attacking_impact,
                        market_sentiment = EXCLUDED.market_sentiment,
                        sentiment_confidence = EXCLUDED.sentiment_confidence,
                        recommendation = EXCLUDED.recommendation,
                        reasoning = EXCLUDED.reasoning,
                        tavily_request_id = EXCLUDED.tavily_request_id,
                        research_confidence = EXCLUDED.research_confidence,
                        researched_at = NOW()
                    """,
                    (
                        ctx.fixture_id,
                        json.dumps(ctx.home_missing),
                        json.dumps(ctx.away_missing),
                        json.dumps(ctx.home_questionable),
                        json.dumps(ctx.away_questionable),
                        result.get("injury_news"),
                        result.get("odds_movement"),
                        result.get("market_volume"),
                        result.get("overall_impact"),
                        result.get("home_defensive_impact"),
                        result.get("home_attacking_impact"),
                        result.get("away_defensive_impact"),
                        result.get("away_attacking_impact"),
                        result.get("market_sentiment"),
                        result.get("sentiment_confidence"),
                        result.get("recommendation"),
                        result.get("reasoning"),
                        result.get("tavily_request_id"),
                        result.get("research_confidence"),
                    ),
                )
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to save: {e}")
        return False
    finally:
        conn.close()


def main() -> int:
    options = parse_args()

    print("=" * 60)
    print("AI Research Enrichment")
    print("=" * 60)

    if options.dry_run:
        print("DRY RUN MODE - no API calls or DB changes")
        print()

    print(f"Finding fixtures to research (next {options.days} days)...")
    fixtures = get_fixtures_to_research(options)

    if not fixtures:
        print("No fixtures with injury data found needing research.")
        return 0

    print(f"Found {len(fixtures)} fixtures to research")
    print()

    success, failed = 0, 0
    for i, ctx in enumerate(fixtures):
        print(f"[{i + 1}/{len(fixtures)}] {ctx.home_team} vs {ctx.away_team}")
        print(
            f"  Home missing: {len(ctx.home_missing)}, Away missing: {len(ctx.away_missing)}"
        )

        result = run_research(ctx, options.dry_run)
        if result:
            if save_research(ctx, result, options.dry_run):
                print(f"  [OK] Recommendation: {result.get('recommendation', 'N/A')}")
                print(
                    f"       Impact: {result.get('overall_impact', 'N/A')}, Confidence: {result.get('research_confidence', 'N/A')}"
                )
                success += 1
            else:
                failed += 1
        else:
            failed += 1
        print()

    print("Summary:")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
