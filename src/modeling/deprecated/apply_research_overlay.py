"""
Apply AI research overlay to model predictions.

Reads the latest ai_research_*.json output and applies:
- confidence_modifier: Adjusts probability
- final_research_verdict: CONFIRM/DOWNGRADE/REJECT

This is Layer 3 of the 3-Layer syndicate architecture:
- Layer 1: Poisson/Dixon-Coles baseline
- Layer 2: Situational ML residual (TODO)
- Layer 3: AI Research overlay (this script)
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportDeprecated=false, reportMissingTypeArgument=false

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

# --- Configuration ---

MODEL_NAME = "market_outcome_gbm"
MODEL_VERSION = "fixtures_first_prematch_v1"

# Verdict multipliers for confidence_modifier
VERDICT_MULTIPLIERS = {
    "CONFIRM": 1.0,      # Use full modifier
    "DOWNGRADE": 0.5,    # Reduce modifier impact
    "REJECT": -1.0,      # Invert (reduce confidence)
}


def get_latest_research_json() -> Path | None:
    """Find the most recent ai_research_*.json file."""
    data_dir = ROOT_DIR / "data" / "v1"
    jsons = list(data_dir.glob("ai_research_*.json"))
    if not jsons:
        return None
    return sorted(jsons, key=os.path.getmtime, reverse=True)[0]


def load_research_results(path: Path) -> list[dict]:
    """Load research results from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_fixture_id_for_match(match_name: str, conn) -> int | None:
    """
    Look up fixture_id from match name.
    Match name format: "Home Team vs Away Team"
    """
    parts = match_name.split(" vs ")
    if len(parts) != 2:
        return None
    
    home_team, away_team = parts[0].strip(), parts[1].strip()
    
    query = """
    SELECT f.fixture_id
    FROM fixtures f
    JOIN teams th ON th.team_id = f.home_team_id
    JOIN teams ta ON ta.team_id = f.away_team_id
    WHERE LOWER(th.team_name) LIKE LOWER(%s)
      AND LOWER(ta.team_name) LIKE LOWER(%s)
      AND f.status = 'scheduled'
    ORDER BY f.match_datetime_utc ASC
    LIMIT 1
    """
    
    with conn.cursor() as cur:
        cur.execute(query, (f"%{home_team}%", f"%{away_team}%"))
        row = cur.fetchone()
        return row[0] if row else None


def apply_research_overlay(
    research_results: list[dict],
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Apply research overlay to predictions.
    
    Returns: (updated_count, skipped_count, rejected_count)
    """
    conn = connect_db()
    updated = 0
    skipped = 0
    rejected = 0
    
    try:
        for result in research_results:
            match_name = result.get("match_id", "")
            market = result.get("prediction_market", "")
            verdict = result.get("final_research_verdict", "CONFIRM")
            modifier = float(result.get("confidence_modifier", 0.0))
            original_conf = float(result.get("original_model_confidence", 0.5))
            
            # Look up fixture_id
            fixture_id = get_fixture_id_for_match(match_name, conn)
            if not fixture_id:
                print(f"  [SKIP] Could not find fixture for: {match_name}")
                skipped += 1
                continue
            
            # Apply verdict multiplier
            effective_modifier = modifier * VERDICT_MULTIPLIERS.get(verdict, 0.0)
            
            # Calculate adjusted probability
            # For REJECT, we reduce confidence toward 0.5 (uncertainty)
            if verdict == "REJECT":
                adjusted_p = original_conf - (original_conf - 0.5) * 0.5
            else:
                adjusted_p = original_conf + effective_modifier
            
            # Clamp to valid probability range
            adjusted_p = max(0.01, min(0.99, adjusted_p))
            
            if dry_run:
                print(f"  [DRY RUN] {match_name} ({market}): {original_conf:.3f} -> {adjusted_p:.3f} [{verdict}]")
                updated += 1
                continue
            
            # Update prediction in DB
            update_query = """
            UPDATE predictions
            SET 
                p_model = %s,
                metadata_json = metadata_json || %s::jsonb,
                created_at = NOW()
            WHERE fixture_id = %s
              AND market_code = %s
              AND model_name = %s
              AND model_version = %s
            """
            
            overlay_metadata = {
                "research_overlay": {
                    "verdict": verdict,
                    "confidence_modifier": modifier,
                    "effective_modifier": effective_modifier,
                    "original_p": original_conf,
                    "adjusted_p": adjusted_p,
                    "applied_at": datetime.utcnow().isoformat(),
                }
            }
            
            with conn.cursor() as cur:
                cur.execute(
                    update_query,
                    (
                        adjusted_p,
                        json.dumps(overlay_metadata),
                        fixture_id,
                        market,
                        model_name,
                        model_version,
                    ),
                )
                if cur.rowcount > 0:
                    print(f"  [UPDATED] {match_name} ({market}): {original_conf:.3f} -> {adjusted_p:.3f} [{verdict}]")
                    updated += 1
                    if verdict == "REJECT":
                        rejected += 1
                else:
                    print(f"  [SKIP] No prediction found for {match_name} ({market})")
                    skipped += 1
        
        conn.commit()
    finally:
        conn.close()
    
    return updated, skipped, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply AI research overlay to predictions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    parser.add_argument("--file", type=Path, default=None, help="Specific research JSON file (default: latest)")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model name to update")
    parser.add_argument("--model-version", type=str, default=MODEL_VERSION, help="Model version to update")
    args = parser.parse_args()
    
    # Find research file
    if args.file:
        research_path = args.file
    else:
        research_path = get_latest_research_json()
    
    if not research_path:
        print("No AI research results found. Run ai_research_pass.py first.")
        return
    
    print(f"Loading research from: {research_path.name}")
    results = load_research_results(research_path)
    print(f"Found {len(results)} research results\n")
    
    if args.dry_run:
        print("=== DRY RUN MODE (no changes will be made) ===\n")
    
    updated, skipped, rejected = apply_research_overlay(
        research_results=results,
        model_name=args.model_name,
        model_version=args.model_version,
        dry_run=args.dry_run,
    )
    
    print(f"\n=== Summary ===")
    print(f"Updated: {updated}")
    print(f"Skipped: {skipped}")
    print(f"Rejected: {rejected}")
    if not args.dry_run and updated > 0:
        print("\nPredictions updated with research overlay.")


if __name__ == "__main__":
    main()
