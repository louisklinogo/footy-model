"""
AI Research Pass using Exa API (alternative to Tavily).

Layer 3 of the 3-Layer syndicate architecture.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from exa_py import Exa

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class PersonnelKPIs(BaseModel):
    absent_minutes_impact: float
    star_player_void: int
    defensive_pillar_out: int
    details: str


class ScheduleKPIs(BaseModel):
    rest_delta_days: int
    travel_distance_km: int | None = None
    fixture_congestion_flag: int


class EnvironmentalKPIs(BaseModel):
    weather_wind_speed_ms: float | None = None
    pitch_surface_type: str | None = None
    heavy_pitch_flag: int


class MarketKPIs(BaseModel):
    market_steam_status: str
    clv_projection: float | None = None


class RefereeKPIs(BaseModel):
    name: str
    yellow_card_avg: float | None = None
    penalty_tendency: float | None = None


class MatchResearchResult(BaseModel):
    match_id: str
    prediction_market: str
    prediction_selection: str
    original_model_confidence: float
    personnel: PersonnelKPIs
    schedule: ScheduleKPIs
    environmental: EnvironmentalKPIs
    market: MarketKPIs
    referee: RefereeKPIs
    verdict_summary: str
    confidence_modifier: float
    final_research_verdict: str


def call_exa_research(prompt: str) -> dict[str, Any]:
    exa = Exa(api_key=os.getenv("EXA_API_KEY"))

    research = exa.research.create(
        instructions=prompt,
        model="exa-research",
        output_schema=MatchResearchResult
    )

    result = exa.research.poll_until_finished(
        research.research_id,
        output_schema=MatchResearchResult,
        poll_interval=3000,
        timeout_ms=300000
    )

    return result.parsed_output


def get_latest_slip_csv() -> Path | None:
    data_dir = ROOT_DIR / "data" / "v1"
    csvs = list(data_dir.glob("daily_slip_*.csv"))
    if not csvs:
        return None
    return sorted(csvs, key=os.path.getmtime, reverse=True)[0]


def run_ai_research_exa(limit: int = 3):
    print(f"[{datetime.now().isoformat()}] Starting AI Research Pass via Exa...")

    csv_path = get_latest_slip_csv()
    if not csv_path:
        print("No daily slip CSV found.")
        return

    print(f"Evaluating edges in: {csv_path.name}")

    target_edges = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ev = float(row.get("ev", 0))
                if ev > 0.10:
                    target_edges.append(row)
            except (ValueError, KeyError):
                continue

    target_edges = sorted(
        target_edges, key=lambda x: float(x.get("ev", 0)), reverse=True
    )

    if not target_edges:
        print("No high-conviction edges (>10% EV) found.")
        return

    if len(target_edges) > limit:
        print(f"   -> Limiting to top {limit} edges")
        target_edges = target_edges[:limit]

    print(f"Researching {len(target_edges)} edges...")

    research_results = []

    for i, edge in enumerate(target_edges):
        match_name = edge["match"]
        print(f"\n[{i + 1}/{len(target_edges)}] {match_name} ({edge['market']})")

        prompt = f"""
You are a football betting analyst. Analyze this match and return a structured verdict.

Match: {match_name} on {edge['kickoff']}
Market: {edge['market']}, Selection: {edge['selection']}
Model confidence: {edge['prob']}

Return a JSON object with:
- personnel: absent_minutes_impact (0-1), star_player_void (0/1), defensive_pillar_out (0/1), details
- schedule: rest_delta_days, travel_distance_km, fixture_congestion_flag (0/1)
- environmental: weather_wind_speed_ms, pitch_surface_type, heavy_pitch_flag (0/1)
- market: market_steam_status ("with"/"against"/"neutral"), clv_projection
- referee: name, yellow_card_avg, penalty_tendency
- verdict_summary: 1-2 sentence analysis
- confidence_modifier: float adjustment (-0.1 to +0.1)
- final_research_verdict: EXACTLY one of: CONFIRM, DOWNGRADE, or REJECT

CONFIRM: Strong signal, proceed with bet
DOWNGRADE: Mixed signals, reduce stake
REJECT: Negative signals, do not bet
"""

        try:
            result = call_exa_research(prompt)
            # result is already a MatchResearchResult Pydantic model
            # Convert to dict and add extra fields
            result_dict = result.model_dump()
            result_dict["match_id"] = match_name
            result_dict["prediction_market"] = edge["market"]
            result_dict["prediction_selection"] = edge["selection"]
            result_dict["original_model_confidence"] = float(edge["prob"])

            validated = MatchResearchResult(**result_dict)
            print(f"   -> Verdict: {validated.final_research_verdict}")
            research_results.append(validated.model_dump())

        except ValidationError as e:
            print(f"   -> Schema error: {e}")
        except Exception as e:
            print(f"   -> Error: {e}")

    if research_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ROOT_DIR / "data" / "v1" / f"ai_research_exa_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(research_results, f, indent=2)
        print(f"\nSaved to {out_path.name}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    run_ai_research_exa(limit=args.limit)
