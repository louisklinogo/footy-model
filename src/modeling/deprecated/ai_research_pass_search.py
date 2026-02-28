import os
import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import from src
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- Pydantic Schema ---

class PersonnelKPIs(BaseModel):
    absent_minutes_impact: float
    star_player_void: int
    defensive_pillar_out: int
    details: str

class ScheduleKPIs(BaseModel):
    rest_delta_days: int
    travel_distance_km: Optional[int] = None
    fixture_congestion_flag: int

class EnvironmentalKPIs(BaseModel):
    weather_wind_speed_ms: Optional[float] = None
    pitch_surface_type: Optional[str] = None
    heavy_pitch_flag: int

class MarketKPIs(BaseModel):
    market_steam_status: Literal["with", "against", "neutral"]
    clv_projection: Optional[float] = None

class RefereeKPIs(BaseModel):
    name: str
    yellow_card_avg: Optional[float] = None
    penalty_tendency: Optional[float] = None

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
    final_research_verdict: Literal["CONFIRM", "DOWNGRADE", "REJECT"]

# --- Comparison Script Logic ---

def run_search_comparison():
    print(f"[{datetime.now().isoformat()}] Starting TAVILY SEARCH (3-Credit) Comparison...")
    
    # Target the same match for head-to-head
    match_name = "Barcelona vs Levante"
    edge = {
        "match": match_name,
        "kickoff": "2026-02-22",
        "market": "O/U 1.5",
        "selection": "OVER",
        "prob": 0.78
    }

    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    client = TavilyClient(api_key=tavily_api_key)

    # Use a prompt that enforces the JSON schema clearly
    prompt = f"""
    Research the football match: {match_name} on {edge['kickoff']}.
    Value Pick: {edge['selection']} in {edge['market']}.
    
    Return a structured JSON object with these EXACT keys:
    personnel: {{absent_minutes_impact: float, star_player_void: 0|1, defensive_pillar_out: 0|1, details: str}}
    schedule: {{rest_delta_days: int, travel_distance_km: int, fixture_congestion_flag: 0|1}}
    environmental: {{weather_wind_speed_ms: float, pitch_surface_type: str, heavy_pitch_flag: 0|1}}
    market: {{market_steam_status: "with"|"against"|"neutral", clv_projection: float}}
    referee: {{name: str, yellow_card_avg: float, penalty_tendency: float}}
    verdict_summary: str
    confidence_modifier: float
    final_research_verdict: "CONFIRM"|"DOWNGRADE"|"REJECT"
    """

    try:
        print(f"   -> Querying Tavily SEARCH + Advanced Answer (Cost: 3 Credits)...")
        response = client.search(
            query=f"{match_name} match preview injuries weather referee status",
            search_depth="advanced",
            include_answer="advanced", # Synthesized AI answer
            max_results=5
        )
        
        answer = response.get("answer", "")
        print(f"\n--- Raw AI Answer ---\n{answer}\n")

        # Try to parse the JSON from the answer
        try:
            if "```json" in answer:
                json_str = answer.split("```json")[-1].split("```")[0].strip()
            elif "```" in answer:
                json_str = answer.split("```")[-1].split("```")[0].strip()
            else:
                json_str = answer.strip()
            
            data = json.loads(json_str)
            
            # Enrich with metadata
            data["match_id"] = match_name
            data["prediction_market"] = edge['market']
            data["prediction_selection"] = edge['selection']
            data["original_model_confidence"] = edge['prob']
            
            # Validate
            validated = MatchResearchResult(**data)
            print(f"   -> [SUCCESS] Search comparison valid! Verdict: {validated.final_research_verdict}")
            
            with open(ROOT_DIR / "data" / "v1" / "comparison_search.json", "w") as f:
                json.dump(validated.model_dump(), f, indent=2)

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"   -> [FAIL] Parsing/Validation failed: {e}")

    except Exception as e:
        print(f"   -> [SYSTEM ERROR] {e}")

if __name__ == "__main__":
    run_search_comparison()
