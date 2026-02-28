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

# --- Pydantic Schema for Senior Data Science Grade JSON ---

class PersonnelKPIs(BaseModel):
    absent_minutes_impact: float = Field(..., description="% of team total minutes missing due to injury/suspension")
    star_player_void: int = Field(..., description="1 if a non-replaceable key player is out, 0 otherwise")
    defensive_pillar_out: int = Field(..., description="1 if 2+ starting defenders are missing, 0 otherwise")
    details: str

class ScheduleKPIs(BaseModel):
    rest_delta_days: int = Field(..., description="Home rest days minus Away rest days")
    travel_distance_km: Optional[int] = None
    fixture_congestion_flag: int = Field(..., description="1 if congested, 0 otherwise")

class EnvironmentalKPIs(BaseModel):
    weather_wind_speed_ms: Optional[float] = None
    pitch_surface_type: Optional[str] = None
    heavy_pitch_flag: int = Field(..., description="1 if heavy/wet, 0 otherwise")

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
    confidence_modifier: float = Field(..., description="Suggested adjustment to model probability (e.g. +0.05)")
    final_research_verdict: Literal["CONFIRM", "DOWNGRADE", "REJECT"]

# --- Main Script Logic ---

def get_latest_slip_csv() -> Path | None:
    data_dir = ROOT_DIR / "data" / "v1"
    csvs = list(data_dir.glob("daily_slip_*.csv"))
    if not csvs:
        return None
    return sorted(csvs, key=os.path.getmtime, reverse=True)[0]

def run_ai_research():
    print(f"[{datetime.now().isoformat()}] Starting Senior-Grade AI Research Pass via Tavily...")
    
    csv_path = get_latest_slip_csv()
    if not csv_path:
        print("No daily slip CSV found to research. Run live_edge_detector.py first.")
        return
        
    print(f"Evaluating edges in: {csv_path.name}")
    
    # Sort by EV descending and pick edges with EV > 10%
    target_edges = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ev_str = row.get('ev', '0').strip('%')
                ev = float(ev_str) / 100 if '%' in row.get('ev', '') else float(ev_str)
                if ev > 0.10: # ONLY HIGH CONVICTION
                    target_edges.append(row)
            except (ValueError, KeyError):
                continue
                
    target_edges = sorted(target_edges, key=lambda x: float(x.get('ev', 0).strip('%')) if '%' in str(x.get('ev',0)) else float(x.get('ev', 0)), reverse=True)
                
    if not target_edges:
        print("No high-conviction edges (>10% EV) found to research.")
        return
        
    # CREDIT SAFETY VALVE: Limit to 10 research tasks per run
    MAX_TASKS = 10
    if len(target_edges) > MAX_TASKS:
        print(f"   -> [NOTE] Found {len(target_edges)} high-EV edges. Limiting to top {MAX_TASKS} to save credits.")
        target_edges = target_edges[:MAX_TASKS]
        
    print(f"Executing Selective AI Research (EV > 10%) for {len(target_edges)} matches...")
    print(f"Initializing Tavily...")

    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    client = TavilyClient(api_key=tavily_api_key)
    research_results = []
    
    # Define the schema for Tavily research output
    output_schema = {
        "properties": {
            "personnel": {
                "type": "object",
                "description": "Analysis of player absences and their impact on the team",
                "properties": {
                    "absent_minutes_impact": {"type": "number", "description": "Percentage of total season minutes represented by missing players"},
                    "star_player_void": {"type": "integer", "description": "1 if a crucial star player is missing, 0 otherwise"},
                    "defensive_pillar_out": {"type": "integer", "description": "1 if 2 or more starting defenders are missing, 0 otherwise"},
                    "details": {"type": "string", "description": "Specific names and roles of key missing players"}
                },
                "required": ["absent_minutes_impact", "star_player_void", "defensive_pillar_out", "details"]
            },
            "schedule": {
                "type": "object",
                "description": "Analysis of fixture congestion, rest days, and travel",
                "properties": {
                    "rest_delta_days": {"type": "integer", "description": "Difference in rest days (Home rest - Away rest)"},
                    "travel_distance_km": {"type": "integer", "description": "Estimated travel distance for the away team in km"},
                    "fixture_congestion_flag": {"type": "integer", "description": "1 if either team played in the last 4 days, 0 otherwise"}
                },
                "required": ["rest_delta_days", "travel_distance_km", "fixture_congestion_flag"]
            },
            "environmental": {
                "type": "object",
                "description": "Weather and pitch conditions",
                "properties": {
                    "weather_wind_speed_ms": {"type": "number", "description": "Estimated wind speed at kickoff in m/s"},
                    "pitch_surface_type": {"type": "string", "description": "Type of pitch (Natural, Hybrid, Artificial)"},
                    "heavy_pitch_flag": {"type": "integer", "description": "1 if heavy rain or poor pitch quality is expected, 0 otherwise"}
                },
                "required": ["weather_wind_speed_ms", "pitch_surface_type", "heavy_pitch_flag"]
            },
            "market": {
                "type": "object",
                "description": "Betting market sentiment and movement",
                "properties": {
                    "market_steam_status": {"type": "string", "enum": ["with", "against", "neutral"], "description": "Direction of odds movement relative to the pick"},
                    "clv_projection": {"type": "number", "description": "Projected closing line value percentage"}
                },
                "required": ["market_steam_status", "clv_projection"]
            },
            "referee": {
                "type": "object",
                "description": "Referee statistical profile",
                "properties": {
                    "name": {"type": "string", "description": "Name of the match referee"},
                    "yellow_card_avg": {"type": "number", "description": "Average yellow cards per game for this ref in this league"},
                    "penalty_tendency": {"type": "number", "description": "Average penalties awarded per game by this ref"}
                },
                "required": ["name", "yellow_card_avg", "penalty_tendency"]
            },
            "verdict_summary": {"type": "string", "description": "Executive summary of the research findings"},
            "confidence_modifier": {"type": "number", "description": "Recommended probability adjustment (e.g., -0.05)"},
            "final_research_verdict": {"type": "string", "enum": ["CONFIRM", "DOWNGRADE", "REJECT"], "description": "Final decision on the original prediction"}
        },
        "required": ["personnel", "schedule", "environmental", "market", "referee", "verdict_summary", "confidence_modifier", "final_research_verdict"]
    }

    for i, edge in enumerate(target_edges):
        match_name = edge['match']
        print(f"\n[{i+1}/{len(target_edges)}] Researching: {match_name} ({edge['market']})")
        
        prompt = f"""
        Analyze the upcoming football match: {match_name} on {edge['kickoff']}.
        Focus: {edge['market']} market, selecting {edge['selection']}.
        Original model confidence: {edge['prob']}.
        
        For flags (star_player_void, heavy_pitch_flag, etc.), use 1 for True and 0 for False.
        """
        
        try:
            print(f"   -> Initiating Tavily 'mini' Research (25 Credits)...")
            research_task = client.research(
                input=prompt,
                model="mini", 
                output_schema=output_schema
            )
            
            request_id = research_task.get("request_id")
            if not request_id: continue

            while True:
                time.sleep(15)
                res = client.get_research(request_id)
                status = res.get("status")
                
                if status == "completed":
                    content = res.get("content")
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                             if "```json" in content:
                                 content = json.loads(content.split("```json")[-1].split("```")[0].strip())
                    
                    content.update({
                        "match_id": match_name,
                        "prediction_market": edge['market'],
                        "prediction_selection": edge['selection'],
                        "original_model_confidence": float(edge['prob'])
                    })
                    
                    try:
                        validated_data = MatchResearchResult(**content)
                        print(f"   -> [SUCCESS] Verdict: {validated_data.final_research_verdict}")
                        research_results.append(validated_data.model_dump())
                    except ValidationError as ve:
                        print(f"   -> [SCHEMA ERROR] {ve}")
                    break
                elif status == "failed":
                    print(f"   -> [FAILURE] {res.get('error')}")
                    break
                else:
                    print(f"      Status: {status}...")

        except Exception as e:
            print(f"   -> [SYSTEM ERROR] {e}")

    # Final Output Generation
    if research_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = ROOT_DIR / "data" / "v1" / f"ai_research_{timestamp}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(research_results, f, indent=4)
        print(f"\n[{datetime.now().isoformat()}] Research Complete! Saved to {out_path.name}")
    else:
        print("\nNo research results generated.")

if __name__ == "__main__":
    run_ai_research()
