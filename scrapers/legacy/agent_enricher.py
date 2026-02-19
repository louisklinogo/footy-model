import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from browser_use import Agent, Browser, ChatBrowserUse

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- Configuration ---
MATCH_ID = "lKNJm8ak"
LEAGUE_CODE = "E0"
API_KEY = "bu_8jMAXaXcmGzG0PIn7MS4ySgPVXspENUNR86ARiz_NYo"

async def main():
    # Initialize Browser (HEADLESS)
    browser = Browser()
    
    # Initialize LLM using BrowserUse native chat model
    llm = ChatBrowserUse(api_key=API_KEY)
    
    task = f"""
    Go to https://www.flashscore.com/match/{MATCH_ID}/#/match-summary
    
    1. Extract all match statistics from the 'Statistics' -> 'Match' tab.
    2. Extract all 'Over/Under' odds lines (0.5, 1.5, 2.5, etc.) with Over and Under values.
    3. Extract all 'Corners' Over/Under odds lines.
    4. Extract all 'Asian Handicap' lines and odds.
    5. Extract 'Both Teams to Score' odds (Yes/No).
    
    Return the data as a clean JSON object.
    """
    
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser
    )
    
    print(f"Starting Agentic Enrichment for {MATCH_ID}...")
    result = await agent.run()
    
    # Process and save result
    output_dir = Path(f"footy-model-v2/scrapers/data/premium/{LEAGUE_CODE}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{MATCH_ID}_agent.json", "w", encoding='utf-8') as f:
        json.dump({"agent_output": str(result.final_result())}, f, indent=4, ensure_ascii=False)
        
    print(f"Enrichment Complete for {MATCH_ID}. Result saved.")
    
    await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
