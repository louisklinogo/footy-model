import subprocess
import time

leagues = [
    "E0", "E1", "E2", "E3", "EC", 
    "SC0", "SC1", "SC2", "SC3", 
    "D1", "D2", "I1", "I2", 
    "SP1", "SP2", "F1", "F2", 
    "N1", "B1", "P1", "T1", "G1"
]

for league in leagues:
    print(f"\n{'='*50}")
    print(f"ENRICHING LEAGUE: {league}")
    print(f"{'='*50}")
    
    try:
        # Running the node script
        subprocess.run(["node", "footy-model-v2/scrapers/premium_enricher_v2.js", league], check=True)
    except Exception as e:
        print(f"Error enriching {league}: {e}")
    
    print(f"Cooling down for 10s...")
    time.sleep(10)

print("\nAll 22 Leagues Enrichment Cycle Complete!")
