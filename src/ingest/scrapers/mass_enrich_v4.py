import subprocess
import time

leagues = [
    "E0", "E1", "E2", "E3", "EC", 
    "SC0", "SC1", "SC2", "SC3", 
    "D1", "D2", "I1", "I2", 
    "SP1", "SP2", "F1", "F2", 
    "N1", "B1", "P1", "T1", "G1"
]

print(f"\n{'='*60}")
print(f"STARTING MASS ENRICHMENT V4 (PLAYWRIGHT CRAWLER)")
print(f"Target Directory: data/scraper/premium_v4/")
print(f"{'='*60}")

for league in leagues:
    print(f"\n[+] ENRICHING LEAGUE: {league}")
    print("-" * 30)
    
    try:
        # Running the new v4 node script
        subprocess.run(["node", "scrapers/premium_enricher_v4.js", league], check=True)
    except Exception as e:
        print(f"Error enriching {league}: {e}")
    
    print(f"Cooling down for 15s...")
    time.sleep(15)

print("\nAll 22 Leagues Enrichment Cycle (V4) Complete!")
