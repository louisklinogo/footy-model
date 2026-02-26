import subprocess
import sys
import time
import argparse

def run_batch(ingest_type, league, limit, total_goal, status):
    script_map = {
        "stats": "src/ingest/ingest_sofascore_stats.py",
        "players": "src/ingest/ingest_sofascore_players.py",
        "availability": "src/ingest/ingest_sofascore_availability.py",
        "odds": "src/ingest/backfill_sofascore_odds_markets_v1.py",
    }
    script_path = script_map[ingest_type]
    
    total_processed = 0
    while total_processed < total_goal:
        batch_size = min(limit, total_goal - total_processed)
        print(f"\n--- Batch {total_processed // limit + 1}: processing up to {batch_size} {ingest_type} (total so far: {total_processed}) ---")
        
        cmd = [
            sys.executable, 
            script_path,
            "--limit", str(batch_size)
        ]
        if league:
            cmd.extend(["--league", league])
            
        if ingest_type == "availability" and status:
            cmd.extend(["--status", status])
            
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            print(f"Batch failed with exit code {result.returncode}. Stopping.")
            break

        total_processed += batch_size

        # If we fetched fewer fixtures than the batch size, there is nothing left to process.
        # The underlying script logs "Found X fixtures needing stats ingestion."
        # We rely on the caller reading the output to notice this, but we can't
        # easily capture and parse that log without storing stdout.
        # The safe signal is: if this batch processed 0 new fixtures (i.e., the
        # ingester found 0), we would have still exit-code 0 but nothing changed.
        # We stop early by letting the user re-run; batches are fully idempotent.

        if total_processed < total_goal:
            print(f"Waiting 10 seconds before next batch...")
            time.sleep(10)

    print(f"\nDone. Processed up to {total_processed} fixture slots total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill Sofascore stats in safe batches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  All leagues:    python scripts/backfill_batch_sofascore.py --total 1000\n"
            "  One league:     python scripts/backfill_batch_sofascore.py --league CL --total 200\n"
        )
    )
    parser.add_argument("--type", type=str, choices=["stats", "players", "availability", "odds"], default="stats", help="Type of data to ingest (default: stats)")
    parser.add_argument("--league", type=str, default=None, help="Optional league code filter. Omit to process all leagues.")
    parser.add_argument("--status", type=str, default="ft", help="Status filter (primarily for availability, e.g. scheduled, ft)")
    parser.add_argument("--limit", type=int, default=50, help="Batch size per API call (default: 50)")
    parser.add_argument("--total", type=int, default=500, help="Total fixture slots to process (default: 500)")
    args = parser.parse_args()
    
    run_batch(args.type, args.league, args.limit, args.total, args.status)
