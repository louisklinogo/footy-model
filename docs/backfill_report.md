# Data Backfill Progress Report
Generated: 2026-02-26 05:32:04

Target Universe: **7811** (Finished matches with Sofascore IDs)

## 🌎 The Big Picture (Database Mapping)
We have **11102** total fixtures in the database.

| Status | Total | Linked | Unlinked | Linked % |
| :--- | :--- | :--- | :--- | :--- |
| ft | 7821 | 7811 | 10 | 99.9% |
| postponed | 16 | 16 | 0 | 100.0% |
| cancelled | 2 | 2 | 0 | 100.0% |
| scheduled | 3263 | 3147 | 116 | 96.4% |


## 📊 Coverage Summary (Target Universe)

| Category | Done | Remaining | Total | Progress | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Match Stats (Presence)** | 7811 | 0 | 7811 | 100.0% | ✅ |
| **H1/H2 Update (Attempted)** | 7686 | 125 | 7811 | 98.4% | 🔄 |
| **H1/H2 Data (Yielded)** | 7037 | 774 | 7811 | 90.1% | 🔄 |
| **Player Stats** | 7810 | 1 | 7811 | 100.0% | 🔄 |
| **Availability (Injuries)** | 5990 | 1821 | 7811 | 76.7% | 🔄 |
| **Match Odds (Backfill)** | 5815 | 1996 | 7811 | 74.4% | 🔄 |

## 🛠️ Active Backfill Scripts

To continue the backfill, use these commands in the project root:

- **Match Stats (Upgrade/Basic)**:
  `python scripts/backfill_batch_sofascore.py --type stats --total 1000 --limit 100`

- **Player Stats**:
  `python scripts/backfill_batch_sofascore.py --type players --total 1000 --limit 100`

- **Availability / Injuries**:
  `python scripts/backfill_batch_sofascore.py --type availability --total 1000 --limit 100 --status ft`

- **Match Odds**:
  `python src/ingest/backfill_sofascore_odds_markets_v1.py --limit 100`
