# Data Backfill Progress Report
Generated: 2026-02-26 17:02:25

Target Universe: **7814** (Finished matches with Sofascore IDs)

## 🌎 The Big Picture (Database Mapping)
We have **11102** total fixtures in the database.

| Status | Total | Linked | Unlinked | Linked % |
| :--- | :--- | :--- | :--- | :--- |
| ft | 7824 | 7814 | 10 | 99.9% |
| postponed | 16 | 16 | 0 | 100.0% |
| cancelled | 2 | 2 | 0 | 100.0% |
| scheduled | 3260 | 3144 | 116 | 96.4% |


## 📊 Coverage Summary (Target Universe)

| Category | Done | Remaining | Total | Progress | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Match Stats (Presence)** | 7814 | 0 | 7814 | 100.0% | ✅ |
| **H1/H2 Update (Attempted)** | 7689 | 125 | 7814 | 98.4% | 🔄 |
| **H1/H2 Data (Yielded)** | 7040 | 774 | 7814 | 90.1% | 🔄 |
| **Player Stats** | 7810 | 4 | 7814 | 99.9% | 🔄 |
| **Availability (Injuries)** | 7813 | 1 | 7814 | 100.0% | 🔄 |
| **Match Odds (Backfill)** | 7654 | 160 | 7814 | 98.0% | 🔄 |

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
