# Data Backfill Progress Report
Generated: 2026-02-25 14:05:44

Target Universe: **7566** (Finished matches with Sofascore IDs)

## 🌎 The Big Picture (Database Mapping)
We have **11102** total fixtures in the database.

| Status | Total | Linked | Unlinked | Linked % |
| :--- | :--- | :--- | :--- | :--- |
| ft | 7576 | 7566 | 10 | 99.9% |
| postponed | 12 | 12 | 0 | 100.0% |
| cancelled | 1 | 1 | 0 | 100.0% |
| scheduled | 3513 | 3397 | 116 | 96.7% |


## 📊 Coverage Summary (Target Universe)

| Category | Done | Remaining | Total | Progress | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Match Stats (Presence)** | 7562 | 4 | 7566 | 99.9% | 🔄 |
| **H1/H2 Update (Attempted)** | 6574 | 992 | 7566 | 86.9% | 🔄 |
| **H1/H2 Data (Yielded)** | 6544 | 1022 | 7566 | 86.5% | 🔄 |
| **Player Stats** | 4198 | 3368 | 7566 | 55.5% | 🔄 |
| **Availability (Injuries)** | 1893 | 5673 | 7566 | 25.0% | 🔄 |

## 🛠️ Active Backfill Scripts

To continue the backfill, use these commands in the project root:

- **Match Stats (Upgrade/Basic)**:
  `python scripts/backfill_batch_sofascore.py --type stats --total 1000 --limit 100`

- **Player Stats**:
  `python scripts/backfill_batch_sofascore.py --type players --total 1000 --limit 100`

- **Availability / Injuries**:
  `python scripts/backfill_batch_sofascore.py --type availability --total 1000 --limit 100 --status ft`
