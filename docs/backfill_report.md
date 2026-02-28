# Data Backfill Progress Report
Generated: 2026-02-27 12:22:19
Database Target: **footy_model** (`172.17.0.2/32:5432` as `footy`)

Target Universe: **7831** (Finished matches with Sofascore IDs)

## The Big Picture (Database Mapping)
We have **11102** total fixtures in the database.

| Status | Total | Linked | Unlinked | Linked % |
| :--- | :--- | :--- | :--- | :--- |
| ft | 7842 | 7831 | 11 | 99.9% |
| postponed | 16 | 16 | 0 | 100.0% |
| cancelled | 2 | 2 | 0 | 100.0% |
| scheduled | 3242 | 3127 | 115 | 96.5% |


## Coverage Summary (Target Universe)

| Category | Done | Remaining | Total | Progress | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Match Stats (Presence)** | 7831 | 0 | 7831 | 100.0% | done |
| **H1/H2 Update (Attempted)** | 7713 | 118 | 7831 | 98.5% | in_progress |
| **H1/H2 Data (Yielded)** | 7057 | 774 | 7831 | 90.1% | in_progress |
| **Player Stats** | 7810 | 21 | 7831 | 99.7% | in_progress |
| **Availability (Injuries)** | 7814 | 17 | 7831 | 99.8% | in_progress |
| **Incidents (Timeline)** | 7684 | 147 | 7831 | 98.1% | in_progress |
| **Match Odds (Backfill)** | 7671 | 160 | 7831 | 98.0% | in_progress |

## Active Backfill Scripts

To continue the backfill, use these commands in the project root:

- **Match Stats (Upgrade/Basic)**:
  `python scripts/backfill_batch_sofascore.py --type stats --total 1000 --limit 100`

- **Player Stats**:
  `python scripts/backfill_batch_sofascore.py --type players --total 1000 --limit 100`

- **Availability / Injuries**:
  `python scripts/backfill_batch_sofascore.py --type availability --total 1000 --limit 100 --status ft`

- **Incidents (Timeline)**:
  `python scripts/backfill_batch_sofascore.py --type incidents --total 1000 --limit 100 --status ft`

- **Match Odds**:
  `python src/ingest/backfill_sofascore_odds_markets_v1.py --limit 100`
