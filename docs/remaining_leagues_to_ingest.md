# Remaining Leagues: Premium Enrich + DB Ingest (After E0-E3)

## Overview

You said you already completed premium enrichment + ingest for `E0`, `E1`, `E2`, `E3`.

Discovery/ID generation is done (the files exist under `data/v1/ids/`). What remains is to, for each remaining league:

1) Scrape premium match pages (stats + odds + result) into `data/v1/premium/<LEAGUE>/`.
2) Ingest those JSON files into the DB (`fixture_stats_premium`, `fixture_odds_snapshots`, `fixture_results`).

## Plan

Run one league at a time:

1) Enrich that league (writes JSON files).
2) Ingest that league (upserts into DB).
3) Move to the next league.

Prereq (once):

```bash
bun install
```

## Troubleshooting (When Things Break)

- If you see `net::ERR_EMPTY_RESPONSE` / timeouts while enriching: rerun the enrich command for the same league. The enricher skips IDs that already have JSON files and will try the missing ones again.
- If you want to force a full refresh for a league: delete `data/v1/premium/<LEAGUE>/*.json` and rerun enrich.
- Ingest is idempotent: it is always safe to rerun `python scrapers/ingest_premium_fixtures_v1.py --league <LEAGUE>`.
- `NO1` is excluded here because your seed discovery run showed it failing; handle it separately after fixing discovery or temporarily disabling it.

## Remaining Leagues (Runbook)

Each section includes the league, its ID input files, and the exact commands.

1) England National League (`EC`) [x]

Inputs:
- `data/v1/ids/match_ids_EC.json`
- `data/v1/ids/upcoming_ids_EC.json`

Commands:

```bash
node scrapers/premium_enricher_v4.js EC --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league EC
```

2) Scotland Premiership (`SC0`) [x]

Inputs:
- `data/v1/ids/match_ids_SC0.json`
- `data/v1/ids/upcoming_ids_SC0.json`

```bash
node scrapers/premium_enricher_v4.js SC0 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league SC0
```

3) Scotland Championship (`SC1`) [x]

Inputs:
- `data/v1/ids/match_ids_SC1.json`
- `data/v1/ids/upcoming_ids_SC1.json`

```bash
node scrapers/premium_enricher_v4.js SC1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league SC1
```

4) Scotland League One (`SC2`) [x]

Inputs:
- `data/v1/ids/match_ids_SC2.json`
- `data/v1/ids/upcoming_ids_SC2.json`

```bash
node scrapers/premium_enricher_v4.js SC2 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league SC2
```

5) Scotland League Two (`SC3`)

Inputs:
- `data/v1/ids/match_ids_SC3.json`
- `data/v1/ids/upcoming_ids_SC3.json`

```bash
node scrapers/premium_enricher_v4.js SC3 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league SC3
```

6) Germany Bundesliga (`D1`) [x]

Inputs:
- `data/v1/ids/match_ids_D1.json`
- `data/v1/ids/upcoming_ids_D1.json`

```bash
node scrapers/premium_enricher_v4.js D1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league D1
```

7) Germany 2. Bundesliga (`D2`) [x]

Inputs:
- `data/v1/ids/match_ids_D2.json`
- `data/v1/ids/upcoming_ids_D2.json`

```bash
node scrapers/premium_enricher_v4.js D2 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league D2
```

8) Italy Serie A (`I1`) [x]

Inputs:
- `data/v1/ids/match_ids_I1.json`
- `data/v1/ids/upcoming_ids_I1.json`

```bash
node scrapers/premium_enricher_v4.js I1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league I1
```

9) Italy Serie B (`I2`) [x]

Inputs:
- `data/v1/ids/match_ids_I2.json`
- `data/v1/ids/upcoming_ids_I2.json`

```bash
node scrapers/premium_enricher_v4.js I2 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league I2
```

10) Spain LaLiga (`SP1`) [x]

Inputs:
- `data/v1/ids/match_ids_SP1.json`
- `data/v1/ids/upcoming_ids_SP1.json`

```bash
node scrapers/premium_enricher_v4.js SP1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league SP1
```

11) Spain LaLiga 2 (`SP2`) [x]

Inputs:
- `data/v1/ids/match_ids_SP2.json`
- `data/v1/ids/upcoming_ids_SP2.json`

```bash
node scrapers/premium_enricher_v4.js SP2 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league SP2
```

12) France Ligue 1 (`F1`) [x]

Inputs:
- `data/v1/ids/match_ids_F1.json`
- `data/v1/ids/upcoming_ids_F1.json`

```bash
node scrapers/premium_enricher_v4.js F1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league F1
```

13) France Ligue 2 (`F2`) [x]

Inputs:
- `data/v1/ids/match_ids_F2.json`
- `data/v1/ids/upcoming_ids_F2.json`

```bash
node scrapers/premium_enricher_v4.js F2 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league F2
```

14) Netherlands Eredivisie (`N1`) [x] **left with ingest

Inputs:
- `data/v1/ids/match_ids_N1.json`
- `data/v1/ids/upcoming_ids_N1.json`

```bash
node scrapers/premium_enricher_v4.js N1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league N1
```

15) Belgium Jupiler League (`B1`) [x]

Inputs:
- `data/v1/ids/match_ids_B1.json`
- `data/v1/ids/upcoming_ids_B1.json`

```bash
node scrapers/premium_enricher_v4.js B1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league B1
```

16) Portugal Liga Portugal (`P1`) [x]

Inputs:
- `data/v1/ids/match_ids_P1.json`
- `data/v1/ids/upcoming_ids_P1.json`

```bash
node scrapers/premium_enricher_v4.js P1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league P1
```

17) Turkey Super Lig (`T1`) [x]

Inputs:
- `data/v1/ids/match_ids_T1.json`
- `data/v1/ids/upcoming_ids_T1.json`

```bash
node scrapers/premium_enricher_v4.js T1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league T1
```

18) Greece Super League (`G1`) [x]

Inputs:
- `data/v1/ids/match_ids_G1.json`
- `data/v1/ids/upcoming_ids_G1.json`

```bash
node scrapers/premium_enricher_v4.js G1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league G1
```

19) Austria Bundesliga (`AT1`)

Inputs:
- `data/v1/ids/match_ids_AT1.json`
- `data/v1/ids/upcoming_ids_AT1.json`

```bash
node scrapers/premium_enricher_v4.js AT1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league AT1
```

20) Brazil Serie A (`BR1`)

Inputs:
- `data/v1/ids/match_ids_BR1.json`
- `data/v1/ids/upcoming_ids_BR1.json`

```bash
node scrapers/premium_enricher_v4.js BR1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league BR1
```

21) Denmark Superliga (`DK1`)

Inputs:
- `data/v1/ids/match_ids_DK1.json`
- `data/v1/ids/upcoming_ids_DK1.json`

```bash
node scrapers/premium_enricher_v4.js DK1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league DK1
```

22) Ireland Premier Division (`IE1`)

Inputs:
- `data/v1/ids/match_ids_IE1.json`
- `data/v1/ids/upcoming_ids_IE1.json`

```bash
node scrapers/premium_enricher_v4.js IE1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league IE1
```

23) Japan J1 League (`JP1`)

Inputs:
- `data/v1/ids/match_ids_JP1.json`
- `data/v1/ids/upcoming_ids_JP1.json`

```bash
node scrapers/premium_enricher_v4.js JP1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league JP1
```

24) Mexico Liga MX (`MX1`)

Inputs:
- `data/v1/ids/match_ids_MX1.json`
- `data/v1/ids/upcoming_ids_MX1.json`

```bash
node scrapers/premium_enricher_v4.js MX1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league MX1
```

25) Poland Ekstraklasa (`PL1`)

Inputs:
- `data/v1/ids/match_ids_PL1.json`
- `data/v1/ids/upcoming_ids_PL1.json`

```bash
node scrapers/premium_enricher_v4.js PL1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league PL1
```

26) Romania Liga 1 (`RO1`)

Inputs:
- `data/v1/ids/match_ids_RO1.json`
- `data/v1/ids/upcoming_ids_RO1.json`

```bash
node scrapers/premium_enricher_v4.js RO1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league RO1
```

27) Russia Premier League (`RU1`)

Inputs:
- `data/v1/ids/match_ids_RU1.json`
- `data/v1/ids/upcoming_ids_RU1.json`

```bash
node scrapers/premium_enricher_v4.js RU1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league RU1
```

28) Switzerland Super League (`CH1`)

Inputs:
- `data/v1/ids/match_ids_CH1.json`
- `data/v1/ids/upcoming_ids_CH1.json`

```bash
node scrapers/premium_enricher_v4.js CH1 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league CH1
```

29) Switzerland Challenge League (`CH2`)

Inputs:
- `data/v1/ids/match_ids_CH2.json`
- `data/v1/ids/upcoming_ids_CH2.json`

```bash
node scrapers/premium_enricher_v4.js CH2 --ids-root data/v1/ids --out-root data/v1/premium
python scrapers/ingest_premium_fixtures_v1.py --league CH2
```
