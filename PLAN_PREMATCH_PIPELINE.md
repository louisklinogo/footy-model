# Plan: Prematch Pipeline & AI Research Integration

**Created:** 2026-02-22  
**Status:** Draft  
**Goal:** Create a robust, automated pipeline that scrapes prematch data (odds + injuries), ingests to DB, and feeds AI research with context.

---

## Executive Summary

### Current Gaps

| Gap | Description |
|-----|-------------|
| **No fixture_availability table** | Injury/lineup data has nowhere to go |
| **Prematch JSONs not ingested** | 117 JSONs exist, DB has no record of them |
| **Stale fixture statuses** | 165 fixtures past match time still "scheduled" |
| **Missing results** | 405 FT fixtures have no scores |
| **AI research disconnected** | Tavily researches from scratch, ignores our data |

### What We'll Build

1. **DB Schema** — `fixture_availability` table + helper functions/views
2. **Migration System** — Simple numbered SQL files
3. **Ingestion Script** — Push prematch JSON → DB
4. **Housekeeping Script** — Fix stale statuses + missing results
5. **AI Research Update** — Pull context from DB, pre-fill Tavily prompt
6. **Pipeline Orchestration** — Daily workflow script

---

## Phase 1: Database Foundation

### 1.1 Create Migration System

**File:** `migrations/migrate.py`

Simple migration runner:
- Track applied migrations in `schema_migrations` table
- Run any `.sql` file in `migrations/` that hasn't been applied
- Order by filename (001, 002, etc.)

### 1.2 Create `fixture_availability` Table

**File:** `migrations/001_fixture_availability.sql`

```sql
CREATE TABLE fixture_availability (
    fixture_id BIGINT PRIMARY KEY REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    
    -- Home team
    home_missing JSONB NOT NULL DEFAULT '[]'::jsonb,
    home_questionable JSONB NOT NULL DEFAULT '[]'::jsonb,
    home_lineup JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Away team  
    away_missing JSONB NOT NULL DEFAULT '[]'::jsonb,
    away_questionable JSONB NOT NULL DEFAULT '[]'::jsonb,
    away_lineup JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Metadata
    scraped_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_fixture_availability_scraped_at 
    ON fixture_availability(scraped_at DESC);
```

### 1.3 Create DB Functions

**File:** `migrations/002_db_functions.sql`

```sql
-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_fixtures_updated_at
    BEFORE UPDATE ON fixtures
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Get fixtures that need prematch scraping
CREATE OR REPLACE FUNCTION get_fixtures_to_scrape(days_ahead INT DEFAULT 3)
RETURNS TABLE (
    flashscore_id TEXT,
    league_code TEXT,
    home_team TEXT,
    away_team TEXT,
    match_datetime_utc TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.flashscore_id,
        f.league_code,
        ht.team_name,
        at.team_name,
        f.match_datetime_utc
    FROM fixtures f
    JOIN teams ht ON f.home_team_id = ht.team_id
    JOIN teams at ON f.away_team_id = at.team_id
    LEFT JOIN fixture_availability fa ON f.fixture_id = fa.fixture_id
    WHERE f.status = 'scheduled'
    AND f.match_datetime_utc BETWEEN NOW() AND NOW() + days_ahead * INTERVAL '1 day'
    AND (fa.fixture_id IS NULL OR fa.scraped_at < NOW() - INTERVAL '12 hours')
    ORDER BY f.match_datetime_utc;
END;
$$ LANGUAGE plpgsql;

-- Atomic fixture settlement
CREATE OR REPLACE FUNCTION settle_fixture(
    p_flashscore_id TEXT,
    p_home_goals INT,
    p_away_goals INT
) RETURNS VOID AS $$
DECLARE
    v_fixture_id BIGINT;
BEGIN
    SELECT fixture_id INTO v_fixture_id 
    FROM fixtures WHERE flashscore_id = p_flashscore_id;
    
    IF v_fixture_id IS NULL THEN
        RAISE EXCEPTION 'Fixture not found: %', p_flashscore_id;
    END IF;
    
    UPDATE fixtures SET status = 'ft' WHERE fixture_id = v_fixture_id;
    
    INSERT INTO fixture_results (fixture_id, home_goals, away_goals, result_status, settled_at)
    VALUES (v_fixture_id, p_home_goals, p_away_goals, 'ft', NOW())
    ON CONFLICT (fixture_id) DO UPDATE SET
        home_goals = EXCLUDED.home_goals,
        away_goals = EXCLUDED.away_goals,
        settled_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Get stale fixtures (scheduled but past match time)
CREATE OR REPLACE FUNCTION get_stale_fixtures(hours_passed INT DEFAULT 3)
RETURNS TABLE (
    flashscore_id TEXT,
    match_datetime_utc TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT f.flashscore_id, f.match_datetime_utc
    FROM fixtures f
    WHERE f.status = 'scheduled'
    AND f.match_datetime_utc < NOW() - hours_passed * INTERVAL '1 hour'
    ORDER BY f.match_datetime_utc;
END;
$$ LANGUAGE plpgsql;
```

### 1.4 Create Views

**File:** `migrations/003_views.sql`

```sql
-- View: Fixtures with full context for AI research
CREATE OR REPLACE VIEW v_fixtures_with_context AS
SELECT 
    f.fixture_id,
    f.flashscore_id,
    f.league_code,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    f.match_datetime_utc,
    f.status,
    fa.home_missing,
    fa.away_missing,
    fa.home_questionable,
    fa.away_questionable,
    fa.home_lineup,
    fa.away_lineup,
    fa.scraped_at AS availability_scraped_at,
    fos.ou_json AS prematch_ou_odds,
    fos.ah_json AS prematch_ah_odds,
    fos.one_x_two_json AS prematch_1x2_odds
FROM fixtures f
JOIN teams ht ON f.home_team_id = ht.team_id
JOIN teams at ON f.away_team_id = at.team_id
LEFT JOIN fixture_availability fa ON f.fixture_id = fa.fixture_id
LEFT JOIN LATERAL (
    SELECT ou_json, ah_json, one_x_two_json
    FROM fixture_odds_snapshots
    WHERE fixture_id = f.fixture_id
    AND snapshot_type = 'prematch'
    ORDER BY snapshot_time_utc DESC
    LIMIT 1
) fos ON true;

-- View: High-conviction edges for AI research
CREATE OR REPLACE VIEW v_edges_for_research AS
SELECT 
    vfc.*,
    p.market_code,
    p.p_model AS model_probability,
    p.metadata_json->>'odds_used' AS odds_used
FROM v_fixtures_with_context vfc
JOIN predictions p ON vfc.fixture_id = p.fixture_id
WHERE vfc.status = 'scheduled'
AND p.p_model > 0.55  -- Adjust threshold as needed
ORDER BY p.p_model DESC;
```

---

## Phase 2: Housekeeping (Fix Current Mess)

### 2.1 Fix Stale Fixture Statuses

**File:** `src/jobs/housekeeping_fix_statuses.py`

```python
"""
Update fixture statuses for matches that have passed.

- scheduled + past match time → check if postponed or ft
- If no result available, keep as scheduled (will need manual check)
"""
```

**Logic:**
1. Call `get_stale_fixtures()` function
2. For each fixture, check Flashscore for result
3. If result found → call `settle_fixture()`
4. If postponed → update status to 'postponed'

### 2.2 Settle Missing Results

**File:** `src/jobs/housekeeping_settle_results.py`

```python
"""
Settle results for FT fixtures that have no result record.

Query: fixtures with status='ft' but no fixture_results row
Action: Scrape final scores from Flashscore, insert to fixture_results
"""
```

---

## Phase 3: Ingestion Pipeline

### 3.1 Ingest Prematch JSON to DB

**File:** `src/ingest/ingest_prematch_v1.py`

```python
"""
Ingest prematch JSON files to database.

For each JSON in data/v1/prematch_json/{league}/{id}.json:
1. Check fixture status (skip if FT)
2. Upsert odds to fixture_odds_snapshots (snapshot_type='prematch')
3. Upsert availability to fixture_availability

Usage:
    python src/ingest/ingest_prematch_v1.py [--all] [--league E0]
    
    --all      Process all JSON files
    --league   Process only specific league
"""
```

**Key Logic:**
```python
def ingest_prematch_json(json_path: Path, cur) -> dict:
    """
    Ingest a single prematch JSON file.
    
    Returns:
        {'status': 'skipped_ft'|'skipped_not_found'|'success', 
         'fixture_id': int|None}
    """
    data = json.loads(json_path.read_text())
    flashscore_id = data['id']
    
    # Get fixture
    cur.execute("SELECT fixture_id, status FROM fixtures WHERE flashscore_id = %s", (flashscore_id,))
    row = cur.fetchone()
    if not row:
        return {'status': 'skipped_not_found'}
    
    fixture_id, status = row
    
    # Skip if already FT
    if status == 'ft':
        return {'status': 'skipped_ft', 'fixture_id': fixture_id}
    
    # Upsert odds
    upsert_prematch_odds(cur, fixture_id, data.get('odds', {}), data.get('scraped_at'))
    
    # Upsert availability
    upsert_availability(cur, fixture_id, data.get('availability', {}), data.get('scraped_at'))
    
    return {'status': 'success', 'fixture_id': fixture_id}
```

---

## Phase 4: AI Research Integration

### 4.1 Update AI Research Pass

**File:** `src/modeling/ai_research_pass.py` (modify existing)

**Changes:**
1. Query `v_edges_for_research` view instead of CSV
2. Pre-fill Tavily prompt with injury context from DB

**New Prompt Structure:**
```python
def build_research_prompt(edge: dict, context: dict) -> str:
    """
    Build Tavily prompt with pre-filled context.
    """
    home_missing = context.get('home_missing', [])
    away_missing = context.get('away_missing', [])
    
    injury_context = ""
    if home_missing:
        injury_context += f"\nHome team missing: {format_players(home_missing)}"
    if away_missing:
        injury_context += f"\nAway team missing: {format_players(away_missing)}"
    
    prompt = f"""
    Analyze the upcoming football match: {edge['match']} on {edge['kickoff']}.
    Focus: {edge['market']} market, selecting {edge['selection']}.
    Original model confidence: {edge['prob']}.
    
    KNOWN INJURY CONTEXT (from pre-match data):
    {injury_context if injury_context else "No injury data available."}
    
    Based on the above injury context, quantify the impact:
    - What % of team minutes are affected?
    - Are any star players or defensive pillars missing?
    
    Also analyze: weather, referee tendencies, market steam.
    """
    return prompt
```

---

## Phase 5: Pipeline Orchestration

### 5.1 Daily Pipeline Script

**File:** `src/pipelines/daily_prematch_pipeline.py`

```python
"""
Daily prematch data pipeline.

Steps:
1. Housekeeping: Fix stale statuses, settle missing results
2. Identify fixtures to scrape (call get_fixtures_to_scrape)
3. Scrape prematch data (prematch_enricher.js)
4. Ingest to DB (ingest_prematch_v1.py)
5. Run AI research pass (ai_research_pass.py)

Usage:
    python src/pipelines/daily_prematch_pipeline.py [--skip-scrape] [--skip-research]
"""
```

### 5.2 Scheduler Setup (trigger.dev)

**Why trigger.dev?**
- Runs in cloud (your machine can be off)
- Automatic retries on failure
- Web UI for job monitoring
- Supports Python via build extensions

**Setup Steps:**

1. Create trigger.dev account: https://trigger.dev
2. Install SDK:
```bash
npm install @trigger.dev/sdk
```

3. Create `trigger.config.ts`:
```typescript
import { defineConfig } from "@trigger.dev/sdk";
import { pythonExtension } from "@trigger.dev/build/extensions/python";

export default defineConfig({
  project: "footy-model",
  build: {
    extensions: [pythonExtension()],
  },
});
```

4. Create scheduled task:
```typescript
// src/trigger/daily_pipeline.ts
import { schedule } from "@trigger.dev/sdk";
import { execSync } from "child_process";

export const dailyPrematchPipeline = schedule({
  id: "daily-prematch-pipeline",
  cron: "0 6 * * *", // 6am daily
}, async () => {
  execSync("python src/pipelines/daily_prematch_pipeline.py", {
    cwd: "/path/to/footy-model",
  });
});
```

5. Deploy:
```bash
npx trigger deploy
```

**Alternative: pg_cron (for SQL-only tasks)**
```sql
-- Requires superuser to install
CREATE EXTENSION pg_cron;

-- Schedule hourly status check
SELECT cron.schedule(
  'fix-stale-statuses',
  '0 * * * *',
  $
  UPDATE fixtures SET status = 'ft'
  WHERE status = 'scheduled'
  AND match_datetime_utc < NOW() - INTERVAL '3 hours';
  $
);
```

---

## Task List

### Phase 1: Database Foundation
- [ ] **1.1** Create `migrations/migrate.py` — migration runner
- [ ] **1.2** Create `migrations/001_fixture_availability.sql` — availability table
- [ ] **1.3** Create `migrations/002_db_functions.sql` — helper functions
- [ ] **1.4** Create `migrations/003_views.sql` — convenience views
- [ ] **1.5** Run migrations

### Phase 2: Housekeeping
- [ ] **2.1** Create `src/jobs/housekeeping_fix_statuses.py`
- [ ] **2.2** Create `src/jobs/housekeeping_settle_results.py`
- [ ] **2.3** Run housekeeping scripts

### Phase 3: Ingestion
- [ ] **3.1** Create `src/ingest/ingest_prematch_v1.py`
- [ ] **3.2** Ingest existing prematch JSONs

### Phase 4: AI Research Integration
- [ ] **4.1** Modify `src/modeling/ai_research_pass.py` to query DB
- [ ] **4.2** Add injury context pre-fill to Tavily prompt
- [ ] **4.3** Test AI research with DB context

### Phase 5: Orchestration
- [ ] **5.1** Create `src/pipelines/daily_prematch_pipeline.py`
- [ ] **5.2** Set up scheduler (Windows Task Scheduler or cron)
- [ ] **5.3** Document pipeline in README

### Phase 6: trigger.dev Setup
- [ ] **6.1** Create trigger.dev account and project
- [ ] **6.2** Create `trigger.config.ts` with Python extension
- [ ] **6.3** Create scheduled task for daily pipeline
- [ ] **6.4** Deploy and test

### Phase 7: Fix Injury Scraping (If Needed)
- [ ] **7.1** Verify scraper captures injuries (spot check on live matches)
- [ ] **7.2** Debug if issues found
- [ ] **7.3** Re-scrape and re-ingest

---

## Estimated Effort

| Phase | Tasks | Time |
|-------|-------|------|
| Phase 1: DB Foundation | 5 tasks | 1.5 hrs |
| Phase 2: Housekeeping | 3 tasks | 1 hr |
| Phase 3: Ingestion | 2 tasks | 45 min |
| Phase 4: AI Research | 3 tasks | 45 min |
| Phase 5: Orchestration | 3 tasks | 30 min |
| Phase 6: Fix Scraping | 3 tasks | 1 hr |
| **Total** | **19 tasks** | **~5.5 hrs** |

---

## Open Questions / Decisions Needed

| Question | Options | Recommendation |
|----------|---------|----------------|
| Use pg_cron for scheduling? | Yes / No | **No** — use external scheduler |
| Keep past prematch JSON odds? | Yes / No | **Yes** — useful for backtesting |
| How often to run pipeline? | Once daily / Twice / Continuous | **Once daily** (can add later) |
| Migration system complexity | Simple files / Alembic | **Simple files** |

---

## Success Criteria

- [ ] All migrations run successfully
- [ ] Housekeeping fixes 165 stale statuses + 405 missing results
- [ ] All 117 prematch JSONs ingested to DB
- [ ] AI research pulls injury context from DB
- [ ] Daily pipeline runs automatically
- [ ] No manual intervention needed for day-to-day operations
