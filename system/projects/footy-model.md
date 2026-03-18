---
description: Key learnings and patterns for the footy-model soccer prediction project
---
# Footy-Model Project Notes

## Database Schema Gotchas

### team_premium_snapshots
- **Correct columns**: `rolling_xg`, `rolling_xg_against`, `rolling_sot`, `rolling_possession`, `sample_size`
- **NOT**: `matches_played`, `goals_scored` (these don't exist)
- **BUG**: Rows with `sample_size=0` exist for first game of each season - these have NULL xG values
- **FIX**: Always filter with `WHERE rolling_xg IS NOT NULL AND sample_size > 0`
- **Query pattern**:
  ```sql
  SELECT DISTINCT ON (t.team_name) 
      t.team_name, tps.rolling_xg, tps.rolling_xg_against, tps.sample_size
  FROM team_premium_snapshots tps
  JOIN teams t ON tps.team_id = t.team_id
  WHERE tps.rolling_xg IS NOT NULL AND tps.sample_size > 0
  ORDER BY t.team_name, tps.sample_size DESC, tps.built_at DESC
  ```
- **Timestamp issue**: All snapshots for a team have same `built_at` timestamp (batch built), so `ORDER BY built_at DESC` alone is unreliable

### team_external_context
- Has rich standings/form data: `overall_rank`, `overall_points`, `form_sequence`, `form_points_last5`
- Also has home/away splits: `home_wins`, `away_wins`, etc.
- Use `DISTINCT ON (team_name)` with `ORDER BY snapshot_time_utc DESC` to get latest

### fixture_results
- Has `home_goals`, `away_goals` (NOT `home_score`, `away_score`)
- Join on `fixture_id` to get match results

### fixtures
- No `home_score`/`away_score` columns - use `fixture_results` table for scores

### Table relationships
- `teams` -> `team_premium_snapshots` via `team_id`
- `fixtures` -> `fixture_results` via `fixture_id`
- `fixtures` -> `teams` via `home_team_id` and `away_team_id`

## Key Scripts

- `scripts/build_accumulators.py` - Builds accumulator slips with validation
- `scripts/validate_accumulators.py` - Validates picks against standings, H2H, form
- `scripts/export_market_model_odds_edges.py` - Exports model predictions with odds

## Output Files

- `storage/reports/edges_today.csv` - Daily edge export
- `storage/reports/accumulators_validated.txt` - Final accumulator output

## Model Pipeline

1. `src/modeling/v2/run_hybrid_prediction_flow.py` - Runs hybrid predictions
2. Models: `scoreline_external_context_v1`, `anytime_direct_monotone_v1`
3. Outputs probabilities for markets: `dc_1x`, `dc_x2`, `dc_12`, `o15`, etc.