# Prediction Risk Assessment Runbook

This document explains the risk system behind the `prediction_risk_assessments` table (sometimes called the "RISC table" in chat). It is written for engineers and operators who need to run, tune, or troubleshoot the workflow safely.

## 1. Purpose

The risk layer is a pre-match decision system that sits on top of `predictions`.

It does three jobs:

1. Applies conservative adjustments to raw model probabilities.
2. Produces a standardized action (`pass`, `watch`, `bet_small`, `bet`).
3. Produces stake guidance (`stake_fraction`) from risk-adjusted edge and Kelly scaling.

It **does not** overwrite rows in `predictions`.

## 2. Where It Lives

- Schema migration: `migrations/014_create_prediction_risk_assessments.sql`
- Scoring job: `src/modeling/evaluation/assess_prediction_risk.py`
- Policy config: `model_artifacts/market_models/risk_policy.json`
- Export integration: `src/modeling/export/export_market_outcomes_fixtures_first.py`
- Pipeline integration:
  - `src/pipelines/daily_pipeline.py`
  - `src/jobs/tick_due_fixtures_v1.py`

## 3. Table Definition

`prediction_risk_assessments` is keyed at prediction granularity.

Important columns:

- Identity and joins:
  - `risk_assessment_id` (PK)
  - `prediction_id` (FK -> `predictions`, unique)
  - `fixture_id` (FK -> `fixtures`)
  - `market_code`, `model_name`, `model_version`
- Core probabilities:
  - `p_model`: raw model probability used by risk job
  - `p_conservative`: probability after risk haircut
  - `implied_probability`: from odds when available
- Edge and decision:
  - `edge_raw = p_model - implied_probability`
  - `edge_adjusted = p_conservative - implied_probability`
  - `risk_score` (0..100)
  - `action` (`pass|watch|bet_small|bet`)
- Sizing:
  - `kelly_fraction`
  - `stake_fraction`
- Explainability:
  - `risk_flags_json` (array)
  - `metadata_json` (object; penalty breakdown and context)
- Timestamps:
  - `created_at`, `updated_at`

Constraints:

- Probability bounds on `p_model`, `p_conservative`, `implied_probability`
- Stake/Kelly bounds `0..1`
- `risk_score` bounds `0..100`
- uniqueness on `prediction_id` and `(fixture_id, market_code, model_name, model_version)`

## 4. Data Flow

The normal daily sequence is:

1. Predict markets into `predictions`.
2. Run risk assessment to upsert `prediction_risk_assessments`.
3. Export predictions CSV (now includes top risk recommendation fields).
4. Score settled predictions (separate post-match flow).

The risk job only scans **scheduled future fixtures** for the target model/version.

## 5. How the Score Is Built

For each candidate prediction row:

1. Read `p_model` from `predictions` (`COALESCE(p_final, p_model)`).
2. Pull latest pre-match odds snapshot and attempt market odds mapping.
3. Add risk penalties from policy:
  - missing odds
  - fallback model usage
  - feature missingness
  - low team sample size
  - stale odds age
  - weak calibration quality / insufficient calibration sample
4. Clamp `risk_score` to `[0, 100]`.
5. Compute haircut:
  - `haircut = min(0.25, risk_score / 400)`
  - `p_conservative = clip(p_model - haircut, 0, 1)`
6. Compute edges when implied probability exists.
7. Assign `action`.
8. Compute `stake_fraction` from scaled Kelly if action is bet-like.

### Current odds mapping scope

Direct implied-probability mapping is currently implemented for:

- `o15`, `u15`, `o25`, `u25` via `ou_json`
- `1x2_h`, `1x2_d`, `1x2_a` via `one_x_two_json`

Markets outside this mapping usually get `missing_odds` penalty and default to `pass`.

## 6. Action Rules

Action decision is rule-based:

1. `pass` if odds are missing.
2. `pass` if `risk_score >= 75`.
3. `pass` if `edge_adjusted < min_edge_watch`.
4. `watch` if edge is between watch and bet_small thresholds.
5. `bet_small` for moderate edge or medium risk.
6. `bet` only for high edge with low risk.

Default thresholds are in `risk_policy.json` and can be tuned without code changes.

## 7. Stake Rules

When action is `bet_small` or `bet` and valid odds exist:

1. Compute base Kelly fraction on `p_conservative`.
2. Scale by:
  - `kelly_scale`
  - `(1 - risk_score/100)`
3. Cap by:
  - `max_small_stake_fraction` for `bet_small`
  - `max_stake_fraction` for `bet`

`stake_fraction` is a bankroll fraction suggestion, not a currency amount.

## 8. Operations

### Apply migration

```powershell
python migrations/migrate.py
```

### Run risk assessment manually

```powershell
python src/modeling/evaluation/assess_prediction_risk.py --days 3
python src/modeling/evaluation/assess_prediction_risk.py --league ECL --days 3
python src/modeling/evaluation/assess_prediction_risk.py --league ECL --days 3 --dry-run
```

### Tune policy

Edit:

- `model_artifacts/market_models/risk_policy.json`

Then rerun `assess_prediction_risk.py` to refresh rows.

## 9. Export Behavior

Export now includes top risk recommendation fields per fixture:

- `risk_market_code`
- `risk_action`
- `risk_score`
- `risk_edge_adjusted`
- `risk_stake_fraction`

The exporter picks one market per fixture by:

1. action priority (`bet > bet_small > watch > pass`)
2. higher `edge_adjusted`
3. lower `risk_score`
4. latest update timestamp

## 10. Validation Queries

### Action distribution

```sql
SELECT action, COUNT(*)
FROM prediction_risk_assessments
GROUP BY action
ORDER BY action;
```

### Top actionable rows

```sql
SELECT
  f.match_datetime_utc,
  f.league_code,
  pra.fixture_id,
  pra.market_code,
  pra.p_model,
  pra.p_conservative,
  pra.edge_adjusted,
  pra.risk_score,
  pra.action,
  pra.stake_fraction
FROM prediction_risk_assessments pra
JOIN fixtures f ON f.fixture_id = pra.fixture_id
WHERE pra.action IN ('bet', 'bet_small')
ORDER BY f.match_datetime_utc ASC, pra.edge_adjusted DESC;
```

### Reasons behind a decision

```sql
SELECT
  prediction_id,
  risk_flags_json,
  metadata_json
FROM prediction_risk_assessments
WHERE fixture_id = :fixture_id
  AND market_code = :market_code;
```

## 11. Do / Don't

### Do

- Do treat `predictions.p_model` as the model output and risk table as a separate decision layer.
- Do tune thresholds and penalties in `risk_policy.json` before changing code.
- Do rerun risk assessment after policy changes.
- Do monitor action distribution after each tuning cycle.
- Do keep migration history and bootstrap schema aligned.

### Don't

- Do not overwrite `predictions` with risk-adjusted probabilities.
- Do not bypass risk scoring and consume raw `p_model` as stake guidance.
- Do not interpret `stake_fraction` as a guaranteed ROI signal.
- Do not compare historical risk outputs across policy versions without tracking config changes.
- Do not assume unsupported markets have valid implied probabilities.

## 12. Known Limitations

- Odds mapping is not yet implemented for all markets.
- Calibration penalty currently uses Brier aggregates; no market-volatility term yet.
- The risk model is deterministic and policy-driven, not learned end-to-end.

## 13. Safe Change Process

Use this order:

1. Change policy JSON only (preferred).
2. Dry-run risk job for one league.
3. Inspect action distribution and top recommendations.
4. Run live for all leagues.
5. Export and validate downstream consumption.

For structural changes (new columns/rules), add migration + tests in the same PR.
