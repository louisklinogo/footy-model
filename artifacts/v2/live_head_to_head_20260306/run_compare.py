from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.db_utils import connect_db
from src.modeling.evaluation.predict_market_outcomes_fixtures_first import (
    add_derived_features,
    apply_imputation,
    build_prediction_rows,
    fetch_candidate_fixtures,
    load_artifacts as load_live_artifacts,
)
from src.modeling.evaluation.score_market_outcomes_fixtures_first import compute_actual
from src.modeling.v2.eval.metrics import binary_metric_values

ROOT = Path('.')
OUT_DIR = ROOT / 'artifacts' / 'v2' / 'live_head_to_head_20260306'
OUT_DIR.mkdir(parents=True, exist_ok=True)
HOLDOUT_PATHS = {
    'scoreline': ROOT / 'model_artifacts' / 'v2' / 'live_verification_scoreline_20260306' / 'holdout_predictions.csv',
    'corners': ROOT / 'model_artifacts' / 'v2' / 'live_verification_corners_20260306' / 'holdout_predictions.csv',
    'anytime': ROOT / 'model_artifacts' / 'v2' / 'live_verification_anytime_20260306' / 'holdout_predictions.csv',
}
FAMILY_MARKETS = {
    'scoreline': {'1x2_h','1x2_d','1x2_a','dc_1x','dc_x2','dc_12','ah_h05','ah_a05','ah_h15','ah_a15','eh_h1','eh_a1','o15','u35'},
    'corners': {'c75','c85','c95','c105','hc25','hc35','hc45','hc55','ac25','ac35','ac45','ac55'},
    'anytime': {'h_1up','a_1up','h_2up','a_2up'},
}
OVERLAP_MARKETS = set().union(*FAMILY_MARKETS.values())


def metric_row(frame: pd.DataFrame, p_col: str) -> dict[str, object]:
    out = binary_metric_values(frame['actual_int'].to_numpy(), frame[p_col].to_numpy())
    return {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in out.items()}


frames = []
for family, path in HOLDOUT_PATHS.items():
    df = pd.read_csv(path).rename(columns={'market': 'market_code', 'p_model': 'p_challenger', 'y_true': 'holdout_y_true'})
    df = df[df['market_code'].isin(OVERLAP_MARKETS)].copy()
    df['family'] = family
    df['match_datetime_utc'] = pd.to_datetime(df['match_datetime_utc'], utc=True, errors='coerce', format='mixed')
    frames.append(df[['family','market_code','fixture_id','holdout_y_true','p_challenger','league_code','match_datetime_utc']])
challenger = pd.concat(frames, ignore_index=True)
challenger['fixture_id'] = challenger['fixture_id'].astype(int)
challenger['holdout_y_true'] = challenger['holdout_y_true'].astype(float)
fixture_ids = sorted(challenger['fixture_id'].unique().tolist())

fixtures = fetch_candidate_fixtures(days=3, league=None, limit=None, backfill_days=60)
fixtures['match_datetime_utc'] = pd.to_datetime(fixtures['match_datetime_utc'], utc=True, errors='coerce')
fixtures['odds_snapshot_time_utc'] = pd.to_datetime(fixtures['odds_snapshot_time_utc'], utc=True, errors='coerce')
fixtures = fixtures[fixtures['fixture_id'].isin(fixture_ids)].copy()
features, imputation, models, model_failures, multiclass_model, multiclass_markets = load_live_artifacts()
featured = add_derived_features(fixtures)
featured['features_missing_count'] = featured.reindex(columns=features).isna().sum(axis=1)
scored = apply_imputation(featured, features, imputation)
live_rows, fallback_rows = build_prediction_rows(scored, features, models, model_failures, multiclass_model, multiclass_markets)
live = pd.DataFrame(live_rows, columns=['fixture_id','market_code','model_name','model_version','p_live','metadata_json'])
live = live[live['market_code'].isin(OVERLAP_MARKETS)].copy()
live['fixture_id'] = live['fixture_id'].astype(int)

conn = connect_db()
try:
    actuals = pd.read_sql(
        """
        SELECT f.fixture_id, f.match_datetime_utc, fr.home_goals, fr.away_goals,
               fs.h_corners, fs.a_corners,
               ils.home_led_by_1_any, ils.away_led_by_1_any,
               ils.home_led_by_2_any, ils.away_led_by_2_any
        FROM fixtures f
        JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
        LEFT JOIN fixture_stats_premium fs ON fs.fixture_id = f.fixture_id
        LEFT JOIN fixture_incident_lead_states ils ON ils.fixture_id = f.fixture_id
        WHERE f.fixture_id = ANY(%s)
        """,
        conn,
        params=(fixture_ids,),
    )
finally:
    conn.close()
actuals['fixture_id'] = actuals['fixture_id'].astype(int)

merged = challenger.merge(live[['fixture_id','market_code','p_live']], on=['fixture_id','market_code'], how='inner')
merged = merged.merge(actuals, on='fixture_id', how='left')
merged['actual'] = merged.apply(lambda r: compute_actual(r.to_dict()), axis=1)
merged['actual'] = pd.to_numeric(merged['actual'], errors='coerce')
merged['actual_available'] = merged['actual'].notna()
merged['holdout_matches_actual'] = np.where(merged['actual_available'], np.isclose(merged['holdout_y_true'], merged['actual']), False)
comparison = merged[merged['actual_available']].copy()
comparison['actual_int'] = comparison['actual'].astype(int)

per_market_rows = []
for (family, market_code), grp in comparison.groupby(['family', 'market_code'], sort=True):
    live_m = metric_row(grp, 'p_live')
    chal_m = metric_row(grp, 'p_challenger')
    per_market_rows.append({
        'family': family, 'market_code': market_code, 'n': int(len(grp)), 'fixtures': int(grp['fixture_id'].nunique()),
        'live_auc': live_m['auc'], 'challenger_auc': chal_m['auc'], 'delta_auc': None if live_m['auc'] is None or chal_m['auc'] is None else chal_m['auc'] - live_m['auc'],
        'live_brier': live_m['brier'], 'challenger_brier': chal_m['brier'], 'delta_brier': chal_m['brier'] - live_m['brier'],
        'live_log_loss': live_m['log_loss'], 'challenger_log_loss': chal_m['log_loss'], 'delta_log_loss': chal_m['log_loss'] - live_m['log_loss'],
        'live_ece': live_m['ece'], 'challenger_ece': chal_m['ece'], 'delta_ece': chal_m['ece'] - live_m['ece'],
        'challenger_better_brier': chal_m['brier'] < live_m['brier'], 'challenger_better_log_loss': chal_m['log_loss'] < live_m['log_loss'], 'challenger_better_auc': None if live_m['auc'] is None or chal_m['auc'] is None else chal_m['auc'] > live_m['auc'],
    })
per_market = pd.DataFrame(per_market_rows).sort_values(['family','market_code']).reset_index(drop=True)

segments = [('full_overlap', OVERLAP_MARKETS)] + [(f'{k}_overlap', v) for k, v in FAMILY_MARKETS.items()]
summary_rows = []
for name, markets in segments:
    grp = comparison[comparison['market_code'].isin(markets)].copy()
    live_m = metric_row(grp, 'p_live')
    chal_m = metric_row(grp, 'p_challenger')
    pm = per_market[per_market['market_code'].isin(markets)]
    summary_rows.append({
        'segment': name, 'rows': int(len(grp)), 'fixtures': int(grp['fixture_id'].nunique()), 'markets': sorted(markets),
        'live': live_m, 'challenger': chal_m,
        'delta': {'auc': None if live_m['auc'] is None or chal_m['auc'] is None else chal_m['auc'] - live_m['auc'], 'brier': chal_m['brier'] - live_m['brier'], 'log_loss': chal_m['log_loss'] - live_m['log_loss'], 'ece': chal_m['ece'] - live_m['ece']},
        'market_win_counts': {'brier': int((pm['challenger_better_brier'] == True).sum()), 'log_loss': int((pm['challenger_better_log_loss'] == True).sum()), 'auc': int((pm['challenger_better_auc'] == True).sum()), 'markets_total': int(len(pm))},
    })
report = {
    'cohort_summary': {
        'challenger_rows_loaded': int(len(challenger)), 'challenger_fixtures_loaded': int(challenger['fixture_id'].nunique()),
        'live_feature_rows_fetched': int(len(fixtures)), 'live_feature_fixtures_fetched': int(fixtures['fixture_id'].nunique()),
        'live_prediction_rows': int(len(live)), 'fallback_rows': int(fallback_rows),
        'matched_rows_before_actual_filter': int(len(merged)), 'matched_rows_scored': int(len(comparison)), 'matched_fixtures_scored': int(comparison['fixture_id'].nunique()),
        'holdout_vs_actual': {'rows_with_actual': int(merged['actual_available'].sum()), 'rows_matching_actual': int((merged['actual_available'] & merged['holdout_matches_actual']).sum()), 'rows_mismatching_actual': int((merged['actual_available'] & ~merged['holdout_matches_actual']).sum()), 'rows_without_actual': int((~merged['actual_available']).sum())},
        'date_min': str(challenger['match_datetime_utc'].min()), 'date_max': str(challenger['match_datetime_utc'].max()),
    },
    'segment_summaries': summary_rows,
}
match_col = 'match_datetime_utc' if 'match_datetime_utc' in comparison.columns else 'match_datetime_utc_x'
per_market.to_csv(OUT_DIR / 'per_market_comparison.csv', index=False)
comparison[['family','fixture_id','market_code','league_code',match_col,'actual_int','p_live','p_challenger']].rename(columns={match_col: 'match_datetime_utc'}).to_csv(OUT_DIR / 'matched_prediction_rows.csv', index=False)
(OUT_DIR / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
print(f'\nSaved: {OUT_DIR}')

