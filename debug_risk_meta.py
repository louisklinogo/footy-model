import sys
from pathlib import Path
import json

ROOT_DIR = Path('C:/Developer/soccer/footy-model').resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.evaluation.assess_prediction_risk import fetch_candidates

cands = fetch_candidates(model='market_outcome_gbm', version='fixtures_first_prematch_v1', league='T1', days=3, limit=1)

for c in cands:
    meta = c.get('metadata_json', {})
    if isinstance(meta, str):
        meta = json.loads(meta)
    print(f"Prediction: {c['prediction_id']} | Fixture: {c['fixture_id']} | League: {c['league_code']}")
    print(f"Home Played: {meta.get('home_played')} | Away Played: {meta.get('away_played')}")
