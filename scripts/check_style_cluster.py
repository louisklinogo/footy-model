import pandas as pd
from pathlib import Path

style = pd.read_parquet('model_artifacts/style_clusters/cluster_labels.parquet')
print(f'Style cluster rows: {len(style)}')
print(f'Unique fixtures: {style["fixture_id"].nunique()}')
print(f'Has style_matchup: {"style_matchup" in style.columns}')
print()
print('Columns:')
for col in style.columns:
    print(f'  - {col}')