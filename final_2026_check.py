import pandas as pd
from src.db.db_utils import connect_db

def audit_2026():
    conn = connect_db()
    
    # 1. Total 2026 fixtures in DB
    query_all = "SELECT fixture_id, status FROM fixtures WHERE season IN ('25/26', '2025-26')"
    df_all = pd.read_sql(query_all, conn)
    
    # 2. Clusters from parquet
    labels_df = pd.read_parquet("model_artifacts/style_clusters/cluster_labels.parquet")
    
    # Merge
    merged = df_all.merge(labels_df, on='fixture_id', how='left')
    merged['has_cluster'] = ~merged['home_style_cluster'].isna()
    
    print("\n--- Final 2026 Style Cluster Coverage (Post-Backfill) ---")
    
    stats = merged.groupby('status').agg(
        total=('fixture_id', 'count'),
        covered=('has_cluster', 'sum')
    )
    stats['percent'] = (stats['covered'] / stats['total'] * 100).round(1)
    print(stats)
    
    total_2026 = len(merged)
    covered_2026 = merged['has_cluster'].sum()
    print(f"\nOverall 2026 Coverage: {covered_2026} / {total_2026} ({ (covered_2026/total_2026*100):.1f}%)")

    conn.close()

if __name__ == "__main__":
    audit_2026()
