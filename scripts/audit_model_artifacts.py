"""
Audit model training and prediction artifacts.

This script checks:
1. Model registry table (if exists) and active models
2. Recent prediction runs from predictions table
3. Model artifact files (.pkl, .joblib) in model_artifacts/
4. V2 main models (scoreline, anytime, corners)
5. Training logs and metadata files
6. Staleness indicators
"""
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, ".")

from src.db.db_utils import connect_db

# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ARTIFACTS_DIR = ROOT_DIR / "model_artifacts"


def format_age(dt: datetime | None) -> str:
    """Format the age of a datetime relative to now."""
    if dt is None:
        return "N/A"
    age = datetime.now(UTC) - dt
    if age.days > 365:
        years = age.days // 365
        return f"{years}y {age.days % 365}d"
    elif age.days > 0:
        return f"{age.days}d {age.seconds // 3600}h"
    elif age.seconds >= 3600:
        hours = age.seconds // 3600
        return f"{hours}h {(age.seconds % 3600) // 60}m"
    elif age.seconds >= 60:
        return f"{age.seconds // 60}m"
    else:
        return f"{age.seconds}s"


def check_model_registry(cur):
    """Check for model_registry table and show active models."""
    print("\n" + "=" * 90)
    print("MODEL REGISTRY")
    print("=" * 90)

    # Check if model_registry table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'model_registry'
        )
    """)
    table_exists = cur.fetchone()[0]

    if not table_exists:
        print("\n  [INFO] model_registry table does not exist in the database")
        print("  [INFO] Models are managed via filesystem artifacts in model_artifacts/")
        return

    print("\n  [INFO] model_registry table exists")

    # Try to get active models
    try:
        cur.execute("""
            SELECT * FROM model_registry 
            WHERE is_active = true 
            ORDER BY trained_at DESC
            LIMIT 20
        """)
        rows = cur.fetchall()
        if rows:
            print(f"\n  Active models ({len(rows)} found):")
            for row in rows:
                print(f"    - {row}")
        else:
            print("\n  No active models found in registry")
    except Exception as e:
        print(f"\n  [WARN] Could not query model_registry: {e}")


def check_predictions(cur):
    """Check recent prediction runs from predictions table."""
    print("\n" + "=" * 90)
    print("PREDICTION RUNS")
    print("=" * 90)

    # Check if predictions table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'predictions'
        )
    """)
    predictions_exists = cur.fetchone()[0]

    # Check if fixture_predictions table exists (as specified in requirements)
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'fixture_predictions'
        )
    """)
    fixture_predictions_exists = cur.fetchone()[0]

    if not predictions_exists and not fixture_predictions_exists:
        print("\n  [WARN] Neither 'predictions' nor 'fixture_predictions' tables exist")
        return

    # Use predictions table if it exists, otherwise fixture_predictions
    table_name = "predictions" if predictions_exists else "fixture_predictions"
    print(f"\n  Using table: {table_name}")

    try:
        # Get column info
        cur.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """)
        columns = cur.fetchall()
        print(f"\n  Table columns: {[c[0] for c in columns]}")

        # Get prediction counts by model_name and model_version
        cur.execute(f"""
            SELECT model_name, model_version, COUNT(*) as count, MAX(created_at) as latest_run
            FROM {table_name}
            GROUP BY model_name, model_version
            ORDER BY latest_run DESC
            LIMIT 15
        """)
        rows = cur.fetchall()
        if rows:
            print(f"\n  Prediction runs by model ({len(rows)} models):")
            print(f"  {'Model Name':<35} {'Version':<30} {'Count':>10} {'Latest Run':<25}")
            print("  " + "-" * 100)
            for row in rows:
                model_name = str(row[0])[:35] if row[0] else "N/A"
                model_version = str(row[1])[:30] if row[1] else "N/A"
                count = row[2] or 0
                latest = row[3]
                latest_str = str(latest)[:19] if latest else "N/A"
                print(f"  {model_name:<35} {model_version:<30} {count:>10} {latest_str:<25}")
        else:
            print("\n  No prediction records found")

        # Recent prediction runs (last 7 days)
        cur.execute(f"""
            SELECT DATE(created_at) as run_date, COUNT(*) as predictions
            FROM {table_name}
            WHERE created_at > NOW() - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY run_date DESC
        """)
        rows = cur.fetchall()
        if rows:
            print(f"\n  Predictions in last 7 days:")
            for row in rows:
                print(f"    {row[0]} | {row[1]:,} predictions")
        else:
            print("\n  [WARN] No predictions in the last 7 days!")

        # Most recent predictions
        cur.execute(f"""
            SELECT model_name, model_version, market_code, created_at
            FROM {table_name}
            ORDER BY created_at DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        if rows:
            print(f"\n  Most recent predictions:")
            for row in rows:
                model_name = row[0] or "N/A"
                version = row[1] or "N/A"
                market = row[2] or "N/A"
                created = row[3]
                age_str = format_age(created)
                print(f"    {model_name}/{version} | {market} | {age_str} ago")

    except Exception as e:
        print(f"\n  [ERROR] Could not query predictions: {e}")


def scan_model_artifacts():
    """Scan model_artifacts/ directory for .pkl and .joblib files."""
    print("\n" + "=" * 90)
    print("MODEL ARTIFACT FILES (.pkl, .joblib)")
    print("=" * 90)

    if not MODEL_ARTIFACTS_DIR.exists():
        print(f"\n  [ERROR] Model artifacts directory not found: {MODEL_ARTIFACTS_DIR}")
        return {}

    # Find all .pkl and .joblib files
    pkl_files = list(MODEL_ARTIFACTS_DIR.rglob("*.pkl"))
    joblib_files = list(MODEL_ARTIFACTS_DIR.rglob("*.joblib"))
    all_model_files = pkl_files + joblib_files

    print(f"\n  Total .pkl files: {len(pkl_files)}")
    print(f"  Total .joblib files: {len(joblib_files)}")
    print(f"  Total model files: {len(all_model_files)}")

    # Group by subdirectory
    subdir_stats: dict[str, dict] = {}
    for f in all_model_files:
        # Get relative path from model_artifacts
        rel_path = f.relative_to(MODEL_ARTIFACTS_DIR)
        parts = rel_path.parts
        subdir = parts[0] if len(parts) > 1 else "root"

        if subdir not in subdir_stats:
            subdir_stats[subdir] = {
                "count": 0,
                "files": [],
                "newest_mtime": None,
                "oldest_mtime": None,
                "newest_file": None,
                "oldest_file": None,
            }

        subdir_stats[subdir]["count"] += 1
        subdir_stats[subdir]["files"].append(f)

        mtime = datetime.fromtimestamp(f.stat().st_mtime, UTC)
        if subdir_stats[subdir]["newest_mtime"] is None or mtime > subdir_stats[subdir]["newest_mtime"]:
            subdir_stats[subdir]["newest_mtime"] = mtime
            subdir_stats[subdir]["newest_file"] = f.name
        if subdir_stats[subdir]["oldest_mtime"] is None or mtime < subdir_stats[subdir]["oldest_mtime"]:
            subdir_stats[subdir]["oldest_mtime"] = mtime
            subdir_stats[subdir]["oldest_file"] = f.name

    # Print per-subdirectory stats
    print(f"\n  Files per subdirectory:")
    print(f"  {'Subdirectory':<50} {'Count':>8} {'Newest File':<30} {'Age':<15}")
    print("  " + "-" * 105)

    # Sort by count descending
    for subdir, stats in sorted(subdir_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        age_str = format_age(stats["newest_mtime"])
        newest_file = stats["newest_file"][:30] if stats["newest_file"] else "N/A"
        print(f"  {subdir:<50} {stats['count']:>8} {newest_file:<30} {age_str:<15}")

    return subdir_stats


def check_v2_main_models():
    """Check model_artifacts/v2/ for main models (scoreline, anytime, corners)."""
    print("\n" + "=" * 90)
    print("V2 MAIN MODELS (scoreline, anytime, corners)")
    print("=" * 90)

    v2_dir = MODEL_ARTIFACTS_DIR / "v2"
    if not v2_dir.exists():
        print(f"\n  [ERROR] V2 directory not found: {v2_dir}")
        return

    main_families = ["scoreline", "anytime", "corners"]

    for family in main_families:
        family_dir = v2_dir / family
        print(f"\n  --- {family.upper()} ---")

        if not family_dir.exists():
            print(f"    [WARN] Directory not found: {family_dir}")
            continue

        # Check for model files
        pkl_files = list(family_dir.glob("*.pkl"))
        joblib_files = list(family_dir.glob("*.joblib"))
        model_files = pkl_files + joblib_files

        print(f"    Model files: {len(model_files)}")
        if model_files:
            for f in sorted(model_files):
                mtime = datetime.fromtimestamp(f.stat().st_mtime, UTC)
                age_str = format_age(mtime)
                size_kb = f.stat().st_size / 1024
                print(f"      {f.name:<40} | {size_kb:>8.1f} KB | {age_str} old")

        # Check for metadata files
        metadata_files = [
            "artifact_metadata.json",
            "features.json",
            "training_report.json",
            "metrics_holdout.json",
            "metrics_walkforward.json",
            "calibration_report.json",
        ]

        print(f"    Metadata files:")
        for meta_file in metadata_files:
            meta_path = family_dir / meta_file
            if meta_path.exists():
                mtime = datetime.fromtimestamp(meta_path.stat().st_mtime, UTC)
                age_str = format_age(mtime)
                print(f"      [OK] {meta_file:<30} | {age_str} old")
            else:
                print(f"      [--] {meta_file:<30} | not found")

        # Check for training timestamp
        artifact_meta = family_dir / "artifact_metadata.json"
        if artifact_meta.exists():
            import json

            try:
                with open(artifact_meta) as f:
                    meta = json.load(f)
                trained_at = meta.get("trained_at_utc")
                model_name = meta.get("model_name", "N/A")
                model_version = meta.get("model_version", "N/A")
                print(f"    Model: {model_name} v{model_version}")
                print(f"    Trained at: {trained_at or 'N/A'}")
            except Exception as e:
                print(f"    [WARN] Could not read metadata: {e}")


def check_training_logs():
    """Look for training logs or metadata files."""
    print("\n" + "=" * 90)
    print("TRAINING LOGS AND METADATA")
    print("=" * 90)

    # Check for training reports in model_artifacts
    training_reports = list(MODEL_ARTIFACTS_DIR.rglob("training_report.json"))
    if training_reports:
        print(f"\n  Found {len(training_reports)} training_report.json files:")
        # Show most recent 10
        reports_with_mtime = [
            (r, datetime.fromtimestamp(r.stat().st_mtime, UTC)) for r in training_reports
        ]
        reports_with_mtime.sort(key=lambda x: x[1], reverse=True)

        for report_path, mtime in reports_with_mtime[:10]:
            age_str = format_age(mtime)
            rel_path = report_path.relative_to(MODEL_ARTIFACTS_DIR)
            print(f"    {str(rel_path)[:60]:<60} | {age_str} old")
    else:
        print("\n  No training_report.json files found")

    # Check for artifact_metadata.json files
    artifact_metas = list(MODEL_ARTIFACTS_DIR.rglob("artifact_metadata.json"))
    if artifact_metas:
        print(f"\n  Found {len(artifact_metas)} artifact_metadata.json files")

    # Check for logs directory
    logs_dir = ROOT_DIR / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            print(f"\n  Log files in logs/: {len(log_files)}")
            for log in log_files[:5]:
                mtime = datetime.fromtimestamp(log.stat().st_mtime, UTC)
                age_str = format_age(mtime)
                print(f"    {log.name:<40} | {age_str} old")
    else:
        print("\n  No logs/ directory found")


def check_staleness(subdir_stats: dict):
    """Report findings about staleness."""
    print("\n" + "=" * 90)
    print("STALENESS REPORT")
    print("=" * 90)

    now = datetime.now(UTC)
    stale_threshold = timedelta(days=7)
    very_stale_threshold = timedelta(days=30)

    print(f"\n  Current time (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stale threshold: {stale_threshold.days} days")
    print(f"  Very stale threshold: {very_stale_threshold.days} days")

    stale_dirs = []
    very_stale_dirs = []

    for subdir, stats in subdir_stats.items():
        if stats["newest_mtime"]:
            age = now - stats["newest_mtime"]
            if age > very_stale_threshold:
                very_stale_dirs.append((subdir, age, stats["newest_file"]))
            elif age > stale_threshold:
                stale_dirs.append((subdir, age, stats["newest_file"]))

    if very_stale_dirs:
        print(f"\n  [CRITICAL] Very stale directories (>30 days old):")
        for subdir, age, newest_file in very_stale_dirs:
            print(f"    {subdir:<50} | {age.days} days old | newest: {newest_file}")

    if stale_dirs:
        print(f"\n  [WARNING] Stale directories (7-30 days old):")
        for subdir, age, newest_file in stale_dirs:
            print(f"    {subdir:<50} | {age.days} days old | newest: {newest_file}")

    if not stale_dirs and not very_stale_dirs:
        print(f"\n  [OK] All model artifacts are fresh (<7 days old)")

    # Check for candidate directories that may need promotion
    print("\n  Candidate directories (potential promotion candidates):")
    v2_dir = MODEL_ARTIFACTS_DIR / "v2"
    if v2_dir.exists():
        candidate_dirs = [d for d in v2_dir.iterdir() if d.is_dir() and "candidate" in d.name.lower()]
        if candidate_dirs:
            # Sort by modification time
            candidates_with_mtime = [
                (d, datetime.fromtimestamp(d.stat().st_mtime, UTC)) for d in candidate_dirs
            ]
            candidates_with_mtime.sort(key=lambda x: x[1], reverse=True)

            print(f"    Found {len(candidate_dirs)} candidate directories:")
            for candidate, mtime in candidates_with_mtime[:10]:
                age_str = format_age(mtime)
                print(f"      {candidate.name:<60} | {age_str} old")
        else:
            print("    No candidate directories found")


def main():
    print("=" * 90)
    print("MODEL ARTIFACTS AUDIT")
    print(f"Run time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Project root: {ROOT_DIR}")
    print("=" * 90)

    # Connect to database
    try:
        conn = connect_db()
        cur = conn.cursor()
        print("\n[OK] Database connected successfully")
    except Exception as e:
        print(f"\n[ERROR] Could not connect to database: {e}")
        cur = None

    # Run checks
    if cur:
        check_model_registry(cur)
        check_predictions(cur)
        cur.close()
        conn.close()

    subdir_stats = scan_model_artifacts()
    check_v2_main_models()
    check_training_logs()
    check_staleness(subdir_stats)

    print("\n" + "=" * 90)
    print("AUDIT COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
