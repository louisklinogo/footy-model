"""Weekly Model Training Job.

Trains all v2 model families (scoreline, anytime, corners) with:
- Current feature contracts
- Walk-forward evaluation
- Calibration
- Automatic candidate creation
- Auto-promotion evaluation

Schedule: Every Sunday at 6 AM UTC
Run manually: python src/jobs/train_weekly_v2.py
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.pipeline_logging import create_pipeline_run, finalize_pipeline_run
from src.common.script_logger import ScriptLogger, default_log_path

JOB_NAME = "train_weekly_v2"

_LOGGER: ScriptLogger | None = None
_LOG_PATH: Path | None = None

FAMILIES = ["scoreline", "anytime", "corners"]


@dataclass(frozen=True)
class Options:
    family: str | None
    skip_promotion: bool
    dry_run: bool
    log_file: str | None


def parse_args() -> Options:
    parser = argparse.ArgumentParser(description="Weekly v2 model training")
    _ = parser.add_argument("--family", choices=FAMILIES, help="Train only this family")
    _ = parser.add_argument("--skip-promotion", action="store_true", help="Skip auto-promotion")
    _ = parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    _ = parser.add_argument("--log-file", default=None)
    ns = parser.parse_args()
    return Options(
        family=ns.family if isinstance(ns.family, str) else None,
        skip_promotion=bool(ns.skip_promotion),
        dry_run=bool(ns.dry_run),
        log_file=ns.log_file if isinstance(ns.log_file, str) else None,
    )


def _configure_logging(log_file: str | None) -> None:
    global _LOGGER, _LOG_PATH
    _LOG_PATH = Path(log_file) if log_file else default_log_path(ROOT, JOB_NAME)
    _LOGGER = ScriptLogger(_LOG_PATH)


def _log_info(message: str) -> None:
    if _LOGGER is not None:
        _LOGGER.info(message)
    else:
        print(message)


def _log_warn(message: str) -> None:
    if _LOGGER is not None:
        _LOGGER.warn(message)
    else:
        print(message, file=sys.stderr)


def train_family(family: str, dry_run: bool = False) -> dict[str, object]:
    """Train a single model family and return results."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "model_artifacts" / "v2" / f"{family}_weekly_{timestamp}"

    if dry_run:
        return {
            "family": family,
            "output_dir": str(output_dir),
            "success": True,
            "dry_run": True,
        }

    # Run training via family-specific training script
    train_script = ROOT / "src" / "modeling" / "v2" / "families" / family / f"train_{family}.py"
    contract_path = ROOT / "model_v2" / "feature_contracts" / f"{family}.yaml"
    
    cmd = [
        sys.executable,
        str(train_script),
        "--output-dir", str(output_dir),
        "--model-version", f"weekly_{timestamp}",
    ]
    
    if contract_path.exists():
        cmd.extend(["--contract", str(contract_path)])

    _log_info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    return {
        "family": family,
        "output_dir": str(output_dir),
        "success": result.returncode == 0,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
    }


def update_production_pointers(family: str, artifact_dir: str, model_version: str) -> None:
    """Update production_pointers.json with the new model for a family.

    Args:
        family: Model family name (e.g., "scoreline", "anytime", "corners")
        artifact_dir: Relative path to the artifact directory from project root
        model_version: Version string for the model
    """
    pointers_path = ROOT / "model_artifacts" / "v2" / "production_pointers.json"

    # Read existing pointers
    if pointers_path.exists():
        pointers = json.loads(pointers_path.read_text(encoding="utf-8"))
    else:
        pointers = {"families": {}}

    # Ensure families dict exists
    if "families" not in pointers:
        pointers["families"] = {}

    # Update the family entry
    pointers["families"][family] = {
        "artifact_dir": artifact_dir,
        "model_version": model_version,
    }

    # Write back
    pointers_path.write_text(json.dumps(pointers, indent=2) + "\n", encoding="utf-8")
    _log_info(f"Updated production_pointers.json: {family} -> {model_version}")


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_file)

    _log_info("=" * 60)
    _log_info(f"WEEKLY TRAINING JOB - {datetime.now(UTC).isoformat()}")
    _log_info("=" * 60)

    details: dict[str, object] = {
        "family": args.family,
        "skip_promotion": args.skip_promotion,
        "dry_run": args.dry_run,
        "log_file": str(_LOG_PATH) if _LOG_PATH is not None else None,
    }
    run_id = create_pipeline_run(JOB_NAME, "weekly training started", details)

    families_to_train = [args.family] if args.family else FAMILIES
    results: list[dict[str, object]] = []

    try:
        for family in families_to_train:
            _log_info("")
            _log_info("=" * 60)
            _log_info(f"Training {family}...")
            _log_info("=" * 60)

            result = train_family(family, dry_run=args.dry_run)
            results.append(result)

            if result["success"] and not args.dry_run:
                # Update production pointers if not skipping promotion
                if not args.skip_promotion:
                    output_path = Path(str(result["output_dir"]))
                    # Get model version from artifact_metadata.json or use default
                    model_version = "weekly_" + datetime.now(UTC).strftime("%Y%m%d")
                    metadata_file = output_path / "artifact_metadata.json"
                    if metadata_file.exists():
                        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                        model_version = metadata.get("model_version", model_version)

                    # Compute relative artifact_dir from project root
                    artifact_dir = str(output_path.relative_to(ROOT)).replace("\\", "/")
                    update_production_pointers(family, artifact_dir, model_version)
                    result["model_version"] = model_version
                    _log_info(f"[OK] {family} training complete (version={model_version})")
                else:
                    _log_info(f"[OK] {family} training complete (promotion skipped)")
            elif result["success"] and args.dry_run:
                _log_info(f"[OK] {family} would be trained to {result['output_dir']}")
            else:
                _log_warn(f"[FAIL] {family} training failed")
                stderr = result.get("stderr")
                if stderr:
                    _log_warn(str(stderr)[-500:])

        # Finalize
        success_count = len([r for r in results if r.get("success")])
        finalize_pipeline_run(
            run_id,
            "success",
            f"trained {success_count}/{len(results)} families",
            {"results": results},
        )

        _log_info("")
        _log_info("=" * 60)
        _log_info("TRAINING COMPLETE")
        _log_info("=" * 60)
        for r in results:
            status = "[OK]" if r.get("success") else "[FAIL]"
            _log_info(f"  {status} {r['family']}: {r.get('output_dir', 'N/A')}")

        return 0

    except Exception as exc:
        import traceback
        error_msg = f"failed: {exc}\n{traceback.format_exc()}"
        finalize_pipeline_run(run_id, "fail", error_msg, {"results": results})
        _log_warn(f"Training failed: {exc}")
        return 1
    finally:
        if _LOGGER is not None:
            _LOGGER.close()


if __name__ == "__main__":
    raise SystemExit(main())
