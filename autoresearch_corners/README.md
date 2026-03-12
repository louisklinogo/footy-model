# autoresearch_corners

Bounded `autoresearch`-style harness for the repo's existing v2 corners family.

This is **not** a new modeling stack. It wraps the real repo entrypoints so corners experiments stay comparable, reviewable, and isolated from production rollout work.

## What it runs

1. `src/modeling/v2/families/corners/train_corners.py`
2. `src/modeling/v2/eval/live_replacement_compare.py`

The driver trains one named corners challenger, then compares it against the current live stack using the repo's existing live-replacement report.

There are now two entrypoints:

- `run_experiment.py`: run one named corners challenger
- `search_runner.py`: run a bounded search over a fixed library of interaction blocks

## Quick start

1. Copy `experiment.example.json` to `experiment.local.json`
2. Edit the dataset path, contract path, `path_version`, and candidate tag
3. Dry-run the command plan:
   - `python autoresearch_corners/run_experiment.py --dry-run`
4. Execute the run:
   - `python autoresearch_corners/run_experiment.py`

## Bounded search

1. Copy `search.example.json` to `search.local.json`
2. Edit the dataset path, base contract, block list, and candidate prefix
3. Preview the generated search batch:
   - `python autoresearch_corners/search_runner.py --dry-run`
4. Execute the bounded search:
   - `python autoresearch_corners/search_runner.py`

Current corners research leader:

- `model_artifacts/v2/corners_totals_first_style_matchup_possession_box_touches_corners_against_featurepass_autoresearch_v1_candidate_20260312/`
- formal evaluation: `model_artifacts/v2/evaluation_corners_totals_first_style_matchup_possession_box_touches_corners_against_featurepass_autoresearch_v1_candidate_20260312/`
- current reusable anchored search config:
  - `autoresearch_corners/search.current_corners_research_leader.json`

Use that config when you want future bounded searches to branch from the current best non-promoted corners challenger instead of starting from the older plain `corners.yaml` surface.

## Outputs

- corners artifact dir: `model_artifacts/v2/<candidate_tag>/`
- live replacement dir: `artifacts/v2/family_replacement/live_replacement_<candidate_tag>/`
- local run logs/summaries: `autoresearch_corners/results/`

## Intended use

Use this to test bounded corners seams such as:

- feature contract variants
- `path_version` variants
- named interaction-block combinations
- totals-first vs calibrated surface variants

Do **not** use it to change schedulers, promotion state, or production model wiring.
