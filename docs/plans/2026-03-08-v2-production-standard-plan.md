# V2 Production-Standard Plan

Date: 2026-03-08  
Owner: Augment Agent / Senior DS loop  
Status: active canonical plan

## Purpose

Make the v2 family architecture the production-grade successor to `market_outcome_gbm / fixtures_first_prematch_v1` without losing the practical strengths that currently help live win on parts of the shared market surface.

This file is the canonical strategy doc for the current v2 push. It is not the live runtime source of truth; see `docs/current_state.md` for that.

## Strategy decision

Proceed with v2 as the successor architecture, but do **not** treat the current pure family implementation as sufficient. The target is a maintainable hybrid system:

- family base models remain the backbone,
- direct odds are used where real market prices exist,
- proxy odds are used only where adjacent liquid markets justify them,
- structural derivation remains primary where no reliable odds anchor exists,
- residual correction, runtime calibration, and explicit backoff tiers become first-class production components.

## Why this plan exists

The latest matched benchmark showed:

- scoreline directional is the closest family to promotion,
- corners still trails live on the corrected overlap,
- anytime still trails live overall,
- internal v2 promotion gates and true live-replacement evidence are not yet the same thing,
- v2 serves a broader market surface than live, so shared markets and v2-only tail markets must be judged differently.

## Replacement standard

v2 is production-ready only when all are true:

1. It beats or at least matches live on shared markets it is claiming ownership of.
2. It remains well calibrated under runtime serving logic, not only offline evaluation logic.
3. It behaves coherently on broader v2-only markets where live is not the right comparator.
4. It degrades safely when feature availability is partial.
5. It can be promoted family-by-family with a reversible rollback path.

## Market-tier policy

### Tier 1: direct-odds shared markets

Examples:

- `1x2_*`, `dc_*`, core `ah_*`, core totals, liquid corners totals

Policy:

- direct odds are first-class inputs,
- promotion requires matched live-replacement evidence,
- benchmark against live, calibration, and league stability.

### Tier 2: proxy-odds markets

Examples:

- adjacent totals/handicaps,
- some derived corners or family derivatives,
- markets with liquid neighboring lines but weak direct pricing.

Policy:

- use proxy odds only when structurally justified,
- judge on calibration, coherence, temporal stability, and replacement evidence where overlap exists.

### Tier 3: structural-first tail markets

Examples:

- v2-only long-tail derived buckets,
- sparse or unsupported market shapes.

Policy:

- do not force fake price anchors,
- optimize for family coherence, calibration, monotonicity, and support,
- do not require live replacement where live never meaningfully owned the market.

## Target production architecture

Each family should converge to the same runtime shape:

1. **Base family model**
   - scoreline generator, corners count model, or anytime path/state model.
2. **Odds / proxy-odds prior layer**
   - direct priors where available, justified proxy priors where not.
3. **Residual correction layer**
   - market-level discriminative correction using strong tabular features.
4. **Runtime calibration layer**
   - loaded and applied in serving, not only in evaluation.
5. **Feature-tier backoff policy**
   - explicit behavior for partial upstream feature availability.
6. **Promotion and monitoring contract**
   - shared markets judged against live, tail markets judged on coherence/stability.

## Feature/backoff policy

Every family should support explicit feature tiers:

- Tier A: full feature set available
- Tier B: situational layer missing
- Tier C: availability / lineup layer missing
- Tier D: structural core + odds/proxy only

Requirements:

- missingness indicators are explicit features,
- prediction artifacts record which tier was used,
- benchmark slices are reported by feature tier,
- no silent downgrade paths.

## Family priorities

### 1. Scoreline v2.1

This is the first promotion target.

Work:

- keep the scoreline base,
- add residual heads for `1x2_*`, `dc_*`, core `ah_*`, `eh_*`, `o15`, `u35`,
- strengthen direct-odds integration,
- improve low-score/dependence handling,
- wire runtime calibration into serving,
- benchmark by shared core market and non-fallback slices.

Success standard:

- core scoreline markets beat or clearly match live on matched rows,
- no hidden reliance on offline-only calibration,
- no regression under reduced feature tiers.

### 2. Corners v2.1

This is the biggest statistical rework.

Work:

- move to overdispersed count logic,
- model home/away corners jointly with shared game-tempo effects,
- add corners-specific features,
- use direct or proxy odds where justified,
- add residual correction and per-market calibration.

Success standard:

- corners no longer loses broadly to live on shared rows,
- team and total corners are better calibrated,
- no unsupported market is treated as promotion-critical just because it exists.

### 3. Anytime v2.1

This is a path/state modeling problem, not just a static derivative.

Work:

- redesign around scoring hazard / state transitions,
- add favorite-home asymmetry features,
- calibrate `h_1up`, `a_1up`, `h_2up`, `a_2up` separately,
- add residual correction.

Success standard:

- eliminate the current home-side losses,
- reach neutral-to-positive replacement evidence on shared anytime markets,
- remain stable by league and side.

## Evaluation and promotion redesign

The promotion system must split three questions that are currently too entangled:

1. **Shared-market replacement gate**
   - challenger must beat or credibly match live on matched rows.
2. **Expanded-market coherence gate**
   - v2-only markets must pass calibration/support/stability/coherence checks.
3. **Runtime realism gate**
   - calibration must be applied in serving and results must hold under true feature-tier availability.

Required additions:

- live-incumbent comparison as a first-class promotion input,
- bootstrap or repeated-slice stability checks for pooled wins,
- per-tier and per-league reporting,
- explicit family ownership and rollback metadata.

## Phased implementation roadmap

### Phase 0 — Lock the contract

- finalize market tiers,
- finalize promotion rules by market class,
- make runtime calibration non-optional where evaluation depends on it.

### Phase 1 — Ship scoreline v2.1

- residual scoreline overlays,
- stronger odds use,
- dependence improvements,
- runtime calibration,
- feature-tier reporting.

### Phase 2 — Harden evaluation and serving parity

- make live-replacement gating canonical,
- ensure prediction-time behavior matches evaluation-time behavior,
- persist tier-aware and league-aware reports.

### Phase 3 — Rebuild corners

- overdispersed joint count family,
- corners-specific features,
- direct/proxy odds policy,
- residual correction and calibration.

### Phase 4 — Rebuild anytime

- path-aware modeling,
- asymmetry features,
- side-specific calibration,
- residual correction.

### Phase 5 — Controlled family-by-family rollout

- shadow mode,
- canary by family and market subset,
- reversible cutover,
- retire live ownership only where v2 has earned it.

## Exit criteria for full successor status

v2 becomes the full successor only when:

- scoreline, corners, and anytime each clear their own replacement gates,
- runtime calibration is active and verified,
- feature-tier degradation is measured and acceptable,
- v2-only tail markets have explicit coherence and support justification,
- rollback-safe family ownership is documented.

## Immediate next actions

1. Implement scoreline residual overlays and runtime calibration serving parity.
2. Redesign the promotion contract so shared markets are judged against live, not only against frozen baseline artifacts.
3. Add explicit market-tier metadata for direct-odds, proxy-odds, and structural-first markets.
4. Start the corners rebuild only after scoreline v2.1 and the new gate logic are in place.

## File placement rules for this program

- Canonical strategy doc: this file
- Execution tracker: `docs/v2_upgrade_tracker.md`
- Live runtime truth: `docs/current_state.md`
- Reproducible benchmark outputs: `artifacts/v2/family_replacement/`
- Candidate artifacts: `model_artifacts/v2/<candidate_name>/`