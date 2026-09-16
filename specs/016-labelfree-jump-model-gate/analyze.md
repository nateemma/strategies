# Pre-implementation Analysis: 016-labelfree-jump-model-gate

**Date**: 2026-09-15 · **Run**: `/speckit-analyze` before `/speckit-implement` (SC-015)
**Artifacts**: spec.md, plan.md, research.md, data-model.md, contracts/jump_gate.md,
quickstart.md, tasks.md

## Pass 1 — findings (14, all remediated)

| ID | Category | Severity | Summary | Resolution |
|----|----------|----------|---------|------------|
| C1 | Constitution/Protocol | CRITICAL | Protocol §4 mandates ≥5 seeds; the jump fit is seed-dependent via k-means initialisation, yet every task ran one seed | FR-022, FR-023, SC-017, research R13, `SeedEnsemble` entity, `fit_ensemble()`/`seed_agreement()`, tasks T008a/T019a/T031a |
| C2 | Constitution/Coverage | CRITICAL | Protocol §5's overfitting-differential kill gate had no SC and no task; `stats.overfitting_differential()` sat unused | SC-018, task T047a |
| A1 | Ambiguity | HIGH | T037's STOP CHECKPOINT fired on "dramatically/materially better" with no threshold | FR-009 fixes it at **≥ 7.633 pp** (`BASELINE_HALF_WIDTH_PP`) |
| A2 | Ambiguity | HIGH | "Material disagreement" between implementations undefined | FR-016 fixes it at agreement **< 0.95** |
| A3 | Ambiguity | HIGH | T032's KILL CHECKPOINT keyed on `redundant_with_rvol` with no threshold | FR-024, SC-019: `novel_block_rate < 0.20` |
| I1 | Inconsistency | HIGH | quickstart ran the 15-backtest leak test *before* the free activity check and *before* the weight-reuse check | reordered; leak test now step 7, after activity (5) and weight reuse (6) |
| G1 | Coverage | HIGH | FR-012 required "Phase C's best fixed choice", which no task covered and no artifact identified | identified at `phase_c_trackability.py:177` as `max(ungated incumbent, cash)`; added to FR-012 and T045 |
| G2 | Coverage | HIGH | DSR computed "over the full trial log" but nothing registered this feature's trials | `register_trials()`, tasks T014a and T049a |
| A4 | Ambiguity | MEDIUM | "unchanged within the floor" undefined | SC-011: `\|Δ annualised return\| < 3.350 pp/yr` |
| G3 | Coverage | MEDIUM | SC-015 needs an analyze report before *and* after; only "after" had a task | task T004a — this file |
| C3 | Constitution | MEDIUM | workflow gate 5 waived on an argument about the wrong thing | plan.md now engages it, and explains why the earlier reasoning was wrong |
| U1 | Underspecification | MEDIUM | `K` unbounded in spec; the 10-of-15 activity rule lived only as a data-model invariant | K in Assumptions; SC-020 |
| T1 | Terminology | LOW | "dramatically" / "materially" used interchangeably | unified; "dramatically" now appears nowhere |
| L1 | Brittleness | LOW | quickstart hard-coded a test count | removed |

## The one worth reading twice

**C1.** The gated arms reuse the ungated fold weights and train nothing, so the backtests are
perfectly deterministic — and that is exactly what made the gap easy to miss. The jump model's own
fit is *not* deterministic: its coordinate descent starts from a k-means initialisation. Every
blocked bar, every block rate, every drawdown delta inherits that randomness.

Determinism in the expensive half of a pipeline reads as determinism in the whole of it. This
repository has already recorded the sharper version of the same failure — multi-task head weights
moved entries **less than a random seed did**, which is what closed that line. A regime gate is
precisely the kind of object that can look decisive and be initialisation noise.

SC-017 now makes the comparison explicit: if the per-fold block rate varies more across seeds than
it does between the gated and ungated arms, the gate is an artifact and the result is a KILL, not
a finding about regimes.

## A measurement taken during this pass

The claim that Phase C contains gates which never fired was previously supported by four rows read
by eye; three attempts to parse `regime/VERDICT.md` had returned 3 rows. The `rule` column
contains `|` characters, which over-splits those lines; rebuilding fields from the row ends fixes
it.

Parsed correctly: **18 of 32 rows (56%) have `n_taken == n_marginal`** — the gate admitted every
marginal trade — all at `edge_pct` exactly 0, with a single `beats_bar True` in the whole table.
Over half of Phase C's evidence is gates that never fired. The earlier claim was conservative, and
spec.md now carries the verified figure in place of the hand-read sample.

## Pass 2 — verification after remediation

| check | result |
|---|---|
| Functional requirements | 24, all referenced in tasks |
| Success criteria | 21, all referenced in tasks |
| Tasks | 62 (55 + 7 inserted), unique, base IDs contiguous T001–T055 |
| Story-label / phase integrity | 0 mismatches |
| Vague gate triggers remaining | 0 — every stop and kill gate fires on a number |
| Constitution principles | I–X all PASS; VII and IX carry written justifications |
| Quickstart ordering | cheap kill gates precede every backtest |

**Critical issues: 0. Ready for `/speckit-implement`.**
