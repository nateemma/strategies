# Implementation Plan: Oracle-Instrumented Exit Diagnosis

**Branch**: `006-oracle-exit-diagnosis` | **Date**: 2026-09-07 | **Spec**: [spec.md](./spec.md)

## Summary

A perfect-entry arm earns LESS than the real model. Use it as an instrument — it
holds entry quality perfect and CONSTANT, so any difference between arms is
attributable to the exit alone — and find out how much the exit stack is leaving
on the table, which mechanism is responsible, and whether any of it transfers to
a real model.

Phase 0 established four things that shape the work: the prior exit study is
STALE rather than redundant, because every one of its arms was measured at a
label that lost 19% (R1); the payoff GEOMETRY may be the entire answer and is
nearly free to test, so it goes first (R3); nothing here requires a
`Framework/` change or a single retrain (R4, R9); and a patient exit is the
intervention most likely to flatter itself on this instrument, which is why
real-model confirmation is P1 and inside the feature (R8).

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`)

**Primary Dependencies**: freqtrade backtesting; existing tooling —
`regime/run_arm.py`, `regime/trade_forensics.py`, `regime/tail/compare.py`,
`regime/ORACLE_GATE.md`, `NNNC/_hold_window.py` (HoldWindowMixin), the
`NNMT_Oracle` / `NNNC_Oracle` pattern, `NNMT/NNMT_MLX_Base.*` as the pinned
control

**Storage**: JSON arm records in `regime/arms/`; freqtrade backtest archives

**Testing**: No unit tests. This is a measurement feature; correctness is
enforced by trade-hash liveness, by refusing metrics from failed or stale runs,
by the instrument-compliance check in Phase 1, and by paired resolution.

**Target Platform**: macOS, Apple Silicon, 14 cores

**Project Type**: Research/measurement study inside an existing strategy repo

**Performance Goals**: Backtests only — minutes per arm. **Expected budget: ~30
arms and ZERO retrains** (R9).

**Constraints**: W1 `20250901-20260831` is the sole acceptance window; W2 and W3
are in-sample and W3 cannot carry a sub-threshold conclusion (R7). `Framework/`
must not be modified. Every arm carries its own parameter file. Entry guards must
not be opened on either family.

**Scale/Scope**: Two strategy families, exit side only. No entry work, no label
work, no architecture, no augmentation.

## Constitution Check

| Principle | Status | Note |
|---|---|---|
| I. Evaluate in the deployment regime | **PASS** | W1 is the deployment window and the sole acceptance basis |
| II. Persistence across distant eras | **PASS-WITH-CAVEAT** | W3 is temporally distant but in-sample AND retains only 7.5% of gross flow, so its sign flips under perturbation (R7). Every window is reported with that ratio; W3 may corroborate direction, never accept |
| III. Learnability is not edge | **PASS** | FR-017 forbids classifier metrics as inputs. No model is trained, so none is produced |
| IV. Verify parameters are actually live | **PASS** | FR-020 requires identical trade hashes to be reported INCONCLUSIVE; R10 records the singleton-contamination trap when reading configuration |
| V. Lookahead-clean is not realistic | **PASS** | Perfect-entry arms are deliberately lookahead, labelled, and non-promotable (FR-019). FR-013 requires the capacity curve before any promotion |
| VI. Diagnose before tuning | **PASS** | The whole feature is diagnosis. US1 validates the instrument before US2 measures anything, and FR-005 makes a small bound re-scope the work |
| VII. Validate cheaply | **PASS** | ZERO retrains (R9). R3 puts the cheapest decisive measurement — the reward:risk arithmetic — first |
| VIII. One parameter space at a time | **PASS-WITH-JUSTIFICATION** | R5 deliberately measures the JOINT bound first, then decomposes strictly one-at-a-time. The joint arm is a bound and is never promotable; every arm that could be ADOPTED moves exactly one mechanism |
| IX. Orthogonality earns a factor its place | **N/A** | No new factor |
| X. Structural fixes, not per-pair overrides | **PASS** | Non-goals exclude per-pair work |

No violations. The one justified deviation (VIII) is recorded in Complexity
Tracking.

## Project Structure

    specs/006-oracle-exit-diagnosis/
    ├── spec.md  plan.md  research.md  data-model.md
    ├── contracts/exit-arms.md
    ├── quickstart.md
    └── checklists/requirements.md

    NNMT/
    ├── NNMT_Ora_Patient.py / .json     NEW — joint bound, all exits relaxed
    ├── NNMT_Ora_NoStop.py / .json      NEW — ATR stop disabled alone
    └── NNMT_Ora_Band*.py / .json       NEW — one per stop band tested
    NNNC/
    └── NNNC_Ora_*.py / .json           NEW only where 004's arm is stale

**Structure Decision**: additive only, confined to the family directories. No
`Framework/` change (R4). `NNNC/_hold_window.py`'s HoldWindowMixin is IMPORTED,
never duplicated. Feature 004's oracle variants are re-read before anything new
is built, and re-run rather than re-derived where their configuration is stale.

## Execution Phases

Strictly sequential. Phase 1 gates everything; Phase 2 gates Phases 3–5.

### Phase 1 — Validate the instrument (US1) — **BLOCKING**
Explain or bound the 84.1% compliance of perfect entries with their own label.
Test R2's causes cheapest-first, eliminate the subset operations by argument and
measurement, and deliver either a named cause with a correction and restated
figures, or a quantified bound carried into every number the feature produces.

**Nothing downstream is trustworthy until this closes.**

### Phase 2 — The geometry, then the bound (US2) — **HARD GATE**
First the reward:risk arithmetic per family (R3), which is nearly free and may
answer the whole question: NNMT promises +2.0% against a −2%…−4% stop and a 2.2%
harvest. Recompute the stop-breach rate per family with the CORRECT band.

Then the joint bound: a patient oracle with the ATR stop off, ROI disabled, the
take-profit lifted clear and the unclog suppressed, held to the label horizon.

**If the bound is not materially larger than the study's resolution, the exit is
not the constraint. Report that and re-scope Phases 3–5** (FR-005).

### Phase 3 — Attribute the bound to mechanisms (US3)
Re-enable each mechanism alone against the patient arm — stop, ROI, take-profit,
unclog, max-hold. Report each contribution and the exit-reason mix it moves, name
any INERT mechanism, and state whether the individual effects sum to the joint
effect. Re-run the stale stop bands from feature 004 at the current label (R1).

### Phase 4 — The target/harvest ordering (US4)
Place the harvest point below, at, and above the label's target on the
perfect-entry arm, and report whether any gain comes from harvesting more moves
or from taking smaller ones. Feature 004's monotonic take-profit result is the
prior; R3 explains why it should be expected.

### Phase 5 — Confirm on the REAL model (US5) — the only path to promotion
Every candidate re-measured on the real model, two seeds, paired by trade,
judged on W1. A candidate that helps the instrument and not the real model is
recorded as instrument-only and is NOT promoted (FR-012). Anything that survives
gets its capacity curve before promotion (FR-013).

### Phase 6 — Conclusion (US6)
The exit's opportunity cost, the responsible mechanism, what transferred, and
what the result forecloses for both families.

## Complexity Tracking

| deviation | why | why the simpler path was rejected |
|---|---|---|
| Constitution VIII — the Phase 2 bound moves several mechanisms at once | It is the feature's gating number. One arm decides whether Phases 3–5 are worth running | Building up one mechanism at a time would spend the whole decomposition budget before learning whether there is anything to decompose. The joint arm is a BOUND and is never promotable; every potentially-adoptable arm moves exactly one mechanism |
