# Implementation Plan: NNMT Operating-Point Programme

**Branch**: `005-nnmt-operating-point` | **Date**: 2026-09-06 | **Spec**: [spec.md](./spec.md)

## Summary

Apply to NNMT the measurement chain that redefined what is knowable on NNNC, and
resolve two configuration changes NNMT inherited untested from that work.

The decisive measurement is first and costs nothing: if a perfect model is worth
about a point on NNMT — as it was on NNNC's old gate — then no model, label or
augmentation work on this family can pay, and the feature reports that and
re-scopes. Everything after it is contingent.

Phase 0 established three things that shape the work: NNMT's control must be
rebuilt rather than taken as-is (R1); the inherited label change reaches an
auxiliary head on this family, so it is not a single-parameter change (R2); and
exit-side questions cannot be settled by the oracle (R9).

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`)

**Primary Dependencies**: freqtrade backtesting; existing tooling —
`regime/run_arm.py`, `regime/trade_forensics.py`, `regime/tail/compare.py`,
`regime/ORACLE_GATE.md`, `NNNC/LabelParamsMixin.py`, and the
`NNNC_Oracle`/`NNNC_NoModel` pattern

**Storage**: JSON arm records in `regime/arms/`; freqtrade backtest archives

**Testing**: No unit tests. This is a measurement feature; correctness is
enforced by trade-hash liveness, by refusing metrics from failed or stale runs,
and by the two-seed rule.

**Target Platform**: macOS, Apple Silicon, 14 cores

**Project Type**: Research/measurement study inside an existing strategy repo

**Performance Goals**: Oracle, bypass and guard arms are backtests — minutes
each. Retrains are 8–45 minutes. Expected budget: ~20 backtest arms and 2–6
retrains, all sequential.

**Constraints**: W1 `20250901-20260831` is the sole acceptance window; W2 and W3
are in-sample. `Framework/` mechanism code must not be left modified. Every arm
carries its own parameter file. Two seeds before any acceptance.

**Scale/Scope**: One strategy family. No augmentation work, no architecture work.

## Constitution Check

| Principle | Status | Note |
|---|---|---|
| I. Evaluate in the deployment regime | **PASS** | W1 is the deployment window and the sole acceptance basis |
| II. Persistence across distant eras | **PASS-WITH-CAVEAT** | W3 is temporally distant but in-sample, AND is historically near-flat on this family (R7). Every W3-based statement must carry its trade count and an explicit flag; W3 may corroborate, never accept |
| III. Learnability is not edge | **PASS** | FR-019 forbids classifier metrics as inputs. Reinforced by NNNC evidence where val_mcc and precision rose monotonically while return fell |
| IV. Verify parameters are actually live | **PASS** | FR-010 requires inert guards to be named as inert; FR-023 requires the model-path override that stops a renamed arm training inside its own backtest |
| V. Lookahead-clean is not realistic | **PASS** | Oracle arms are deliberately lookahead and are labelled and non-promotable (FR-024). Fills are already volume-capped rather than rejected; FR-029 requires the capacity curve before any promotion |
| VI. Diagnose before tuning | **PASS** | US1 and US3 are diagnosis and precede every tuning story; FR-004 makes a negative diagnosis re-scope the feature |
| VII. Validate cheaply | **PASS** | FR-003 requires the decisive measurement to need no training. The first UNCONDITIONAL retrain is T015; Phase 1 trains only in the contingency that the control's model does not match its label |
| VIII. One parameter space at a time | **PASS** | R1's staged control and R2's pinning of the profit-head threshold both exist to keep changes single-variable |
| IX. Orthogonality earns a factor its place | **N/A** | No new factor |
| X. Structural fixes, not per-pair overrides | **PASS** | Non-goals exclude per-pair work |

No violations. Complexity Tracking is empty.

## Project Structure

    specs/005-nnmt-operating-point/
    ├── spec.md  plan.md  research.md  data-model.md
    ├── contracts/nnmt-arms.md
    ├── quickstart.md
    └── checklists/requirements.md

    NNMT/
    ├── NNMT_MLX_Base.json          NEW — pins the pre-inheritance control
    ├── NNMT_Oracle.py / .json      NEW — perfect entry signal
    ├── NNMT_NoModel.py / .json     NEW — model bypassed
    └── NNMT_<variant>.py / .json   NEW — one per screened variant

**Structure Decision**: additive only, and confined to `NNMT/`. The shared
tooling in `regime/` is reused unchanged; `NNNC/LabelParamsMixin.py` is imported
rather than duplicated. No `Framework/` change is anticipated — if one proves
necessary it must be a parameter default, never mechanism.

## Execution Phases

Strictly sequential. Phase 2 is a HARD GATE on **Phase 5 only** — that is
FR-004's definition of re-scoping, and it is stated identically here, in
tasks.md's Phase 2 callout, in T009 and in tasks.md's Dependencies section.
Phases 3 and 4 run regardless of the headroom result: Phase 3 EXPLAINS it and
Phase 4 fixes a live inconsistency that stands either way.

### Phase 1 — Build a valid control
Pin NNMT's pre-inheritance configuration (label 0.007, take-profit 0.032) in its
own parameter file, and establish its three-window baseline. Confirm whether the
models on disk match that label; if not, retrain once so the control is coherent.

### Phase 2 — HEADROOM (US1) — **HARD GATE**
Oracle arm, model-bypass arm, and the real model, all at the control
configuration, all three windows. Report the three-way comparison and the
headroom figure.

**If headroom is not materially larger than the resolution implied by the
observed trade count, the feature reports that and re-scopes.** That was the
outcome on NNNC's old gate and it is the expected case here. Re-scoping affects
Phase 5 only; Phases 3, 4, 6 and 7 still run.

### Phase 3 — Attribution (US3)
Guard leave-one-out at the control configuration. Identify inert guards. State
whether individual effects sum to the joint effect.

### Phase 4 — The inherited changes (US2)
*The label change AND the take-profit change: FR-030 covers the second, which
US2's requirements formerly omitted.*
Introduce the label change and the take-profit change **separately**, each
against the Phase 1 control, each on two seeds, each with its own operating
point rather than a fixed threshold. Pin the profit-head threshold so the label
arm is single-variable (R2). Choose values and leave configuration and models
consistent.

### Phase 5 — Operating point and exit side (US4, US5, US6)
Only if Phase 2 found headroom. The gate itself is chosen first, from Phase 3's
attribution by a stated rule (FR-031) — the operating point is the threshold AND
the guard settings. Then the confidence-threshold sweep with a stated decision
rule for per-window disagreement; then the exit-rvol gate; then the exit
signal, measured at both the chosen exit-gate value and fully open, and on
the real model rather than the oracle (R9).

### Phase 6 — Conclusion (US7)
Headroom, chosen configuration, what it forecloses, and — if a change is
recommended — its capacity curve.

## Complexity Tracking

No constitutional violations requiring justification.
