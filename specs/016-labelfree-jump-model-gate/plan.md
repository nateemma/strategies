# Implementation Plan: labelfree-jump-model-gate

**Branch**: `016-labelfree-jump-model-gate` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/016-labelfree-jump-model-gate/spec.md`

## Summary

Build an **online (filtered) sparse jump model** over causal volatility features, use its state
as a **hard entry gate** on the incumbent `NNNC_MLX`, and measure what the gate does to
**drawdown** through the pooled walk-forward — with the deliberate **smoothed-state leak run
first** so the size of the avoided leak is published beside the honest number.

This is the study's last brief and an explicit **negative control**: `regime/VERDICT.md` already
returned FAIL over eight regime instruments. The brief's own expectation is that the gate buys
drawdown, not return, so acceptance is a **deployment** modification and never an alpha claim.

Two measurements taken during planning shape the whole design:

1. **Peak trade concurrency is 9 of a cap of 10** across all 15 folds (0 folds reach the cap).
   Gating can therefore only *remove* trades via the concurrency channel — it can never free a
   slot that manufactures a trade the ungated run never had.
2. **But freqtrade also refuses a second position on a pair that already holds one**, and that
   suppression channel *is* reachable by gating: block a pair's first trade and a later signal on
   the same pair, previously dropped, becomes takeable. So a cheap filter over the recorded 378
   trades is **not provably exact**, and real gated backtests are the primary instrument.

Feature 013 is the precedent that settles this: a single real backtest exposed four independent
fidelity bugs in its exit surrogate. The gated arms here need no retraining — a hard entry gate
does not change the model — so real backtests cost 15 evaluations per arm and nothing else.

## Technical Context

**Language/Version**: Python 3.12.13 at `/Users/philprice95/projects/freqtrade/.venv/bin/python`.
The pyenv shim `~/.pyenv/shims/python3` is a *different* 3.12.13 without pandas/scipy and MUST NOT
be used; every run asserts `sys.executable`.

**Determinism**: the backtests are deterministic (gated arms reuse ungated weights, no
retraining), but the **jump-model fit is not** — k-means initialisation makes it seed-dependent.
The seed ensemble is therefore part of the design, not a robustness afterthought.

**Primary Dependencies**: numpy 1.26.4, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.0 (all
present). `jumpmodels 0.1.1` to be installed as the cross-check — dry-run verified 2026-09-15 as a
pure-Python wheel pulling **no new transitive dependencies**. No MLX and no torch: the jump model
is a coordinate-descent fit over a handful of features, not a neural network.

**Storage**: JSON artefacts under `research/labelfree/jump/out/`; backtest archives via the
existing `regime/arms/*.json` + `user_data/backtest_results/*.zip` path that `evaluation.pool`
already reads.

**Testing**: pytest, `research/labelfree/tests/test_jump_*.py`, joining the existing 201 tests.

**Target Platform**: local macOS (Darwin 25.6.0), Apple Silicon.

**Project Type**: research module inside `user_data/strategies` — the only repository that
receives commits.

**Performance Goals**: not a throughput feature. The binding budget is **evaluation count**: 15
folds × {online-gated, smoothed-gated} = 30 backtests, no retraining, against an ungated baseline
that already exists (`evaluation/pooled_nnnc.json`, 378 trades).

**Constraints**: all runs sequential; brief budget is one session; the pooled harness and its
pinned constants (`BOOTSTRAP_SEED = 0`, floor 3.350 pp/yr, noise floor 7.633 pp) are used as-is
and never recomputed.

**Scale/Scope**: 15 anchored folds, ~12 pairs, 15-minute bars, 378 pooled incumbent trades.

## Constitution Check

*GATE: evaluated before Phase 0, re-evaluated after Phase 1.*

| Principle | Status | How this plan satisfies it |
|---|---|---|
| **I. Deployment regime** | PASS | Verdict read from the pooled walk-forward across W1/W2/W3; no full-history headline. |
| **II. Persistence (NON-NEGOTIABLE)** | PASS | 15 anchored folds spanning three temporally distant windows; SC-006 requires the sign to hold in each. Drawdown acceptance requires ≥10 of 15 folds (SC-011), never an aggregate. |
| **III. Learnability ≠ edge** | PASS | No state-quality metric (BIC, silhouette, state persistence) may decide acceptance. They are reported as diagnostics only; the verdict is cost-aware backtest P&L and drawdown. |
| **IV. Parameters actually live** | **PASS — and it is the feature's first gate** | US1/FR-001–003 make "the gate demonstrably fires" a precondition. An inert gate is reported as a defect (SC-012), never as evidence about regime gating. |
| **V. Lookahead-clean ≠ realistic** | PASS | Volume/liquidity constraints active via `config_wf.json`, unchanged from the incumbent baseline. Causality proven by an end-to-end truncated-vs-full deletion test (FR-004), not by framework lookahead analysis. |
| **VI. Diagnose before tuning** | PASS | Gate-activity and state diagnostics are produced before any performance number is read; no jump-penalty tuning without a diagnostic localizing a fault. |
| **VII. Validate cheaply; trackability before adaptivity** | **PASS with written justification — see below** | |
| **VIII. One parameter space at a time** | PASS | The jump penalty λ and the number of states are the only free parameters; they are fixed from the literature and a single calibration pass, never co-searched with the incumbent's gate parameters (which stay frozen at rvol 2.0 / guard 0.3). |
| **IX. Orthogonality** | **PASS — and it is a named risk** | See "The rvol redundancy risk" below. |
| **X. Structural, not per-pair** | PASS | One model specification for the whole universe; per-pair fits use identical hyperparameters. No pair-specific thresholds. |

### Principle VII — explicit justification for building despite Phase C's FAIL

Principle VII requires an adaptive or regime-conditional scheme to be gated behind a cheap
feasibility measurement, and forbids fitting a regime-to-parameter mapping on a handful of window
observations. `regime/phase_c_trackability.py` already ran that gate across eight instruments and
returned **FAIL**.

This feature proceeds anyway, and the justification is written rather than silent:

1. The brief commissions it as a **negative control**, not as a candidate. Its expected outcome is
   a KILL consistent with Phase C.
2. A jump model is **mechanically distinct** from the eight tested instruments — it penalises
   switching rather than thresholding a statistic — which is the one property Phase C did not
   sample.
3. The cheap gate Principle VII demands is **not skipped, it is reordered to the front**: the
   leak test and the gate-activity check both run and must pass before any performance
   evaluation. Either can kill the feature in an afternoon.
4. The gate is **binary admit/block**, not a regime-to-parameter mapping fitted on window
   observations, so the specific prohibition in VII's second paragraph is not engaged.

### Principle IX — the rvol redundancy risk, named up front

The incumbent already gates entries on **relative volume ≥ 2.0**, and its guard chain rejects
roughly 97% of candidates with rvol dominant. A jump model fitted on realised and downside
volatility is at serious risk of being **that same axis in disguise** — the dominant factor
wearing a different name.

This is not a footnote, it is the most likely way the feature fails while looking like a result.
Principle IX requires the redundancy check to run **FIRST, before performance is evaluated**, so
`redundancy.py` is a Phase-2 foundational task, not a polish task. Two outcomes are anticipated
and both are informative:

- **The gate blocks bars rvol already blocked** → the gate is inert *because the incumbent got
  there first*. That is a defect report under Principle IV, and specifically not "regime gating
  does not help".
- **The gate blocks a distinct set** → the comparison is meaningful and proceeds.

## Project Structure

### Documentation (this feature)

```text
specs/016-labelfree-jump-model-gate/
├── plan.md              # This file
├── research.md          # Phase 0 — R1..R11 decisions
├── data-model.md        # Phase 1 — entities and invariants
├── quickstart.md        # Phase 1 — runnable validation sequence
├── contracts/
│   └── jump_gate.md     # Phase 1 — module interface contract
├── checklists/
│   └── requirements.md  # from /speckit-specify, 16/16
└── tasks.md             # /speckit-tasks output — NOT created here
```

### Source Code (repository root = `user_data/strategies`)

```text
research/labelfree/
├── jump_gate.py             # thin entry point named by the brief; delegates to jump/
└── jump/
    ├── __init__.py
    ├── features.py          # causal realised vol, downside vol, sign persistence
    ├── model.py             # in-repo sparse jump model (fit; coordinate descent)
    ├── online.py            # FILTERED inference + the SMOOTHED leak reference
    ├── refit.py             # 90-day expanding-window refit schedule + volatility state mapping
    ├── crosscheck.py        # jumpmodels 0.1.1 agreement + filtered-vs-smoothed audit
    ├── redundancy.py        # Principle IX: overlap with the incumbent rvol guard
    ├── gate.py              # state -> entry permission; blocking counts per fold
    ├── arms.py              # generate gated fold classes reusing ungated fold weights
    ├── drawdown.py          # max drawdown + time-underwater, block-bootstrap CIs
    └── report.py            # §6 result report

research/labelfree/tests/
├── test_jump_model.py       # fit determinism, penalty behaviour, label stability
├── test_jump_causality.py   # deletion test: no bar's state moves when the future is truncated
├── test_jump_gate.py        # gate activity, inert/total refusal, redundancy
└── test_jump_drawdown.py    # drawdown + CI correctness on constructed curves
```

**Structure Decision**: a package `research/labelfree/jump/` mirroring the established
`es/`, `stress/` and `ssl/` layout, with `research/labelfree/jump_gate.py` as the thin entry point
the brief names by path. Existing instruments are **reused, not reimplemented**:
`research/labelfree/folds.py` (fold geometry, purge, embargo), `evaluation/` (pooled walk-forward,
pinned constants), `research/labelfree/admissibility.py` (the shared abstention gate),
`research/labelfree/stats.py` (bootstrap CI, IQM, DSR, PBO), and
`research/labelfree/stress/bootstrap.py::block_index` for the path-statistic CIs.

## Constitution Check — re-evaluated after Phase 1 design

No principle moved to FAIL. Three became *more* binding once the design was concrete:

- **Principle IV (liveness)** is now structural rather than procedural. `gate.activity()` returns
  `active | inert | total`, and the contract requires callers to check it **before** computing any
  performance statistic. An inert gate cannot silently become a clean null, because the code path
  that would report it does not exist.
- **Principle V (lookahead measured end to end)** sharpened into a named artefact. The causality
  test targets the `StateSequence` file **the backtest actually reads**, not a parallel in-memory
  computation. Brief 07's encoder trained on the wrong split while the splitting code read
  correctly; pointing the test at a different object than the run consumes is exactly how that
  survives.
- **Principle IX (orthogonality)** produced a Phase-2 foundational task, `redundancy.py`, rather
  than a polish-phase check. The incumbent's rvol ≥ 2.0 guard is the dominant axis, and a
  volatility-fitted jump model is the most plausible way this feature produces a confident
  non-result.

One design element exists purely to satisfy a principle and is worth naming, because it inverts
the usual reading: the **smoothed** state sequence must **fail** the causality test. A leak
reference that passes is not reassuring — it means the traceback never ran and the measured gap is
not a leak size at all. Test `test_smoothed_prefix_CHANGES` asserts the failure.

### Workflow gate 5 — why seed-robustness is engaged rather than waived

An earlier draft of this plan waived paired seed-robustness on the grounds that the gated arms
reuse the ungated fold weights, so the backtests are deterministic and carry no run-to-run
variance. That reasoning is correct about the *backtest* and wrong about the *feature*: the jump
model's own fit is stochastic through its **k-means initialisation**, so a single-seed run could
produce a gate whose regime structure is an artifact of where the centroids started.

This repository has already recorded the sharper version of that failure — head weights moved
entries **less than a random seed did**, which is what closed that line. A regime gate is exactly
the kind of object that can look decisive and be initialisation noise.

So `00-evaluation-protocol.md` §4 applies in full: **≥5 seeds**, interquartile mean, stratified
bootstrap, never the best seed (FR-022, SC-017), with cross-seed state agreement required at
≥0.95 (FR-023).

**Workflow gates** (constitution §"Experiment Workflow and Quality Gates"): pinned time ranges via
the fold plan (1); 15 folds across three distant windows (2); volume/liquidity active, unchanged
from the incumbent (3); liveness via `GateActivity` (4); paired seed-robustness **engaged, and it had to be** (5) — see
below; trade counts reported per fold with zero-trade folds marked inconclusive (6); scalers
untouched, because the feature changes **no** feature include list — the gate consumes a separate
artefact and the incumbent's inputs are unchanged (7).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Two implementations of the same model (in-repo + `jumpmodels`) | Clarification Q1=C. The verdict turns on a filtered-vs-smoothed distinction that must be inspectable, and an independent implementation is the only check that our fit is the published one. | A single in-repo fit leaves "our code is wrong" unfalsifiable; the package alone leaves the inference mode unauditable — and a library defaulting to smoothed would silently reintroduce the exact leak this feature exists to measure. |
| Real gated backtests rather than filtering the recorded 378 trades | freqtrade's one-position-per-pair rule means gating can promote a previously-suppressed signal into a real trade. The filter cannot see that. | Measured: concurrency peaks at 9/10 so the *cap* channel is closed, but the per-pair channel is not. Feature 013's surrogate passed review and still carried four fidelity bugs that one real backtest found. |
