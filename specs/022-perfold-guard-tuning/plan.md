# Implementation Plan: Per-Fold Guard Tuning

**Branch**: `022-perfold-guard-tuning` | **Date**: 2026-10-10 | **Spec**: [spec.md](./spec.md)

## Summary

Test whether the quality-guard chain is a model-free strategy when tuned per fold, and whether
the model adds anything on top. The work is **gate-first**: a cheap trackability test
(Principle VII) decides whether the expensive arm runs at all, and a failed gate is a terminal
result rather than a stalled feature.

Almost nothing is built. The repository already contains every mechanism this needs:
per-fold leakage-safe selection (`evaluation/pick_threshold.py`), the plan-key route by which a
per-fold choice reaches a fold run (`evaluation/run_folds.py`'s `thresholds`), paired
symmetric-difference statistics (`evaluation/bootstrap.py::bootstrap_delta`), the no-model
strategy scaffold (`SimpleStrategies/TriggerOnly.py`), and a per-bar outcome table keyed by
exit hash (`research/outcomes/`). Two new modules and one plan key complete it.

## Technical Context

**Language/Runtime**: Python 3.12, the virtualenv at `.venv` (never conda).

**Primary dependencies**: freqtrade (backtesting, `advise_all_indicators`/`advise_entry`),
numpy/pandas/scipy. No new third-party dependency.

**Reused, not rebuilt**:
| need | existing asset |
|---|---|
| 15-fold plan + date ranges | `evaluation/folds/plan_NNNC_A_Raw_Open.json` |
| pooled statistic | `evaluation/pool.py` → `wallet_profit_pct` |
| power / minimum resolvable effect | `evaluation/power.py` (378 trades ↔ 3.350 pp/yr) |
| **paired symmetric-difference bootstrap** | `evaluation/bootstrap.py::bootstrap_delta` — already documents the "entry set itself moves" case, which is exactly arm C vs A |
| **leakage-safe per-fold selection pattern** | `evaluation/pick_threshold.py`'s `TrainWindow.from_fold`, which *asserts* the window ends at or before the test window starts |
| route for a per-fold choice into a run | `evaluation/run_folds.py`'s `thresholds` plan key (`run_folds.py:194,237`) |
| running one arm | `regime/run_arm.py` — already records `entry_enable_guards_written` and a per-fold `trades_hash` |
| no-model strategy scaffold | `SimpleStrategies/TriggerOnly.py` (`entry_enable_guards: True`) |
| per-bar outcome under the fixed exit | `research/outcomes/table.py` (exit-hash keyed), `leaderboard.py::_select_concurrent` |
| arms A / B / D | `regime/arms/wf_gbbo_*`, `wf_gbbg_*`, `wf_aro_*` |
| arm E models | `saved_data/NNNC_ARO_f{00..14}/*.safetensors` (all 15 present) |

**Storage**: JSON arm records under `regime/arms/`; a guards map under `evaluation/folds/`
mirroring the existing `thresholds_*.json` convention; grid scores cached as feather beside the
outcome table.

**Testing**: pytest under `evaluation/` and `research/outcomes/tests/`, matching existing layout.

**Target platform**: macOS / Apple Silicon, local.

**Project type**: offline research. Nothing here runs inside a live strategy.

**Performance goals**: the trackability gate MUST complete in one working session (SC-008).
Grid scoring is therefore table-based, not backtest-based — see research.md R2.

**Constraints**: all code under `user_data/`; the exit is held fixed and proven by
`exit_config_hash` equality; the live dry-run (`MomentumRegimeBasket15mFast`) must see no
behaviour change; `saved_data/main_scaler.pkl` and `gan_scaler_a.pkl` untouched; **no model is
fitted anywhere in this feature**; one backtest at a time.

**Scale**: 64 grid configurations × 15 folds = 960 training-window scorings (table-based,
minutes) plus 15 harness backtests for arm C (~30 min) and ~8 for the inertness proof.

## Constitution Check

| principle | status | how discharged |
|---|---|---|
| I. Evaluate in the deployment regime | PASS | the 15-fold harness IS the deployment-regime instrument; no full-history average is quoted |
| **II. Persistence across distant eras** | PASS | FR-016 / SC-007 require early-half and late-half values for every effect |
| III. Learnability is not edge | PASS | FR-018; no classifier metric is an acceptance input, and no model is fitted |
| **IV. Verify parameters are actually live** | PASS | FR-013 via per-fold `trades_hash`, **with the known-inert `guard_metric` as a negative control** proving the check can detect inertness (research.md R3) |
| V. Lookahead-clean is not realistic | PASS | arm C is a real backtest under the volume-headroom check; the table proxy is used only for *selection*, never quoted as return (research.md R2) |
| VI. Diagnose before tuning | PASS | the gate and the proxy-fidelity audit both precede any tuning |
| **VII. Trackability before adaptivity (NON-NEGOTIABLE)** | PASS | US2 is a blocking P1 gate: rolling optimum + trailing-best vs the best constant, regret over the 64 actual candidates (research.md R4). FR-006 forbids arm C if it fails |
| **VIII. One parameter space at a time** | **DEVIATION, justified** | a joint 3-D coarse grid. See Complexity Tracking |
| IX. Orthogonality earns a factor its place | N/A | no new factor is proposed |
| X. Structural fixes, not per-pair overrides | PASS | guards are universal; nothing is per-pair |
| XI. No data leakage | PASS | FR-007 + `TrainWindow.from_fold`'s assertion; the test window is scored once |

**Gate result: PASS with one documented deviation.** No unjustified violation.

## Project Structure

### Documentation (this feature)

```
specs/022-perfold-guard-tuning/
├── spec.md
├── plan.md                 # this file
├── research.md             # R1-R6, the design decisions
├── data-model.md           # GuardConfig, GridScore, TrackabilityResult, Arm
├── contracts/
│   └── guard-cli.md        # pick_guards / guard_trackability command contracts
├── quickstart.md           # how to run the gate, then the arm
└── checklists/requirements.md
```

### Source Code (repository root = `user_data/strategies/`)

```
evaluation/
├── pick_guards.py          # NEW. Per-fold grid scoring on TRAIN windows only.
│                           # Mirrors pick_threshold.py and REUSES its TrainWindow.
├── guard_trackability.py   # NEW. The Principle VII gate: rolling optimum,
│                           # trailing-best vs best constant, regret.
├── run_folds.py            # MODIFIED. One optional `guards` plan key, exactly
│                           # mirroring the existing `thresholds` key.
└── tests/
    ├── test_pick_guards.py         # NEW
    └── test_guard_trackability.py  # NEW

SimpleStrategies/
└── GbbGuardGridOpen.py     # NEW. Thin TriggerOnly subclass whose guard values
                            # come from the generated per-fold params file.

regime/run_arm.py           # UNCHANGED (already records trades_hash and
                            # entry_enable_guards_written as of feature 021)
Framework/                  # NO RUNTIME CHANGE -- precisely BECAUSE this is runtime code
                            # a live dry-run inherits. A test-allowlist entry may be added
                            # (T008); nothing a strategy executes is modified.
```

**Why `evaluation/` and not `regime/` or `Framework/`**: `evaluation/` already owns the fold
plan, pooling, power and per-fold parameter selection — `pick_guards.py` is the direct sibling
of `pick_threshold.py` and shares its leakage-safety type. `regime/` owns running a single arm,
which `run_arm.py` already does unchanged. `Framework/` is runtime code a live dry-run
inherits, and this feature is offline research, so placing anything there would risk the
no-behaviour-change constraint for no benefit.

## Complexity Tracking

| deviation | why needed | simpler alternative rejected because |
|---|---|---|
| **Joint 3-D coarse grid instead of Principle VIII's one-space-at-a-time** | `regime/GUARD_ABLATION.md` records the guards as "strongly super-additive… not independent filters": individual removals cost −0.17 to −7.33pp summing to ~−17.9pp while removing all six costs −24.91pp | A sequential tune-and-lock search assumes the parameters compose. They provably do not, so the locked-in first value would be wrong once the others moved, and the search would answer a question nobody asked. Mitigation: only **3** axes (the heaviest lifters), only **4** coarse levels each, and the 64 points are enumerated in advance so FR-005's "best of the actual candidates" is a finite, inspectable set |
| **A table-based proxy for grid scoring rather than 960 backtests** | SC-008 requires the gate inside one working session; 960 backtests is ~30 hours | Running real backtests for *selection* would make the gate as expensive as the thing it is meant to gate, defeating Principle VII's "kill or confirm in an afternoon". Mitigation: the proxy is **audited against arms A and B first** (research.md R2) and is never quoted as a return — arm C is a real harness run |
