# Implementation Plan: Family Coexistence and Reversion Late-Half Fix

**Branch**: `025-family-coexistence` (spec directory; commits go to master) | **Date**: 2026-10-11 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/025-family-coexistence/spec.md`

## Summary

024's combined arm (gbb reversion + breakout_gbb momentum) fell short of momentum alone. The
families pre-empt each other's pairs: freqtrade allows one open trade per pair, and the slot cap
never binds.

This feature adds two pre-declared, structural coexistence rules:
- **A, a per-family pair cap:** an entry-confirmation hook in a strategy-agnostic `utils` mixin.
- **B, a momentum-recency guard on reversion entries:** a causal row filter in SimpleTrigger's
  entry tagging.

It then tunes the reversion exit set one parameter at a time, inside the combined arm (Q1 → A).

Every choice is made on folds f00–f07 and judged on f08–f14 by a stage runner that mechanically
cannot see late folds when choosing. Liveness counters reach the run record through 023's
entry-condition sink. There is no `Framework/` edit.

## Technical Context

**Language/Version**: Python 3.12 (`.venv` at the repo root)

**Primary Dependencies**: freqtrade (`confirm_trade_entry`, `Trade.get_trades_proxy(is_open=True)`
in backtests), pandas, numpy. The 023/024 harness: `foldplan`, `run_folds`, `pool`, `halves`,
`compare_pooled`, `contention`, `fingerprint`, `regime/run_arm`.

**Storage**: JSON arm records, plans, backtest zips, and `decisions.md`

**Testing**: pytest. Walk-forward reproduction by `trades_hash` (SC-001).

**Target Platform**: macOS / Apple Silicon. No training.

**Project Type**: research codebase

**Performance Goals**: about 16 arms × ~5.5 min ≈ 90 min of backtests in 5 gated chains

**Constraints**:
- No `Framework/` edits.
- No pair-specific code.
- Choices use early folds only.
- One backtest chain at a time.
- 023/024 arms stay reproducible.
- Every result is PROVISIONAL.

**Scale/Scope**: 2 new modules (`utils/FamilyCoexistence.py`, `evaluation/stage.py`), 3 edited
(`SimpleTrigger/SimpleTriggerStrategy.py`, `regime/run_arm.py`, `evaluation/fingerprint.py`),
and about 16 configs.

## Constitution Check

*GATE: checked before Phase 0 and re-checked after Phase 1; it is unchanged.*

| principle | status | how |
|---|---|---|
| I. Deployment regime | PASS | Same walk-forward span; halves reported |
| II. Distant era (NON-NEGOTIABLE) | PASS, every result PROVISIONAL | No distant era; nothing promoted |
| III. Learnability ≠ edge | N/A | No model; walk-forward P&L only |
| IV. Parameters proven live | PASS | Refusal counters (R3) and governed exit-reason counts (R5); INERT reported |
| V. Realistic execution | PASS | The volume check runs first and unchanged (C1) |
| VI. Diagnose before tuning | PASS | 024's contention diagnosis motivates US1; tuning follows it |
| VII. Validate cheaply | PASS | Neutral reproductions gate the stages; 024 is the feasibility evidence |
| VIII. One parameter space at a time | PASS | Coexistence then 4 single-parameter steps; small pre-declared sets; early-choose/late-judge enforced in code (C5) |
| IX. Orthogonality | PASS | Per-fold family correlation reported |
| X. No per-pair code | PASS | Rules read family tags, position state and trigger history, never pair identity |
| XI. No leakage | PASS | Rule B is a causal rolling window; triggers already pass causality |
| Framework strategy-agnostic | PASS | No `Framework/` edit; the rule A mixin is in `utils/` and reads tags only |
| Params merge | PASS | Tuning goes through `FAMILY_EXIT_OVERRIDES` (merge per family) |

## Project Structure

### Documentation

```text
specs/025-family-coexistence/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/interfaces.md
├── checklists/requirements.md
├── decisions.md          # written by evaluation/stage.py, one block per stage
├── baseline.md           # reproduction + run evidence
├── report.md             # US3
└── tasks.md              # /speckit-tasks
```

### Source Code (`user_data/strategies`)

```text
utils/FamilyCoexistence.py           # NEW C1: rule A mixin (confirm_trade_entry), sink logging
SimpleTrigger/SimpleTriggerStrategy.py  # EDIT C2: rule B in _tags; MRO adds the coexistence mixin
regime/run_arm.py                    # EDIT C3: aggregate_coexistence -> coexistence_refusals
evaluation/fingerprint.py            # EDIT C4: optional coexistence members
evaluation/stage.py                  # NEW C5: early-choose / late-judge stage runner -> decisions.md
SimpleTrigger/SimpleTrigger_Coex*.py # NEW: 2 neutral + 6 active coexistence configs
SimpleTrigger/SimpleTrigger_Tune*.py # NEW: 8 tuning configs, created stage by stage
*/tests/                             # NEW tests for C1–C5
```

**Structure Decision**: the same layout as 023/024. The rule A mixin is strategy-agnostic for a
later NNNC adoption. Rule B lives with SimpleTrigger's entry tagging because it needs the
per-trigger masks.

## Phases (for /speckit-tasks)

1. **Foundation:**
   - C1 / C2 / C3 / C4 with tests;
   - neutral configs reproduce `st_fam_combo` (SC-001);
   - 023/024 shas unchanged.
2. **US1 coexistence:**
   - 6 active arms → `stage.py --stage coexistence` → adopt or not.
3. **US2 tuning:**
   - per step, create the candidate configs on top of the previous arm → run → `stage.py`;
   - 4 steps, strictly sequential, because each depends on the previous decision.
4. **US3 report** and polish.

## Risks

| risk | mitigation |
|---|---|
| `Trade.get_trades_proxy` behaves differently in backtests | Unit test with a stubbed Trade API; SC-001 neutral reproduction; non-zero refusal counts on active arms |
| Rule A refusals are invisible (in-process) | Sink JSONL (R3), the same path as 023's entry counts |
| Selection leaks late-fold information | `stage.py` computes the choice from f00–f07 rows only; planted-candidate test (C5) |
| The multiplier is inert inside the narrow ATR clamp | Tune the floor instead (R5); liveness check reports INERT if it is anyway |
| Small late effects (momentum late +2.50 on ~2.7k trades) | Paired verdicts with CIs; nothing claimed from point estimates |

## Complexity Tracking

None. There are no constitution violations.
