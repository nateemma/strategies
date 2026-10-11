# Implementation Plan: Regime Gate on the Reversion Family

**Branch**: `026-reversion-regime-gate` (spec directory; commits go to master) | **Date**: 2026-10-11 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/026-reversion-regime-gate/spec.md`

## Summary

**The question.** In 025, the reversion family (`gbb`) lost in the late half in every arm, while
momentum earned. Can a causal, market-wide regime state switch reversion entries off where reversion
loses, and beat momentum alone, the better of the two fixed choices?

**The instruments.** Four are pre-declared, one fixed construction each:
- breadth: the share of pairs above an 8-day EMA;
- a cross-pair volatility regime;
- trend persistence: the cross-pair median variance ratio VR(16);
- BTC trend: BTC against its 8-day EMA.

**US1 — feasibility check, no backtests.** The gate's on/off rule is chosen on early reversion
entries: 3 percentiles × 2 directions, with ≥50% of early trades kept "on". Each instrument then
PASSES only if "off" captures losing reversion P&L in BOTH halves. If none passes, the feature
stops.

**US2 — only if something passes.** A gated SimpleTrigger arm per passing instrument (the same
drop-and-retag mechanism as 025 rule B). The winner is chosen early by 025's stage runner and judged
late against `st_bgo_mx`.

## Technical Context

**Language/Version**: Python 3.12 (`.venv`)

**Primary Dependencies**:
- pandas / numpy;
- `research.clfstudy.data.load_pair` (15m feathers, including `BTC_USDT-15m`, 2022-01 to
  2026-10);
- the 023–025 harness: `pool`, `stage`, `contention`, `fingerprint`, `run_folds`;
- freqtrade's `informative_pairs` and `dp.get_pair_dataframe` (US2 only).

**Storage**: `feasibility.{md,json}`, `decisions.md`, arm records

**Testing**: pytest (instrument causality by truncation, an early-only rule choice); walk-forward
reproduction

**Target Platform**: macOS / Apple Silicon. No training.

**Project Type**: research codebase

**Performance Goals**: US1 under 10 minutes of compute. US2 ≤ 5 arms, about 30 minutes.

**Constraints**:
- causal and market-wide only;
- rules chosen early only;
- the bar is momentum alone;
- one backtest chain at a time;
- 023–025 arms stay reproducible;
- PROVISIONAL.

**Scale/Scope**:
- New: `utils/RegimeState.py`, `research/regime_gate/feasibility.py`.
- US2 edits: SimpleTrigger, fingerprint, and `stage.py` (`--bar`).
- Up to 5 configs.

## Constitution Check

*GATE: checked before Phase 0 and re-checked after Phase 1. It is unchanged.*

| principle | status | how |
|---|---|---|
| I. Deployment regime | PASS | The whole point is a regime split; halves are reported |
| II. Distant era (NON-NEGOTIABLE) | PASS, every result PROVISIONAL | Nothing is promoted; the halves split is the only persistence test available |
| IV. Parameters proven live | PASS | Refusal and off-bar counters; INERT reported |
| VI. Diagnose before tuning | PASS | 025's diagnosis (reversion late, entry-side) motivates this |
| **VII. Validate cheaply; trackability before adaptivity** | **PASS, central** | US1 is a no-backtest feasibility gate with a STOP. The bar is the better of the two real constants (memory `feedback_trackability_baseline_must_be_a_real_choice`). A regime→parameter mapping is NOT fitted per window: the rule is one early-chosen threshold. |
| VIII. One space at a time | PASS | One gate, rules from a small fixed grid, early choice |
| IX. Orthogonality | N/A | No new factor; reversion on/off only |
| X. No per-pair code | PASS | The state is market-wide; pair names appear only as dict keys |
| XI. No leakage | PASS | Truncation causality test per instrument (R5); a planted control must fail |
| Framework strategy-agnostic | PASS | Instruments live in `utils/`; the gate lives in SimpleTrigger; no `Framework/` edit |

## Project Structure

### Documentation

```text
specs/026-reversion-regime-gate/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/interfaces.md
├── checklists/requirements.md
├── feasibility.md / feasibility.json   # US1 output
├── decisions.md                        # US2 stage (if reached)
├── baseline.md  report.md
└── tasks.md
```

### Source Code (`user_data/strategies`)

```text
utils/RegimeState.py                    # NEW C1: the 4 instruments, pure and causal
utils/tests/test_regime_state.py        # NEW: truncation causality + planted control
research/regime_gate/__init__.py        # NEW
research/regime_gate/feasibility.py     # NEW C2: US1 feasibility (no backtests)
research/regime_gate/tests/test_feasibility.py  # NEW: early-only choice, PASS logic
SimpleTrigger/SimpleTriggerStrategy.py  # EDIT C3 (US2 only): REVERSION_REGIME_GATE
evaluation/fingerprint.py               # EDIT C4 (US2 only)
evaluation/stage.py                     # EDIT C5 (US2 only): --bar
SimpleTrigger/SimpleTrigger_RegGate*.py # NEW (US2 only)
```

**Structure Decision**: the US1 code is research tooling under `research/`, alongside 021's
`outcomes`. The instrument functions go in `utils/` because the strategy reuses them in US2. The
US2 edits are deferred until US1 passes, so a STOP leaves no strategy code changed.

## Phases (for /speckit-tasks)

1. **Foundation:** instruments (C1) with causality tests.
2. **US1:** the feasibility script with tests, then run it. **STOP gate.**
3. **US2 (conditional):**
   - the gate in SimpleTrigger;
   - fingerprint;
   - `stage --bar`;
   - a neutral reproduction;
   - gated arms;
   - the stage decision.
4. **US3:** report, memory, test gate.

## Risks

| risk | mitigation |
|---|---|
| An instrument is "causal" in code but leaks through `dp.get_pair_dataframe` full history | Instrument functions use backward windows only; truncation test on real data (R5) |
| A gate that is a calendar switch (one flip, stays off late) | Off-bar share and off-run counts are reported per half (R3) |
| The bucket split overstates the gain (substitution) | US1 only selects; US2 arms decide; the caveat is stated in the report |
| Multiple comparisons (4 × 6 options) | Early-only choice per instrument; the late-half PASS condition is the out-of-sample test; all PROVISIONAL |
| BTC trend repeats the old gate's failure | It is scored on separating reversion LOSSES with an early-chosen direction, not "require uptrend"; if it fails, it fails on the record |

## Complexity Tracking

None.
