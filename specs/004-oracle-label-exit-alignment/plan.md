# Implementation Plan: Oracle-First Label/Exit Alignment for NNNC

**Branch**: `004-oracle-label-exit-alignment` | **Date**: 2026-09-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/004-oracle-label-exit-alignment/spec.md`

## Summary

A perfect-prediction oracle on NNNC's own label loses money once the entry guards
are removed (−18.89% on W1), and the whole span from no model to a perfect model
with guards on is 1.45pp (11.122% vs 9.671%). The guards are therefore not filtering bad predictions
— they are hiding a disagreement between the label and the exit stack.

This feature (a) locates the exit mechanism responsible, (b) turns the oracle
into a standing, training-free screening gate, and (c) searches for a label or
exit configuration the oracle can profit from. Phase 0 research established that
(a) is a **read over backtest archives already on disk** — no measurement campaign
— and that two candidates already stand out: the ATR adaptive stop (40% of
trades, the only large negative contributor) and trading cost (a first-order
estimate exceeds the entire net loss).

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`)

**Primary Dependencies**: freqtrade backtesting; existing repo tooling —
`regime/run_arm.py`, `regime/tail/compare.py`, `NNNC/NNNC_Oracle.py`,
`NNNC/NNNC_NoModel.py`, `NNNC/_hold_window.py`, `Framework/TrainingSignals.py`

**Storage**: JSON arm records in `regime/arms/`; freqtrade backtest archives
(zip) in `user_data/backtest_results/`

**Testing**: No unit tests. This is a measurement feature; its outputs are arm
records and written findings. Correctness is enforced by liveness checks
(trade-hash comparison) and by refusing metrics from failed or stale runs.

**Target Platform**: macOS, Apple Silicon, 14 cores

**Project Type**: Research/measurement study inside an existing strategy repo

**Performance Goals**: An oracle arm is a backtest with no training. Guards-on
arms run in minutes; guards-off arms carry 1760–6015 trades and take longer.
Budget expectation: baseline completion (2 arms) plus 8–14 screened variants ×
3 windows, all sequential, no retrains.

**Constraints**: W1 `20250901-20260831` is the sole acceptance window; W2
`20240601-20250531` and W3 `20230101-20231231` are reported and labelled.
`Framework/` must be left unmodified — a live dry run imports `BaseStrategy`.
Every arm carries its own params file. Sequential execution only. Every oracle
figure is a lookahead ceiling and nothing measured with it is promotable.

**Scale/Scope**: One strategy family (NNNC). No training, no GAN work, no
architecture change.

## Constitution Check

| Principle | Status | Note |
|---|---|---|
| I. Evaluate in the deployment regime | **PASS** | W1 is the deployment window and the sole acceptance window |
| II. Persistence across distant eras | **PASS-WITH-CAVEAT** | W3 (2023) is temporally distant but falls inside the model's training window. For ORACLE arms no weights are fitted, so W3 corroborates a mechanism; it may not confer acceptance (R10, FR-022). The caveat must be carried into every task, and no task may treat W3 as out-of-sample evidence |
| III. Learnability is not edge | **PASS** | FR-008 forbids classifier metrics as acceptance input. This feature is the strongest expression of the principle yet: it measures the trading ceiling directly rather than any proxy |
| IV. Verify parameters are actually live | **PASS** | Liveness by trade-hash comparison (FR-010); `run_arm.py` refuses failed/stale metrics; R6 records which knobs are unreachable via flags and need subclasses, and that `--stoploss` is inert |
| V. Lookahead-clean is not realistic | **PASS, INVERTED** | The oracle is deliberately lookahead. The principle's concern — mistaking a phantom return for a real one — is met by FR-012/SC-009 making every oracle figure a labelled ceiling and forbidding promotion. Execution realism is additionally addressed by US4, which quantifies cost rather than assuming it away |
| VI. Diagnose before tuning | **PASS** | FR-006 is a hard ordering gate: no label or exit parameter may be varied until the diagnosis is reported. Phase 2 below enforces it |
| VII. Validate cheaply | **PASS** | The oracle gate IS this principle instantiated — a training-free screen that kills or confirms a change in minutes instead of 8–45 |
| VIII. One parameter space at a time | **PASS** | FR-015 requires label-side and exit-side variants separately before any combination |
| IX. Orthogonality earns a factor its place | **N/A** | No new factor is introduced |
| X. Structural fixes, not per-pair overrides | **PASS** | Non-goals exclude per-pair special-casing. US4's pair-mix check is diagnostic, not a licence to special-case |

No violations. Nothing to record in Complexity Tracking.

## Project Structure

### Documentation (this feature)

    specs/004-oracle-label-exit-alignment/
    ├── spec.md
    ├── plan.md              # this file
    ├── research.md          # R1-R11, all resolved
    ├── data-model.md
    ├── contracts/
    │   └── forensics-cli.md
    ├── quickstart.md
    └── checklists/requirements.md

### Source Code (repository root = `user_data/strategies/`)

    regime/
    ├── run_arm.py                 # EXISTING, unchanged by this feature
    ├── tail/compare.py            # EXISTING, reused
    ├── trade_forensics.py         # NEW — the archive reader (US1, US4)
    └── ORACLE_GATE.md             # NEW — the standing process (US2)
    NNNC/
    ├── NNNC_Oracle.py / .json     # EXISTING, the baseline oracle host
    ├── NNNC_NoModel.py / .json    # EXISTING
    ├── _hold_window.py            # EXISTING, reused for hold-window variants
    └── NNNC_Ora_<variant>.py/.json  # NEW, one per variant needing a subclass

**Structure Decision**: additive only. The one new module is a read-only reader
over `user_data/backtest_results/`; `run_arm.py` is deliberately NOT extended
because 480+ existing arm records depend on its output shape (R4).

## Execution Phases

Strictly sequential. Phase 2 is a HARD GATE on Phase 4.

### Phase 1 — Complete the baseline
Fill the missing W2 arms for `oracle_nog` and `gpool` (R8) so the baseline the
variants are judged against is as complete as FR-014 demands of them. Two
backtests, no training.

### Phase 2 — DIAGNOSE (US1, US4) — **HARD GATE**
Build `regime/trade_forensics.py` and run it over the six existing arms plus the
two new ones. Produce, per arm: exit-reason breakdown (count, summed P&L, mean
P&L, mean duration, share of total), cost burden, pair mix, and the
excursion-based stop-before-payoff and truncation measures from `max_rate`/
`min_rate`. Apply the R2 mapping so `trailing_stop_loss` is reported as the ATR
adaptive stop.

Report the dominant mechanism with its trade count and share of destroyed P&L,
and state whether the guards-off loss survives removal of cost.

**No label or exit parameter may be varied until this is reported (FR-006).**

### Phase 3 — Write the gate (US2)
Write `regime/ORACLE_GATE.md`: what is measured, on which windows, the R9 pass
condition (W1 guards-off oracle return > 0 AND >= 200 trades), fail and
inconclusive conditions, the lookahead-ceiling labelling rule, and the
requirement that a failing change is not trained. Apply it to one worked example
end to end.

### Phase 4 — Screen variants (US3)
Only variants addressing the mechanism the diagnosis identified, cheapest-first:
first those reachable via `--set-sell`/`--roi` on the existing oracle host, then
those needing a subclass (label knobs, ATR band, hold window). Label-side and
exit-side separately before any combination (FR-015). Each variant on all three
windows, each with its own params file, each judged by the gate.

### Phase 5 — Conclude (US5)
Record whether any configuration makes the oracle profitable guards-off on W1,
the model-headroom conclusion and what it forecloses, and update the repository's
findings documents and memory.

## Complexity Tracking

No constitutional violations requiring justification.
