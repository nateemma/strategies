# Implementation Plan: Target-Aware Exits

**Branch**: `027-target-aware-exits` (spec directory; commits go to master) | **Date**: 2026-10-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/027-target-aware-exits/spec.md`

## Summary

**The question.** The exit doesn't know what the trigger's label promised: at least 2% within 48
bars (12 hours). The stop after grace sits at −4% to −6% and holds last up to 10 days. Is the exit
inconsistent with that target, so that it kills trades on their way to the target? And if it is,
which coupling fixes it?

**US1: feasibility, with no backtests.** This uses the existing trades of `st_bgo_mx` (momentum) and
`st_book_rev2` (gated reversion), replayed on price paths. It measures four things per family and
half:
- (a) stop exits whose price later reached the target;
- (b) winners killed by the stop on the way;
- (c) time to target;
- (d) gain given back after the target.

Five couplings (C1–C5) each have a pre-declared rule and an early-chosen value (research R3).
Only SUPPORTED couplings go further. If none is supported, the feature STOPS.

**US2: switches in the one shared exit** (`ExitPolicy` + `BaseStrategy`), off by default.
- With every switch off, all recorded arms keep their fingerprint and trades.
- The target touch is read from closed candles only (research R4).

**US3: one walk-forward arm per supported coupling per family.** Each arm is judged late against
its own reference arm; adoption requires an IMPROVED P&L verdict, and the risk verdict is reported
beside it. Adopted couplings are re-run together.

**US4: report**, including the exit NNNC should inherit.

## Technical Context

**Language/Version**: Python 3.12 (`.venv`)

**Primary Dependencies**:
- pandas / numpy;
- freqtrade backtesting (pinned checkout);
- `utils.ExitPolicy` (the single exit definition);
- `research.clfstudy.data.load_pair`;
- the 023–026 harness: `pool`, `halves`, `risk`, `fingerprint`, `run_folds`, `run_arm`.

**Storage**: `feasibility.{md,json}`, `decisions.md`, arm records, `report.md`

**Testing**: pytest:
- synthetic-path unit tests for the measurements and replays;
- ExitPolicy decisions;
- stop invariants;
- fingerprint stability over all recorded classes;
- switch-off reproduction on f00.

**Target Platform**: macOS / Apple Silicon. No training.

**Project Type**: research codebase

**Performance Goals**: US1 in minutes. US3 ≤ 9 coupling arms plus ≤ 2 combinations, about 6–8 min
each, run sequentially.

**Constraints**:
- couplings are switches, off by default;
- values fixed early;
- no pair-specific code;
- framework strategy-agnostic;
- stop invariants in every mode;
- one backtest chain at a time;
- PROVISIONAL.

**Scale/Scope**:
- New: `research/exit_coupling/feasibility.py` (+ tests).
- Edits: `utils/ExitPolicy.py`, `Framework/BaseStrategy.py`, `utils/StopInvariants.py`,
  `evaluation/fingerprint.py`, `regime/run_arm.py`.
- New arm classes only for SUPPORTED couplings.

## Constitution Check

*GATE: checked before Phase 0 and re-checked after Phase 1. Unchanged.*

| principle | status | how |
|---|---|---|
| I. Deployment regime | PASS | Late half judges; halves reported |
| II. Distant era (NON-NEGOTIABLE) | PASS, every result PROVISIONAL | Nothing is promoted beyond "the exit NNNC inherits next", stated as provisional |
| III. Learnability ≠ edge | N/A | No model |
| IV. Parameters proven live | PASS | Per-arm coupling liveness counts (FR-010); switch-off reproduction proves the off state is inert |
| V. Lookahead-clean ≠ realistic | PASS, explicit | Target touch from CLOSED candles only, never `trade.max_rate` (R4); ambiguous same-candle touch/breach excluded from US1 |
| VI. Diagnose before tuning | **PASS, central** | US1 measures what the stop does to label-positive paths before any switch is built; prior stop sweeps plateaued, so nothing is swept blind |
| **VII. Validate cheaply** | **PASS, central** | US1 is a no-backtest gate with a STOP and pre-declared rules (R3) |
| VIII. One space at a time | PASS | One coupling per arm; each value from a ≤ 2-point grid fixed early; combinations only after single verdicts |
| IX. Orthogonality | N/A | |
| X. No per-pair code | PASS | Values are per family at most, identical for every pair |
| XI. No leakage | PASS | Early-only choices (a planted late effect test); closed-candle touch; signal-bar ATR for C2 |
| Framework strategy-agnostic | PASS | `BaseStrategy` reads only `ExitParams` fields and declared targets, with no family or trigger names |
| Parameter dicts merge | PASS | Overrides go through the existing merging `FAMILY_EXIT_OVERRIDES` / `SELL_OVERRIDES` paths |

## Project Structure

### Documentation

```text
specs/027-target-aware-exits/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/interfaces.md
├── checklists/requirements.md
├── feasibility.md / feasibility.json   # US1 output
├── decisions.md                        # US3 verdicts (if reached)
├── report.md
└── tasks.md
```

### Source Code (`user_data/strategies`)

```text
research/exit_coupling/__init__.py            # NEW
research/exit_coupling/feasibility.py         # NEW C1: US1 measurements + replays (no backtests)
research/exit_coupling/paths.py               # NEW: path metrics + replays (pure)
research/exit_coupling/tests/test_paths.py    # NEW: synthetic paths
research/exit_coupling/tests/test_feasibility.py  # NEW: early-only choice, rules
utils/ExitPolicy.py                           # EDIT C2 (US2 only): optional coupling fields
utils/tests/test_exit_policy_coupling.py      # NEW (US2)
Framework/BaseStrategy.py                     # EDIT C3 (US2 only)
utils/StopInvariants.py                       # EDIT C4 (US2 only)
utils/tests/test_stop_invariants.py           # EDIT (US2)
evaluation/fingerprint.py                     # EDIT C5 (US2 only)
evaluation/tests/test_fingerprint_027.py      # NEW (US2): recorded shas unchanged
regime/run_arm.py                             # EDIT C6 (US2 only)
SimpleTrigger/SimpleTrigger_*_C<n>.py         # NEW C7 (US3, supported couplings only)
```

**Structure Decision**:
- **US1 is research tooling under `research/`,** like 026's feasibility check, so a STOP leaves
  no framework code changed.
- **US2 edits the shared exit itself** (`utils/ExitPolicy.py` and `Framework/BaseStrategy.py`). The
  couplings are framework behaviour that NNNC must inherit, and the constitution's
  strategy-agnostic rule is met because only declared targets and `ExitParams` fields are read.

## Phases (for /speckit-tasks)

1. **Foundation:** path metrics and replays (pure), with synthetic-path tests.
2. **US1:** the feasibility script, then run it. **STOP gate.**
3. **US2 (conditional), only for SUPPORTED couplings:**
   - ExitPolicy fields and decision inputs;
   - BaseStrategy branches;
   - invariants;
   - fingerprint;
   - run_arm record;
   - switch-off reproduction (fingerprints + f00 trades).
4. **US3:** one arm per supported coupling per family, judged late; then the combination per
   family if ≥ 2 are adopted.
5. **US4:** report, memory, full test gate.

## Risks

| risk | mitigation |
|---|---|
| A coupling looks good in the replay but loses in the arm (re-sequencing) | Replays only choose (R2); arms judge |
| Same-candle high/low ordering creates phantom lock profits | Touch from closed candles only (R4); US1 excludes ambiguous candles and reports their count |
| New ExitParams fields silently change recorded fingerprints | Dropped from the hash while `None` (R5); a test over all recorded classes |
| C1 loosens the stop into larger losses | Hard bound −10% inside the invariants; risk verdict (max drawdown) reported per arm |
| C4 is arm-wide, because grace is shared by definition | Judged per reference arm separately; NNNC inherits it only if both families agree |
| The post-grace stop ratchet (latest-bar ATR) is mis-modelled offline | US1 recomputes it with `ExitPolicy.atr_stop_fraction` itself; checked against archive stop-exit prices (a fidelity check in the US1 report) |
| `simulate_forward` diverges from live once couplings exist | It refuses any coupling field (R4), so 021 tables can't describe an exit that isn't run |
| Too many arms | Only SUPPORTED couplings get arms; each value is fixed by US1, with no stage grid |

## Complexity Tracking

None.
