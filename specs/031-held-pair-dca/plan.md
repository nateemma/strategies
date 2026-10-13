# Implementation Plan: Held-Pair Position Add (DCA)

**Branch**: `031-held-pair-dca` (spec directory; commits go to master) | **Date**: 2026-10-13 |
**Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/031-held-pair-dca/spec.md`

## Summary

This feature adds a default-off switch to the shared entry framework. With the switch on, a fresh
entry signal on a pair that already holds a trade adds one stake to that trade through freqtrade's
native position adjustment. The pre-registered arms then test it on the four NNNC models and the
two SimpleTrigger books. It runs in five steps.

1. **Switch (US1).**
   - `HELD_PAIR_ADD` is a merged class declaration on `EntryFrameworkMixin`. It sets
     `position_adjustment_enable` / `max_entry_position_adjustment`.
   - `adjust_trade_position` returns one stake when the signal candle's `enter_long` is set, the
     last fill is at least 1 h old, and liquidity allows it. The stake is sized like a fresh entry
     and wallet-bounded by freqtrade.
   - One generic `trade_clock_start(trade)` in `Framework/BaseStrategy` makes every age-based stop
     and exit rule measure from the last fill when position adjustment is on.
   - Freqtrade averages `open_rate` itself.
2. **Off means unchanged.**
   - The test suites pass.
   - The reference arms reproduce `trades_hash` 15/15, with fingerprints and model keys unchanged.
   - The harness gains additive `n_adds` fields and a multi-fill accounting test.
3. **Fidelity (US2).** The ARO arm's early half (f00–f07, already seen by the screen) is
   backtested with the switch on. `fidelity.py` matches its adds to the screen's DCA simulation.
   The gate is ≥ 95% of adds matched.
4. **Pre-register, then judge (US3).**
   - Commit the pre-registration. It names the smoke and fidelity records (ARO f00–f07) as exempt.
   - Run the late half for ARO, and all 15 folds for GBR/GFM/MLX.
   - Run SimpleTrigger `_G24` + DCA on 2022 (confirmatory) and on 2023–26 (exploratory).
   - `judge.py` applies `judge_one` / the 2022 rule, plus exposure, drawdown and the leverage
     control.
5. **Report (US4).** Write the report and update memory. The switch stays off by default.

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`). Freqtrade backtesting is run via
`regime/run_arm.py` / `evaluation/run_folds.py`.

**Primary Dependencies**:
- freqtrade position adjustment (upstream, read-only; research R1);
- `utils/EntryFramework.py`, `utils/StopInvariants.py`, `NNNC/_hold_window.py`,
  `Framework/BaseStrategy.py`;
- `evaluation/` (foldplan, run_folds, pool, compare_pooled, risk, fingerprint);
- `research/held_pair/screen.py`, `research/nnnc_framework/{entries,build_arms}.py`,
  `research/nnnc_switches/judge.py`, `research/st_grace/judge.py`.

**Storage**:
- plans in `evaluation/folds/`;
- arm records in `regime/arms/`;
- archives in `user_data/backtest_results/`;
- study outputs in `research/held_pair/`.

**Testing**: pytest (`utils/tests/`, `Framework/tests/`, `evaluation/` tests, `SimpleTrigger/tests/`,
`research/exit_coupling/tests/`), plus backtest byte-identity checks.

**Target Platform**: macOS (Apple Silicon), local backtests. One chain at a time.

**Project Type**: framework switch plus research harness.

**Performance Goals**:
- off-checks: 7 reference arms × 15 folds ≈ 25 min;
- arms: NNNC 4 × 15 folds ≈ 13 min, SimpleTrigger 2 × (3 + 15) folds ≈ 15 min;
- each `adjust_trade_position` call is O(1): one dataframe tail read.

**Constraints**:
- default off and byte-identical;
- no retraining, model keys unchanged;
- Framework/ strategy-agnostic;
- merged parameter dicts;
- no pair-specific code;
- the exit-parameter guard test passes;
- pre-registration before late-half/2022 records;
- the live bot is untouched.

**Scale/Scope**: 6 books, 1 setting (DCA:all:1h, one add), 24 h grace.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | How this plan complies | Status |
|---|---|---|
| I. Deployment regime | NNNC is judged on 2023–26 with the late half as the test, as for the adopted 24 h grace. 2022 is the distant era for SimpleTrigger. | PASS |
| II. Distant era (NON-NEGOTIABLE) | NNNC has no model covering 2022, and retraining is out of scope. NNNC verdicts are therefore labelled PROVISIONAL (two halves of one era), the same status as the 24 h grace adoption. The mechanism's distant-era evidence is SimpleTrigger 2022, which is pre-registered as confirmatory. | PASS (labelled) |
| III. Learnability ≠ edge | Judged on backtest P&L and risk verdicts only. | PASS |
| IV. Parameters live | Per-fold add counts. A zero-add arm is INCONCLUSIVE. AddDecision refusal counters. `trades_hash` differs from the control. | PASS |
| V. Realistic execution | The add applies the same liquidity cap and quote-volume floor as a fresh entry (R3) and the wallet bound. | PASS |
| VI. Diagnose before tuning | The screen and decomposition localised the cost (pair-held blocking) before this build. No tuning here: one fixed setting. | PASS |
| VII. Validate cheaply | The screen (an afternoon) preceded the build. The fidelity gate precedes the judged runs. | PASS |
| VIII. One space at a time | Only the held-pair policy varies. Grace, threshold and exits are fixed. | PASS |
| IX. Orthogonality | Not a factor addition (N/A). | N/A |
| X. No per-pair code | The switch is uniform across the universe. | PASS |
| XI. No leakage | No model fitting. Weights resolve to the existing per-fold 030 keys. | PASS |
| Repo: Framework agnostic | Framework/ gains only `trade_clock_start` and `_fillable_stake`, both generic. | PASS |
| Repo: params merge | `HELD_PAIR_ADD` merges along the MRO. | PASS |
| Repo: staging / master | Named files only, committed to master. | PASS |
| Gates 1–7 | Pinned fold plans; 2022 plus two halves; liquidity on; liveness; a < 1 pp adopted gain gets a repeat run; trade counts; no feature change. | PASS |

Post-design re-check: PASS. Nothing in research.md, data-model.md or the contracts adds
strategy-specific code to Framework/, changes a recorded fingerprint, or touches the model.

## Project Structure

### Documentation (this feature)

```text
specs/031-held-pair-dca/
├── spec.md
├── plan.md              # this file
├── research.md          # R1–R11: freqtrade mechanics, signal visibility, sizing, clocks, harness, arms, rule
├── data-model.md
├── quickstart.md
├── contracts/interfaces.md
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks
```

### Source Code (user_data/strategies)

```text
Framework/BaseStrategy.py            # + trade_clock_start, _fillable_stake; 4 age sites + ATR-TP key use them
utils/EntryFramework.py              # + HELD_PAIR_ADD (merged, validated), adjust_trade_position, refusal counters
utils/StopInvariants.py              # grace test uses trade_clock_start
NNNC/_hold_window.py                 # hold window uses trade_clock_start
NNNC/NNNC_HP_{ARO,GBR,GFM,MLX}_DCA_f{00..14}.py (+ .json)              # generated per-fold arms (60), as build_arms does
SimpleTrigger/SimpleTrigger_{RevOnlyVolreg2_P2,BreakoutGbb_MomExit}_G24_DCA.py
evaluation/fingerprint.py            # HELD_PAIR_ADD in OPTIONAL_ATTRIBUTE_MEMBERS
evaluation/pool.py                   # + n_entries (additive)
regime/run_arm.py                    # summarize: + n_adds, n_trades_with_adds (only when > 0), hp_counters (sidecar)
evaluation/folds/plan_hp_*.json, plan_st_e22_hp_*.json, plan_st_hp_*.json
utils/tests/test_held_pair_add.py    # unit: AddDecision, sizing, clocks, ATR TP, stop anchor, invariants, validation
evaluation/test_pool_multifill.py    # pool/pair/risk on a two-fill trade
Framework/tests/test_exit_policy_single_source.py  # allowlist entries for new files naming exit params
research/held_pair/
├── screen.py                        # + add_e per trade (additive)
├── build_arms.py                    # arm classes/.json/plans from the controls
├── fidelity.py                      # backtest adds vs screen adds
├── exposure.py                      # avg/peak stake, leverage control
├── judge.py                         # pre-registered rule -> verdicts.json
├── preregistration.json / .md
└── report.md
```

**Structure Decision**:
- The generic clock and sizing helpers go in Framework/.
- The switch goes in the shared entry framework in utils/.
- The arms follow the 023–030 generated-class pattern.
- The study code goes in research/held_pair/, next to the screen.

## Phases

### Phase 0 — Research: done ([research.md](research.md))

All unknowns are resolved from the code:
- add timing and fill (R1);
- signal visibility (R2);
- sizing (R3);
- clock sites (R4);
- profit reference (R5);
- declaration and hashing (R6);
- fidelity (R7);
- harness accounting (R8);
- exposure and the leverage control (R9);
- arms (R10);
- rule (R11).

No NEEDS CLARIFICATION remain.

### Phase 1 — Design: done

[data-model.md](data-model.md), [contracts/interfaces.md](contracts/interfaces.md),
[quickstart.md](quickstart.md).

### Phase 2 — Tasks (`/speckit-tasks`)

The expected order is:
1. tests;
2. switch + clock;
3. off-identity checks;
4. harness fields;
5. arm builder;
6. fidelity run + gate;
7. pre-registration commit;
8. judged chain;
9. judge + exposure;
10. report + memory.

## Risks

| risk | mitigation |
|---|---|
| The signal row seen in `adjust_trade_position` is not the signal candle | Date check on every call, a `row_misaligned` counter, and a smoke-test assertion (R2). A non-zero count fails the fidelity step. |
| Add-candle exits double the losing exits (the backtest adds before the exit check) | Counted in fidelity. If material, report it. The rule is fixed, and the arm is judged as built. |
| Restarting grace widens a tightened stop | This is the screen's behaviour, and the invariants still bound it. Reported. |
| A clock-site change alters an off strategy | Gated on `position_adjustment_enable`, plus the 7-arm byte-identity check. |
| The wallet binds and adds starve fresh entries | Exposure report: adds vs fresh entries refused for the wallet. |

## Complexity Tracking

No constitution violations to justify.
