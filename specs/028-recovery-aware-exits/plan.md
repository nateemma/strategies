# Implementation Plan: Recovery-Aware Reversion Exits

**Branch**: `028-recovery-aware-exits` (spec directory; commits go to master) | **Date**: 2026-10-12 |
**Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/028-recovery-aware-exits/spec.md`

## Summary

This feature is the confirmatory test of 027's three exploratory reversion exit couplings, plus
their C4+C2 combination, on a window none of them has seen: 2022. It runs in four steps.

1. **Freeze and pre-register.** Add the one missing candidate class (C4+C2) and run it on f00–f14,
   so the rule's 2023–26 sign condition can be checked for every candidate. Write the 2022 fold
   plans. Commit a pre-registration (`preregistration.md` and `.json`) that fixes the candidates,
   values, window, universe, feasibility floor, decision rule and minimum trades. Nothing touches
   2022 before that commit.
2. **US1: mechanism check on 2022.** Run the control (`st_book_rev2`) on the three 2022 folds. Then
   replay its trades with 027's simulator: a fidelity gate of at least 90%, the recovery share of
   stop exits, the P&L those recoveries forgo, and each candidate's replay gain.
3. **US2: arms.** Each passing candidate runs on the same three folds. `confirm.py` applies the
   pre-registered rule (a)–(d) and writes ADOPT, NOT ADOPTED or INCONCLUSIVE.
4. **US3: promotion, only if adopted.**
   - C4 is an existing parameter, so adopting it means declaring 12 h grace; no framework code.
   - C1 and C2 become default-off `ExitParams` fields, resolved once in the shared exit.
   - Promotion is proven two ways: switched off, every recorded arm is byte-identical; switched
     on, the trades equal the exploratory subclass's on all 18 folds.

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`), freqtrade backtesting via `regime/run_arm.py`.

**Primary Dependencies**: `evaluation/` (foldplan, run_folds, pool, halves, compare_pooled, risk,
fingerprint), `research/exit_coupling/` (paths, feasibility), `utils/ExitPolicy.py`,
`utils/FamilyExits.py`, `utils/StopInvariants.py`, `Framework/BaseStrategy.py`, SimpleTrigger arms.

**Storage**: JSON fold plans in `evaluation/folds/`, arm records in `regime/arms/`, backtest
archives in `user_data/backtest_results/`, and markdown/JSON outputs in the spec directory.

**Testing**: pytest (`research/exit_coupling/tests/`, `utils/tests/`, `Framework/tests/`,
`evaluation/` tests).

**Target Platform**: macOS (Apple Silicon), local backtests. One backtest chain at a time.

**Project Type**: research harness plus framework switch.

**Performance Goals**: control and 4 candidates × 3 folds ≈ 15 fold backtests (about 5 min). C4+C2
on f00–f14 is about 3 min. The replay over about 1,000 trades takes minutes.

**Constraints**:
- pre-registration before any 2022 run;
- no re-tuning;
- no pair-specific code;
- every recorded arm's fingerprint unchanged;
- stop invariants in every switch combination.

**Scale/Scope**: reversion family only; 4 candidates; 3 folds of 100 days.

## Constitution Check

*GATE: checked before Phase 0 and re-checked after Phase 1. Unchanged.*

| principle | status | how |
|---|---|---|
| I. Deployment regime | PASS, with a caveat | 2022 is a bear regime, not today's. It judges persistence; the 2023–26 halves stay the deployment-regime evidence, and rule (d) requires both. |
| **II. Distant era (NON-NEGOTIABLE)** | **PASS, central; results stay PROVISIONAL** | 2022 is the first never-tested window for these arms. It adjoins f00, so it is a different regime but not a fully distant era. The report keeps the PROVISIONAL label (R8). |
| III. Learnability ≠ edge | N/A | No model |
| IV. Parameters proven live | PASS | Per-candidate count of folds with changed trades; a zero-change arm is INCONCLUSIVE. Switch-off reproduction proves the off state is inert. |
| V. Lookahead-clean ≠ realistic | PASS | Same liquidity cap, costs and stake as 023–027. The replay uses closed candles only. |
| VI. Diagnose before tuning | PASS | Nothing is tuned; values are frozen at 027's |
| **VII. Validate cheaply** | **PASS** | The US1 replay gates the arms. It is cheap, but kept because it re-checks the mechanism on a new regime. |
| VIII. One space at a time | PASS | One coupling per arm; the combination is its own arm |
| IX. Orthogonality | N/A | |
| X. No per-pair code | PASS | Values are the same for every pair |
| XI. No leakage | PASS | Values are fixed before 2022 is seen. Signal-bar ATR for C2. A 4,096-bar warm-up starts after every pair's listing (R1). |
| Framework strategy-agnostic | PASS | Promoted switches are `ExitParams` fields with no family or trigger names |
| Parameter dicts merge | PASS | Through the existing merging `SELL_OVERRIDES` and `FAMILY_EXITS` |
| Quality gate 1 (pinned range) | PASS | Fold windows are fixed in the pre-registration |

## Project Structure

### Documentation (this feature)

```text
specs/028-recovery-aware-exits/
├── spec.md
├── plan.md               # this file
├── research.md           # R1–R9
├── data-model.md
├── quickstart.md
├── contracts/interfaces.md
├── preregistration.md    # US0, committed before any 2022 run
├── preregistration.json  # machine-readable copy, read by confirm.py
├── baseline_c42.md       # C4+C2 on f00–f14 (before pre-registration)
├── feasibility.md/.json  # US1 on 2022
├── verdicts.md/.json     # US2
├── promotion.md          # US3 (only if adopted)
└── report.md
```

### Source Code

```text
SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_XC42.py     # C4+C2 exploratory class (setup)
evaluation/folds/plan_st_xpl_rev_c42.json              # C4+C2 on f00–f14
evaluation/folds/plan_st_e22_{rev2,c4,c2,c1,c42}.json  # 2022 folds e00–e02
research/exit_coupling/confirm.py                      # US1 replay + US2 decision rule
research/exit_coupling/tests/test_confirm.py
# US3, only for an adopted C1/C2:
utils/ExitPolicy.py           # ExitParams.stop_min_depth / take_profit_atr_mult, resolvers
Framework/BaseStrategy.py     # call sites (_resolve_normal_stop; take-profit resolution)
utils/StopInvariants.py       # bound includes stop_min_depth
evaluation/fingerprint.py     # None-valued new fields omitted from the hash
utils/tests/, Framework/tests/  # equivalence, invariants, single-source allowlist
```

**Structure Decision**:
- Everything decision-related lives in `research/exit_coupling/confirm.py`. It reuses
  `feasibility.trade_rows`, `fidelity`, `paths.replay`, `halves.split` and `risk.compare`, with no
  second implementation of a verdict.
- Framework edits happen only in US3 and only for an adopted C1 or C2.

## Complexity Tracking

None. The only new framework surface (two `ExitParams` fields) is conditional on adoption.
