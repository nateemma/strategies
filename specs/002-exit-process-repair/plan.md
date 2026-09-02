# Implementation Plan: Exit-Process Characterisation and Repair

**Branch**: `002-exit-process-repair` (spec directory; commits go directly to master) | **Date**: 2026-09-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/002-exit-process-repair/spec.md`

## Summary

The stop is half of all exits and the only loss-making one. Phase 0 research
resolved *why*, and the answer reshapes the feature: the exits labelled
`trailing_stop_loss` come from a `return 1.0` in the grace branch that freqtrade
applies as a ratcheting profit-lock — not from the grace level (inert), and not
from freqtrade's trailing stop (inert). That accidental ratchet is **load-bearing**:
replacing it with the semantically correct `None` costs 9.24pp across three
windows.

So this is no longer "characterise, then tune the stop". Four of nine exit-side
settings are inert, the two most attractive interventions are already refuted,
and the remaining leverage is concentrated in one place: the ratchet is a
constant that happens to work, with no tunable distance. Making it explicit is
the main candidate.

Approach: finish characterisation from recorded trades (no backtests), screen
candidates by counterfactual replay (no backtests), and spend full evaluations
only on survivors. Nothing retrains at any point.

## Technical Context

**Language/Version**: Python 3.12 under `.venv/bin/python` with `PYTHONPATH` set — the strategy imports `Framework` and `utils` as top-level packages.

**Primary Dependencies**: pandas, numpy, `freqtrade.data.btanalysis` for reading recorded trades. freqtrade itself only for Phase 4. No new third-party dependency.

**Storage**: Recorded trades in `user_data/backtest_results/*.zip` (three baseline archives already produced). Price history in `user_data/data/binanceus/` at 15m. Outputs are CSV plus a written report under `regime/exit/`.

**Testing**: Correctness of the counterfactual screen is established by reproducing the recorded baseline outcomes under the current rule (FR-014). Phase 4 is validated by measurement, not unit tests.

**Target Platform**: macOS on Apple Silicon, local execution.

**Project Type**: Research study plus, if a candidate survives, a change to one strategy-side mechanism.

**Performance Goals**: Phases 1–3 complete in seconds to minutes — they read 167 trades and price history, nothing more. Phase 4 backtests are the only expensive step and are budgeted per surviving candidate.

**Constraints**: **Nothing retrains.** No entry-side change. Pinned literal timeranges. Realistic execution constraints active. The baseline is captured explicitly in code, never read from the strategy JSON, which has already drifted once mid-feature.

**Scale/Scope**: 167 baseline trades across three windows, 11 traded pairs, one host strategy, a candidate list that research has already cut from nine to five.

## Constitution Check

*GATE: evaluated before Phase 0 research, re-evaluated after Phase 1 design. Constitution v1.0.0.*

| Principle | Verdict | How this plan satisfies it |
|---|---|---|
| I. Evaluate in the deployment regime | **PASS** | W1 carries the headline verdict; W2/W3 are persistence checks. |
| II. Persistence across temporally distant eras | **PASS** | W3 decides. Already applied in research: the hyperopt candidate was rejected on a W3 sign reversal. |
| III. Learnability is not edge | **PASS (N/A)** | No model metric appears anywhere; nothing trains. |
| IV. Verify parameters are actually live | **PASS — the feature's core** | Research found the 8th inert parameter (`stoploss_grace_level`) and confirmed three more. Four of nine exit settings do nothing. FR-023/FR-027 make an unchanged result a defect, not a finding. |
| V. Lookahead-clean is not realistic | **PASS** | Phase 4 keeps volume constraints active. Phases 1–3 read only *recorded* trades, so there is no lookahead surface — the outcomes already happened. |
| VI. Diagnose before tuning | **PASS — exemplary** | Research diagnosed the mechanism before any tuning, and that diagnosis killed four of the nine anticipated candidates outright. |
| VII. Validate cheaply before building | **PASS** | The counterfactual screen rejects candidates for the cost of a calculation. R2 spent one throwaway backtest to refute the most attractive wrong move. |
| VIII. One parameter space at a time | **PASS** | Sell space only; entry is frozen by feature 001. Candidates are discrete arms, measured one at a time. |
| IX. Orthogonality earns a factor its place | **PASS (N/A)** | No multi-factor book here. |
| X. Structural fixes, not per-pair overrides | **PASS** | FR-026. No candidate may reference a pair. |

**Post-Phase-1 re-evaluation**: unchanged. The design adds a study directory and one extension to an existing runner; no `Framework/` change is made until a candidate has survived the screen, and any such change is reverted if it does not.

**Result: no violations.** Complexity Tracking is empty and has been removed.

## Project Structure

### Documentation (this feature)

```text
specs/002-exit-process-repair/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 — R1..R6; the central puzzle is RESOLVED here
├── data-model.md        # Phase 1 — entities and validation rules
├── quickstart.md        # Phase 1 — runnable validation guide
├── contracts/
│   └── study-cli.md     # Script invocations and output schemas
├── checklists/
│   └── requirements.md  # Spec quality checklist (all items pass)
└── tasks.md             # Phase 2 — created by /speckit-tasks, NOT here
```

### Source Code (repository root = `user_data/strategies/`)

```text
regime/exit/                        # NEW — this feature's study
├── baseline.py                     # the pinned baseline config + archive paths (FR-028b)
├── characterise.py                 # exit mix, stop attribution, excursions (FR-001..FR-008)
├── replay.py                       # counterfactual screen; must reproduce baseline first (FR-014)
├── candidates.py                   # candidate exit rules, each citing its motivating measurement
└── README.md                       # findings, including a "nothing improves it" outcome

regime/
└── run_arm.py                      # MODIFIED — add sell-space, ROI and stoploss overrides

# Only if a candidate survives the screen:
Framework/
└── BaseStrategy.py                 # the ratchet lives here; changed only under a surviving candidate,
                                    # and reverted if it does not hold across all three windows
```

**Structure Decision**: the study lives under `regime/exit/` rather than a new
top-level directory, because it reuses `regime/regimedata.py` and
`regime/run_arm.py` directly and shares the same pinned windows and host. Phases
1–3 touch no strategy code at all. `Framework/BaseStrategy.py` is modified only
once a candidate has earned it, using the edit-measure-revert pattern that
research R2 already used successfully — the file is committed and clean, so
`git checkout` is a reliable undo.

## Execution Phases

### Phase 1 — Characterise (no backtests, no model load)

Most of this is already done in research. What remains:

- Per-trade excursion analysis using the recorded `min_rate` / `max_rate`
  columns plus forward price history, over ~1/3/6/12/24h extensions.
- Whether each stopped trade would have reached the take-profit or ROI level.
- **Profitable and losing stops reported as separate populations throughout**
  (FR-004) — the 50.6% whose stop ended *above* entry are a different
  phenomenon from those that fell.
- Excluded-trade counts where subsequent history is insufficient (FR-008).

**Exit condition**: a stated verdict on whether the stop cuts noise or cuts
losers, with the figure behind it. Research already establishes the mechanism;
this quantifies its cost.

### Phase 2 — Enumerate candidates

Research already eliminated four of nine. The live list, each with its motivating
measurement:

1. **Make the ratchet explicit and tunable** — it produces 50% of exits and is
   currently the constant `1.0` with no distance parameter.
2. **Grace duration** — the one remaining live dimension of the grace mechanism.
3. **ROI ladder instead of flat 3%** — 22% of exits land at exactly +3.00%.
4. **`cexit_take_profit` upward** — R4 showed lowering it is harmful; upward is untested.
5. **Re-enable model exit signals** — no retrain; a change of kind, flagged as higher risk.

**Exit condition**: each candidate cites its measurement and is confirmed
no-retrain. Anything research killed is listed as excluded with the reason.

### Phase 3 — Screen (no backtests)

`replay.py` re-derives each recorded trade's outcome under a candidate rule from
price history. **It must first reproduce the recorded baseline outcomes under the
current rule** (FR-014) — otherwise it silently mis-scores everything. Candidates
that do not beat the baseline under the screen are rejected here.

### Phase 4 — Confirm (backtests, survivors only)

Full evaluation on all three windows with volume constraints active. Judged on
realized profit and risk-adjusted return. W1 headline, W3 decides persistence.
Every arm checked for inertness against the baseline hash.

## Per-Arm Protocol (Phase 4)

| Gate | Mechanism |
|---|---|
| Pinned windows | W1 `20250901-20260831`, W2 `20240601-20250531`, W3 `20230101-20231231` |
| Baseline | Pinned in `baseline.py`, never read from the strategy JSON |
| Realistic execution | Volume/liquidity constraints active |
| Liveness | Identical `trades_hash` to baseline ⇒ INERT, a defect, not a result |
| Judge on | Realized profit, Calmar, Sortino, drawdown — never a model metric |
| Distant window | Sign reversal in W3 ⇒ rejected |
| Trade count | Reported; a zero- or few-trade arm is inconclusive |
| Entry untouched | Entry configuration byte-identical to baseline |

## Known Hazards

1. **The strategy JSON drifts.** It changed mid-feature (15:46), replacing the
   baseline exit config with a hyperopt output. The baseline is therefore pinned
   in code. Never read it from disk.
2. **`return 1.0` looks like a bug and is not.** Research R2 refuted the
   correction at a cost of 9.24pp. Anyone reading `custom_stoploss` will form
   the same hypothesis; the record exists to stop them acting on it.
3. **Four of nine exit settings are inert.** `stoploss_grace_level`,
   `trailing_*`, `exit_guard_threshold`, `exit_close_norm_threshold`. Tuning any
   of them yields a guaranteed null that reads like a finding.
4. **A candidate can shift trades between exit categories without improving
   them.** Converting stops into equally-bad time-based exits looks like success
   on the exit mix. Judge on total outcome.
5. **The two stop populations may want opposite remedies.** Widening helps the
   ones that recovered and hurts the ones that fell further.
6. **A live dry run is executing** (`MomentumRegimeBasket15mFast`). Phase 4
   modifies `Framework/BaseStrategy.py`, which that strategy also imports — do
   not leave an experimental edit in place, and revert immediately after each arm.

## Next Command

`/speckit-tasks` — to generate the dependency-ordered task breakdown.
