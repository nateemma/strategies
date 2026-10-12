# Report: Recovery-Aware Reversion Exits (028)

*2023–26 results are EXPLORATORY (chosen after their late-half replay was seen). 2022 results are
CONFIRMATORY (pre-registered), and PROVISIONAL under quality gate 2: 2022 adjoins f00, so it is a
different regime, not a distant era.*

## SUMMARY

| candidate | 2023–26 gain early / late (EXPLORATORY) | 2022 mechanism check | 2022 gain (CONFIRMATORY) | verdict |
|---|---|---|---|---|
| **C2** take-profit = 2.897 × signal-bar ATR, clamped [1%, 5%] | +4.46 / +4.17 | replay +0.97, PASS (floor n/a) | **+4.95 pp** (3/3 folds positive) | **ADOPT** |
| C4 grace 6.5 h → 12 h | +8.44 / +6.27 | recovery 0.469 < 0.50, replay −0.04 | not run | STOPPED_AT_FEASIBILITY |
| C1 stop never tighter than −3.71% | +3.29 / +1.88 | recovery 0.469 < 0.50, replay −1.29 | not run | STOPPED_AT_FEASIBILITY |
| C42 (C4 + C2) | +4.23 / +4.08 | recovery 0.469 < 0.50, replay −2.93 | not run | STOPPED_AT_FEASIBILITY |

**The reversion exit NNNC should inherit: today's exit with C2 switched on.** That is, set
`take_profit_atr_mult = 2.8970711921574543` for the reversion family. It is now a default-off
switch in the shared exit (`ExitParams` / `BaseStrategy`). Its promoted form reproduces the
exploratory arm's trades in 18 of 18 folds, and every recorded fingerprint is unchanged.

**Why the stop-side couplings failed.**
- They act on a mechanism: stopped trades that later recover. That mechanism is regime-dependent.
- In 2023–26, 63% / 57% of reversion stop exits recovered to +2% within the max hold. In the 2022
  bear market only 47% did, about momentum's level, where C4 had already failed.
- Giving stopped trades more room pays only while most of them recover.
- C4 had the largest 2023–26 effect and still failed on the fresh window, which is what a
  pre-registered confirmatory test is for.

## Pre-registration and deviations

- Committed as `e645016` before any 2022 backtest or replay. Every 2022 record is newer than that
  commit; `confirm.py` checks this and refuses otherwise.
- Contents: candidates and frozen values, folds e00–e02 (100 days each), universe, fidelity 0.90,
  recovery floor 0.50, at least 200 control trades, rule (a)–(d), the verdict-value mapping, and
  gate 5. See [preregistration.md](preregistration.md).
- **Deviations: none.**
- Recorded at T008, before commit: **XRP has no binance.us data from 2021-01-13 to 2023-07-14.**
  9 pairs trade in 2022, the same 9 for every arm, as in f00–f02.
- Freshness (R2):
  - no SimpleTrigger arm had been backtested on 2022, and the reversion gate's threshold came from
    2023+ trades;
  - the shared base exit (grace 6.5 h / −27%, ATR band, take-profit 2.2%) was settled in NN-era
    runs that spanned 2022 (325 backtests). That applies to control and candidates alike.

## US1 — mechanism check on 2022 ([feasibility.md](feasibility.md))

| | 2023–26 early | 2023–26 late | **2022** |
|---|---|---|---|
| control trades | 2,092 | 1,292 | 969 |
| null-replay fidelity | 0.996 | 0.996 | **0.997** |
| stop exits | 699 | 466 | 343 |
| … recovering to +2% within the max hold | 63.4% | 57.5% | **46.9%** |
| forgone P&L of those recoveries | | | +52.92 pp |

C2 passed with a replay gain of +0.97 pp. Its floor doesn't apply, because it changes the
take-profit, not the stop. The three stop-side candidates failed the floor, and also had replay
gains ≤ 0.

## US2 — verdicts ([verdicts.md](verdicts.md))

**C2 vs the control on 2022:**
- **Totals:** −0.47 pp vs −5.42 pp. Both arms lose in the bear year.
- **Per-fold gain:** e00 +2.02, e01 +1.15, e02 +1.78; total **+4.95 pp**.

| rule | value | result |
|---|---|---|
| (a) 2022 gain > 0 | +4.95 pp | pass |
| (b) paired P&L | INCONCLUSIVE_ARM_LOSES (p 0.55) | pass, flagged (pre-registered mapping) |
| (c) risk | Sharpe UNCHANGED (−0.03 vs −0.32); max drawdown UNCHANGED (15.66 vs 17.59) | pass |
| (d) 2023–26 sign | +4.46 / +4.17 | pass |
| liveness | trades changed in 3/3 folds | |
| power | CI half-width ±16.25 pp over the window | |

- **Not a significance test.** It is a persistence verdict: positive in three windows, two regimes
  and every 2022 fold, with no risk measure worse. Gate 5 doesn't apply (gain ≥ 1 pp).
- **How C2 works:** the take-profit scales with volatility, so it takes gains earlier on calm bars
  and lets them run on volatile ones.
  - Take-profit exits fall from 534 to 419; ROI exits rise from 11 to 66; stop exits are about
    level.
- **Replay vs arm:** the arm realised 5.1× the replay prediction in 2022, versus about 0.5× in
  2023–26. The replay can't see re-sequencing, and that effect changes sign between windows. Use
  the replay to choose what to test, never to size a gain.

## US3 — promotion ([promotion.md](promotion.md))

- **What was built.**
  - `ExitParams.take_profit_atr_mult` (None = off), with validation.
  - `ExitPolicy.atr_take_profit`, the one copy of the clamp; the 027 replay delegates to it.
  - `BaseStrategy.take_profit_atr_mult`, and `BaseStrategy._apply_atr_take_profit`, the per-trade
    resolution on the signal bar, cached per trade.
  - `FamilyExitMixin`: re-resolves after family overrides, and refuses the switch set both
    strategy-wide and per family.
  - The fingerprint omits the switch while it is off.
- **C4 and C1 were not built.** C4 was always only a parameter value, and C1 was not adopted.
- **How a strategy turns it on:**
  - `FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit_atr_mult": 2.8970711921574543}}`
    (SimpleTrigger); or
  - `take_profit_atr_mult = k` strategy-wide.
  - Unlike the 027 method override, the promoted switch is visible to the fingerprint.

**Equivalence and inertness (T021):**
- The promoted `SimpleTrigger_RevOnlyVolreg2_P2` reproduces XC2's `trades_hash` in **18 of 18
  folds** (f00–f14, e00–e02).
- The control class, re-run after the change under a fresh tag, reproduces its recorded f00 hash
  (`26869046bc547532`).
- Every recorded fingerprint is unchanged (10 pinned by test).
- 0 stop-invariant violations.

Details are in [promotion.md](promotion.md).

## Tests

- 39 new tests for `confirm.py` and its decision rule, including every verdict value.
- 9 for the exit switch.
- 12 for the fingerprints.
- **Single-source exit guard:** extended to the new parameter name. It had been failing since 027's
  replay files were added without classification. Fixing it moved the replay's grace-or-ATR choice
  onto `ExitPolicy.effective_stop_fraction`; the 2022 replay re-ran byte-identical.
- **Full suites:** `utils`, `Framework`, `research/exit_coupling`, `evaluation` and `SimpleTrigger`
  give **621 passed, 3 failed**.
  - The 3 failures are the training-pipeline goldens
    (`Framework/tests/test_training_pipeline_characterization.py`).
  - They fail identically with this feature's changes removed, and they don't touch exits.

## What this does and does not license

- **It licenses** making C2 the reversion exit for the next NNNC feature. Its evidence is three
  windows and two regimes, all positive, but no single window is significant.
- **It does not license:**
  - C4, C1 or C42 for reversion;
  - C2 for momentum (out of scope; momentum has no take-profit);
  - any claim beyond PROVISIONAL. A genuinely distant era is still missing (quality gate 2).
- **The stop-side result is a regime finding, not a dead end.**
  - Recovery after a stop is high in the 2023–26 regime and low in 2022.
  - A regime-conditioned exit is the natural follow-up: room for stopped trades only while the
    recovery rate is high, using an instrument such as 026's volreg.
  - It would need its own pre-registration and a fresh window.

## Tasks

- T001–T016 and T018–T025 done.
- T014a N/A (gain ≥ 1 pp).
- T017 N/A (C1 not adopted).
