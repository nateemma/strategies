# Report: 023 SimpleTrigger strategy family

**Date**: 2026-10-10 · **Spec**: [spec.md](spec.md) · **Evidence**: [baseline.md](baseline.md)

> **Every delta in this report is PROVISIONAL, not a finding (quality gate 2).** No distant era
> exists for this data, so persistence was tested by the early/late halves split only
> (f00–f07 / f08–f14).

## SUMMARY

1. **The family exists and is validated.** `SimpleTrigger/` runs any measured-causal trigger
   through NNNC's framework with no model, and it reproduces every recorded 021/022 no-model arm
   trade-for-trade: 7 arms × 15 folds, identical `trades_hash` in every fold (SC-001).
2. **The twin is exact.** `NNNC_TriggerGate` is arm D's real NNNC code path, with arm D's own
   fold models loaded and the `gbb` trigger as its entry. It reproduces `SimpleTrigger_Gbb`'s
   trades in 15/15 folds, and all three arms have identical framework fingerprints in every fold
   (SC-003). So the frameworks really are the same, and the model is the only difference.
3. **What the model adds to arm D (provisional).**

   | arm | trades | per-fold sum | positive folds |
   |---|---|---|---|
   | arm D: `gbb` model, ARM_D_OPEN framework | 1,420 | **+42.71pp** | 14/15 |
   | twin: `gbb` trigger, same framework, no model | 9,056 | **+11.04pp** | 7/15 |

   - **Paired bootstrap:** UNCHANGED. Delta +31.67pp, CI −1.09..+18.30pp/yr, p = 0.083. It
     can't resolve because the arms share only 617 trades (6.8% of the twin's).
   - **Matched-count selection test:** arm D's 1,420 trades against same-count random draws
     from the twin's trades, stratified by fold.
     **z = +6.55 pooled, +5.65 early, +3.79 late**, each at the test's minimum p (≈5e-5).
     The selection effect holds in **both** halves.
   - **Wallet by half:** early, arm D +29.72pp (8/8) against the twin +30.33pp (4/8), which is a
     tie. Late, arm D +12.99pp (6/7) against the twin −19.28pp (3/7), where the paired verdict
     is IMPROVED.
   - The same shape as 022: the selection advantage persists in both halves; the wallet
     advantage comes from the late half, where the unselected trigger collapses.
   - **Caveat:** only 43.5% of arm D's trades are also twin trades. The model enters
     *outside* the `gbb` trigger more than half the time, so "selection within the trigger" is
     only part of what it does.
4. **021/022's no-model numbers were not model-free comparisons with arm D (research R1).** Their
   `TriggerOnly` framework ran a **static −5% stop** with the ATR-adaptive stop off, which also
   made the volume-confirmation stop inert, and a 200-bar warm-up. The same `gbb` trigger made
   **+2.64pp** under that framework and **+11.04pp** under arm D's. That ≈+8.4pp is the stop, not
   the model.
5. **Arm D's headline number was pre-fix (research R2).** +39.98pp / 15/15 predates the
   2026-10-08 post-grace stop fix. Current code gives **+42.71pp / 14/15**.

## Success criteria

| SC | status | evidence |
|---|---|---|
| SC-001 legacy reproduction | **PASS**: 7 arms × 15/15 folds; `wf_gbbr` 458 trades | `SimpleTrigger/tests/test_legacy_reproduction.py`, baseline.md T030 |
| SC-002 new arm = trigger + overrides, no training | **PASS** | `SimpleTrigger_Gbb.py` is 1 line; no `saved_data/SimpleTrigger*` |
| SC-003 equivalence | **PASS**: 15/15 | `evaluation/twins/st_gbb.check.json` |
| SC-004 model contribution reported | **PASS** (provisional) | `evaluation/twins/st_gbb.compare.{json,md}` |
| SC-005 attribution rows with liveness | **PASS**: every component has a row; 2 INERT flagged (max_hold; legacy volume-confirm) | `evaluation/attribution_st_gbb.{json,md}`, `attribution_stl_gbbo.*` |
| SC-006 causality | **PASS**: 16 PASS, `trends`/`bands` UNSCOREABLE, `random_matched` FAIL, planted control FAILS | `research/outcomes/trigger_causality.json` |
| SC-007 coupled == decoupled | **PASS**: 15/15; 0 stop-invariant violations in any run | `regime/arms/st_gbbc_*`, run logs |
| SC-008 dry-run untouched | **PASS**: class dump identical at `9e82a45` and now | baseline.md T032 |

## Component attribution (US2)

Full configuration `st_gbb` (`gbb` + ARM_D_OPEN): 9,056 trades, +11.04pp. "Contribution" is
full minus neutralised, so positive means the component helps. Many paired verdicts read
INCONCLUSIVE_ARM_LOSES: the neutralised arm loses money, so the bootstrap refuses to quote a
delta against it (compare_pooled admissibility). The sums are still reported.

| component | status | neutralised arm | contribution | liveness evidence |
|---|---|---|---|---|
| trigger (→ `empty`) | LIVE | −91.35pp | **+102.4pp** | signal bars 206,715 → 1,282,080 |
| guards ×6, exit_signal | ALREADY_NEUTRAL | | | ARM_D_OPEN is guards-open |
| take_profit | LIVE | −1.85pp | **+12.9pp** | take_profit exits 3,839 → 0 |
| **max_hold** | **INERT** | +11.04pp | 0 | **max_hold exits 0 → 0**: the 10-day hold never binds, because unclog_12h/1d always close first |
| roi | LIVE | +9.55pp | +1.5pp | roi exits 63 → 0 |
| stop_atr_band (→ static) | LIVE | +2.82pp | **+8.2pp** | trades differ in every fold |
| **stop_grace** | LIVE | −36.92pp | **+48.0pp** | trades differ in every fold |
| stop_volume_confirm | LIVE | +11.28pp | −0.2pp | trades differ, P&L flat |

Under LEGACY_TRIGGERONLY (`stl_gbbo`, +2.64pp):
- `stop_grace` is LIVE (+13.1pp).
- `stop_atr_band` is ALREADY_NEUTRAL (adaptive is already off).
- **`stop_volume_confirm` is INERT**: byte-identical trades. This is research R1, now measured.

Reading:
- The 6.5h grace stop is the single largest framework lever, worth ~48pp.
- The take-profit (~13pp) and the ATR band (~8pp) also matter.
- max_hold is dead weight under this exit stack.
- The volume-confirmation tightening does nothing measurable under the arm D framework, and
  literally nothing under the legacy one.

## What was built

- **`SimpleTrigger/`**, a sibling of NNNC (clarification Q1):
  - `SimpleTriggerStrategy` (contract C1, G1–G6);
  - `presets.py`: `ARM_D_OPEN`, the default, and `LEGACY_TRIGGERONLY`;
  - `label_map.py`: all 29 `LabelMethod`s mapped;
  - `components.py` and `coupling.py`;
  - configs `SimpleTrigger_Gbb`, `_GbbCoupled` and 7 `_Legacy_*`.
- **`NNNC/NNNC_TriggerGate.py` + `.json`**: the equivalence arm, which borrows arm D's fold
  models and can never train.
- **`utils/StopInvariants.py`**: a strategy-agnostic stop-invariant mixin (sign, price-space
  bound, grace direction).
- **`evaluation/`**:
  - `fingerprint.py`: the resolved-framework hash;
  - `halves.py`, `matched_count.py`, `twins.py`, `attribution.py`, `dump_strategy.py`;
  - `foldplan derive` plus the no-model path in `run_folds` (with its causality preflight);
  - exact arm-record matching;
  - `compare_pooled.compare_data`.
- **`regime/run_arm.py`**: every record now carries `framework_sha`, entry-condition counts,
  exit-reason counts and the 023 declarations.
- **`Framework/BaseStrategy.py`**: a generic entry-condition sink (off unless the env var is set).
  It was proven byte-neutral on recorded arm D and `gbb` folds.
- **`research/outcomes/causality.py`**: the truncation causality checker with a planted control.

## Deviations from the task list

- **T015(b):** the literal clause can't happen, so it was implemented as |ratio| > 1. The (c)
  bound uses min(ATR floor, `stoploss`), because the stop falls back to the static value without
  ATR. The no-change `1.0` and `STOP_AT_MARKET_RATIO` values are exempt.
- **T026(f):** `exit_profit_only` is not forced False. Arm D inherits True, and parity wins.
- **`enter_tag`** stays `"model_entry "`.
- **Exit codes:** `twins check` adds exit 4 (incomplete) and `twins compare` exit 5 (refused) to
  the contract's 2/3.
- **Coupled take-profit** is rounded to 1e-12, so a declared round trip is exact.

## Findings along the way (outside 023's scope, filed)

- **The paired bootstrap is not reproducible across processes.** `bootstrap_delta` builds its
  difference list by iterating Python sets, so CI bounds move about ±0.02pp/yr and p about ±0.01
  with `PYTHONHASHSEED`. The file forbids retuning, and a fix (sorting keys) would shift every
  recorded value slightly. **This is the user's decision.**
- **XRP's binance.us 15m history has a 2.5-year delisting gap** (2021→2023). Any per-era slice
  that touches 2021–23 XRP silently spans it.
- **`random_matched` is not causal.** It's seeded from `len(df)`, so live it reshuffles every
  candle. That's fine as a leaderboard control and wrong as an entry rule.
- **Stale records corrected:** `future_sharpe/_sortino/_expectancy` reduce to `adx` (the
  inventory test said they had no entry condition).
- **`evaluation/pick_guards.floor_for_window` has zero callers.** L177 still applies the absolute
  25-trade floor (022's fix is unwired).
- **`run_folds.py` refers to `evaluation/guard_trackability.py`, which does not exist.**

## Follow-up (FR-007a)

**Merge: restructure NNNC as SimpleTrigger plus a model gate.** FR-008 has now passed (SC-003,
15/15), so the two families can share one framework definition instead of being held together by
an equivalence test. Recommended next. It also puts `StopInvariantMixin` on every NNNC arm, which
FR-017 currently covers only for SimpleTrigger and the twin.

Further investigations this enables:
- **The model's entries outside the trigger** (56.5% of arm D's trades): what are they, and do
  they carry the edge?
- **Coupled-mode experiments with real label targets.**
- **A 5-axis guard grid** run directly on SimpleTrigger arms.

## Addendum (2026-10-10): every causal trigger under ARM_D_OPEN

All 16 causality-PASS triggers through arm D's framework, no model, same 15 folds. Every
fold of every arm has `framework_sha 303456c1fa322b4c`, the same as arm D, so the trigger is the
only thing that differs between rows. Per-fold sum of wallet %; early = f00–f07, late = f08–f14.
**PROVISIONAL (quality gate 2).**

| trigger | arm | trades | per-fold sum | positive folds | early | late |
|---|---|---|---|---|---|---|
| `gbb_tight` | `st_gbbt` | **89** | +13.84pp | 7/15 | +7.01 | +6.83 |
| `gbb` | `st_gbb` | 9,056 | +11.04pp | 7/15 | +30.33 | −19.28 |
| `breakout_bb` | `st_bbo` | 7,351 | −1.09pp | 6/15 | +4.18 | −5.27 |
| `breakout_gbb` | `st_bgo` | 8,563 | −5.86pp | 6/15 | +4.99 | −10.85 |
| `votes_consensus` | `st_vcon` | 13,704 | −14.99pp | 8/15 | +2.96 | −17.95 |
| `geometry` | `st_geo` | 5,400 | −17.09pp | 6/15 | +2.28 | −19.37 |
| `gbb_stale` | `st_gbbst` | 5,895 | −27.58pp | 4/15 | −13.07 | −14.51 |
| `breakout_squeeze` | `st_bsq` | 4,824 | −42.97pp | 2/15 | −28.37 | −14.60 |
| `votes_ind3` | `st_vi3` | 13,116 | −43.80pp | 5/15 | +4.41 | −48.20 |
| `breakout_vol` | `st_bvo` | 9,544 | −43.85pp | 7/15 | −30.54 | −13.31 |
| `votes_tech` | `st_vtech` | 13,276 | −53.51pp | 4/15 | −4.15 | −49.36 |
| `breakout` | `st_bo` | 10,869 | −54.26pp | 3/15 | −31.76 | −22.50 |
| `adx` (also `future_*`) | `st_adx` | 19,035 | −57.01pp | 6/15 | −4.33 | −52.69 |
| `votes_ind2` | `st_vi2` | 13,822 | −60.32pp | 5/15 | −7.59 | −52.73 |
| `votes_ind4` | `st_vi4` | 15,690 | −72.85pp | 3/15 | −13.06 | −59.79 |
| `empty` (no trigger) | `st_empty` | 23,226 | −91.35pp | 6/15 | −18.74 | −72.61 |

Reading:
- Every trigger beats `empty`, so each one adds selection over "enter on every bar". Only `gbb`
  and `gbb_tight` are net positive.
- **`gbb_tight` is the only trigger positive in both halves, but at 89 trades it is
  unresolvable.** Four folds have zero trades, and the result rests on 5 folds (f04, f05, f07,
  f08, f11) with 76–100% win rates. This is consistent with the threshold finding (the ceiling
  rises with gbb strictness while trade count collapses). It is a candidate to examine, not a
  result.
- `gbb`'s early-half edge (+30.33) is the largest of any trigger. Every trigger except
  `gbb_tight` is negative in the late half, so the late-half collapse is not specific to `gbb`.
  It is market-wide under this exit stack.
- The `votes_*` triggers all hold up early (−13 to +4) and collapse late (−18 to −60), the
  same shape as `adx`.
- Consistency check: `st_empty` reproduces `st_gbb__trigger` (the attribution arm) with
  identical `trades_hash`.
