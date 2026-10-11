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

## Addendum 2 (2026-10-10): component attribution for every trigger

Same method as US2, applied to all 15 non-empty triggers under ARM_D_OPEN. 70 new arms × 15 folds.
"Contribution" = full minus neutralised (positive = the component helps). In brackets: the
early/late split of that contribution. ✓ means both halves have the same sign. The `trigger`
column is full minus `st_empty` (−91.35pp), which is the same neutralised arm for every trigger.
`max_hold` is not run: it had **0 exits in every fold of every trigger**, so it is INERT
everywhere. **PROVISIONAL (gate 2), and this is 75 cells read post hoc. Treat single cells as
leads, not findings.**

| trigger | full | trigger | grace | TP | atr | roi | volc |
|---|---|---|---|---|---|---|---|
| `gbbt` | +13.84 | +105.2 | +6.3 (+2/+4)✓ | -1.1 (-2/+0)✗ | -0.2 (-1/+0)✗ | +1.8 (+1/+1)✓ | -0.7 (-1/+1)✗ |
| `gbb` | +11.04 | +102.4 | +48.0 (+29/+19)✓ | +12.9 (-1/+14)✗ | +8.2 (+10/-1)✗ | +1.5 (+1/+1)✓ | -0.2 (+3/-3)✗ |
| `bbo` | -1.09 | +90.3 | +59.4 (+40/+20)✓ | -20.9 (-14/-7)✓ | +20.0 (+12/+8)✓ | +2.7 (-1/+4)✗ | +11.8 (+10/+2)✓ |
| `bgo` | -5.86 | +85.5 | +71.2 (+54/+17)✓ | -30.2 (-22/-9)✓ | +19.6 (+13/+7)✓ | +4.3 (-0/+4)✗ | +9.2 (+9/+0)✓ |
| `vcon` | -14.99 | +76.4 | +60.6 (+44/+17)✓ | +5.9 (-3/+9)✗ | +12.1 (+8/+4)✓ | +12.4 (+4/+9)✓ | +2.3 (+6/-4)✗ |
| `geo` | -17.09 | +74.3 | +33.8 (+19/+15)✓ | +0.0 (-5/+5)✗ | +15.7 (+6/+10)✓ | +6.8 (+2/+5)✓ | +6.4 (+3/+3)✓ |
| `gbbst` | -27.58 | +63.8 | +14.3 (+17/-3)✗ | +2.1 (-4/+7)✗ | +2.0 (-2/+4)✗ | +4.0 (+1/+3)✓ | -1.4 (-4/+3)✗ |
| `bsq` | -42.97 | +48.4 | +11.7 (+9/+2)✓ | -0.2 (-1/+1)✗ | +14.1 (+4/+10)✓ | -0.3 (-1/+1)✗ | +13.7 (+7/+7)✓ |
| `vi3` | -43.80 | +47.6 | +35.4 (+34/+1)✓ | +0.6 (+3/-2)✗ | -15.9 (-3/-12)✓ | +2.2 (+2/+0)✓ | -3.8 (+5/-9)✗ |
| `bvo` | -43.85 | +47.5 | +52.2 (+33/+19)✓ | +4.9 (-5/+10)✗ | +19.4 (+19/+1)✓ | +6.1 (+0/+6)✓ | +14.6 (+14/+1)✓ |
| `vtech` | -53.51 | +37.8 | +37.3 (+31/+6)✓ | -1.4 (-2/+1)✗ | -14.5 (-6/-9)✓ | +3.9 (+2/+2)✓ | -2.7 (-0/-3)✓ |
| `bo` | -54.26 | +37.1 | +51.1 (+31/+20)✓ | -2.8 (-3/-0)✓ | +16.1 (+13/+3)✓ | +8.6 (+1/+7)✓ | +3.9 (+3/+1)✓ |
| `adx` | -57.01 | +34.3 | +81.1 (+57/+24)✓ | -14.7 (-7/-8)✓ | -30.6 (-7/-24)✓ | +4.6 (+2/+3)✓ | -18.3 (-1/-18)✓ |
| `vi2` | -60.32 | +31.0 | +65.4 (+46/+19)✓ | +6.9 (+3/+4)✓ | -9.7 (-4/-6)✓ | +6.5 (+3/+3)✓ | -5.9 (+3/-9)✗ |
| `vi4` | -72.85 | +18.5 | +61.1 (+49/+12)✓ | +0.7 (+1/-0)✗ | -27.1 (-11/-16)✓ | +2.7 (+1/+2)✓ | -4.8 (+2/-7)✗ |

| component | helps in | same sign both halves | median |
|---|---|---|---|
| `stop_grace` | **15/15** | 14/15 | **+51.1pp** |
| `roi` | 14/15 | 12/15 | +4.0pp |
| `stop_atr_band` | 9/15 | 7/15 positive in both | +8.2pp |
| `take_profit` | 8/15 | 1/15 positive in both | +0.0pp |
| `stop_volume_confirm` | 7/15 | 6/15 positive in both | −0.2pp |

Reading:
- **The grace stop is general, not specific to `gbb`.** It helps under every trigger, is
  positive in both halves for 14 of 15, and is worth +6 to +81pp. It is the framework's one
  robust lever.
- **The ATR stop band splits by trigger family.** It helps the breakout/geometry triggers in both
  halves (+14 to +20pp) and hurts `adx`/`votes_*` in both halves (−10 to −31pp). So one band
  is not right for all entries.
- **The take-profit is right for `gbb` and wrong for the breakout triggers.** With it removed,
  `breakout_bb` goes from −1.09 to +19.87pp (early +18.01 / late +1.86) and `breakout_gbb`
  goes from −5.86 to +24.38pp (+26.72 / −2.34). Neither paired verdict resolves (p 0.16 and
  0.07). For `gbb`, the take-profit is worth +12.9pp, all of it in the late half.
- The volume-confirmation stop helps the breakout family (+9 to +15pp) and hurts `adx`
  (−18pp). Its median is about 0.
- **Coupling implication.** The exit components that matter interact with the trigger (ATR
  band, take-profit, volume-confirm), so a single global exit stack is a compromise across
  triggers. This argues for per-trigger exit parameters, which is coupled mode's purpose.

## Addendum 3 (2026-10-11): momentum-exit cheap check

Question: do momentum triggers want a different exit cycle from arm D's shared stack? For each of the
5 momentum triggers, three arms were compared: full; no take-profit (`__take_profit` from Addendum 2);
and `_mx`, which is no take-profit plus `unclog_12h` deferred to 24h via `NNNC/_hold_window.HoldWindowMixin`.
The hold window is LIVE in every arm: `unclog_12h` exits fall by about two thirds and `unclog_1d`
exits rise about 3–5×. **PROVISIONAL (gate 2).**

| trigger | full | no TP | no TP + 24h hold | `_mx` early / late |
|---|---|---|---|---|
| `breakout` | −54.26 | −51.43 | −52.84 | −35.15 / −17.69 |
| `breakout_vol` | −43.85 | −48.78 | −46.27 | −32.33 / −13.95 |
| `breakout_squeeze` | −42.97 | −42.79 | −38.06 | −26.38 / −11.68 |
| `breakout_bb` | −1.09 | +19.87 | **+20.89** | **+16.22 / +4.67** |
| `breakout_gbb` | −5.86 | +24.37 | **+25.19** | **+22.68 / +2.50** |

Paired (`PYTHONHASHSEED=0`):
- `st_bbo_mx` vs `st_bbo`: +5.96pp/yr, CI [−2.84, +14.96], p 0.18. UNCHANGED.
- `st_bgo_mx` vs `st_bgo`: +8.42pp/yr, CI [−1.08, +18.06], p 0.08. UNCHANGED.
- Against `st_gbb`, both are UNCHANGED (p 0.68 / 0.57). They share only ~190 trades with `gbb`,
  so the entry sets are almost disjoint.
- Per-fold correlation with `st_gbb` is 0.54 (`bgo_mx`) and 0.39 (`bbo_mx`): moderate, not orthogonal.

Reading:
- **The split is not "breakout vs reversion".** The true price breakouts (`breakout`, `_vol`,
  `_squeeze`) are not rescued by a longer cycle. They stay at −38 to −53pp, negative in both halves.
- **The band-position momentum triggers (`breakout_bb`, `breakout_gbb`) are rescued by removing the
  take-profit.** The 24h hold adds about 1pp in total but moves late-half gains to the positive
  side. Both `_mx` arms are the only no-model arms besides `gbb_tight` that are positive in both
  halves, and they beat `gbb` in the late half (+4.7 / +2.5 vs −19.3).
- None of these deltas resolves (p ≥ 0.08). Direction is consistent in both halves for both triggers.

**Instrument gap found:** `evaluation/fingerprint.py` does not capture `HOLD_WINDOW_HOURS`, so
`_mx` and `__take_profit` arms hash identically despite different trades. Liveness was proven
from exit counts instead. The fingerprint needs the mixin's attribute before a family-preset feature
relies on it.

## Addendum 4 (2026-10-11): momentum guard on the true breakouts

New triggers `breakout_gm`, `breakout_vol_gm`, `breakout_squeeze_gm` (each breakout AND
`guard_metric > 0.2`, `research/outcomes/audit_triggers.py`; causality PASS, every earlier verdict
unchanged). Run on the momentum exits (no TP, 24h hold):

| trigger | no guard | + guard | early / late | signal bars kept |
|---|---|---|---|---|
| `breakout` | −52.84 | −44.48 | −32.93 / −11.54 | 92% |
| `breakout_vol` | −46.27 | −43.26 | −29.35 / −13.91 | 90% |
| `breakout_squeeze` | −38.06 | −34.15 | −23.28 / −10.88 | 86% |

**The guard is not what the breakouts are missing.** It rarely binds, because a close above the
20-bar high is almost always high in the band. Guards under ARM_D_OPEN are 100% pass for every
trigger (entry-condition sink), so they were never in the way either.

## Addendum 5 (2026-10-11): pricing `gbb` / `breakout_gbb` overlap entries both ways

New arm: `st_gbb_mx` (`gbb` on the momentum exits), completing the trigger × exit-set grid. Overlap
= entries opened by both `st_gbb` and `st_bgo`, the same bar under the same exits. There are 206
such entries, **2.3% of either trigger's trades** (1 of 206 differs in profit between the two
arms, a fill artefact). Matched per-trade return, momentum minus reversion exits, fold-blocked
bootstrap (`scripts/overlap.py`):

| set | n | reversion exits | momentum exits | diff/trade | 95% CI | p |
|---|---|---|---|---|---|---|
| overlap (via gbb arms) | 120 | +0.546% | +0.565% | +0.02pp | [−0.37, +0.38] | 0.86 |
| overlap (via bgo arms) | 182 | +0.373% | +0.510% | +0.14pp | [−0.23, +0.55] | 0.49 |
| all `gbb` entries | 7,125 | −0.004% | −0.040% | −0.04pp | [−0.10, +0.02] | 0.26 |
| all `breakout_gbb` entries | 5,422 | −0.011% | +0.036% | +0.05pp | [−0.03, +0.11] | 0.22 |

- **The overlap priority rule does not matter.** The overlap is small and indifferent to the
  exit set. Use **gbb priority**: the reversion side then reproduces `st_gbb` exactly, which is a
  mechanical correctness check for the combined arm.
- **The overlap entries are unusually good** (+0.4–0.55% per trade against ~0 for each trigger
  overall), almost all of it in the late half. `0.2 < guard_metric < 0.3` with a wide band may
  deserve its own look.
- Per-trade direction agrees with the family-exit hypothesis: `gbb` is worse on momentum exits
  (late −0.10pp/trade) and `breakout_gbb` is better (+0.05). Neither is resolved per trade.

## Addendum 6 (2026-10-11): breakout timing screen. Breakouts FADE on these pairs.

Path-blind, pre-fee, close-to-close screen (`scripts/breakout_timing.py`): forward return from
each bar's close, minus the pair's unconditional mean in the same half. It is conditioned on k,
the number of bars since the most recent 20-bar breakout, and on whether every close since has
held above the broken level. 10 pairs, 2023-01 to 2026-09. **A screen only. An earlier offline
leaderboard got the sign right 1 time in 6 against live arms.**

| bucket | 12h excess, early | 12h excess, late | pairs positive (12h) e/l |
|---|---|---|---|
| k = 0, break bar | −0.081% | −0.065% | 4/2 |
| k = 0, first break of an episode | −0.183% | −0.137% | 1/2 |
| k = 2, held | −0.221% | −0.191% | 0/0 |
| k = 4, held | −0.264% | −0.229% | 1/1 |
| k = 8, held | −0.383% | −0.234% | 0/3 |
| k = 16, held | −0.329% | −0.352% | 2/3 |
| k = 2, failed (back below level) | +0.069% | +0.163% | 9/10 |
| k = 4, failed | +0.040% | +0.117% | 8/9 |
| no break in the last 24h | +0.183% | +0.257% | 8/7 |

- **The confirmed-breakout idea is refuted by the screen.** A breakout that HOLDS gets worse
  with every bar it holds, in both halves and on nearly every pair. A breakout that FAILS
  (closes back below the level) is followed by above-average returns, on 8–10 of 10 pairs in
  both halves. At 15m these alts mean-revert through breakouts. That explains why every true
  breakout trigger loses, and why no exit or guard rescued them.
- **Cross-check on real trades** (`scripts/bgo_by_k.py`): 45% of `st_bgo_mx`'s entries are
  break bars (mean +0.002%/trade; late −0.07%), and 16% are held breakouts (late −0.23%). The
  failed-breakout entries (+0.11%, both halves) and the no-break entries (+0.06%) carry it. So
  **excluding fresh/held breakouts from `breakout_gbb`** is the screen's prediction to test.
- A "failed breakout" is itself a reversion entry condition, positive in both halves on 8–10/10
  pairs. It is a candidate reversion trigger for the reversion exit set.

## Addendum 7 (2026-10-11): the timing screen's two predictions, tested. Both FAIL.

New triggers `breakout_gbb_nofresh` and `failed_breakout` (`audit_triggers.py`, causality PASS).
**PROVISIONAL (gate 2).**

| arm | trades | per-fold sum | positive folds | early | late | against |
|---|---|---|---|---|---|---|
| `st_bgo_mx` (`breakout_gbb`, momentum exits) | 6,156 | +25.19 | 9/15 | +22.68 | +2.50 | (reference) |
| `st_bgonf` (`breakout_gbb_nofresh`, momentum exits) | 5,145 | **+13.11** | 9/15 | +15.61 | **−2.50** | `st_bgo_mx`: −3.28pp/yr, CI [−14.2, +7.4], p 0.55, UNCHANGED |
| `st_fbo` (`failed_breakout`, ARM_D exits) | 10,974 | **−18.46** | 3/15 | +1.30 | −19.76 | loses to cash. It beats `empty` (−91.35) but not `gbb` (+11.04) |

Why the screen was wrong:
- **Removed entries get SUBSTITUTED, not deleted.** In `st_bgo_mx`, the 1,130 failed-breakout
  entries averaged +0.113%/trade. Dropping the break bars freed each pair earlier, so `st_bgonf`
  took 3,652 failed-breakout entries, and those averaged **+0.009%** (late −0.098%). The bucket
  only looked good in `st_bgo_mx` because of which failed-breakout bars a pair happened to be free
  for. A per-bucket average of an arm's trades does not predict what removing a bucket does.
- The break-bar entries were doing useful work. Their `metric_overbought` exits (10.4% of
  `st_bgo_mx` exits, 3.7% of `st_bgonf`'s) are quick high-band exits.
- `failed_breakout`'s screen edge (+0.04–0.16% excess, 12h, pre-fee) is below the ~0.2%
  round-trip fee, and path-blind. Under stops and fees it loses −0.086%/trade in both halves.

**Second offline screen in this project to mis-sign against walk-forward arms** (after the
EV leaderboard's 1/6). Screens can choose what to test. Only arms decide. The true breakouts
remain unrescued. Neither confirmation nor fading them works under this framework.
