# Phase 0 Research: Exit-Process Characterisation and Repair

**Feature**: 002-exit-process-repair | **Date**: 2026-09-02

The spec's central unknown — what actually closes the 83 trades labelled
`trailing_stop_loss` — is **resolved here**, by reading freqtrade's own
application logic and testing the resulting hypothesis. The answer changes the
shape of the feature, so it belongs in research rather than in Phase 1.

---

## R1. What fires the stop exits — RESOLVED, and it is not what the code says

### The grace level is inert (8th recorded inert parameter)

`BaseStrategy.custom_stoploss` reads:

```python
if age_h < grace_hours:
    return grace_level if after_fill else 1.0
```

`freqtrade/strategy/interface.py:1017` declares `_ft_stop_uses_after_fill = False`,
and line 1533 returns early when `after_fill and not self._ft_stop_uses_after_fill`.
**The strategy never sets that flag**, so `custom_stoploss` is never called with
`after_fill=True`, and the `grace_level` branch is unreachable.

**Measured confirmation**: across all 167 baseline trades, `initial_stop_loss_abs`
is **−5.00% from entry** — the static `self.stoploss` — never the −0.27 grace
level. `stoploss_grace_level` cannot affect anything in this configuration.

**Consequence**: the grace mechanism has ONE live dimension (`stoploss_grace_hours`),
not two. Any search that varies `stoploss_grace_level` is spending budget on a
parameter that does nothing.

### The live branch is `return 1.0`, and it ratchets the stop upward

With the `after_fill` branch unreachable, the grace window always executes
`return 1.0`. The comment calls this "leave unchanged", but freqtrade treats a
returned float as a stoploss value:

```python
if stop_loss_value_custom and not (isnan(...) or isinf(...)):
    trade.adjust_stop_loss(bound or current_rate, stop_loss_value, ...)
```

`1.0` is truthy and finite, so it is applied. `adjust_stop_loss` only ever
tightens, so the stop ratchets upward as price rises.

**Measured on the 83 stopped trades**:

| observation | value |
|---|---|
| final stop above entry price | **42 of 83 (50.6%)**, median +2.19%, max +7.99% |
| stop at or above the exit price | 15 of 83 (18.1%) |
| exited within one bar (≤0.25h) | 21 of 83 (25.3%), p10 age = 0.00h |

The correct "no change" sentinel is `None` — that is what
`strategy_safe_wrapper(default_retval=None)` exists for.

---

## R2. The apparent bug is LOAD-BEARING — hypothesis raised and REFUTED

Reading R1, the obvious conclusion is that `return 1.0` is a defect and `None`
is the fix. **That was tested and is wrong.** One-line change, identical
everything else, all three windows:

| window | current (`1.0`) | fix (`None`) | Δ wallet | Δ Calmar |
|---|---|---|---|---|
| W1 | +3.95%, Calmar 26.36 | +2.07%, Calmar 16.66 | **−1.88pp** | −9.70 |
| W2 | +5.75%, Calmar 13.13 | +0.27%, Calmar 0.34 | **−5.48pp** | −12.79 |
| W3 | +1.72%, Calmar 17.83 | −0.16%, Calmar −1.32 | **−1.88pp** | −19.14 |
| **summed** | **11.43%** | **2.19%** | **−9.24pp** | |

Worse in every window, and W3 turns negative.

**Why**, from the exit mix under the fix: ROI exits collapse from **36 to 1**,
while take-profit rises from 42 to 97 at a *lower* mean (+1.13% vs +2.69%).

So the ratcheting stop is not corrupting outcomes — it is a **profit-lock that
carries trades up to the +3% ROI level**. Without it, trades are harvested
earlier and smaller, and almost none survive to ROI.

**Decision**: `return 1.0` MUST NOT be replaced by `None`. The profit-lock stays.

**CORRECTION (2026-09-02, after review).** The conclusion originally drawn from
this — that the ratchet is settled and needs no further work — was WRONG, and
the test does not support it. `1.0` versus `None` compares *keeping the accident*
against *deleting the mechanism entirely*. Removing a profit-lock from a strategy
whose winners cap at +3% was always going to be bad; that says nothing about
whether `1.0` is the right way to obtain the lock.

The untested and correct comparison is `1.0` versus a **deliberately implemented
ratchet with a tunable distance**. That remains a LIVE candidate, and is the
strongest one in the feature.

Correctness matters here independently of P&L. The behaviour depends on two
things freqtrade does not contract: that a returned float is applied as a
stoploss regardless of magnitude, and that `adjust_stop_loss` only ever tightens.
A version changing either silently alters the strategy, and the symptom would be
drifting live P&L with no code change to attribute it to.

**Rationale for recording a refuted hypothesis**: it is the most attractive
wrong move available in this feature. The code comment says one thing, the code
does another, and the "obvious correction" costs 9.24pp. Anyone revisiting this
file will form the same hypothesis; this record is what stops them acting on it.

---

## R3. What the exit stack actually is

Three mechanisms are live, and one of them is not what its name suggests:

| mechanism | status | effect |
|---|---|---|
| `return 1.0` ratchet, during the grace window | **LIVE, load-bearing** | Profit-locking trailing stop; produces the exits labelled `trailing_stop_loss` |
| anchored tighten to −5%, after the grace window | LIVE | Fixed stop at `open_rate × 0.95`, no trailing |
| ROI table `{"0": 0.03}` | LIVE | Exits at exactly +3.00% (36 of 167) |
| `cexit_take_profit` | LIVE | Exits above the threshold (42 of 167) |
| time-based unclog | LIVE | 6 of 167 at 12h |
| `stoploss_grace_level` | **INERT** | after_fill branch unreachable |
| `trailing_stop_positive` and the rest of `trailing_*` | **INERT** | Verified: 0.01 vs 0.03 gives byte-identical backtests |
| `exit_guard_threshold`, `exit_close_norm_threshold` | **INERT** | Behind the `enable_exit_signal` early return |
| `use_atr_adaptive_stoploss` block | **INERT** | Flag is False |

**Four of nine exit-side settings do nothing.** Phase 2 must not propose
interventions on any of them, and Phase 4 must treat an unchanged result as an
inertness defect rather than a finding.

---

## R4. The hyperopt output, screened as the first candidate

Already measured, so it does not need re-running. Same entry config, same model:

| window | baseline | hyperopt exit | Δ wallet | Δ Calmar |
|---|---|---|---|---|
| W1 | +3.95%, Calmar 26.36 | +2.91%, Calmar 32.54 | −1.04pp | +6.17 |
| W2 | +5.75%, Calmar 13.13 | +5.62%, Calmar 13.21 | −0.13pp | +0.08 |
| W3 | +1.72%, Calmar 17.83 | +0.48%, Calmar 6.20 | −1.24pp | **−11.63** |
| **summed** | **11.43%** | **9.02%** | **−2.41pp** | |

**Verdict: REJECTED.** Return falls in all three windows, and the Calmar gain
reverses sign in the temporally distant window — the same test that killed the
`rvol 2.5` candidate in feature 001.

Mechanism: win rate rises everywhere (64.7→76.9, 69.8→81.7, 75.0→85.0) because
`cexit_take_profit` at 0.005 harvests at +0.5%, converting would-be losers into
small winners. That flatters win rate and W1 Calmar while cutting total return,
and it is why the parameter pinned to the floor of its allowed range.

**This also validates the FR-028 baseline choice.** Had the hyperopt output been
adopted as the baseline, every later candidate would have been measured against
a configuration 2.41pp worse.

---

## R5. Where the remaining leverage is

R1–R4 eliminate most of the anticipated candidate list before Phase 2 begins:

| anticipated candidate | status after research |
|---|---|
| Reshape the grace **level** | **DEAD — confirmed by test.** Setting `_ft_stop_uses_after_fill = True`, the one-line enablement, gives BYTE-IDENTICAL results (51/96/20 trades, identical hashes, initial stop still −5.00%). The grace level is inert more deeply than the flag. |
| Replace `return 1.0` with `None` | **DEAD** — refuted, costs 9.24pp (R2) |
| **Implement the ratchet DELIBERATELY, with a tunable distance** | **LIVE — strongest candidate.** R2 only tested deleting the mechanism, never replacing it with a proper one. |
| **Native trailing stop via `use_custom_stoploss = False`** | **LIVE.** `trailing_*` is inert *under the current architecture*, not inert as an option — freqtrade's own trailing stop has a distance and an offset parameter. |
| Adopt the hyperopt exit config | **DEAD** — rejected (R4) |
| Reshape the grace **duration** | LIVE — the one remaining stop dimension |
| Widen the `cexit_take_profit` range | LIVE — but R4 shows lowering it is harmful; the untested direction is *upward* |
| Replace the flat ROI 3% with a ladder | LIVE — ROI is 22% of exits at exactly +3.00%, and the ratchet exists to reach it |
| Re-enable model exit signals | LIVE — no retrain needed; changes kind, not degree |
| Time-based exits | LIVE — but only 6 of 167 exits, so low ceiling |

**The ratchet is the most interesting untested axis.** It is currently an
accidental profit-lock with no tunable distance — `1.0` is a constant that
happens to work. Making the ratchet explicit and giving it a distance parameter
is a no-retrain change that targets the mechanism actually producing 50% of
exits, which nothing in the anticipated list did.

---

## R6. Environment and reuse

- Recorded trades: `freqtrade.data.btanalysis.load_backtest_data` on the three
  archives named in `regime/arms/applied_W{1,2,3}.json`. Columns include
  `initial_stop_loss_abs`, `stop_loss_abs`, `min_rate`, `max_rate` — the last
  two give per-trade excursions without touching price history.
- Price history: `regime.regimedata.load_ohlcv` (tz-aware UTC), `window_slice`,
  `WINDOWS`, `TRADED_PAIRS`.
- Backtest arms: `regime/run_arm.py`. **Needs extending** — it currently
  overrides only buy-space parameters via `--set`; exit work needs the sell
  space, the ROI table, and the stoploss block.
- Determinism verified: identical config run twice gives an identical
  `trades_hash`, so an unchanged hash proves a parameter did nothing.
- `BaseStrategy.py` is committed and clean, so mechanism experiments can be made
  in-place and reverted with `git checkout` — that is how R2 was tested.

No `NEEDS CLARIFICATION` items remain.
