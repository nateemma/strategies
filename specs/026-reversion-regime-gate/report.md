# Report: 026 regime gate on the reversion family

**Date**: 2026-10-11 · **Spec**: [spec.md](spec.md) · **Feasibility**: [feasibility.md](feasibility.md) ·
**Decisions**: [decisions.md](decisions.md) · **Evidence**: [baseline.md](baseline.md)

> **Every figure is PROVISIONAL (quality gate 2).** There is no distant era. The gate's rule was
> chosen on f00–f07 reversion trades, so **the early half is in-sample for the gate and only the
> late half (f08–f14) is out-of-sample.**

## SUMMARY

1. **Feasibility (US1, no backtests): one of four instruments passed, `volreg`.** That is the
   cross-pair volatility regime: "off" when cross-pair median 1-day volatility is below 1.033× its
   own 30-day median, i.e. in calm markets.
   - Its "off" state lost reversion P&L in both halves (−14.65 early, −11.79 late).
   - The `st_gbb` cross-check agreed.
   - It is not a calendar switch: the longest late "off" run is 3% of late bars.
   - `breadth`, `persist` (variance ratio) and `btctrend` stopped, because their "off" states were
     profitable early.
2. **Walk-forward (US2): the gated combined arm is the best total of any arm in 023–026, but it is
   not adopted.**

   | arm | total | positive folds | early (in-sample for the gate) | late (out-of-sample) |
   |---|---|---|---|---|
   | **gated `st_reg_volreg`** | **+38.14** | 8/15 | +44.19 | −6.05 |
   | always-on, same warm-up `st_reg_on4096` | +15.54 | 8/15 | +35.92 | −20.38 |
   | momentum alone `st_bgo_mx` (the bar) | +25.19 | 9/15 | +22.68 | **+2.50** |
   | reversion alone `st_gbb` | +11.04 | 7/15 | +30.33 | −19.28 |

   - **Against the bar, momentum alone, the late verdict is INCONCLUSIVE_ARM_LOSES:** −8.56pp,
     CI [−20.6, +10.6], p 0.53. Not adopted (FR-008).
   - Whole span against momentum alone: +3.52pp/yr, CI [−6.9, +13.8], p 0.50, UNCHANGED.
   - Against the always-on arm: +6.13pp/yr, CI [−0.5, +12.7], **p 0.069**. The late half is +14.32
     (p 0.13). It is close, but not resolved.
3. **The gate does what feasibility predicted, out of sample.** The reversion side's LATE
   contribution moved from **−24.70 to −9.20pp**, a 63% cut in the out-of-sample half. The momentum
   side is essentially unchanged (late +4.33 → +3.14). The gate is live: ~80k reversion signal
   rows dropped, 4.5k overlap rows retagged, and "off" on ~45% of bars.
4. **Why it still is not enough:** reversion loses in the late half even in the gate's "on" state
   (feasibility: on-state late −12.91). The volatility regime explains part of gbb's late collapse,
   not all of it.
5. **The early gain is mostly in-sample.** The reversion side's early contribution rose from
   +24.00 to +33.51 because the rule was fitted to remove early losers. That part must not be
   counted as evidence.

## SC answers

| SC | status | evidence |
|---|---|---|
| SC-001 thresholds early-fixed and instruments causal | **PASS**: truncation at 16 cuts plus a planted control failing; live-data check identical to 2024-12-21 | `utils/tests/test_regime_state.py`; baseline.md T006 |
| SC-002 feasibility table, per state and half, with PASS/STOP | **PASS**: 1 PASS / 3 STOP | feasibility.md |
| SC-003 gate disabled reproduces `st_fam_combo`; gated arm live | **PASS**: `st_reg_off` 15/15 identical; ~80k refusals | baseline.md |
| SC-004 gated arm vs momentum alone, by half, with intervals | **Answered: it does NOT beat momentum alone in the late half** (−8.56pp, p 0.53). It is higher overall (+12.95pp span, p 0.50, unresolved), but that is driven by the in-sample early half. | above; decisions.md |
| SC-005 023–025 fingerprints and reproduction unchanged | **PASS**: fresh f00 re-runs of `st_fam_combo` and `st_gbb` are identical, and 3 fingerprints are unchanged | baseline.md T020 |

**Warm-up control (H1).** The longer 4096-bar startup alone changed 4 of 15 folds slightly
(+15.54 vs +15.39). As pre-declared, `st_reg_on4096` replaced `st_fam_combo` as the always-on
comparison.

## What was built

- `utils/RegimeState.py`: 4 causal, market-wide instruments, strategy-agnostic.
- `research/regime_gate/feasibility.py`: the no-backtest feasibility gate, with an early-only rule
  choice and a calendar-switch check.
- `SimpleTrigger`:
  - `REVERSION_REGIME_GATE`, which drops reversion-only entries in "off" and gives overlap bars to
    momentum;
  - `informative_pairs` for BTC;
  - validation requiring a ≥ 4096-bar startup;
  - sink counters.
- `evaluation/stage.py`: `--bar` (judge against a named arm), and the regime-stage degeneracy rule
  (H2).
- `evaluation/fingerprint.py`: the gate is hashed when set.

## Recommended next (fix-oriented)

The volatility gate removes about 63% of reversion's out-of-sample late loss. What remains is
reversion losing even in high-volatility markets in the late period. Options:

1. **Stack a second, orthogonal gate on the "on" state.** Re-run the feasibility check on the
   *volreg-on* reversion trades only. An instrument that separates the remaining late losers would
   compose with this gate. The one-time early-choose / late-judge discipline applies again.
2. **Treat the gated combined arm as the new candidate, and get an out-of-sample read on its early
   half.** The early gain is in-sample. A distant era (constitution II) is the only way to confirm
   the gate. That needs older data than this binance.us set provides (2022+).
3. **Keep momentum alone as the deployment reference** until a combined arm beats it in the late
   half. Here it falls short by −8.56pp, unresolved.
