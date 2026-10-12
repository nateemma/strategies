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

## Addendum 1: a second gate on the volreg-"on" trades (2026-10-11)

**Question.** Is there a second instrument that catches the reversion trades still losing late
once the volreg gate lets them through? The method is option 1 above: no backtests until
something passes.

**Feasibility** ([feasibility_stack.md](feasibility_stack.md)):
`feasibility.py --on-gate volreg:lt:1.0332` scored the same 4 instruments, with the same rule
choice and PASS test, on `st_reg_volreg`'s reversion trades (4,422 of 4,424 were "on").

- `breadth`, `persist` and `btctrend` **STOP**: their "off" states were profitable early.
- `btctrend` was the near miss: its "off" state lost late (−8.87) but earned early (+4.79).
- **Only `volreg` itself PASSES, at a tighter cut**: off when volreg < 1.3658 (p50 of the
  "on" trades).
  - Off: −4.10 early / −14.07 late. On: +37.78 / +4.88.
  - The cross-check, `st_gbb` restricted to the first gate's "on" trades, agrees late
    (−17.08 off vs +13.29 on), but its early "off" is +3.39 (reported only, not a veto).
- So this is a **tighter version of the same instrument, not an orthogonal second gate**. The
  first pass could not pick it because no rule may switch off more than 50% of the early trades.

**Walk-forward:** `st_reg_volreg2` = `SimpleTrigger_RegGateVolreg2`, the same arm with the
threshold changed from 1.0332 to 1.3658.

| arm | total | positive folds | early (in-sample) | late (out-of-sample) | reversion pp early / late | momentum pp early / late |
|---|---|---|---|---|---|---|
| always on `st_reg_on4096` | +15.54 | 8/15 | +35.92 | −20.38 | +24.00 / −24.70 | +11.92 / +4.33 |
| one cut `st_reg_volreg` | +38.14 | 8/15 | +44.19 | −6.05 | +33.51 / −9.20 | +10.68 / +3.14 |
| **tighter cut `st_reg_volreg2`** | **+47.38** | 9/15 | +53.68 | **−6.29** | +44.67 / **−1.29** | +9.00 / **−5.01** |
| momentum alone `st_bgo_mx` | +25.19 | 9/15 | +22.68 | **+2.50** | — | — |

The late verdicts ([decisions_stack.md](decisions_stack.md)) are both unresolved:
- against momentum alone: −8.80pp, CI [−19.1, +8.2], p 0.45. **Not adopted.**
- against the one-cut arm: −0.24pp, p 0.99.

**Why the late total did not move: substitution on the momentum side.** The tighter cut does what
feasibility predicted on reversion: its late loss falls from −9.20 to −1.29 on 945 trades.
Momentum gives all of that back.

- Late, 2,272 momentum trades are common to both arms.
- 189 momentum trades exist only in the one-cut arm. They made **+4.95**, and the freed pairs
  displaced them.
- 320 momentum trades exist only in the tighter-cut arm. They made **−4.73**. Of these, 95 are
  "+ov" retags and 225 are plain `breakout_gbb` entries.
- That is a −9.7pp swing from re-sequencing which momentum entries take a pair. It is the 023
  Addendum 7 lesson again: a bucket split prices the removed trades, not what replaces them.

**Reading.** The volatility instrument now has reversion close to break-even out of sample.
The remaining gap to momentum alone sits on the momentum side of the combined arm: late −5.01
inside the combo vs +2.50 alone. That is path dependence in the one-position-per-pair book, not
regime. The early gain (+53.68) is in-sample twice over, because both cuts were chosen on
f00–f07. PROVISIONAL.

**Next (fix-oriented).** The lever has moved from reversion's entries to how the two families
share the book:

1. **Give momentum's own entries precedence.** Run the combined arm so a momentum signal is
   never pre-empted by a pair held, or recently freed, by reversion. One way is to price
   momentum as in `st_bgo_mx` and add reversion only on pairs momentum is not using.
2. **Separate books.** Run momentum and gated reversion as two sub-portfolios with fixed slot
   counts, so neither re-sequences the other. This also makes "sum of singles" the honest
   reference.

## Addendum 2: how the two families share the book (2026-10-11)

**Question.** Addendum 1 found the late gap to momentum alone sits on the momentum side of the
combined arm, from pair re-sequencing. This addendum tries the two fixes it proposed.

- **Momentum first.** `MOMENTUM_PREEMPT` closes a reversion trade when momentum signals on its
  pair.
- **Separate books.** Momentum and gated reversion run as independent sub-portfolios.

### Momentum first (`st_pre_volreg`, `st_pre_volreg2`). Not adopted.

**Mechanism.** A fillable momentum entry signal on a pair held by a reversion trade exits that
trade at the next open (`momentum_preempt`). Momentum then enters on its own signal if the signal
is still live on the next bar.

- Same-bar handover is impossible: freqtrade evaluates a bar's entry before its exits.
- **Liquidity floor.** A thin signal bar does not preempt. In smoke f00, 27 of 66 handovers had
  been lost to a zero stake: `custom_stake_amount` sizes from the last candle's volume, so these
  were signals momentum alone could not have filled either.
- **Late copy dropped.** A one-bar-late copy of the signal was tried and removed. Any extra entry
  signal suppresses ROI on the open trade (`ignore_roi_if_entry_signal`), so it broke the null
  check (0/15 identical). It also added only 2 of 46 handovers.
- **Null check passes:** `st_pre_momonly` (reversion disabled) reproduces `st_fam_combo_mom`
  15/15.
- **Reference check:** a fresh `st_fam_combo_mom` f00 (`st_momonly_fresh`) reproduces the 024
  record. The reference is current.

| arm | total | early | late | momentum pp late | reversion pp late | preempts (preempted reversion pp) |
|---|---|---|---|---|---|---|
| `st_reg_volreg` (cut 1.0332) | +38.14 | +44.19 | −6.05 | +3.14 | −9.20 | — |
| `st_pre_volreg` | +37.04 | +42.02 | −4.98 | +4.27 | −9.26 | 1,021 (+51.03) |
| `st_reg_volreg2` (cut 1.3658) | +47.38 | +53.68 | −6.29 | −5.01 | −1.29 | — |
| `st_pre_volreg2` | +47.24 | +54.04 | −6.80 | −0.59 | −6.21 | 615 (+34.77) |

Late vs `st_bgo_mx` ([decisions_preempt.md](decisions_preempt.md)): −7.49pp (p 0.59) and
−9.31pp (p 0.42). **Not adopted.**

**Why it does not help: preempted reversion trades are winners.** A breakout signal tends to fire
while a reversion trade is working (price recovering through the band), so preempting cuts
reversion's profit-taking.

- Momentum gains in the late half: +1.1 on cut 1, +4.4 on cut 2.
- Reversion loses about the same: −0.1 on cut 1, −4.9 on cut 2.
- Even with precedence, only 85–92% of the combined arm's late momentum trades match
  `st_bgo_mx`'s. Handovers need the signal to persist one bar, and the one-bar shift re-times
  later exits.

### Separate books (`st_bgo_mx` + `st_book_rev[2]`). Not adopted, but closest so far.

- `SimpleTrigger_RevOnlyVolreg[2]` is the gated combined arm with `breakout_gbb` disabled: the
  reversion book alone.
- Momentum's book is `st_bgo_mx` itself, unchanged.
- `evaluation/books.py` adds the books per fold and pools their trades.
- It also measures capacity:
  - at the config stake (900, 10 slots), the two books exceed 10 open positions on 12–14% of
    entries, peaking at 18–19, so the plain sum needs about twice the capital;
  - **the fair, same-capital version halves the stake** (450, 20 slots): no overflow (peak
    18–19 ≤ 20).
- Two books also hold the same pair at once (2,010–3,533 entries), which needs two
  bots or sub-accounts, not one freqtrade bot.

| books (stake x0.5, same capital) | total | positive folds | early | late | late vs `st_bgo_mx` | whole span vs `st_bgo_mx` |
|---|---|---|---|---|---|---|
| momentum + gated reversion, cut 1.0332 | +31.68 | 9/15 | +32.25 | −0.57 | INCONCLUSIVE_ARM_LOSES, p 0.53 | UNCHANGED, p 0.74 |
| **momentum + gated reversion, cut 1.3658** | **+39.81** | 9/15 | +36.46 | **+3.35** | UNCHANGED, p 0.96 | UNCHANGED, p 0.30 |
| momentum alone `st_bgo_mx` (the bar) | +25.19 | 9/15 | +22.68 | +2.50 | — | — |

Full-stake sums, which need about twice the capital, are in [books.md](books.md):
+79.62 total, +6.70 late.

**The gated reversion book alone** is the clearest result of this addendum:

| arm | total | positive folds | early (in-sample for the gate) | late (out-of-sample) |
|---|---|---|---|---|
| `st_book_rev2` (gbb, volreg ≥ 1.3658) | **+54.43** | **10/15** | +50.23 | **+4.20** |
| `st_book_rev` (gbb, volreg ≥ 1.0332) | +38.17 | 10/15 | +41.82 | −3.65 |
| `st_gbb` (gbb, ungated) | +11.04 | 7/15 | +30.33 | −19.28 |
| `st_bgo_mx` (momentum alone) | +25.19 | 9/15 | +22.68 | +2.50 |

- `st_book_rev2` vs `st_gbb`, late: **IMPROVED**, +23.48pp, p 0.05. This is the first resolved
  out-of-sample improvement in 024–026. The gate's cut was chosen on f00–f07 trades only.
- `st_book_rev2` vs `st_bgo_mx`, late: UNCHANGED, +1.69pp, p 0.89.
- Caveats:
  - The gated book runs the 4,096-bar startup. In 026, the startup alone moved 4 of 15 combined
    folds slightly (H1).
  - The tighter cut was chosen in two early-only passes, so the early half is in-sample twice
    over.
  - PROVISIONAL (quality gate 2).

### Reading

The families do not share one book well. Whichever family yields a pair loses trades that pay:

- reversion's working trades, under momentum-first;
- momentum's entries, under reversion-first (the 024–026 combined arms).

Run apart, each keeps its own sequence. The gated reversion book then earns in both halves, and
the sum at equal capital is the best late figure of any combined construction (+3.35 vs +2.50).
It is still unresolved against momentum alone. The volatility gate turns reversion from a
late-half loser into a small out-of-sample winner on its own.

**Next (fix-oriented).**

1. **Size the books by risk, not 50/50.** The reversion book's median drawdown is 2.82 against
   6.10 for momentum, so equal capital under-weights it. A pre-declared risk-parity weight could
   be chosen on early folds and judged late.
2. **Get a distant era.** The volreg gate now carries a resolved out-of-sample improvement over
   ungated reversion. Only a distant era can turn the in-sample early gain, and the two-pass
   threshold, into a finding.
3. **Build a deployment path for two books:** two dry-run bots or sub-accounts, momentum
   (`st_bgo_mx`) and gated reversion, before any capital decision.

## Addendum 3: risk-weighted books (2026-10-11)

**Rule, pre-declared before scoring:** `evaluation.books --weights risk`.

- Each book's weight is proportional to 1/sd of its daily realised P&L (pp, by close date,
  zero-filled), measured on **f00–f07 only**.
- Weights are normalised to sum to 1, so total capital matches a single-book arm.
- Correlation is ignored.
- The late half (f08–f14) judges.

**Capacity** is now measured in capital, not slots. Peak committed stake is 8,259 (cut 1.3658)
and 8,584 (cut 1.0332), both within the bar's 9,000 (10 × 900). The `over_max_open` slot count
in [books_risk.md](books_risk.md) assumes full-size stakes and does not apply to weighted books.

**Weights chosen on the early folds:**
- cut 1.3658: momentum 0.412 / reversion 0.588 (early daily sd 0.776 vs 0.543);
- cut 1.0332: momentum 0.463 / reversion 0.537.

| books, same capital | total | positive folds | early | late | late vs `st_bgo_mx` | whole span |
|---|---|---|---|---|---|---|
| **risk 41/59, cut 1.3658** | **+42.39** | 9/15 | +38.89 | **+3.50** | UNCHANGED, p 0.997 | UNCHANGED, p 0.26 |
| 50/50, cut 1.3658 (addendum 2) | +39.81 | 9/15 | +36.46 | +3.35 | UNCHANGED, p 0.96 | UNCHANGED, p 0.30 |
| risk 46/54, cut 1.0332 | +32.16 | 9/15 | +32.97 | −0.80 | INCONCLUSIVE_ARM_LOSES, p 0.55 | UNCHANGED, p 0.72 |
| momentum alone `st_bgo_mx` | +25.19 | 9/15 | +22.68 | +2.50 | — | — |

**Return: no resolved change, and none is possible.** With fixed stakes, the books' P&L is the
weighted average of the two singles, late +2.50 (momentum) and +4.20 (gated reversion). So no
weighting can exceed the better single book, and that book is itself unresolved against momentum
alone (p 0.89, addendum 2).

**Risk is where combining pays** (cut 1.3658; descriptive, no significance test):

| late half (out-of-sample) | sum pp | daily sd | annualised Sharpe | max drawdown |
|---|---|---|---|---|
| momentum alone | +2.50 | 0.663 | +0.12 | 18.15 |
| gated reversion alone | +4.20 | 0.411 | +0.31 | 9.19 |
| 50/50 books | +3.35 | 0.413 | +0.25 | 9.54 |
| **risk 41/59 books** | +3.50 | **0.388** | +0.27 | **8.60** |

- The two books' daily P&L correlation is +0.32 early and **+0.14 late**.
- The risk-weighted books have the lowest late volatility of any construction, and under half of
  momentum alone's late drawdown, for slightly more return.
- Gated reversion alone has the highest late Sharpe, but its gate threshold is fitted on the
  early half (two passes), so it carries more selection risk than the books.
- PROVISIONAL (quality gate 2).

**Reading.** Weighting by risk rather than 50/50 changes little: +0.15pp late, slightly lower
risk. The books construction is the useful step. Two nearly uncorrelated books (momentum, and
volatility-gated reversion) roughly halve momentum alone's drawdown at the same capital, with
similar or better return.

**Next (fix-oriented):**
1. **Make risk a first-class verdict.** The harness judges P&L only. A paired bootstrap on daily
   Sharpe or drawdown would say whether the halved drawdown is resolved.
2. **Get a distant era**, still the only way to confirm the gate's fitted early half.
3. **Run two dry-run bots** (momentum `st_bgo_mx`, gated reversion `st_book_rev2`) at 41/59
   stake, to check fills and same-pair holding live.

## Addendum 4: a risk verdict in the harness (2026-10-11)

**Built:** `evaluation/risk.py` — the paired risk verdict, reported BESIDE the P&L verdict
(never instead of it).

**Statistics** (arm minus control):
- annualised daily Sharpe — higher is better;
- maximum drawdown — lower is better.

Both are computed on each side's realised daily pp, by close date, zero-filled over the shared
span.

**Bootstrap** (pinned):
- paired stationary block bootstrap over days, so drawdown's day ordering survives;
- mean block 10 days, 2,000 resamples, seed 20261011;
- 95% interval → IMPROVED / WORSENED / UNCHANGED;
- UNRESOLVABLE when trades carry no close dates.

**Where it shows up:**
- `halves.split` adds a `risk` key to each half, so every caller gets it: `stage.py` prints it
  under the late verdict, and `books.py` prints the late and whole-span lines;
- `halves` CLI.

**Exposure:** `books.py` now also reports mean committed capital. Max drawdown scales with
position size, and Sharpe does not. The unit tests pin this: a half-size copy of a series is
drawdown IMPROVED, Sharpe UNCHANGED.

**Results on the 026 arms.** Full detail is in [books_risk.md](books_risk.md) and
[risk_singles.md](risk_singles.md).

| late half vs `st_bgo_mx` | Sharpe | max drawdown | mean committed capital (bar 2,123) |
|---|---|---|---|
| books, risk 41/59 (same capital) | 0.27 vs 0.12, UNCHANGED, p 0.68 | **8.60 vs 18.15, IMPROVED, p 0.040** | 1,189 |
| books, 50/50 (same capital) | 0.25 vs 0.12, UNCHANGED, p 0.67 | **9.54 vs 18.15, IMPROVED, p 0.025** | 1,329 |
| books, **exposure-matched** (x1.79) | 0.27 vs 0.12, UNCHANGED, p 0.68 | 15.36 vs 18.15, UNCHANGED, p 0.71 | 2,124 |
| gated reversion alone `st_book_rev2` | 0.31 vs 0.12, UNCHANGED, p 0.78 | 9.19 vs 18.15, UNCHANGED, p 0.28 | 536 |
| combined, gated `st_reg_volreg2` | −0.26 vs 0.12, UNCHANGED, p 0.42 | 23.05 vs 18.15, UNCHANGED, p 0.37 | — |

- Over the whole span (early half in-sample), the risk-weighted books are IMPROVED on both:
  Sharpe 1.30 vs 0.49 (p 0.012) and drawdown p 0.008.
- `st_book_rev2` early: Sharpe IMPROVED, 2.45 vs 0.78 (p 0.018).

**Reading — the drawdown win is mostly lower exposure.**
- At the same capital, the books' smaller late drawdown is resolved (p 0.04), but the books keep
  only 56% of momentum alone's average capital in the market.
- Matched to the same average exposure, the drawdown gain is unresolved. That would also need a
  14,752 peak commitment, more than the 9,000 capacity.
- Sharpe, the exposure-neutral measure, is higher for the books in the late half (0.27 vs 0.12)
  but unresolved.

So, out of sample: the books lose less in their worst stretch **at the same capital**, which is
a real property for a same-capital deployment. They are not shown to be more efficient per unit
of risk. PROVISIONAL.

**Next (fix-oriented):**
1. **Lift the books' exposure within capacity.** A larger stake for the reversion book, bounded
   by peak committed ≤ 9,000, closes part of the gap without leverage; choose the size on
   f00–f07.
2. **Get a distant era** — the whole-span Sharpe win rests on the in-sample early half.
