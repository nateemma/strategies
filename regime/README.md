# Market-Phase Framework and Phase-Adaptive Entry Gate

Study for spec-kit feature `001-market-phase-entry-gate`. See
`specs/001-market-phase-entry-gate/` for the spec, plan, and task list.

Extends the `btc_corr/` regime work (Exp 4) into a three-level phase framework —
asset, market breadth, and BTC — then asks one question: does a phase label
predict which of two entry-aggressiveness settings wins the coming period,
better than always using the better fixed setting?

## Evaluation windows (pinned; research R3)

| Window | Range | Role |
|---|---|---|
| W1 | `20250901-20260831` | Deployment regime — **carries the headline verdict** |
| W2 | `20240601-20250531` | Near-era confirmation |
| W3 | `20230101-20231231` | Temporally distant era |

W1 and W2 are only three months apart and share a regime; they cannot establish
persistence between them. W3 is the window that can.

## Running

All scripts run from the freqtrade root — the system Python has no pandas:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/<script>.py

## Results

_Pending._ Phase A (classify) not yet run.

## Phase A observations (T007-T009, 2026-09-02)

Preliminary distributions from running the instruments on real data. These are
observations, not verdicts - Phase B does the formal characterisation.

**Alts are strongly mean-reverting at these horizons.** `asset_vr24` over full
history: Reverting 56.4% (ETH), 67.2% (SOL), 71.7% (LINK); Trending only
18.5% / 10.8% / 10.7%. This corroborates the project's accumulated evidence that
dip-buying labels work on these pairs while breakout labels do not, and it is
measured here independently of any model.

**`asset_hurst` "Persistent" is a rare state and may not survive Phase B.**
Full-history share: ETH 6.3%, SOL 3.5%, LINK 4.3%. Two of three are already
below the 5% minimum-state-share rule, so that state would be excluded from
verdict-driving comparisons on those pairs. A switch keyed on "be aggressive
when trending" would fire rarely. Flagged now because it bears directly on
actionability - see T015.

**`btc_exp4` at 15m is more Sideways-heavy than the original 1h study**
(35.4% vs 29.8%), with the other four states correspondingly thinner. Expected:
ADX is noisier on 15m bars, and the span here is 5.7 years rather than 2.2. The
taxonomy reproduces bit-identically on the original 1h data, so this is a
timeframe effect and not a transcription difference.

**Defect found and fixed in `market_correlation`.** One bar (2026-08-31 03:45)
had every one of 75 pairs unchanged, so the cross-sectional denominator was
exactly zero and the ratio went NaN - which then propagated through the whole
96-bar averaging window and produced 18 UNKNOWN labels *inside W1*. Fixed with
`min_periods` on the rolling mean and quantiles, so an isolated dead bar is
skipped rather than blanking a day. All three market instruments now emit zero
UNKNOWN inside W1/W2/W3, which is the rule working as designed: UNKNOWN inside
an evaluation window is a defect to investigate, never a state to interpret.

## Universe scoping (T010) — a design correction

Running the classifier over every feather on disk surfaced a scoping error in
the original plan.

**The host trades 11 pairs, not 77.** `config/config.json` uses a StaticPairList
of ZEC, XRP, SOL, LINK, NEAR, AAVE, SUI, AVAX, LTC, BCH, DOT. Asset-level
instruments are only meaningful there, since Phase C aligns phase labels to the
host's realized trades and those occur nowhere else. Classifying all 77 was
wasted work that also polluted the defect check.

**Most of the wider universe is not tradeable on this venue.** Of the 77 pairs
with 15m data, 61 have >50% zero-volume bars in W1 and 16 exceed 90%. OCEAN is
97.1% zero-volume in W1 with a 3,078-bar (32-day) flat run; KAITO and SKL are
similar. These carry stale forward-filled prices, and 480 consecutive flat bars
drive rolling variance to zero, making the estimators genuinely undefined — the
UNKNOWN labels were correct, not a bug. **All 16 problem pairs are outside the
traded universe; the 11 traded pairs produce zero UNKNOWN.**

Consequences, both applied:

- Market-level instruments are built from a LIQUID universe (29 pairs, ≤80%
  zero-volume bars in *all three* windows) rather than all 77. Breadth over
  mostly-dead pairs measures listing artifacts, not market state. The filter is
  applied across all three windows so the universe is FIXED across eras — a
  universe that changed definition per window would manufacture instability and
  corrupt the Principle II comparison. The 29 are a superset of the traded 11.
- Asset-level classification defaults to the traded universe.

**Second defect found and fixed**: the final bar of the dataset
(2026-08-31 17:00) is a PARTIAL candle — the download stopped mid-bar, so only
17 of 29 pairs had posted where every preceding bar had all 29. That produced a
spurious low-coverage UNKNOWN at the right-hand edge of W1. Trailing incomplete
bars are now trimmed from the panel.

Final state: **all 11 instruments emit zero UNKNOWN inside W1/W2/W3.**

## Phase A gate result (T011-T012) — PASS

End-to-end lookahead validation at three cut points (2023-06-30, 2024-12-31,
2026-03-31), comparing labels computed from truncated history against
full-history labels over the overlap:

**22,481,397 label-bars compared across 213 comparisons. Zero differences.**
All 11 instruments are causal and may proceed to Phase B.

This is the measurement Principle V requires. Reasoning line-by-line about
causality is not evidence — centred windows, `ffill` reaching backwards, and
cross-sectional operations over a panel containing future rows all pass casual
inspection and all leak. The market-level instruments were the highest risk
here because breadth, dispersion and correlation each touch the whole panel;
they pass.

### KNOWN LIMITATION — the liquid universe is chosen with full-sample information

`liquid_pairs()` selects the 29-pair market-level universe using volume across
*all three* windows, so it "knows" which pairs stayed liquid. That is
survivorship bias in the market-level instruments, and it is deliberately NOT
what the lookahead test measures — the universe is held fixed on both sides so
the test isolates per-bar causality.

Scope of the concern:

- **Asset level: unaffected.** Those instruments run on the 11 traded pairs,
  which are fixed a priori by the strategy config, not selected by this study.
- **Market level: affected.** Breadth, dispersion and correlation are computed
  over a set chosen with hindsight.

Why it was still the right call: the alternative — all 77 pairs — is worse,
because 61 of them are >50% zero-volume in W1 and breadth over stale
forward-filled prices measures listing artifacts rather than market state. A
true point-in-time liquid universe would be better still, but a universe that
changes membership per era would manufacture instability and corrupt the
Principle II comparison, which is the one test the whole feature turns on.

**How this could bite**: if a market-level instrument wins the Phase C verdict,
its edge may partly reflect knowing which coins survived. Re-check any
market-level PASS against a point-in-time universe before believing it. An
asset-level or BTC-level PASS carries no such caveat.

## T013 — baseline measurements, and two design problems they expose

`NNNC_MLX`, W1 (`20250901-20260831`), config.json (11 pairs), model reused.

| threshold | trades | pairs | summed profit | win rate | median duration | trades/day |
|---|---|---|---|---|---|---|
| 0.55 | 49 | 10 | +39.59% | 65.3% | 4.50h | 0.186 |
| 0.69 (tuned) | 49 | 10 | +39.59% | 65.3% | 4.50h | 0.186 |
| 0.80 | 29 | 8 | +29.29% | 65.5% | 2.75h | 0.119 |

**Dwell kill threshold for T015 = p25 of host duration = 1.75h = 7 bars** at 15m.
Note how far that is from the 1.4–31h figure that was circulating from the
momentum-basket dry run — a different strategy family. R6's insistence on
measuring the host rather than borrowing was right.

### Problem 1 — the R5 bracket is wrong: 0.55 is INERT

`prediction_threshold` 0.55 and 0.69 produce **byte-identical backtests** — same
49 trades, same timestamps, same profits, same hash. The parameter file is
demonstrably being read (freqtrade logs the scratch path), so this is not a
plumbing failure: every argmax-BUY bar is *already* ≥0.69 confident, so lowering
the bar selects nothing new. This reproduces recorded prior art exactly — "the
0.7–0.8 buy-prob cluster IS where the threshold bites; below it, argmax==BUY
bars are already >0.5 confident so 0.15–0.50 are identical".

**The live band is (0.69, 0.80]**, not 0.55–0.80. An "aggressive" arm below the
tuned value does not exist for this model.

### Problem 2 — trade counts cannot support block-based regret

R12 sized decision blocks at H ∈ {14d, 30d, 60d}. At the measured rate:

| H | blocks/window | trades/block (0.69) | trades/block (0.80) |
|---|---|---|---|
| 14d | 26 | 1.9 | 1.1 |
| 30d | 12 | 4.0 | 2.4 |
| 60d | 6 | 8.1 | 4.8 |

A risk-adjusted return computed from 2–8 trades is noise, and 6–26 blocks give
almost no power. **`N`, the minimum trades per qualifying block, cannot be set
to any value that is both satisfiable and meaningful.** This is the T013
derivation doing its job — it was specified precisely so this would surface
before Phase C, not after.

### What the measurement offers instead — a strictly better-posed question

**The 0.80 trade set is a strict SUBSET of the 0.69 set.** The conservative
config takes no trades of its own; it simply declines 20 of the 49.

| | trades | mean profit | win rate | summed |
|---|---|---|---|---|
| kept by 0.80 | 29 | +1.010% | 65.5% | +29.29% |
| **declined by 0.80** | **20** | **+0.515%** | **65.0%** | **+10.31%** |

The higher bar keeps the better trades — 2× mean profit at an identical win
rate. Model confidence tracks *magnitude*, not direction.

That nesting gives Phase C a well-defined marginal set and turns it into a
per-trade question: **does the phase state at entry predict which of the 20
declined trades were worth taking?** That is 49 observations rather than 6–26
noisy blocks, it needs no arbitrary block length, and it matches what M3
actually does — move the bar per bar. Proposed as a revision to R12; not yet
adopted.

## CORRECTION — the guards are the binding constraint, not the z-gate

The Problem-1 explanation above ("every argmax-BUY bar is already ≥0.69
confident") is **WRONG**. Measured by running with `entry_enable_guards` off:

| guards | threshold | trades | summed profit | win rate |
|---|---|---|---|---|
| **on** | 0.55 | 49 | +39.59% | 65.3% |
| **on** | 0.69 | 49 | +39.59% | 65.3% |
| **on** | 0.80 | 29 | +29.29% | 65.5% |
| **off** | 0.55 | 2287 | −512.42% | 57.1% |
| **off** | 0.69 | 1917 | −449.54% | 56.8% |
| **off** | 0.80 | 699 | −102.81% | 60.7% |

Two facts follow:

1. **The guard chain rejects ~97.4% of what the model would take** at the tuned
   threshold (1917 → 49). It, not the z-gate, is the binding constraint. This
   revises the recorded "guards barely filter, ~12%; the NN z-gate is the real
   filter" — that was measured on a different configuration. Here the guards are
   tuned hard (`rvol > 3.1`, `adx > 38`, `guard_metric < −0.4`).
2. **The threshold IS live and strongly discriminating** — 2287 → 1917 → 699
   with guards off. Its apparent inertness at 0.55↔0.69 was the guards removing
   those bars before the threshold could act on them.

Also confirmed: **`enable_exit_signal` is False**. The model is entry-only; exits
run on ROI 3% / `cexit_take_profit` 2.4% / stoploss grace 5.6h at −27%. So every
contrast in this study is an entry-selection contrast, which is what Phase C
wants — but it also means trade durations are exit-rule artifacts, not model
decisions.

### The operating-point problem this creates

The recorded rule is to judge interventions at a *loosened* gate, or they look
falsely inert. But here full loosening does not merely loosen — it **destroys**
the strategy (−449% summed at 0.69). So neither end is usable for Phase C:

- **guards on**: exactly one usable contrast (0.69 vs 0.80 = 49 vs 29 trades),
  too few for block-based statistics.
- **guards off**: ample trades, but a strategy that loses heavily is not the one
  the feature is meant to improve.

The study needs an intermediate operating point — enough trades for statistical
power, still profitable. That means finding which guard binds hardest and
relaxing it partially, rather than the all-or-nothing flag. NOT yet run.

## Guard sweep — the powered operating point (W1, all at 15m)

Prompted by the observation that the JSON overrides the class `buy_params` with
a far harsher config: `entry_rvol_threshold` 3.1 vs 0.5, `entry_guard_threshold`
−0.4 vs 0.3.

| config | thr | trades | wallet % | maxDD % | PF | win % |
|---|---|---|---|---|---|---|
| JSON (rvol 3.1, guard −0.4) | 0.69 | 49 | +3.45 | 0.79 | 2.17 | 65.3 |
| JSON | 0.80 | 29 | +2.00 | 0.80 | 2.12 | 65.5 |
| guard = 0.3 | 0.69 | 51 | +3.95 | 0.79 | **2.33** | 64.7 |
| guard = 0.3 | 0.80 | 31 | +2.50 | 0.80 | 2.39 | 64.5 |
| **rvol = 2.5** | 0.69 | **73** | **+4.45** | **1.10** | **1.93** | **68.5** |
| **rvol = 2.0** | 0.69 | **98** | **+2.85** | 2.26 | 1.33 | 64.3 |
| rvol = 2.0 | 0.75 | 93 | +2.94 | 2.28 | 1.36 | 65.6 |
| rvol = 2.0 | 0.80 | 56 | +1.19 | 2.32 | 1.20 | 66.1 |
| rvol = 1.0 | 0.69 | 140 | −3.65 | 5.47 | 0.78 | 57.9 |
| rvol = 0.5 | 0.69 | 160 | −4.73 | 6.55 | 0.74 | 56.9 |
| guard 0.3 + rvol 0.5 | 0.69 | 170 | −2.94 | 5.11 | 0.84 | 58.2 |
| adx 20 + guard 0.3 + rvol 0.5 | 0.69 | 403 | −5.45 | 10.67 | 0.87 | 60.0 |

**`rvol` is the dominant guard.** It alone moves trade count 49 → 160 and is
what separates profitable from unprofitable. `guard_metric` barely gates at all
here (49 → 51) — relaxing it adds three trades while slightly improving every
metric. `adx` relaxation only adds losing trades.

**The profitability cliff sits between rvol 2.0 and 1.0.** Everything at or
above 2.0 is profitable with PF > 1.3; everything at or below 1.0 loses.

### Chosen operating point for Phase C: `entry_rvol_threshold = 2.0`

It satisfies all three requirements simultaneously:

1. **Powered** — 98 trades vs 49, double the baseline.
2. **The threshold discriminates** — 0.69 / 0.75 / 0.80 → 98 / 93 / 56 trades,
   monotone, with distinct trade hashes. At the JSON operating point only 0.80
   moved at all, giving a single usable contrast.
3. **Still the strategy** — +2.85% wallet, PF 1.33. Not the broken
   guards-off regime (−449% summed) that the "loosen the gate" rule would
   otherwise have pushed us into.

### Side finding, NOT yet believed — rvol 3.1 may be over-tightened

`rvol = 2.5` beats the tuned 3.1 on every axis at once: more trades (73 vs 49),
higher return (+4.45% vs +3.45%), lower drawdown relative to return, higher win
rate (68.5% vs 65.3%). That is a production-relevant result, but it is **one
window**. Per Constitution II it means nothing until it holds in W3 — the
day-of-week failure looked exactly this convincing on two windows before
reversing. Recorded as a lead, not a finding.

### `exit_guard_threshold` is dead code here (7th inert parameter)

Entry uses `guard_metric < entry_guard_threshold`; exit mirrors it with
`guard_metric > exit_guard_threshold` at `BaseStrategy.py:805`. But
`populate_exit_trend` returns early on `if not self.enable_exit_signal.value`,
and `enable_exit_signal` is **False**, so the exit guard never executes. The
mirror is architecturally real and would need maintaining if exit signals were
ever enabled; today, relaxing the entry guard cannot break it, and FR-024 (no
arm alters exits) is satisfied trivially.

## rvol 2.5 — REJECTED. The W1 "improvement" was window-specific.

Tested the lead across all three windows at threshold 0.69:

| window | rvol 3.1 (tuned) | rvol 2.5 | Δ wallet | Δ PF | Δ win |
|---|---|---|---|---|---|
| W1 | 49 tr, +3.45%, PF 2.17, 65.3% | 73 tr, +4.45%, PF 1.93, 68.5% | **+1.00pp** | −0.24 | +3.2pp |
| W2 | 94 tr, +5.56%, PF 1.64, 70.2% | 141 tr, +5.19%, PF 1.36, 67.4% | −0.37pp | −0.28 | −2.8pp |
| W3 | 13 tr, +0.96%, PF 3.30, 84.6% | 34 tr, +0.78%, PF 1.28, 70.6% | −0.18pp | **−2.03** | **−14.0pp** |

**Profit factor falls in all three windows** (−0.24, −0.28, −2.03), and wallet
return improves only in W1. The apparent free win was a W1 artifact; the sign
flips in both other eras and the collapse is worst in the distant one.

This is the day-of-week failure mode reproduced exactly, and it is why
Constitution II exists: on W1 alone the result looked unambiguous — more trades,
more return, higher win rate, low drawdown. One distant window killed it. Do not
re-propose relaxing rvol below 3.1 on this evidence.

`rvol = 3.1` stands. The tuned value is not over-tightened; it is trading fewer
but materially better signals, exactly as intended.

### Consequence — W3 is severely trade-starved

Trade counts at the tuned config vary enormously by era: **W1 = 49, W2 = 94,
W3 = 13**. The temporally distant window — the one Principle II makes decisive —
has the fewest trades by a factor of four. Any Phase C verdict must clear that
bar in W3, so the operating point has to be chosen for W3's sparsity, not W1's.

## Phase C arm sizing at the study operating point (rvol = 2.0)

| window | thr 0.69 | thr 0.80 | **marginal set** | wallet % @0.69 | PF @0.69 |
|---|---|---|---|---|---|
| W1 | 98 | 56 | **42** | +2.85 | 1.33 |
| W2 | 180 | 136 | **44** | +9.35 | 1.59 |
| W3 | 51 | 25 | **26** | +2.09 | 1.58 |

**Phase C is viable.** All three windows are profitable with PF > 1.3, the
threshold discriminates in each, and the marginal set — the trades the
conservative config declines, which is what the phase label must sort — holds
26–44 observations per window, 112 in total.

W3 goes from 13 trades at the production guard to 51 here, which is what makes a
distant-era verdict possible at all. 26 marginal trades still only supports
detecting a large effect; a subtle one will not clear the noise floor, and that
is an honest bound on what Phase C can conclude.

### rvol = 2.0 is a STUDY operating point, not a production recommendation

Profit factor is *lower* than the tuned 3.1 in every window (1.33 vs 2.17,
1.59 vs 1.64, 1.58 vs 3.30). It is selected for statistical power, not
performance — it buys the trade volume Phase C needs while keeping the strategy
profitable and recognisably itself. **Production stays at rvol = 3.1.**

Note also that the rvol optimum is **non-monotonic across eras**: 2.5 beats 2.0
on W1 but 2.0 beats 2.5 on W2 (+9.35% vs +5.19%). The best rvol wanders by
window with no stable ordering — the same signature that killed the dynamic
lookback. This is a further reason to treat any rvol value as a fixed study
choice rather than something to tune.

## Phase B — actionability (T014–T015)

### The dwell criterion was measured by the wrong statistic, and it mattered

The spec's rationale for the dwell rule is explicit: *"if a quarter or more of
trades would span a phase change, the phase in force at entry does not describe
the conditions the trade actually experiences."* The first implementation used
**median run length** as a proxy for that. It is a bad proxy, and the direct
measurement — take the host's 49 real trades, look at each instrument's labels
between entry and exit, count how often the phase is not constant — inverts
several verdicts:

| instrument | trades spanning a phase change | median dwell | proxy said | direct says |
|---|---|---|---|---|
| asset_trend | **8.9%** | 1.50h | ELIMINATED | **SURVIVING** |
| market_breadth | 11.1% | 1.25h | ELIMINATED | **SURVIVING** |
| btc_vol | 15.6% | 2.50h | SURVIVING | SURVIVING |
| asset_vr24 | 15.6% | 3.75h | SURVIVING | SURVIVING |
| asset_hurst | 15.6% | 6.06h | SURVIVING | SURVIVING |
| asset_autocorr | 20.0% | 2.12h | SURVIVING | SURVIVING |
| asset_vr4 | 22.2% | 1.25h | ELIMINATED | **SURVIVING** |
| market_correlation | 24.4% | 1.25h | ELIMINATED | **SURVIVING** |
| asset_adx | 42.2% | 3.00h | SURVIVING | **ELIMINATED** |
| **btc_exp4** | **48.9%** | 3.00h | SURVIVING | **ELIMINATED** |
| market_dispersion | 84.4% | 0.25h | ELIMINATED | ELIMINATED |

Median run length punishes instruments that chop briefly around a boundary
(`asset_trend` crossing its EMA) and rewards ones with rare long runs. Neither is
the question being asked. Median dwell is retained as a reported diagnostic only.

### Result: 8 of 11 SURVIVING

Eliminated, all on directly measured span:

- **`btc_exp4` — 48.9%.** The prior study's own taxonomy, and the designated
  baseline instrument, is the **second-worst** on this criterion: nearly half of
  real trades straddle a regime change. At 1h with passive exposure it was a
  reasonable description; against 15m trades averaging 4.5 hours it is not
  something an entry gate can act on. That is a substantive finding about the
  prior work, not a defect in it.
- **`asset_adx` — 42.2%.** The free trend proxy is too fast at this timeframe.
- **`market_dispersion` — 84.4%.** Essentially a per-bar quantity.

Surviving: `btc_vol`, `asset_vr4`, `asset_vr24`, `asset_hurst`,
`asset_autocorr`, `asset_trend`, `market_breadth`, `market_correlation`.

Notable: the **direct trend-vs-reversion measures all survive** (VR at both
horizons, Hurst, autocorrelation) while both **ADX-based proxies fail** —
`asset_adx` outright, and `btc_exp4`, which is built on ADX. The instruments
added because they measure the property directly are the ones that turn out to
be actionable; the proxies are not.

`asset_hurst` carries one rare state (<5% share) as anticipated, so its
"Persistent" state is excluded from verdict-driving comparisons but the
instrument itself stands.
