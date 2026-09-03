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

## LIMITATION — outcomes are truncated by the exit policy

`enable_exit_signal` is **False**, so only the entry half of the model runs.
Outcomes are decided entirely by ROI 3% / `cexit_take_profit` 2.4% / stoploss
grace. Measured on the 49 baseline trades:

| profit band | trades | share |
|---|---|---|
| < −5% | 2 | 4.1% |
| −5…−1% | 12 | 24.5% |
| −1…0% | 3 | 6.1% |
| 0…2.4% | 5 | 10.2% |
| **2.4…3.1% (cap band)** | **22** | **44.9%** |
| > 3.1% | 5 | 10.2% |

Exit reasons: `trailing_stop_loss` 21, `roi` 13, `take_profit` 10, `unclog_12h` 5.
**84% of winners (27/32) sit at or above 2.4%**, pinned to the ceiling, while
losses are uncapped and run to −9.02%.

**Why Phase C is still valid.** The exit policy is identical across every arm, so
it is a fixed transformation applied equally to both configurations. The
marginal-set contrast — "should this entry have been taken?" — is well defined
under a constant exit rule, and that is exactly the question. FR-024 requires
exits unchanged regardless, so disabled exit signals make that trivially true.

**What is compressed.** Upside is capped near 3% while downside runs to −9%, so
the measurable effect is asymmetric. Phase predicting *direction* is fully
visible; phase predicting *magnitude of gains* is largely invisible.

**Why the asymmetry favours the hypothesis under test.** Phase B's premise is
that phase discriminates *risk* better than *return* (a 4× drawdown spread
against a 2.6pp win-rate spread). With gains capped and losses uncapped, the
outcome variable is dominated by loss avoidance — precisely that axis. A phase
signal that steers away from the −9% trades will show clearly.

**Consequence for the T013 result.** "Kept +1.010% vs declined +0.515% at
identical win rate" was measured under this cap, and 9 of 13 marginal-set
winners are capped. That 2× gap is a **floor on the true effect, not an
estimate**. The direction stands; the magnitude is understated.

Also note the dwell/span timescale is an exit-rule artifact: the 4.5h median
duration is how fast ROI/stop fires, not a model-chosen holding period. The span
criterion is still correct — a phase must outlast the actual position, whatever
closes it — but it is calibrated to the exit machinery, not to the model.

## T016–T017 — the exp4 risk discrimination is CONFIRMED, but smaller than reported

| state | % time | n bars | maxDD raw | maxDD time-equalised | mean bp |
|---|---|---|---|---|---|
| Strong Down | 19.3 | 9530 | −117.5% | −102.2% | +0.46 |
| Weak Down | 16.4 | 8093 | −108.1% | −111.8% | −0.64 |
| **Sideways** | 30.4 | 14999 | **−187.8%** | **−118.1%** | **−1.00** |
| **Weak Up** | 18.4 | 9093 | **−44.6%** | **−43.5%** | **+2.14** |
| Strong Up | 15.6 | 7683 | −39.2% | −56.1% | +1.13 |

| | worst / best |
|---|---|
| raw | **4.79×** |
| time-equalised (2000 bootstrap draws, n = 7683 each) | **2.71×** |

**VERDICT: CONFIRMED.** Roughly **43% of the apparent spread was a time-in-state
artifact** — Sideways holds 1.95× the bars of Strong Up and had more opportunity
to draw down — but a 2.7× discrimination survives equalising exposure. The
ordering is unchanged: Sideways worst, Weak Up best.

This is the first quantitative check of the prior study's headline risk claim,
and it lands in between the two obvious answers: not the clean 4× that motivated
this feature, but not an artifact either.

**Reproduction note.** The absolute drawdowns differ from the original report
(Sideways −187.8% here vs −118.3% there) because the 1h dataset has extended by
roughly fourteen months since that run, and drawdown is path- and
length-dependent. The *structure* reproduces closely, and the mean-return column
reproduces almost exactly (Weak Up +2.14bp vs +2.21bp; Sideways −1.00bp vs
−1.52bp), which is the part that does not depend on sample length.

**What it cannot do.** `btc_exp4` was eliminated in Phase B — 48.9% of real
trades span one of its regime changes — so this result cannot feed the gate. It
settles a question about the prior work rather than advancing this one. The risk
hypothesis itself survives and now has to be carried by an instrument that is
actually actionable, from the eight that passed.

**Bug caught here**: the first implementation computed the spread as
`max/min` on a column of negative drawdowns, which inverts the ratio and turned
4.79× into 0.21×, printing REFUTED. Worst = column minimum when the values are
negative.

---

# PHASE C VERDICT: FAIL — and why

**No phase label produces an entry rule that beats taking every marginal trade.**
32 fit/evaluate comparisons across 8 surviving instruments and 4 temporally
distant window pairs. One row beat the bar (`market_breadth`, fit W3 → eval W1,
+0.81%) — 1 in 32, against roughly 1.6 expected by chance at a 5% level.

Per FR-017, Phase D does not proceed. Per the constitution's experiment gate,
this is a completed finding, not an abandoned build.

## The structural reason

The marginal set — the trades the conservative configuration declines — is
**strongly profitable in every window**: +22.79% (W1), +25.62% (W2), +26.96%
(W3) summed. So `always-aggressive` is the better fixed choice everywhere, and
any rule that declines part of the set is removing profit. Every phase rule that
declined anything scored negative edge, from −0.50% to −9.01%.

There is nothing to filter, because **the marginal trades are profitable in
essentially every phase state.**

## What the diagnostic DID find — a real signal that is not a gate

Mean marginal return by state, all three windows:

| instrument | best state | W1 | W2 | W3 |
|---|---|---|---|---|
| asset_vr4 | **Reverting** | +0.97 | +0.70 | +1.48 |
| | Trending | −0.73 | −0.40 | +0.45 |
| asset_vr24 | **Reverting** | +0.86 | +0.77 | +1.22 |
| asset_hurst | **Anti-persistent** | +0.53 | +1.04 | +2.05 |
| | Random | +0.44 | +0.16 | +0.45 |
| asset_autocorr | **Neg Autocorr** | +0.89 | +0.78 | +1.34 |

**The reversion-flavoured state is the best state on all four direct
trend-vs-reversion instruments, in all three windows.** That is a consistent,
era-stable ordering — not noise. It also corroborates the Phase A measurement
that these alts are strongly mean-reverting, and it is exactly what a dip-buying
model should show.

But it is a **ranking, not a sign change**. The non-reverting states are mostly
profitable too, so the ordering supports *sizing* — press harder in reverting
phases — and not *gating*, which is what this feature set out to test. A gate
needs a state whose trades are reliably bad, and there isn't one.

## Two instruments carry no information here

`asset_trend` and `market_breadth` do not partition the marginal set at all:
96%+ of marginal trades fall in a single state (Below EMA: 46/56/24; Narrow:
44/55/26). Their apparent results are properties of one bucket. `market_breadth`'s
lone win was fit on a window where a competing state held **n = 2**.

## "No effect detected", not "no effect exists" (FR-018d)

Three bounds on this verdict:

1. **W3 holds 26–28 marginal trades.** Only a large effect clears that.
2. **Outcomes are truncated near +3%** by ROI/take-profit while losses run to
   −9%. The signal the diagnostic found is precisely a *magnitude* ordering, and
   magnitude is the axis the exit policy compresses. A phase effect on winner
   size is largely invisible to this measurement.
3. **Gating was tested; sizing was not.** The design tested one mechanism —
   decline or accept a marginal entry. The ordering above is real and stable and
   would be expressed through position size, which this feature explicitly
   scoped out.

## What would be worth testing next, if anything

The honest read is that the reverting-state ordering is the only survivor, and
the two things that would let it be evaluated properly are both currently out of
scope: **un-truncated outcomes** (so magnitude is measurable) and a **sizing
mechanism** rather than a gate. Neither is a small change, and neither should be
started on this evidence alone — a stable ranking across three windows is
suggestive, not a validated edge.

---

# BREAKOUT FEASIBILITY GATE: FAIL (well-powered)

Ran before adding a breakout `TrainingSignal`, per Constitution VII. No training,
no model — a price-based test of the *phenomenon*, because a label form can be
redesigned but an absent phenomenon cannot.

## The mirror hypothesis is not just unsupported, it is inverted

Phase C found reverting states best for the DIP model. The mirror claim was that
breakout entries should do best in TRENDING states. Mean forward return at
H=48, conditioned on `asset_vr24` at the signal bar:

| state | breakout W1 | W2 | W3 | | dip W1 | W2 | W3 |
|---|---|---|---|---|---|---|---|
| Reverting | −8.3 | **+15.0** | **+17.7** | | +10.0 | +27.0 | +35.6 |
| Neutral | +0.3 | −19.3 | −1.4 | | +17.0 | +5.2 | +41.5 |
| **Trending** | −3.7 | **−1.7** | **−8.8** | | −9.2 | +57.1 | +32.3 |

`Trending − Reverting` for breakouts: **+4.6 / −16.7 / −26.5 bp**. Negative in
two of three windows including the distant era. **Trending is the *worst* state
for breakouts in W2 and W3** — the opposite of the hypothesis.

`asset_hurst` agrees: the Persistent (trending) state gives breakout EV
+37.2 / −6.1 / **−18.7** — sign-flipping, worst in the distant era.

## Dips beat breakouts in 26 of 27 configurations

Robustness across lookback L and horizon H, pooled forward EV in bp:

| L | H | breakout W1/W2/W3 | dip W1/W2/W3 |
|---|---|---|---|
| 24 | 24 | −5.0 / −0.0 / +1.4 | +9.8 / +21.5 / +30.0 |
| 24 | 48 | −9.4 / +4.6 / +12.1 | +10.2 / +26.5 / +32.7 |
| 24 | 96 | −8.7 / +13.5 / +33.6 | +6.5 / +42.2 / +55.1 |
| 48 | 24 | −0.1 / +0.7 / +3.3 | +12.9 / +25.8 / +35.0 |
| 48 | 48 | −6.6 / +8.0 / +13.2 | +9.1 / +26.3 / +36.4 |
| 48 | 96 | −5.0 / +19.0 / +35.9 | +5.0 / +46.8 / +62.7 |
| 96 | 24 | +7.0 / +5.4 / +7.0 | +13.0 / +23.0 / +41.6 |
| 96 | 48 | +5.7 / +8.8 / +15.3 | +11.3 / +24.7 / +45.5 |
| 96 | 96 | **+15.9** / +20.8 / +30.4 | +7.9 / +46.6 / +69.3 |

The single exception is W1 at L=96/H=96 (breakout +15.9 vs dip +7.9) — and in
that same configuration dip wins W2 by 26pp and W3 by 39pp.

Breakout EV does improve at longer lookback and horizon, turning positive
everywhere at L=96. So momentum is not *absent* on these assets — it is simply
dominated by reversion at every horizon tested.

## Why this is a strong FAIL rather than a weak one

**~17,000–22,000 signals per window**, against Phase C's 26 marginal trades.
This is not a power-limited null. "No effect detected" is close to "no effect
exists" here, which is the opposite of the Phase C caveat.

It also corroborates the 2026-07 verdict ("breakout has no tradeable long edge at
ANY timeframe, with ANY exit") and supplies the mechanism that verdict lacked:
it is not that breakout *labels* are unlearnable — `labels_breakout_gbb` reached
MCC 0.73–0.77 — it is that **momentum does not pay on these assets at these
horizons, in any phase**. A better label cannot fix that.

## On the basket-strategy result that motivated this

Breakout strategies being greatly helped by market phase in the basket work is
consistent with all of this. Those are **cross-sectional** momentum books over a
**wide universe** at **slow cadence**, where the edge is relative ranking. This
is **single-pair** entry timing at **15m** on **11 alts measured as strongly
mean-reverting** (`asset_vr24` Reverting 56–72% of bars). The phase→momentum
relationship does not transfer across that gap, and this gate is the cheap way
that was established — an afternoon instead of a training cycle.

## Entry gate RE-MEASURED on the repaired exit stack (2026-09-02)

The 4x4 grid below was scored on P&L that included the native-trailing defect
(see `regime/exit/README.md`). After that was fixed, both entry dimensions were
re-swept at the new exit config (trailing off, ROI 0.05, grace 6.5h). The two
findings did NOT fare the same way.

### guard_metric 0.3 — SURVIVES, unchanged

Re-measured at rvol 1.5 (the most trade-rich cell, so the best powered):

| guard | W1 | W2 | W3 | summed | worst Calmar |
|---|---|---|---|---|---|
| −0.4 | 9.24% | 12.82% | 3.02% | 25.09% | 9.11 |
| 0.0 | 9.31% | 13.28% | 3.68% | 26.27% | 11.50 |
| **0.3** | **9.57%** | **13.28%** | **4.13%** | **26.98%** | **12.96** |
| 0.6 | 9.57% | 13.28% | 4.13% | 26.98% | 12.96 |

Same shape as before: monotone in guard, 0.3 ≥ 0.0 ≥ −0.4 in every window on
both return and Calmar, and 0.6 is byte-identical to 0.3 (verified by trade
hash) — still a saturating plateau, not a peak. A plateau does not move when
something downstream changes, which is why this one held.

### entry_rvol_threshold 3.1 — OVERTURNED

| rvol | W1 | W2 | W3 | summed | worst Calmar | trades |
|---|---|---|---|---|---|---|
| 1.5 | 9.57% | **13.28%** | **4.13%** | **26.98%** | 12.96 | 397 |
| **2.0** | 10.40% | 11.19% | 3.73% | 25.32% | 10.72 | 331 |
| 2.5 | **11.91%** | 7.37% | 2.02% | 21.29% | 5.75 | 254 |
| 3.1 | 7.44% | 8.74% | 2.11% | 18.29% | **18.09** | 163 |
| 4.0 | 2.65% | 5.75% | 0.83% | 9.22% | 16.89 | 78 |

**3.1 was an artifact of the broken exit.** It was selected on worst-window
Calmar, and it won that criterion only because loose rvol collapsed in W1 —
rvol 1.5 scored **+0.04%** on W1 under the old stack. That collapse *was* the
trailing stop: the marginal trades a looser gate admits were exactly the ones
being cut on noise. The same cell now scores **+9.57%**.

Return is now monotone decreasing in rvol with no interior optimum, and 3.1 is
the worst live setting tested except 4.0. It still wins worst-window Calmar, but
only because it barely trades — 163 trades against 331.

**Minimax regret** (pp below each window's own best cell) is the right rule here,
because the per-window return optimum is still not identifiable (W1 wants 2.5,
W2 and W3 want 1.5 — the same disagreement as before):

| rvol | W1 | W2 | W3 | **max regret** |
|---|---|---|---|---|
| 1.5 | 2.34 | 0.00 | 0.00 | 2.34 |
| **2.0** | 1.51 | 2.09 | 0.39 | **2.09** |
| 2.5 | 0.00 | 5.91 | 2.11 | 5.91 |
| 3.1 | 4.47 | 4.54 | 2.02 | 4.54 |
| 4.0 | 9.26 | 7.53 | 3.30 | 9.26 |

What changed is not the disagreement but its cost: choosing low rvol used to be
catastrophic in W1 (4.96pp of regret), and now costs 2.34pp.

**rvol 2.0 recommended**, on three independent grounds:
- lowest max regret (2.09pp vs 4.54pp for 3.1);
- it dominates 1.5 on **every** metric in W1, the deployment regime that carries
  the headline verdict — 10.40% vs 9.57%, drawdown 1.50% vs 2.25%, Calmar 36.45
  vs 22.30, Sortino 1.82 vs 0.97;
- it was *also* the best summed-return cell under the OLD exit stack (16.50%),
  so it is the one value that is good under both — 3.1 is good under neither.

Drawdown stays small throughout: worst cell is rvol 1.5 on W2 at 5.18%, against
2.54% at 3.1. This buys ~7pp of return for ~2.5pp of drawdown.

**META-LESSON.** Two studies were each measured against the other's broken half.
The entry grid scored entries on P&L that a defective exit was destroying, and
concluded the entry gate should be tight. When a study concludes "take fewer
trades", check whether the trades are bad or whether something downstream is
mishandling them.

### Full 4x4 grid on the repaired stack — guard column confirmed

Completing the grid (27 further arms). Rows = rvol, cols = `guard_metric`.

**Summed return %**

| rvol | −0.4 | 0.0 | **0.3** | 0.6 |
|---|---|---|---|---|
| 1.5 | 25.09 | 26.27 | **26.98** | 26.98 |
| 2.0 | 22.87 | 23.67 | **25.32** | 25.32 |
| 2.5 | 19.45 | 19.89 | **21.29** | 21.29 |
| 3.1 | 16.50 | 16.89 | **18.29** | 18.29 |

**Worst-window Calmar**

| rvol | −0.4 | 0.0 | **0.3** | 0.6 |
|---|---|---|---|---|
| 1.5 | 9.11 | 11.50 | **12.96** | 12.96 |
| 2.0 | 7.90 | 8.55 | **10.72** | 10.72 |
| 2.5 | 3.18 | 3.63 | **5.75** | 5.75 |
| 3.1 | 14.83 | 14.12 | **18.09** | 18.09 |

`guard_metric = 0.3` is dominant in **all 16 cells on both metrics**, and 0.6 is
byte-identical to 0.3 at every rvol (verified by trade hash). Identical structure
to the pre-fix grid. The plateau did not move, which is the expected behaviour of
a saturating threshold and the reason this finding survived the exit repair while
the rvol finding did not.

### rvol curve extended below 1.5 — a real interior optimum

| rvol | 0.5 | 1.0 | **1.5** | 2.0 | 2.5 | 3.1 | 4.0 |
|---|---|---|---|---|---|---|---|
| trades | 523 | 466 | 397 | 331 | 254 | 163 | 78 |
| summed % | 22.26 | 22.24 | **26.98** | 25.32 | 21.29 | 18.29 | 9.22 |
| max regret | 4.05 | 4.07 | 2.34 | **2.09** | 5.91 | 4.54 | 9.26 |

Return peaks at 1.5 and falls off **both** sides, so this is a genuine optimum
rather than a monotone preference for more trades. **rvol 2.0 APPLIED**: lowest
max regret of all seven points, and it dominates 1.5 on every metric in W1.
Verified by FR-030 — the applied config reproduces the measured arm with
identical trade hashes in all three windows (94/179/58 trades, 25.32% summed).

### rvol is FAMILY-SPECIFIC — not promoted to the base class

Measured on NNMT_MLX at its own defaults before touching the shared value:

| NNMT | rvol 0.5 | rvol 2.0 | Δ |
|---|---|---|---|
| W1 | 169 tr, 9.64% | 104 tr, 6.08% | −3.56pp |
| W2 | 340 tr, 9.79% | 208 tr, 4.85% | −4.94pp |
| W3 | 125 tr, 0.05% | 63 tr, 0.92% | +0.87pp |
| **sum** | **19.48%** | **11.85%** | **−7.62pp** |

**`BaseStrategy.buy_params` keeps `entry_rvol_threshold: 0.5`.** The two families
agree on the *direction* — loose rvol is better — and NNMT is already there. NNNC
at 3.1 was the outlier. Moving the shared default to 2.0 would push every other
family the wrong way on the only cross-family evidence available.

Contrast with `trailing_stop`, which WAS promoted: that one improved both
families in all six windows. A change earns the base class by being measured on
more than one family, not by being large on one.
