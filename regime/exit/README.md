# Exit-Process Characterisation and Repair

Study for spec-kit feature `002-exit-process-repair`. See
`specs/002-exit-process-repair/` for the spec, plan, research and task list.

**Read `specs/002-exit-process-repair/research.md` before changing anything
here.** It resolves what actually fires the stop exits, and it records one
refuted hypothesis that costs 9.24pp if acted on.

## Baseline exit configuration (pinned)

| setting | value | note |
|---|---|---|
| `stoploss_grace_hours` | 5.6 | the one live grace dimension |
| `stoploss_grace_level` | −0.27 | **INERT** — the `after_fill` branch is unreachable |
| `cexit_take_profit` | 0.024 | live |
| `cexit_max_days` | 10 | live |
| ROI | 3% | live; 22% of exits land here exactly |
| `stoploss` | −0.05 | the post-grace anchored level |
| `enable_exit_signal` | False | the model is entry-only |

Pinned as literals in `baseline.py`. **Never read from `NNNC/NNNC_MLX.json`** —
that file is untracked and was overwritten by a hyperopt mid-study, which
confounded one measurement before a pinned run caught it (FR-028b).

## Windows

W1 `20250901-20260831` (deployment regime, headline) · W2 `20240601-20250531` ·
W3 `20230101-20231231` (decides persistence).

## Running

From the freqtrade root — the system Python has no pandas:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/exit/<script>.py

## Results

_Pending._

---

## Phase 1 findings (T005–T008)

Baseline verified: 51 / 96 / 20 = 167 trades. Initial stop is within 1pp of −5%
on every trade, nearest approach to the −27% grace level being 22pp — **the
grace level is still unreachable, research R1 holds.**

### The stop category contains two different animals

| population | n | share | mean | median age | stop above entry |
|---|---|---|---|---|---|
| ended in **profit** | 31 | 37.3% | **+3.33%** | 0.50h | **100%** |
| ended at a **loss** | 52 | 62.7% | **−3.38%** | 6.12h | 21.2% |

Those are not one phenomenon with a spread. The profitable stops are the ratchet
working — every one has the stop above entry, they close in half an hour, and
they average **+3.33%**. The losing stops are slower (6.12h) and mostly have the
stop still below entry. The −72.45% headline is 31 × +3.33% netted against
52 × −3.38%.

### Counterfactual: what price did after each stop

Losing stops — **the load-bearing result**, n=52:

| extension | would reach ROI | median best | median worst |
|---|---|---|---|
| 1h | 23.1% | +2.12% | −0.39% |
| 3h | 30.8% | +3.59% | −0.86% |
| 6h | 32.7% | +4.17% | −1.34% |
| 12h | 57.7% | +6.83% | −2.14% |
| **24h** | **63.5%** | **+7.86%** | **−2.70%** |

Profitable stops, n=31: median **+11.19%** further upside after an exit taken at
**+2.71%** median.

### VERDICT: the stop is cutting NOISE, not losers

Both populations say the same thing, which is the useful part — an earlier
concern was that they would want opposite remedies, and they do not:

- **Losing stops recover.** 63.5% reach the ROI level within 24h of being
  stopped, with median worst excursion only −2.70% — comfortably inside the −5%
  static stop. They were cut on noise, not on a thesis breaking.
- **Profitable stops are cut short.** The ratchet locks +2.71% while the trade
  goes on to +11.19% median. It is protecting profit far too eagerly.

Both want the same thing: **more room and more time.**

### Measurement caveat, stated rather than buried

For the profitable population "would reach ROI" is near-tautological — 11 of 31
were already at or above the 3% ROI level when stopped, so the metric is
answering a question they had already passed. Their honest figure is the
**+11.19% of further upside forgone**, not a recovery rate. The pooled 77.1%
headline the script prints inherits that flattery; **the 63.5% on the loss
population is the number to trust**, because for those trades reaching ROI is a
genuine reversal rather than a restatement.

## Candidate reprioritisation (T009–T011, revised after Phase 1)

Phase 1's attribution reshaped the ranking. **The ratchet cut 72 of 83 stopped
trades (87%)** — 31/31 of the profitable stops and 41/52 of the losing ones fired
*above* the −5% static stop. Only **11 of 83** ever reached the configured stop.

| population | inside the 5.6h grace window | age p25/p50/p75 | final stop vs entry |
|---|---|---|---|
| profit (31) | **100%** | 0.25 / 0.50 / 0.75h | **+2.64 / +2.92 / +4.08%** |
| loss (52) | 21.2% | 5.75 / **6.12** / 8.31h | −4.39 / **−2.59** / −1.98% |

The losing stops fire against a stop at **−2.59% from entry**, roughly half the
−5% that was configured. That is the tighten-only behaviour: the ratchet raises
the stop during grace, freqtrade never lowers it, so the elevated level
**persists after grace expires** and the trade dies on an ordinary pullback.

### Revised ranking

| # | candidate | why it moved |
|---|---|---|
| 1 | **ratchet_explicit** | Unchanged at the top, but now overwhelming: it owns 87% of stops and is the only lever with a distance parameter to give. |
| 2 | **native_trailing** | Promoted. freqtrade's trailing carries an **offset** — "don't trail until up X%" — which is exactly the missing control for stops firing at +2.92% after 0.50h. |
| 3 | **ratchet_cap** | **NEW.** 100% of profitable stops fire with the stop above entry, so bounding how far it may rise targets the 31 trades that forfeited +11.19% median. |
| 4 | grace_duration | Demoted and reframed. It looked strong because losing stops cluster past the boundary, but the elevated stop persists regardless — so a **shorter** grace may protect by freezing the ratchet earlier. Direction is non-obvious; test both ways. |
| 5 | time_based_exit | Unchanged. "More time" is literally the finding, but only 6 of 167 exits take this path. |
| 6 | roi_ladder | **Demoted.** ROI is not the binding constraint — trades are cut before reaching it. Fixing the ratchet is upstream. |
| 7 | take_profit_upward | **Demoted, possibly harmful.** Raising it keeps 42 trades open longer under the ratchet that is already cutting 87% of stops, converting take-profit exits into ratchet exits. |
| 8 | enable_model_exits | **Counter-indicated.** The finding is that the strategy exits too eagerly; adding the model's exit signal adds more exit pressure. Retained only so the evidence against it is on record. |

The top three all target the same mechanism from different angles, and all three
require a `Framework/BaseStrategy.py` change — edit, measure, revert per arm.

## Deviation: Phase 3 screen skipped for the top three candidates

**What was specified**: score every candidate by counterfactual replay (T012–T017)
before spending a backtest, with T014 requiring the screen to reproduce the
recorded baseline first.

**What is being done instead**: candidates 1–3 go straight to backtest arms.

**Why**. The reprioritisation put all three top candidates on the *same
mechanism* — the ratchet — and all three need a `Framework/BaseStrategy.py`
change. A price-history replay can faithfully model rule changes like an ROI
ladder or a time exit, but the ratchet's behaviour depends on freqtrade's
`adjust_stop_loss` tighten-only semantics and on how a returned float is applied.
Reproducing that well enough to pass the T014 gate approaches reimplementing the
backtester, and a screen that *nearly* models it would mis-score every ratchet
variant — the exact failure T014 exists to prevent.

Each arm is one cheap backtest per window, and the edit-measure-revert pattern is
already proven, so the screen would cost more than it saves here.

**What is preserved**: the screen still gates candidates 4–7 (grace duration, time
exit, ROI ladder, take-profit), which it *can* model faithfully. T014 still
applies before it scores anything.

**The risk accepted**: the top three consume 3 backtests each without a cheap
pre-filter. Bounded and visible.

---

# RESULT: `trailing_stop: false` — +4.53pp across three windows

## The mechanism, corrected

Phase 1 attributed 87% of stops to "the ratchet". That attribution was right
about *where* the stops fired (above the −5% level) and **wrong about what put
them there.**

`ft_stoploss_adjust` runs freqtrade's native trailing block **in addition to**
`custom_stoploss`, not instead of it:

```python
if self.trailing_stop and dir_correct:
    if not (self.trailing_only_offset_is_reached and bound_profit < sl_offset):
        if self.trailing_stop_positive is not None and bound_profit > sl_offset:
            stop_loss_value = self.trailing_stop_positive
        trade.adjust_stop_loss(bound or current_rate, stop_loss_value)
```

With `trailing_only_offset_is_reached: false` the guard never skips, so this ran
on **every candle of every trade**. That is what was raising the stop above the
configured −5%.

This also explains why `trailing_stop_positive` measured as inert: it applies
only when `bound_profit > 0.03`, and ROI fires at exactly +3.00%, so a trade
essentially never survives above the offset for the positive-trailing branch to
engage. The block always took the fallback path.

## The explicit-ratchet sweep, which pointed the way

Replacing `return 1.0` with a deliberate distance was **rejected** — every value
lost, monotonically improving toward baseline as the distance widened:

| distance | summed | vs baseline |
|---|---|---|
| baseline (`1.0`) | 11.43% | — |
| 0.05 | 2.50% | −8.93pp |
| 0.08 | 4.47% | −6.95pp |
| 0.12 | 6.25% | −5.17pp |
| 0.20 | 7.96% | −3.47pp (W3 INERT) |

The trend said the optimum is "don't let `custom_stoploss` bind at all", which
`1.0` already achieves — so `return 1.0` is a **no-op**, not a ratchet, and the
cutting had to be coming from somewhere else. That is what sent the search to
the trailing block.

## The result

| window | trailing ON | trailing OFF | Δ wallet | Δ Calmar |
|---|---|---|---|---|
| W1 | +3.95%, PF 2.33, Calmar 26.4, win 64.7% | **+7.11%, PF 3.70, Calmar 48.7, win 75.5%** | **+3.16pp** | **+22.4** |
| W2 | +5.75%, PF 1.66, Calmar 13.1, win 69.8% | **+6.84%, PF 1.81, Calmar 13.7, win 73.7%** | +1.09pp | +0.6 |
| W3 | +1.72%, PF 3.86, Calmar 17.8, win 75.0% | **+2.01%, PF 4.57, Calmar 20.8, win 85.0%** | +0.28pp | +3.0 |
| **summed** | **11.43%** | **15.95%** | **+4.53pp** | |

Return, profit factor, Calmar and win rate all improve in **all three windows**,
including the temporally distant one — the test that rejected both `rvol 2.5` and
the hyperopt exit config.

Exit mix, pooled:

| reason | ON | OFF |
|---|---|---|
| stop | 83 @ −0.87% = −72.45% | **40 @ −4.23% = −169.39%** |
| take-profit | 42 @ +2.69% = +112.98% | **73 @ +3.57% = +260.84%** |
| ROI | 36 @ +3.00% = +107.90% | **45 @ +3.00% = +134.89%** |

Exactly the Phase 1 prediction: trades formerly cut on noise now survive to reach
their targets. Half as many stops, and the ones that remain take the full −5%
because they are genuine losers rather than noise casualties — more than paid for
by 31 extra take-profits at a higher mean.

## Caveats

- **W2 drawdown worsens** 2.299% → 2.617%. W1 and W3 are flat or better.
- The surviving stops are **larger individual losses** (−4.23% vs −0.87% mean).
  Total stop losses rise from −72.45% to −169.39%; the gain comes from the win
  side, not from losing less.
- LIVE in all three windows (trade hashes differ from baseline), so this is not
  an inert-parameter null.
- `native_trailing` as a *candidate* is resolved in the opposite direction from
  the one proposed: the fix is to turn native trailing OFF and keep
  `custom_stoploss`, not to switch to native trailing.

## ROI ladder test — every variant beats flat 3%, but ROI is not the real ceiling

Control = baseline exit params + `trailing_stop: false` (the accepted change).
Trade counts are **identical (49/95/20) across every variant**, so this isolates
where trades exit, not which are taken.

| ROI | summed | vs control | worst-window Calmar | W1 / W2 / W3 |
|---|---|---|---|---|
| flat 0.03 (control) | 15.95% | — | 13.72 | 7.11 / 6.84 / 2.01 |
| flat 0.04 | 16.86% | +0.90pp | 15.79 | 7.47 / 7.32 / 2.07 |
| flat 0.05 | 17.19% | +1.24pp | 16.30 | 7.55 / 7.53 / 2.11 |
| flat 0.06 | 17.28% | +1.33pp | 16.60 | 7.51 / 7.66 / 2.11 |
| ladder 6/4/3 | 17.13% | +1.17pp | 16.67 | 7.37 / 7.66 / 2.09 |
| ladder 8/5/3 | **17.36%** | **+1.41pp** | **16.68** | 7.44 / 7.83 / 2.09 |

Improvement is consistent in **all three windows for every variant**, drawdown is
essentially unchanged (W1 0.763–0.765%, W3 0.507–0.508%), and worst-window Calmar
rises from 13.72 to ~16.7.

**The ladder is not worth its parameters.** Flat 0.06 (+1.33pp) and ladder 8/5/3
(+1.41pp) differ by 0.08pp, far below what this test resolves. A ladder adds two
degrees of freedom for no measurable gain.

**Recommended: flat 0.05.** W1 peaks there and W3 saturates there; only W2 gains
from 0.06, and by 0.13pp. It captures ~93% of the available improvement with the
smallest departure from the validated value.

### Why the gain plateaus: take-profit is now the binding ceiling

| exit reason | ROI 3% | ROI 5% | ROI 6% |
|---|---|---|---|
| roi | 45 @ +3.00% | **5** @ +5.00% | **3** @ +6.01% |
| take_profit | 73 @ +3.57% | **111** @ +3.55% | **113** @ +3.56% |
| stop | 40 @ −4.23% | 41 @ −4.28% | 41 @ −4.28% |

Raising ROI does not let winners run — it hands control to `cexit_take_profit`,
which caps them at a **+3.55% mean regardless**. ROI exits collapse from 45 to 5.
The +1.3pp comes entirely from the 40 trades that used to exit at exactly 3.00%
now exiting at ~3.55%.

**So `take_profit_upward` — demoted after Phase 1 — is now the live lever.** With
ROI out of the way at 5%, `cexit_take_profit` at 0.024 is what stands between the
strategy and the +11.19% median upside the counterfactual identified.

This also answers the trailing question that prompted the test: at ROI 5% there
is finally a real band above the 3% offset for a positive trailing stop to work
in — but only once take-profit stops capping at 3.55%.

## Take-profit sweep — REJECTED, and it qualifies the counterfactual

Control = ROI 0.05 + trailing off (17.19%).

| take-profit | summed | vs control | worst-window Calmar | W2 maxDD |
|---|---|---|---|---|
| **0.024 (control)** | **17.19%** | — | **16.30** | 2.43% |
| 0.030 | 14.84% | −2.35pp | 17.10 | 2.06% |
| 0.035 | 13.19% | −4.00pp | 9.30 | 3.16% |
| 0.040 (range max) | 14.15% | −3.04pp | 9.14 | 3.45% |
| 0.060 (beyond range) | 13.49% | −3.70pp | 6.06 | 4.57% |
| profit checks OFF | 12.66% | −4.53pp | 6.06 | 4.57% |

At 0.060 the arm is identical to disabling profit checks — so the value took
effect (the declared 0.005–0.04 range does not clamp a JSON override) and
take-profit simply never fires. Disabling it outright is the **worst** arm, so
the 2.4% ceiling is doing real work. Drawdown rises monotonically as the ceiling
lifts (W2 2.43% → 4.57%).

**`cexit_take_profit = 0.024` is already optimal.** Candidate 7 is closed.

### What this says about the Phase 1 counterfactual

Phase 1 measured a median **+11.19%** of further upside after profitable stops,
and that number is real — but it is a **maximum favourable excursion**, the best
price reachable, which you would have to time perfectly to collect. This sweep is
the direct test of whether a simple ceiling can harvest it, and the answer is no:
every attempt to let winners run further loses more on the trades that give back
than it gains on the ones that continue.

**Methodological note worth carrying forward: an MFE-based counterfactual
overstates capturable upside.** It bounds what was theoretically available, not
what any implementable rule can take. Treat it as a screen for whether to look,
never as an estimate of the gain.

## Final three candidates — one win, one dead, one null

Control = ROI 0.05 + trailing off (17.19%).

### grace_duration → **6.5h, ACCEPTED (+1.10pp)**

| grace | sum% | worst Calmar | W1 | W2 | W3 | ΔW3 |
|---|---|---|---|---|---|---|
| 2.0h | 15.67 | 17.96 | 5.99 | 7.95 | 1.74 | −0.37 |
| 3.5h | 15.06 | 13.66 | 5.59 | 7.34 | 2.13 | +0.02 |
| 5.6h (control) | 17.19 | 16.30 | 7.55 | 7.53 | 2.11 | — |
| **6.5h** | **18.29** | **18.09** | 7.44 | 8.74 | **2.11** | **+0.00** |
| 7.0h | 18.61 | 15.82 | 8.01 | 8.64 | 1.96 | −0.15 |
| 8.0h | 18.44 | 17.42 | 7.94 | 8.49 | 2.01 | −0.10 |
| 9.0h | 16.43 | **0.15** | 7.46 | 8.91 | **0.07** | −2.04 |
| 12.0h | 18.11 | **0.44** | 8.82 | 9.09 | **0.20** | −1.91 |

**A cliff sits between 8h and 9h**: W3 collapses from +2.01% to +0.07% and its
Calmar from 17.42 to 0.15. 12.0h looks strong on W1/W2 (8.82 / 9.09, the best of
any arm) and is a trap — the classic recent-window win that dies in the distant
one.

6.5h is chosen over 7.0h despite lower raw return (18.29 vs 18.61) because it has
the better worst-window Calmar (18.09 vs 15.82), leaves W3 exactly unchanged, and
keeps margin from the cliff.

### time_based_exit → **DEAD (inert)**

`cexit_max_days` at 1 and 2 both give **byte-identical** results to the control at
10. Median trade duration is ~3.5h, so `max_hold` never fires. The real time
thresholds — `unclog_12h` and `unclog_1d` — are **hardcoded**, not parameters, so
there is no tunable time lever at all. Another inert parameter for the register.

### enable_model_exits → **NULL (+0.03pp)**

LIVE (trade hashes differ) but indistinguishable from noise: 17.19% → 17.22%,
worst-window Calmar 16.30 → 16.29. **My "counter-indicated" prediction was wrong
in direction** — enabling model exits does not harm, it simply does nothing worth
having. Left off, since it adds a subsystem for no measured gain.

---

# FINAL CONFIGURATION — 9.02% → 18.29%

| change | verdict | contribution |
|---|---|---|
| `trailing_stop: false` | ACCEPTED | +4.53pp |
| ROI 0.03 → 0.05 | ACCEPTED | +1.24pp |
| grace 5.6h → 6.5h | ACCEPTED | +1.10pp |
| revert hyperopt exit block | — | +2.41pp |
| take-profit raise | REJECTED | every value lost |
| explicit ratchet | REJECTED | every distance lost |
| `grace_level` | DEAD | inert, confirmed by test |
| `cexit_max_days` | DEAD | inert |
| model exits | NULL | +0.03pp |

Verified by re-running with **no overrides**: identical trade hashes in all three
windows (49 / 94 / 20 trades, +7.44 / +8.74 / +2.11%). No retraining at any point.

## Framework promotion + NNMT transfer check

`trailing_stop = False` promoted to `BaseStrategy` (was `True`). Validated on a
second family first.

**NNMT_MLX** has a trained model and **no params file**, so it runs purely on
class defaults — the cleanest possible transfer test.

| window | trailing ON | trailing OFF | Δ wallet | Δ Calmar |
|---|---|---|---|---|
| W1 | +0.57%, PF 1.03, Calmar 0.61 | **+9.64%, PF 1.62, Calmar 23.62** | **+9.08pp** | +23.01 |
| W2 | +7.70%, PF 1.28, Calmar 8.14 | **+9.79%, PF 1.37, Calmar 11.04** | +2.08pp | +2.91 |
| W3 | **−0.77%**, PF 0.94, Calmar −0.63 | **+0.05%**, PF 1.00, Calmar 0.05 | +0.82pp | +0.67 |
| **summed** | **7.50%** | **19.48%** | **+11.97pp** | |

Better in all three windows, and **W3 flips from negative to positive**. NNMT
gains nearly 3× what NNNC did (+11.97pp vs +4.53pp) — consistent with it trading
2–4× more, so more trades were exposed to the defect.

Two families, six windows, same direction. The framework-wide promotion is
justified on evidence rather than on the mechanism argument alone.

**Not promoted**: ROI 0.05 and grace 6.5h stay NNNC-only. Both were swept against
NNNC_MLX's specific trade profile — grace especially, which has a cliff at 9h on
this strategy's duration distribution. NNMT holds trades differently and would
need its own sweep.

## Exit signals, re-tested at the final config

| window | off | ON | Δ |
|---|---|---|---|
| W1 | +7.439% | +7.317% | −0.12pp |
| W2 | +8.742% | +8.899% | +0.16pp |
| W3 | +2.109% | +2.105% | −0.00pp |
| **summed** | **18.29%** | **18.32%** | **+0.03pp** |

**Identical to the earlier result at grace 5.6h (+0.03pp).** Trade counts
unchanged (49/94/20), deltas in both directions, LIVE but null. The exit stack
changing by +9pp did not make the model's exit opinion matter. Consistent with
the recorded NNMT finding that the model wins only ~8% of exits.

**Answer: no, enabling exit signals still makes no difference.** Left off — it
adds a subsystem for nothing.
