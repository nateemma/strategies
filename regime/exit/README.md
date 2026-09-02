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
