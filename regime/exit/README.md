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
