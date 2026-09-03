# NNMT_MLX entry-gate sweep (2026-09-02)

Run after the exit repair, on NNMT's OWN defaults — no NNNC tuning imposed.
NNMT has a trained model and no params file, so it runs purely on class
defaults. Windows W1 `20250901-20260831`, W2 `20240601-20250531`, W3 `20230101-20231231`.

## rvol — optimum is 0.25, far looser than NNNC's 2.0

| rvol | trades | W1 | W2 | W3 | summed | worst Calmar |
|---|---|---|---|---|---|---|
| 0.0 | 649 | 8.96 | 6.21 | 0.42 | 15.60 | 0.43 |
| **0.25** | 654 | **10.71** | 8.45 | **1.04** | **20.20** | **1.11** |
| 0.5 *(current)* | 634 | 9.64 | **9.79** | 0.05 | 19.48 | 0.04 |
| 1.0 | 551 | 6.67 | 8.64 | 0.48 | 15.79 | 0.50 |
| 1.5 | 457 | 6.06 | 5.55 | 0.12 | 11.73 | 0.12 |
| 2.0 | 375 | 6.08 | 4.85 | 0.92 | 11.85 | 0.95 |

0.25 wins W1 (headline) and W3 (persistence), summed return and worst-window
Calmar; 0.5 wins only W2. Max regret marginally favours 0.5 (1.06 vs 1.34) but
on W2 alone. **Recommend 0.25** — the gain is modest in return (+0.72pp) but W3
goes 0.05% -> 1.04% and its Calmar 0.04 -> 1.11, on the family's weakest window.

rvol 0.0 (guard fully off) is WORSE than 0.25, so a light volume filter does earn
its place — the peak is just much looser than NNNC's.

## guard_metric 0.3 — dominant here too, second family

| guard | trades | W1 | W2 | W3 | summed | worst Calmar |
|---|---|---|---|---|---|---|
| −0.4 | 603 | 10.06 | 7.80 | 0.59 | 18.45 | 0.64 |
| 0.0 | 646 | 10.05 | 7.97 | 0.62 | 18.64 | 0.66 |
| **0.3** | 654 | **10.71** | **8.45** | **1.04** | **20.20** | **1.11** |
| 0.6 | 654 | 10.71 | 8.45 | 1.04 | 20.20 | 1.11 |

Dominant in all three windows on both metrics, and 0.6 byte-identical to 0.3 by
trade hash — the same saturating plateau as NNNC. **`guard_metric = 0.3` is now
validated on two families and is correctly already the `BaseStrategy` default.**
It is structural in the way rvol is not.

## apply_task_filters — strongly HARMFUL at the current heads

Inference-time entry filter, so this needs no retrain. At rvol 0.25 / guard 0.3:

| filter | W1 | W2 | W3 | summed | trades |
|---|---|---|---|---|---|
| OFF *(default)* | 10.71% | 8.45% | 1.04% | **20.20%** | 654 |
| ON | 3.68% | 1.15% | 1.15% | **5.99%** | 408 |

**−14.21pp**, worse in W1 and W2, LIVE (hashes differ in all three windows).

This **reverses** the recorded finding that the 3-of-4 task filter is the
dominant entry-quality lever. It is dominant, but in the harmful direction at
this operating point. The old measurement was taken under the broken exit and at
a much tighter gate (it reported disabling the filter giving 4.5x the trades;
here enabling it removes only 38%), so the two are not directly comparable.

**This is consistent with the task audit**, which found NNMT trains the three
LOWEST-relevance tasks and excludes the highest (NMI 0.130). The filter is gating
entries on heads that do not carry entry information, so it is discarding good
trades. That is evidence about the *current* heads, not about the concept.

## What this says about re-running the task study

The cheap gate proposed before spending retrains has now run, and it argues for a
**targeted** retrain rather than a combinatorial one. The aux heads as currently
trained are not merely uninformative for entries — using them actively destroys
14pp. Re-sweeping combinations OF THOSE HEADS is unlikely to find an edge. The
question worth a retrain is the one the audit already identified: **train the
high-NMI head that is currently excluded**, then re-test the filter.
