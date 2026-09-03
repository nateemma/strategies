# NNNC_BreakoutOpen — breakout label, open gate, model exits (2026-09-03)

Re-test of the breakout signal on the REPAIRED exit stack. The 2026-07-08 final
verdict ("no tradeable long edge at any timeframe with any exit", best −1.11%)
was reached entirely under the native-trailing-stop defect fixed on 2026-09-02,
and its core argument was about stops outnumbering winners — so it was worth
re-measuring once the stop stopped cutting winners on noise.

## Configuration

| | value | note |
|---|---|---|
| `TRAINING_TYPE` | 24 `breakout_gbb` | `guard_metric > +0.2`, the sign flip vs gbb's `< −0.2` |
| `HORIZON` | 48 **pinned** | would otherwise inherit gbb-tuned `TrainingConfig.HORIZON` |
| `MIN_BUY_GAIN_THRESHOLD` | 0.005 **pinned** | gbb-tuned default is 0.007 |
| `entry_enable_guards` | **False** | the dip guards reject every breakout bar |
| `enable_exit_signal` | **True** | model participates in exits |
| sell label | method 17 `labels_gbb_sell` | sell the RIP, not the mirror's sell-the-dip |

Trained `20220101-20250831`, so **W1 is out-of-sample**; W2/W3 in-sample.
Training was healthy — `val_mcc 0.576`, early-stopped at 46 epochs, ~5.5K buy and
~6.8K sell labels, no collapse.

## Result: still negative

Threshold sweep on W1 (guards off, so this is the ONLY entry filter):

| `prediction_threshold` | trades | W1 return |
|---|---|---|
| 0.69 (NNNC's gbb-tuned value) | **0** | — |
| 0.50 | 1336 | **−5.05%** |
| 0.30 | 1995 | −9.64% |
| 0.15 | 1995 | −9.64% (saturated) |

Monotone: **trade less, lose less**, with no interior optimum and zero trades at
the point where the loss would reach zero. That is the signature of no edge, and
it reproduces the 2026-07-08 finding exactly.

At its best threshold (0.50), across all three windows:

| window | trades | return | DD | PF | win% | Calmar |
|---|---|---|---|---|---|---|
| W1 (OOS) | 1336 | −5.05% | 10.10% | 0.92 | 58.3 | −2.62 |
| W2* | 1881 | −14.73% | 19.69% | 0.85 | 58.5 | −3.93 |
| W3* | 1532 | +2.22% | 7.89% | 1.03 | 60.4 | 1.48 |
| **sum** | | **−17.55%** | | | | |

Against NNNC gbb tuned at **+25.32%**.

**Win rate is 58–60% but profit factor is below 1** in the two windows that
matter — many small wins against fewer, much larger losses. That is the
false-breakout payoff shape, and it is the real problem: the entry is directionally
right more often than not, but wrong expensively.

## What the exit signal was worth

Enabling it helped, and in the predicted direction — **+0.66pp on W1**
(−5.71% → −5.05%, 1232 → 1336 trades). Real, consistent with the +10.4pp measured
on NNMT at an open gate, and nowhere near enough to matter here.

## Verdict

**The 2026-07-08 verdict survives the exit repair.** The exit defect was not what
was holding breakout back.

**The one untested lever left** is the risk side, and it is untested for a real
reason: `stoploss = −0.05` and `minimal_roi = 0.05` are both gbb-tuned, carried
over unchanged. With a 58–60% win rate and PF < 1, the arm loses on loss
MAGNITUDE, not on direction — which is exactly what a stop controls. A breakout
arm plausibly wants a tighter stop than a dip-buyer, since a failed breakout
reverts hard. That has never been swept for this label. It would not have to
recover 17pp to be interesting, only to show the payoff shape responding.
