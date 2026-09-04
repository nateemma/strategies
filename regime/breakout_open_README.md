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

## The learnability screen passed on BOTH axes — and was wrong

Independent gate run (`DebugSignalLearnability`, methods 17 vs 24, H ∈ {24,36,48},
thresholds {0.007, 0.010, 0.015}, XRP/SOL/LINK). It confirms H=48 was the right
pin — MCC rises monotonically with H for both labels — and then says something
much more useful.

At H=48, mean over the three pairs:

| label | thr | MCC | EV/signal | signals |
|---|---|---|---|---|
| gbb | 0.007 | **0.800** | 3.90% | 13777 |
| gbb | 0.010 | 0.747 | 4.19% | 12618 |
| gbb | 0.015 | 0.665 | 4.69% | 10805 |
| breakout_gbb | 0.007 | 0.753 | **4.50%** | 13505 |
| breakout_gbb | 0.010 | 0.699 | **4.87%** | 12334 |
| breakout_gbb | 0.015 | 0.609 | **5.48%** | 10635 |

**`breakout_gbb` has HIGHER EV per signal than gbb at every threshold**, and is
nearly as learnable (0.753 vs 0.800; per-pair LINK 0.787 / SOL 0.790 / XRP 0.681).

The screen was green on both axes. The backtest is **−17.55%** against gbb's
**+25.32%** — a ~43pp gap in opposite directions, from the label that screened
BETTER on expected value.

### Why the screen lies, specifically

`ev_per_signal` is computed from `future_gain` = the **maximum favourable
excursion** over the horizon. It is PATH-BLIND: it asks whether price ever
reached a good level, never whether it got there without hitting the stop first.

Breakouts have excellent MFE and an adverse path — they revert hard before
continuing, so the stop fires long before that upside is reachable. The backtest
shape confirms it exactly: **58–60% win rate with PF below 1**, i.e. frequently
right about direction, expensively wrong about the route.

gbb does not have this problem because a dip-buy enters AFTER the adverse
excursion has happened; the breakout enters BEFORE it.

**Methodological consequence.** An MFE-based EV screen systematically flatters
momentum labels and is fair to mean-reversion ones. It cannot rank the two
against each other. To screen a momentum label, the EV metric needs to be
path-aware — maximum ADVERSE excursion before the favourable one, or the
counterfactual replay built for feature 002, which already computes exactly this
from recorded trades.

This is the SIXTH independent confirmation that learnability does not equal edge,
and the sharpest: previous cases had the metric merely failing to predict P&L.
Here both screening metrics pointed the WRONG WAY.

## Risk-side sweep: both levers exhausted, and a tooling correction

### `stoploss` is INERT for every NN strategy — the real lever is the ATR band

`BaseNNStrategy:139` sets `use_atr_adaptive_stoploss = True` (it is False in
`BaseStrategy`, which is why this was missed). `_resolve_normal_stop()` therefore
returns an ATR-derived stop and **ignores `self.stoploss`**, which only sets the
cosmetic `initial_stop_loss_abs`.

Verified: `--stoploss -0.01` vs `-0.05` gives **byte-identical** backtests on both
NNNC_MLX and the breakout arm, while the recorded initial stop dutifully changes
from −1.00% to −5.00%. This corrects `specs/002-exit-process-repair/research.md`,
which recorded the ATR block as inert. No conclusion in that study depended on it
— every arm held `--stoploss -0.05` — but it made the wrong parameter look live.

The real lever is `atr_stoploss_multiplier` / `_floor` / `_cap`, swept here via
subclasses that override `get_model_path` to reuse the trained model (no retrain).

### ATR stop band — an inverted U peaking at the EXISTING value

| multiplier | band | trades | W1 | PF | win% |
|---|---|---|---|---|---|
| 1.5 | [−2.5, −1.0] | 1348 | −6.25% | 0.88 | 43.5 |
| 2.0 | [−3.0, −1.5] | 1342 | −5.92% | 0.89 | 50.2 |
| 3.0 | [−4.5, −2.5] | 1339 | −5.37% | 0.91 | 56.4 |
| **4.0** | **[−6.0, −4.0]** | 1336 | **−5.05%** | **0.92** | 58.3 |
| 6.0 | [−9.0, −6.0] | 1334 | −7.58% | 0.88 | 60.2 |

**The hypothesis is refuted.** Tightening the stop does not fix the
loss-magnitude problem — it converts winners into losers faster than it saves
losers, with win rate collapsing 58.3% → 43.5%. Loosening is worse still. The
gbb-tuned band is already the optimum for the breakout arm, so the carry-over was
never the problem.

### Upside caps — flat

| take_profit | roi | W1 | PF |
|---|---|---|---|
| 0.024 (ctl) | 0.05 | −5.05% | 0.92 |
| 0.04 | 0.05 | −5.11% | 0.91 |
| 0.06 | 0.05 | −4.91% | 0.92 |
| 0.06 | 0.10 | −5.96% | 0.90 |

Raising the take-profit cap is worth +0.14pp — noise. Raising ROI hurts.

### Conclusion

Both risk-side levers are exhausted and the existing gbb-tuned values are at or
next to the optimum for this arm too. The −5% is structural: 557 stop exits at a
−2.67% mean total −1485% against +1268% of wins, and no stop or target setting
changes that balance. The breakout entry is simply wrong often enough, and
expensively enough, that exit-side tuning cannot rescue it.

Worth recording on the positive side: `model_exit` fires 416 times at **+1.79%
mean with a worst case of 0.00%** — the exit signal is genuinely productive here.
It is the entry that fails.

## Exit label: level vs crossing — the fix worked mechanically and lost more

`labels_breakout_gbb_sell` was rewritten from the level condition to the
transition "was above the upper Bollinger band, closed back inside" — momentum
FAILING rather than momentum PRESENT. Motivated by a real defect: with the
level-based sell, model exits fired at a 135-minute median for +1.67% while
take_profit reached +3.49% at 465 minutes.

Training changed in the predicted direction. Sell labels fell 5x (≈6,600 → ≈1,330,
2.22% of bars), and although `val_mcc` dipped 0.576 → 0.546, **precision more than
tripled (0.129 → 0.428)** and F1 on the signal class went 0.216 → 0.564.

| | OLD level-sell | NEW failure-sell |
|---|---|---|
| trades (W1) | 1336 | 2149 |
| model_exit hold | 135 min | **255 min** |
| model_exit mean | +1.79% | +1.56% |
| take_profit exits | 35 | **100** |
| stop exits | 557 @ −2.67% | 829 @ −3.21% |
| **W1 (OOS)** | **−5.05%** | **−12.40%** |
| summed | −17.55% | −25.16% |

**The mechanism worked and the premise was wrong.** Trades held 1.9× longer and
take-profit exits tripled, exactly as designed. But holding longer did not convert
losers into winners — it let them run further against, with mean stop loss
deepening −2.67% → −3.21%. Two trades even reached the static `stop_loss` at
−17.15%, an exit reason that had never appeared before.

Two confounds worth recording, because they make this a weaker test than it looks:
1. The sparser sell class loosened the 3-class decision boundary, so **entries
   rose 61%** (1336 → 2149). The comparison is not exit-only.
2. Lowering the threshold to 0.30 was worse still (−28.63% summed), so the extra
   entries are the marginal-quality ones, consistent with everything else here.

**Another precision-does-not-predict-edge instance**, and a sharper one than MCC:
precision tripled and P&L halved.

## Status: closed

Five independent interventions have now been tested on this arm — prediction
threshold, ATR stop band, take-profit/ROI caps, exit signal on/off, and
level-vs-crossing sell label. All fail the same way: they redistribute the same
negative expectancy. The 2026-07-08 verdict stands, now with the exit defect,
the risk parameters and the exit label form all separately ruled out.

## Retrained under the peak-filter fix

The breakout model had been trained under the peak-filter bug (effective 6-bar
labels). Retrained with the fix so `HORIZON = 48` genuinely applies. Identical
config, guards, threshold and training window — only the model differs.

| | buy labels | sell labels | val_mcc | precision |
|---|---|---|---|---|
| pre-fix | 5940 | 1375 | 0.5456 | 0.4275 |
| FIXED | **9951** (+68%) | **2554** (+86%) | **0.7116** | **0.6352** |

| arm | trades | W1 (OOS) | W2* | W3* | summed | PF | win% |
|---|---|---|---|---|---|---|---|
| pre-fix (labels @6) | 291 | −1.09 | +0.12 | +1.95 | +0.98 | 1.05 | 57.4 |
| FIXED (labels @48) | 293 | −0.74 | +0.34 | +1.71 | **+1.31** | 1.06 | 57.6 |

W1 payoff: win 58.2 → 58.8%, avgWin 2.99 → 2.95%, avgLoss 3.96 → 4.03%,
margin +1.2 → **+1.0pp**.

**+0.33pp summed — directionally positive, materially nothing.** Unlike the NNNC
pair, the trades genuinely DIFFER in all three windows, because two of seven
guards are neutralised here so the model has room to express a change. It used
that room to move the win rate 0.6pp when the arm needs ~4pp, and average loss
rose enough that the break-even MARGIN actually got slightly worse.

So the fourth large metric shift of the study (+0.166 MCC, +0.208 precision)
again converted to nothing. Keep the retrained model — it is the correct one —
but the fix does not rescue the breakout entry.

The sell-label sparsity concern does ease: the momentum-failure crossing label
was 2.2% of bars and flagged HIGH collapse risk; at the true horizon it is ~86%
denser, so that specific risk is gone.
