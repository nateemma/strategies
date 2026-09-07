# The sell label's path behaviour — the mirror of the buy-side stop check

Measured 2026-09-07 (`regime/path_label_sell.py`), prompted by the question of
whether the buy side's stop check has a sell-side equivalent.

## First, the premise correction

**Neither side has a stop/path check.** The production labeler
(`TRAINING_TYPE 17`, `labels_gbb` / `labels_gbb_sell`) is path-BLIND on both
sides, and the two are exact mirrors of each other:

    BUY   future_gain = (rolling MAX over next H - close)/close >= min_gain
    SELL  future_loss = (close - rolling MIN over next H)/close >= min_loss

Both are extremum-over-window. Neither knows the ORDER in which prices arrived,
so neither can tell whether a barrier fired first.

Where a first-touch check DOES exist it IS mirrored, but both instances are
inactive: `labels_triple_barrier`/`_sell` is `TRAINING_TYPE 1` (unused;
`TrainingConfig` annotates it "poorly learnable"), and NNMT's
`get_profit_target` is genuine first-touch and symmetric — but it feeds the
PROFIT head, which carries loss weight 0.0 and is untrained (feature 005 T013).

## The mirror risk is NOT a stop

The buy label promises a rise and its path risk is the ATR STOP firing first.
The sell label promises a fall and is used to EXIT a long, so its mirror risk is
the **TAKE-PROFIT firing first**: the position closes profitably on the way
down, and the sell signal never gets to act. So the mirror of "worst adverse
excursion before the target" is "best FAVOURABLE excursion before the target".

## Result (126,408 labelled bars at min_loss 0.02, 11 pairs, H=48)

| min_loss | labelled bars | never drops | pre-empted @2.2% | @3.0% | @3.2% |
|---|---|---|---|---|---|
| 0.7% (pre-inheritance) | 187,547 | 0.0% | 9.2% | 5.1% | 4.4% |
| **2.0% (current)** | 126,408 | 0.0% | **11.4%** | 6.5% | 5.7% |

MFE before the drop arrives: p50 **+0.29%**, p75 +1.18%, p90 +2.38%,
p95 +3.39%, p99 +6.07%.

## Read against the buy side

| | buy label | sell label |
|---|---|---|
| path risk | ATR stop fires first | take-profit fires first |
| affected | **4.2%** @ −4% stop | **11.4%** @ +2.2% take-profit |
| median excursion on the way | **−0.29%** | **+0.29%** |

The median excursions are mirror images to the decimal, which is a good sign the
two labels really are symmetric constructions.

**The sell label is pre-empted 2.7× more often than the buy label is killed** —
not because it is worse, but because the barrier it faces is much closer: a
+2.2% take-profit against a −4% to −6% stop.

## What it means, and what it does not

**It is not a P&L defect.** A pre-empted sell label costs nothing in trading —
the take-profit closed the position profitably, which is the better outcome. The
cost is in TRAINING: roughly 1 in 9 sell-labelled bars teaches the sell head to
predict a drop that could never have been acted on. That is target noise, not
lost return.

**The sell side avoids the buy side's larger problem entirely.** The buy label's
worst defect was never the stop — it was that 12.6% of labelled bars never reach
the take-profit at all, because the label certifies +2.0% while the exit needs
+2.2%. The sell label has no analogue: it is a SIGNAL, not a price promise, and
the strategy exits at market when it fires. There is no level it must reach, so
"never reached" is 0.0% by construction.

**Verdict: the sell label is in better shape than the buy label.** Its only path
defect is mild, costs no return, and is bounded at ~11% of labels. This is
mildly reassuring for the exit-side promotion, since that change makes the sell
head live for the first time on this family.

**Not worth acting on yet.** A path-aware sell label would clean up ~11% of the
training target for an effect that cannot exceed that, on a head whose whole
measured contribution is +0.78pp. It is a candidate only if the exit-side line
is pursued further — and it should then be built as a mirrored pair with the
buy side, not bolted onto one side.
