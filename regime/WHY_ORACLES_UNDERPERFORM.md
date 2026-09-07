# Why a PERFECT model underperforms a real one

Measured 2026-09-07 on NNMT W1. This corrects how every oracle figure in this
repository should be read, including feature 005's headline result.

## The paradox

The oracle's entry signal IS the training label. It knows, with certainty, that
every bar it enters reaches **+2% within 48 bars**. The real model only guesses.
Yet:

| W1 | real model | ORACLE (perfect entry) |
|---|---|---|
| trades | 176 | 155 |
| return | **+10.708%** | **+6.150%** |
| % stopped out | 44.3% | **48.4%** |
| % reaching take-profit | **22.2%** | 15.5% |
| win rate | 55.7% | 51.6% |
| avg win / avg loss | +4.2% / −3.6% | +3.9% / −3.3% |
| **avg max favourable excursion** | **4.0%** | **3.7%** |

**Perfect foreknowledge stops out MORE often and converts FEWER entries into
take-profits.** Broken down by mechanism, the oracle's 21 missing trades are
almost entirely WINNERS:

| mechanism | real | oracle | difference |
|---|---|---|---|
| ATR stop (moved) | 77 trades, −1406.8 | 75, −1337.6 | **avoids only 2 losers** |
| ROI table | 50, +842.0 | 47, +803.0 | −3 |
| **take-profit** | **39, +1627.5** | **24, +1088.8** | **−15 winners, −538.7** |

## The explanation: a binary FLOOR versus a continuous RANKING

The label is `max forward close over 48 bars ≥ +2%`. Three properties make it a
weak proxy for profit:

1. **It is a FLOOR, not a target.** A bar qualifies at exactly +2.0%. The exit
   needs +2.2% (`cexit_take_profit`). So a labelled bar can be certain to rise
   2% and still never trigger a profitable exit — feature 004 measured 12.6% of
   labelled bars never reaching the take-profit at all.
2. **It is PATH-BLIND.** Reaching +2% says nothing about the −4% to −6% ATR stop
   firing first.
3. **It is INDIFFERENT among qualifying bars.** A bar that rises 2.01% and one
   that rises 12% carry the identical label.

The real model does not reproduce this binary certificate. It learns a
**continuous score**, and thresholding that score at 0.5 selects bars where the
score is HIGH — which correlates with the SIZE of the coming move, not merely
with clearing 2%.

**The measured consequence is the last row of the table: the real model's trades
have a larger average maximum favourable excursion, 4.0% against the oracle's
3.7%.** The model picks stronger moves than the label requires. The oracle, being
indifferent, takes the marginal ones too — and marginal moves stop out.

## What this means for the word "headroom"

**Negative headroom does NOT mean "the model is already perfect".** It means
**"the model has learned something BETTER than the label"** — a ranking, where
the label offers only a threshold.

Feature 005 measured NNMT's headroom as **−4.558pp** and read it as "there is no
room above the current model". The arithmetic is right; the interpretation needs
this correction. The honest statement is: *a model that perfectly reproduced
NNMT's training label would trade WORSE than the model we have, because the
label is a coarser instrument than the score the model actually learned.*

It also explains, cleanly, why the label work failed:

- `gbb` 0.007 → 0.02 moved the ceiling +45.9pp and realised −0.157pp.
- `gbb` → `gbb_path` moved the ceiling +6.2pp and realised −0.091pp.

**Improving the label's FIDELITY cannot help a model that is already extracting
more information than the label contains.** Both label changes made the
certificate more accurate; neither made the ranking better.

## What the oracle gate is still good for — and its new caveat

The gate is not invalidated. It answers a narrower question correctly: *with the
guards off and no ranking available, can this label make money at all?* That is
exactly the right screen before a retrain, and it killed `gbb` 0.007 (−28.1%
guards-off) before anyone trained on it.

**But add this caveat wherever an oracle number is used:**

> An oracle bounds what PERFECT LABEL-FOLLOWING is worth, NOT what perfect
> trading is worth. When the label is a coarse proxy — a floor rather than a
> target, path-blind, indifferent among qualifying bars — a well-calibrated
> model can and does BEAT its own oracle. A ceiling below the real model's
> return is evidence about the LABEL, not about the model.

## The practical inversion

The programme spent this session asking "how do we help the model learn the
label better?" The measurement says the model is already ahead of the label. So
the productive questions are the other two:

- **Make the label rank, not just certify.** The model's edge over the oracle
  comes from ordering bars by expected move size. A regression or ordinal target
  on forward excursion would give it that signal directly, instead of making it
  infer ordering from a binary label. (Note: `feedback_pnl_weighted_loss_raises
  _mcc_not_pnl` records that P&L-weighting the loss raised MCC without moving
  P&L — so this is not a free win, and would need its own gate.)
- **Fix the exits.** 44-48% of trades stop out in BOTH arms, and the oracle
  avoids only 2 of 77 stops. That number is nearly independent of entry quality,
  which makes it an exit-side problem — and the exit side is where this
  session's only positive result came from.
