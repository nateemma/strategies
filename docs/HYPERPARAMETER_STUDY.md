# Before you tune it, find out whether it can move

**A practical guide to examining strategy hyperparameters.**

Audience: anyone about to sweep, hyperopt, or otherwise search a strategy's parameters in this
repo. Companion to `AGENT_GUIDE.md`'s "Silent failure modes" and "The noise floor" sections —
read those first for the mechanics; this is about the *method*.

Every number below was measured in this repository. Sources are cited so you can re-derive them.

---

## The short version

Most parameter tuning in a trading strategy is spent on parameters that cannot change the
outcome, measured on windows that cannot resolve the change, against an evaluator that has never
been checked against reality. Three cheap measurements, taken **before** any search, tell you
which of those three problems you have — and usually the answer is "all of them, so don't run
the search".

The single most useful question is not *what is the best value?* It is **can this parameter move
the objective at all?** Those are different questions, and only the second one is cheap.

---

## 1. Measure the span before you search the space

A parameter sweep produces a best value whether or not the parameter does anything. Rank the
rungs, take the top one, and you have a number — with no indication of whether it beat the
others or merely came first in a shuffle.

The test that separates the two is the **span**: sweep the parameter across its entire range,
with everything else fixed, and record `max(objective) − min(objective)`. Then compare that span
to your noise floor.

In this repo, on the pooled 15-fold walk-forward harness:

```
acceptance floor (pooled walk-forward)        3.35 pp/yr
single-window noise floor (W1, 99 trades)     7.633 pp
span of an ENTIRE rvol parameter sweep        4.93 pp
```

**A whole sweep fits inside the noise floor.** Whatever the best rung was, it was a ranking of
noise. Confirmed independently: full sweeps of two parameters across three windows disagreed on
the optimum every time, with rank correlations between windows of −0.46 and +0.71, neither
significant (`AGENT_GUIDE.md`, "The noise floor").

Ten parameters measured this way in the exit and guard stack:

| parameter | span (pp/yr) | verdict |
|---|---|---|
| `stoploss_grace_hours` | 1.370 | weak |
| `entry_rvol_threshold` | 1.335 | weak |
| `stoploss_grace_level` | 1.104 | weak |
| `cexit_take_profit` | 0.140 | weak |
| `entry_guard_threshold` | 0.072 | **inert** |
| `atr_stoploss_cap` | 0.059 | **inert** |
| `atr_stoploss_floor` | 0.052 | **inert** |
| `atr_stoploss_multiplier` | 0.039 | **inert** |
| `hold_window_hours` | 0.035 | **inert** |
| `cexit_max_days` | 0.000 | **inert** |

**Six of ten cannot move the objective at all, and not one reaches the acceptance floor.** The
widest spans 1.37 pp against a floor of 3.35. That is the whole result: no search over this
space can succeed, and it cost ~5 minutes of compute to learn.

**Pre-register your thresholds.** Decide what counts as "inert" *before* you see the spans, or
you will pick the cut that tells the story you already believe. The thresholds used above were
fixed in the design document: `inert < 0.10 pp/yr` (numerically indistinguishable from nothing),
`weak < 3.35` (live, but cannot clear the floor on its own).

---

## 2. A mechanism mattering is not the same as its knobs mattering

This is the distinction that most often gets lost, and it is counter-intuitive enough to be
worth a section.

Census of 378 archived walk-forward trades by exit reason:

| exit reason | trades | share | share of P&L |
|---|---|---|---|
| `take_profit` | 195 | 51.6% | +179.7% |
| `trailing_stop_loss` | 98 | 25.9% | **−120.2%** |
| `model_exit` | 45 | 11.9% | +21.9% |
| `unclog_12h` | 27 | 7.1% | +7.6% |
| `roi` | 10 | 2.6% | +14.8% |
| `stop_loss` | 1 | 0.3% | −4.7% |

The ATR-adaptive stop stack fires on 26% of trades and accounts for essentially the **entire
loss side of the ledger**. It is, by any reasonable reading, the most consequential mechanism in
the exit stack.

Its three tunable parameters — `atr_stoploss_multiplier`, `_floor`, `_cap` — are all **inert**.
Varying the cap eightfold (−0.04 → −0.005) moves P&L by 0.03 pp/yr.

Both facts are true. The stop decides the outcome; its knobs do not, because the level is
clamped for ~87% of bars and moving the clamp just shifts which bar triggers. **"This mechanism
is important" is not an argument for tuning its parameters.** It is an argument for measuring
whether its parameters reach anything.

The same pattern recurs across this repo: gate and mechanism changes move P&L, model-side
parameters do not — seven interventions across two strategy families, every win structural,
every null model-side, with +77% more labels and +0.155 MCC producing *identical trades*.

---

## 3. Read the label before you believe the census

The census above has a trap in it, and it inverts the conclusion.

`trailing_stop_loss` is **not** freqtrade's native trailing stop, which is switched off
(`BaseStrategy.trailing_stop = False`). `freqtrade/strategy/interface.py:1654-1658` labels a stop
exit `STOP_LOSS` and **upgrades it to `TRAILING_STOP_LOSS` whenever the stop that fired differs
from the initial one** — i.e. whenever `custom_stoploss` moved it.

So a naive reading of the table says the ATR stop governs the `stop_loss` row: 1 trade, 0.3%,
irrelevant. The correct reading is that it governs the `trailing_stop_loss` row: 98 trades,
26%, the entire loss side. **Two opposite conclusions from the same table**, separated only by
knowing what the label means.

Before drawing conclusions from any categorical breakdown — exit reasons, entry tags, rejection
counts — read the code that assigns the labels. Ten minutes there is cheaper than a week spent
optimising the wrong mechanism.

---

## 4. Calibrate your evaluator against reality, not against itself

Any fast evaluator — a vectorised simulator, a surrogate, a cached approximation — is a
re-implementation of the real execution path, and re-implementations drift.

A surrogate built for the study above was validated with ten unit tests, all passing. Then it was
compared against **one real backtest** of the same configuration over the same range. That single
comparison found four bugs, every one on the two mechanisms that decide the P&L:

| # | bug | ground truth |
|---|---|---|
| 1 | Exit sequence inverted — the custom exit is evaluated *before* the stoploss, so a bar where both fire resolves as the custom exit | `interface.py:1503-1507` |
| 2 | Wrong price per check — `current_profit` (every custom exit) comes from the **open**, `current_profit_best` from the **high** for ROI only, the stop triggers on the **low** | `backtesting.py:992-1000` |
| 3 | Stop fill too punitive — the open is used only when the stop sits above the candle's **high**; otherwise the fill is *at the stop* | `backtesting.py:614-616` |
| 4 | Exit signal ungated — it is not just `predict_sell`; it is gated on `exit_rvol_threshold` and `exit_guard_threshold` | `BaseStrategy.py:1009-1015` |

Trade counts by exit reason, before and after:

```
                      real   before    after
  model_exit            36      143       44
  take_profit          211      169      235
  trailing_stop_loss   176      139      187
  stop_loss              2        2        2
  TOTAL                478      538      521
```

**None of the four failed any unit test**, because every unit test was self-consistent with the
surrogate's own logic. Only comparison against the real thing discriminates.

Two rules follow:

- **Gate on rank, not level.** A fast evaluator is used to *order* candidates. A constant level
  offset is harmless; a reordering is fatal. Demanding level agreement fails on differences that
  no change can fix — an earlier feature's "bit-for-bit" gate tripped on an MLX-vs-numpy `log()`
  difference of ~5 ulps. The calibrated surrogate here reached Spearman **+0.9766** on rank while
  levels differed by −5.29 ± 23.46 pp, and that is a pass.
- **Check *where* it disagrees, not just how much.** A global correlation of 0.98 is reassuring
  about the wrong thing if the disagreements concentrate on one parameter — the evaluator would
  then be least reliable exactly where the search pushes hardest. Regress rank error against each
  parameter's distance from the incumbent. Here the worst was +0.20 and six of ten were negative.

---

## 5. Beating the control does not mean the space has an edge

Give an optimiser a budget and it will find the best point in the space. That says the optimiser
works. It says nothing about whether the best point is worth having.

In the study above, CMA-ES was selected over uniform random search in **14 of 15 folds** — a
clean win on the control gate. The result was still a kill:

```
best vs incumbent            +0.19 pp/yr        (noise band 4.93 pp — 25x larger)
probability of backtest overfitting   0.7128    (threshold 0.2)
evaluations entering it              48,015
distinct configurations              23,175
```

**Always run the equal-budget random control**, and always count your evaluations. Without the
control, "we found a good configuration" and "we looked 48,000 times" are indistinguishable.

A subtlety worth internalising: a per-configuration overfitting check *passed* here (none of the
15 selected configurations collapsed from train to test) while PBO was 0.71. Those measure
different things. The first asks whether your chosen configurations held up; the second asks
whether the *ranking that produced them* is stable. **A search can pick configurations that
individually survive while the ranking that selected them is noise.** That is the multiple-testing
problem in one line, and only the second measurement sees it.

To compute PBO at all you need a **time axis** — per-block returns, not a summary scalar. Four
consecutive studies in this repo reported overfitting statistics as "unevaluable" because the
trial log stored only scalars. A first attempt here returned PBO = 0.0000, which is not a result:
with constant columns, CSCV finds every split's ranking identical. Sixteen block returns per
trial fixed it, and the honest number was 0.7128.

---

## 6. Your gates were calibrated on something; check it was this

A gate is a measuring instrument, and instruments are calibrated against a reference. Use one
outside its calibration range and it produces confident nonsense.

Two instances from this study, both caught by pointing the gate at **production** and seeing what
it said:

**A participation floor that rejected production.** A `mean_exposure ≥ 0.05` floor existed to
catch policies that abstain. Applied to a discrete-trade strategy on a 90-day window, it marked
the *production strategy* degenerate (0.0196 and 0.0474). Production is not abstaining — it takes
13–20 trades in 90 days at ~4 h each, so 2–5% occupancy is simply what a selective strategy looks
like. The floor had been calibrated on continuous-exposure policies where a weight is held every
bar. Left unfixed, every fold would have come back inadmissible and the study would have reported
**a miscalibrated instrument as a research finding**.

**A return convention that flipped the sign.** An evaluator compounded `prod(1 + pnl)` over a
pooled trade list — correct for sequential full-wallet bets, wrong for 297 *concurrent* trades
across 9 pairs under a fixed stake. The config runs a fixed 900 on a 10,000 wallet, so returns are
additive and one unit of `profit_ratio` is worth exactly 9 pp. A real +23.46 pp/yr read as −19.06.

**The general technique: run your gate on the incumbent.** If your acceptance gate rejects the
thing you are already running in production, the gate is wrong, not production. This costs one
evaluation and catches a class of error that is otherwise invisible, because a miscalibrated gate
fails *quietly* and in the direction of a null result — which looks like rigour.

A related trap: `nan < threshold` is `False` in both Python and numpy, so a NaN slips through
every floor **silently**. Guard for it explicitly.

---

## 7. Reading the evidence honestly

Four ways to over-read a result, each seen in this study:

**Nested folds are not independent votes.** Fifteen anchored walk-forward folds all began
2021-01-01, so the last fold's training data contains the first fold's entirely. When all 15
agreed a parameter should move up — with an interquartile range of 0.016 — that is much closer
to *one* observation than fifteen. A tight spread across nested folds measures shared data, not
replication.

**An optimum on a bound means the space was drawn too narrowly.** Report it; do not quietly
accept the edge value. The converse also matters: if a parameter is inert, the optimiser has no
gradient and lands on a bound *arbitrarily* — so check liveness before you interpret a boundary
hit as a signal.

**Abstention is not performance.** Roughly half of uniformly drawn candidates in this space
described a policy that barely traded, and 20.6% of all evaluations were inadmissible. A search
that minimises cost drag will drift toward not trading, and unless something forbids it, "trade
less" wins on the training range and generalises badly. An earlier study measured **92%** of an
arm's apparent advantage to be turnover rather than signal.

**Persistence beats magnitude.** A result that holds in one era and not another is a property of
that era. Split across temporally distant windows and require *both*. Here the found
configuration returned +0.61 and +0.78 pp/yr in two windows and **−0.02** in the third, which is
a fail regardless of the pooled average.

---

## 8. A checklist

Before any sweep, hyperopt, or search:

1. **Census the mechanism.** What actually fires? Read the code that assigns the labels.
2. **Measure the span of each parameter** across its full range, one at a time. Pre-register what
   counts as inert.
3. **Compare the widest span to your noise floor.** If the whole space fits inside the band, stop
   — no search can resolve it.
4. **Run your gates on the incumbent.** If they reject production, fix the gate.
5. **Calibrate your fast evaluator against one real backtest.** Gate on rank; check *where* it
   disagrees.
6. **Record every evaluation before you score it**, including the ones you abandon — the
   deflation counts attempts, not successes.
7. **Run an equal-budget random control**, and report the evaluation count beside every
   overfitting statistic.
8. **Require persistence across distant eras**, not just a pooled average.

Steps 1–4 are minutes of compute and routinely make steps 5–8 unnecessary.

---

## Why negative results are the normal output

Five studies have run under this protocol. None produced an adoptable operating point. That is
not a failure of the method — it is the method working. Each delivered a number that closes a
line of enquiry:

| study | verdict | the number |
|---|---|---|
| direct policy optimisation | NOT RESOLVED | no configuration reached a pooled run |
| label warm-start | NOT RESOLVED | 92% of the advantage was turnover, not signal |
| meta-labelling | KILL | nested AUC 0.4978 — the confidence→outcome map is non-stationary |
| RL in a pooled environment | KILL | pooled reward IQM −0.1301, positive in 5/15 folds |
| gradient-free policy search | KILL | PBO 0.7128 over 48,015 evaluations |

A quantified negative is worth more than an unquantified positive, because it is *reusable*: the
span table in §1 means nobody needs to sweep the ATR stop parameters again, and the reason is on
the record. An unmeasured "we tried it and it didn't seem to help" gets re-tried every six months.

**The corollary for parameter defaults**: when a study cannot beat the incumbent, the finding is
usually not "we need a better search". It is that the incumbent sits in a flat region, and the
useful output is the map of that flatness — which parameters are inert, which are weak, and how
far the whole space is from mattering.

---

## Sources

- `AGENT_GUIDE.md` — "Silent failure modes", "The noise floor"
- `research/label-free-study/README.md` — the five studies and their verdicts
- `research/label-free-study/briefs/results/05-2026-09-13.md` — the study this article generalises
- `regime/wf/README.md` — the pooled walk-forward harness
- `research/labelfree/es/liveness.py` — the span probe, with its pre-registered thresholds
- `research/labelfree/admissibility.py` — the participation gate and why it is centralised
