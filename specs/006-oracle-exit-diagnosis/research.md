# Phase 0 Research — 006 Oracle-Instrumented Exit Diagnosis

All items resolved. Nothing carried into Phase 1 as NEEDS CLARIFICATION.

---

## R1. The prior exit work is STALE, not redundant — DECIDED: re-ask at the current label

Feature 004 explored the ATR stop band and the take-profit on NNNC and concluded
the stop was "the dominant destroyer". Those arms still exist. **Every one of
them was measured at label 0.007**, and the label moved to 0.02 on 2026-09-06:

| arm | label | trades | W1 return |
|---|---|---|---|
| `ora_lbl02` (label change alone) | **0.02** | 1333 | **+26.54%** |
| `ora_stight` (tight band 2.0/−3%/−2%) | 0.007 | 1760 | −13.58% |
| `ora_smid` (3.0/−4.5%/−3%) | 0.007 | 1764 | −17.68% |
| `ora_swide` (6.0/−10%/−6%) | 0.007 | 1714 | −18.68% |
| `oracle_nog` (baseline) | 0.007 | 1760 | −18.89% |
| `ora_tp004/006/010` (take-profit) | 0.007 | ~1700 | −28.06 … −23.34% |

**Decision**: the stop-band question is re-opened, because it has never been
asked at a label that pays. At 0.007 the oracle loses ~19% guards-off and every
stop band is a rearrangement of a losing hand — the best (tight, −13.58%) is
still deeply negative. At 0.02 the same guards-off oracle makes **+26.54%**. A
band tested against a broken label says nothing about the band.

**This is not a contradiction of feature 004** and must not be written as one.
Its conclusions were correct for its configuration; the configuration changed
underneath them. Its take-profit finding — raising the cap made things
monotonically WORSE — is the one result that should transfer, and R3 explains
why it now looks inevitable rather than surprising.

**Alternatives considered**: treating 004 as settled and skipping the stop
entirely. Rejected — that would inherit a conclusion drawn in a regime that no
longer exists.

---

## R2. The instrument-compliance puzzle — DECIDED: bounded measurement task, not an open hunt

Only 84.1% of perfect entries satisfy their own label when reconstructed from raw
candle data. A simple off-by-one is REJECTED: the entry bar gives 84.1%, the
signal bar 81.4%, entry−2 64.1%, so the entry bar is already the best alignment.

Candidate causes, in the order the phase will test them (cheapest first):

1. **The reconstruction is not the strategy's dataframe.** `add_indicators` runs
   before labelling and freqtrade prepends startup candles and trims to the
   timerange, so both bar indexing and the available forward window differ from a
   raw feather read. **Most likely cause.**
2. **Truncated horizon at the window edge.** A bar within 48 of the end cannot
   see its full forward window; the strategy sees a shorter one than the feather
   does.
3. **Fill timing.** `entry_pricing` uses `price_side: "other"` with
   `order_book_top: 1`, so an unfilled limit could open the trade on a later
   candle than the signal.
4. **Label composition** (test to ELIMINATE, not to confirm): NNMT's trading
   label is not the raw gbb signal — `get_trading_classes` applies
   `_filter_trading_by_tasks`, and `get_train_buy_signals` then applies
   `filter_peaks_by_future_performance`. Both are SUBSET operations and therefore
   **cannot add** a non-compliant entry, so they can be ruled out by argument and
   confirmed cheaply.

**Decision**: the deliverable is either a named cause with a correction and a
restatement of the affected figures, or a quantified bound carried into every
number this feature produces (FR-002). The phase does not proceed to the headline
measurement until one of those two exists.

**Why it blocks**: every ceiling figure in this repository inherits this slop.
An exit conclusion built on an instrument that is 16% not-what-it-claims would be
worse than no conclusion.

---

## R3. The payoff geometry may be the whole answer — TEST IT FIRST

Measured effective values, per family (isolated processes — see R8):

| | ATR band | ROI | label target | take-profit | reward : risk |
|---|---|---|---|---|---|
| **NNMT** | **−2% … −4%** (mult 2.5) | 3% | 2.0% | 2.2% | **1:1 to 1:2** |
| **NNNC** | −4% … −6% (mult 4.0) | 5% | 2.0% | 2.4% | 1:2 to 1:3 |

**A perfectly-labelled NNMT trade is certain of a +2.0% excursion and risks 2–4%
to capture 2.2%.** That is a negative-expectancy bet before costs, and it needs
no further mechanism to explain the paradox. On NNNC the geometry is worse still.

This reframes feature 004's refuted take-profit result. Raising the take-profit
cap made things monotonically worse **because the label only guarantees +2%** —
lifting the harvest point above the guaranteed move converts certain winners into
trades that run to a stop. That is now an expected consequence of the geometry
rather than an anomaly.

**Consequence for the plan**: the reward:risk arithmetic is Phase 2's first
measurement, before any mechanism is disabled. It is nearly free and it may make
the rest of the feature unnecessary. **Correction to carry**: the session
analysis reporting "6.6% of entries breach a −4% stop" used NNNC's band on NNMT
data. NNMT's floor is −0.04 and its cap −0.02, so the true breach rate is higher
and must be recomputed per family.

---

## R4. Every exit lever is reachable WITHOUT touching shared framework code

Verified by reading the source:

| lever | mechanism | reachable by |
|---|---|---|
| ATR stop, on/off | `use_atr_adaptive_stoploss` (BaseNNStrategy:145 True) | subclass class attribute |
| ATR band | `atr_stoploss_multiplier` / `_floor` / `_cap` | subclass class attributes |
| take-profit | `cexit_take_profit` | `--set-sell` |
| max hold | `cexit_max_days` | `--set-sell` |
| exit signal + its gate | `enable_exit_signal`, `exit_rvol_threshold` | `--set-sell` |
| ROI table | `minimal_roi` | `--roi` |
| 12-hour unclog | hardcoded in `BaseStrategy.custom_exit` (~1040) | **`NNNC/_hold_window.py` HoldWindowMixin — already exists, DO NOT REBUILD** |

**Decision**: no `Framework/` modification is anticipated or permitted. The
unclog is the only mechanism that is not a parameter, and feature 003 already
built the per-strategy mixin for it precisely to avoid an edit-measure-revert on
a file the live dry run imports.

---

## R5. Decomposition order — DECIDED: joint bound FIRST, then one-at-a-time

Two options for the "patient oracle" that waits for the label horizon:

- **(a) One mechanism at a time**, building up to fully patient. Gives the US3
  attribution for free and never moves two things at once.
- **(b) Joint first** — disable the stop, the ROI, the take-profit and the unclog
  together to establish the bound, then re-enable one at a time to decompose.

**Decision: (b).** The bound is the feature's gating number (FR-005): if a fully
patient exit is worth little, the exit is not the constraint and Phases 3–5 are
re-scoped. Establishing it first risks one arm to answer the question that
decides whether the other arms are worth running at all. The decomposition that
follows is still strictly one-at-a-time, so Constitution VIII is satisfied where
it matters — at the point where a change might be ADOPTED, not at the point where
a bound is being measured.

**Cost**: one extra arm against option (a). **Explicitly noted**: the joint arm
is a bound, never a candidate. Nothing that disables the stop entirely is
promotable under any result.

---

## R6. Resolution for exit arms — DECIDED: paired BY TRADE, not by set membership

Exit arms share their entries and differ only in outcome, so the symmetric-
difference pairing used elsewhere in this programme returns **zero differing
trades** and is the wrong statistic. The correct one keys trades by
`(pair, open_date)` and bootstraps the **per-trade P&L difference** over the
matched set.

Demonstrated on the exit sweep of 2026-09-07: matched 176/176 trades on W1 with
33 of them changing P&L, giving a 1.42pp band — where the membership statistic
would have reported nothing to test.

Both are needed: membership pairing when the entry set changes, per-trade pairing
when it does not. The phase states which it used for each arm.

---

## R7. Window informativeness — report NET/GROSS, not just trade count

| NNMT window | trades | net/gross | usable for |
|---|---|---|---|
| W1 | 176 | **29.8%** | acceptance |
| W2 | 347 | 18.7% | corroboration |
| W3 | 126 | **7.5%** | direction only |

W3 retains 7.5% of its gross flow as profit and its four worst trades exceed the
entire window's profit, so its sign flips under small perturbations — it has
disagreed with W1 on three consecutive exit-side changes for that reason alone
(`regime/W3_INSTABILITY.md`). **Every window in this feature is reported with its
net/gross ratio**, and no sub-threshold conclusion may rest on W3.

---

## R8. An oracle WIN is not evidence FOR an exit change — the mirror of R9

Feature 005's R9 established that an oracle null is not evidence AGAINST an exit
change, because a perfect model only enters bars that reach their target. **This
feature needs the mirror, and it is the sharper risk here**: an exit that waits
patiently is rewarded under perfect entries — where every trade is guaranteed to
reach its target — and punished on a real model, whose entries frequently never
get there. A patient exit is exactly the intervention most likely to look
brilliant on the instrument and lose money in production.

**Decision**: US5 (real-model confirmation) is P1 and inside the feature. No
candidate is promoted on a perfect-entry result, ever. `regime/ORACLE_GATE.md`
rule 0 already records the general form of this caution and gains this specific
case.

---

## R9. Training budget — DECIDED: ZERO retrains

This is an exit feature. Every lever in R4 is a class attribute or a runtime
parameter, and none of them changes the LABEL, so no model needs retraining.
Existing weights are reused throughout via `get_model_path` aliasing.

**One exception, found during analysis and recorded rather than glossed**: NNNC's
CLOSED-gate production (`saved_data/NNNC_MLX/`) has a single model on disk and no
seed replicate, so FR-018's two-seed rule cannot be satisfied there from existing
weights. Every other arm can: NNMT has `Ctl_s42/s7`, `Lbl02_s42/s7` and
`Path_s42/s7`; NNNC's OPEN gate has `MLX_Open_s1/s7`. The feature therefore
spends zero retrains as planned, and a promotion on NNNC's closed gate
specifically would cost one (~7 minutes at NNNC's model size). It is spent only
if a candidate is otherwise promotable there — never speculatively (T022a).

The other circumstance requiring a retrain would be a decision to change the label
in response to R3's geometry finding — which is **out of scope** (label fidelity
is a non-goal, and `gbb_path` was already built, gated and found null on both
families). If the geometry points at the label, that is a FINDING to record and
hand on, not work to do inside this feature.

---

## R10. Isolated-process rule for reading configuration

freqtrade `Parameter` objects are **class-level singletons on `BaseStrategy`**.
Loading two strategies in one process contaminates the second: values set by the
first persist wherever the second does not override them. Loading NNNC then NNMT
reported NNMT's take-profit as 0.024/10 days when it is 0.022/21.

**Every configuration dump in this feature loads ONE strategy per process**, and
`ft_load_hyper_params()` must be called (it normally runs inside `ft_bot_start`),
or the values read are raw parameter defaults rather than the live merged config.

---

## Summary of decisions

| id | decision |
|---|---|
| R1 | Prior exit arms are STALE (all at label 0.007); re-ask the stop question at 0.02 |
| R2 | Compliance is a bounded measurement task with a definite deliverable; it gates everything |
| R3 | Test the reward:risk geometry FIRST — it may be the whole answer, and it is nearly free |
| R4 | Every lever is reachable without a `Framework/` change; reuse HoldWindowMixin |
| R5 | Joint bound first, then strictly one-at-a-time decomposition |
| R6 | Paired BY TRADE for exit arms; membership pairing only when the entry set changes |
| R7 | Report net/gross per window; W3 cannot carry a sub-threshold conclusion |
| R8 | An oracle WIN is not evidence for an exit change — the mirror of 005 R9 |
| R9 | ZERO retrains |
| R10 | One strategy per process when reading configuration |
