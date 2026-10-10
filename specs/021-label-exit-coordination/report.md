# Feature 021 — Label/exit coordination: verdicts and what they imply

**Status**: all 40 tasks closed. **Exit held fixed throughout** (FR-019);
`exit_config_hash 5a18305d1fafa83e` for every number in this document. Every P&L figure comes
from the pooled walk-forward harness in `evaluation/` (FR-017), which resolves ~3–4pp/yr.

---

## SUMMARY — read this and stop, unless you need the evidence

### What was built, and it works

Evaluating a candidate label is now a **query taking seconds instead of a 1.5-hour training
run**. Three durable artefacts:

* **`utils/ExitPolicy.py`** — one exit model. The arithmetic was duplicated **four** ways, not
  the two the spec named; all four are callers; live behaviour provably unchanged over 1,440
  cases. Two real bugs fell out of the offline copy: an **additive fee model** where
  freqtrade's is multiplicative, and a **missing 1-hour minimum-hold veto**. Both were
  silently wrong in every stop-aware label this repository has built.
* **The per-bar outcome table** — what a trade opened at any bar actually returned under that
  one exit. Cached, keyed by an exit-config hash so a changed exit invalidates it, and
  structurally barred from reaching a model as a feature.
* **The trigger registry + enforced inventory** — 29 buy-side `LabelMethod`s classified, with
  a test that fails if a new one appears unclassified.

### The four conclusions

1. **The open question is answered: explanation (b), decoration.** ~90% of every
   trigger-defined label's learnability is the model recovering the trigger from its own input
   columns — **13 of 13 triggers**, mean 9.6% retained (range 6.2–15.1%), with the `empty`
   control at 101.1% and an LSTM at −0.0044 ruling out a linearity artefact.
2. **No trigger is tradeable alone.** Six trigger-only harness arms: five lost **21–71pp**,
   `gbb` was flat (+2.64pp ≈ +0.7pp/yr, below the floor). Includes `geometry`, which the spec
   named as never tested.
3. **Selectivity does not rescue them.** Tail precision at the top 1% **inverts between
   windows** (A vs B rank correlation **−0.648, p = 0.017**); 10 of 13 are sign-unstable and
   the 3 stable ones are exactly what chance predicts (P = 0.216).
4. **So as labels they are equivalent, and none carries stable individually-exploitable
   signal under this exit.** Not "worthless" — *indistinguishable*.

### What was NOT achieved

* **The economic-value column does not work as a label screen.** Sound as a statistic,
  **1 of 6 on sign** against live arms, and its verdict population (fillable bars only) is not
  one a capital-constrained strategy enters. `selection_value` is withdrawn as a verdict word.
* **The instrument was never validated against the expensive ground truth.** We never showed
  that probe ranking predicts trained-model harness ranking. The one column that *was* checked
  against reality failed.
* **The study is blind to outcome-definition differences by design** — 11 of 29 labelers
  differ only in their outcome test, so under one shared outcome they are the same label.

### The unresolved puzzle, and it is the important one

`gbb` + model + per-fold threshold earns **+39.98pp in 15 of 15 folds**. Nothing measured here
explains it: not the trigger (flat alone), not average learnability (collapses), not tail
ranking (unstable). Two clues point the same way — the `empty` control's tail lift is
**+0.112/+0.117 in windows B and C at z ≈ +10**, so outcome predictability exists but is *not
trigger-specific*; and the incumbent **re-selects its threshold per fold on that fold's own
training window**, which is exactly what a fixed-split probe cannot do. The hypothesis this
study cannot test: the signal is real but only exploitable **adaptively**. See
`followup-perfold-guard-tuning.md`.

### Retractions — claims made during this feature and later withdrawn

| claim | why withdrawn |
|---|---|
| "Two independent arms show the board's EV column inverted" | the `gbb` +39.98pp arm is a **model** arm at 1.4735% of bars vs the trigger's 14.19% — never a valid anchor |
| "The inversion is a weighting artefact; the board is not anti-predictive" | six arms later: **1/6 on sign**. Stake-weighting agreed with the board because both up-weight liquid bars — a **shared defect, not corroboration** |
| "The EV column has *no predictive validity*" | over-claimed at n=6; the four rows are ~2 independent conditions. The defensible claim is the **population** one |
| "The mechanism is selectivity — guards and threshold cutting 14.19% of bars to 1.47%" | the incumbent ran **guards-OPEN** (`spec.md:13`); the 1.4735% is the **model threshold alone** |
| "The guarded no-model arm will be unpowered noise" | wrong: **+18.76pp (in-sample — 97.8% from folds inside its own tuning window; see 022/provenance.md), 12/15 folds**, ~+5.1pp/yr, above the floor |
| "The breakout family carries tail signal; `breakout_gbb` clears both bars" | **regime ghost** — −0.088 at z = −4.90 in window B |
| "Break-even precision is ~0.25" | **~0.58** here; average loss is ~1.4× average win. Cited from memory before being computed |
| "`volume_filter_active: True` shows the filter was on" | that field was a **hardcoded literal**; the conclusion survived on other grounds, the evidence did not |
| "Concurrency is immaterial" | true only for a **sparse** trigger; decisive for a dense one |

### Reading guide

The sections below are **in the order the findings arrived**, not in order of importance, and
several contain reasoning later corrected by a section further down. The table above is
authoritative where they conflict.

---

**Status**: complete. **Date**: 2026-10-09. **Exit held fixed throughout** (FR-019);
`exit_config_hash 5a18305d1fafa83e` for every number below.

This feature was commissioned to settle one question. Across four arms, `val_mcc` tracked
pooled return almost perfectly (0.204 / 0.521 / 0.580 / 0.622 against −0.8 / +21.8 / +35.4 /
+40.0pp), and two incompatible explanations predict exactly that ordering:

* **(a) selection** — a trigger picks moments when a forward move is collectable;
* **(b) decoration** — a trigger hands the model an easy target it can partly read off the
  trigger's own input columns, so the apparent edge is the entry rule and the model is ornament.

They imply opposite strategies and no experiment then run could separate them.

---

## The answer: (b), and it is not close

One number separates them. The probe was fitted twice on identical features, splits, seeds
and protocol — once on the label `trigger AND positive outcome`, once **conditionally**, on
`positive outcome` among the bars where the trigger already fired.

**EVERY scoreable trigger, 3 windows x 3 seeds each** (9 unconditional / 18 conditional fits
per row; the unconditional column was backfilled 2026-10-09 after it was found to be stale —
the original campaign ran when only 6 triggers were registered):

| trigger | unconditional MCC | **conditional** MCC | drop | learnability retained |
|---|---|---|---|---|
| `gbb` | +0.5927 | +0.0504 | −0.5424 | 8.5% |
| `votes_tech` | +0.5906 | +0.0466 | −0.5440 | 7.9% |
| `breakout_gbb` | +0.5622 | +0.0498 | −0.5124 | 8.9% |
| `breakout` | +0.5440 | +0.0465 | −0.4975 | 8.6% |
| `votes_ind3` | +0.5387 | +0.0405 | −0.4981 | 7.5% |
| `votes_ind4` | +0.5142 | +0.0320 | −0.4823 | 6.2% |
| `votes_consensus` | +0.4600 | +0.0521 | −0.4079 | 11.3% |
| `breakout_bb` | +0.4358 | +0.0656 | −0.3702 | 15.1% |
| `votes_ind2` | +0.4206 | +0.0408 | −0.3797 | 9.7% |
| `adx` | +0.3883 | +0.0364 | −0.3518 | 9.4% |
| `breakout_squeeze` | +0.3798 | +0.0358 | −0.3439 | 9.4% |
| `breakout_vol` | +0.3738 | +0.0516 | −0.3222 | 13.8% |
| `geometry` | +0.3286 | +0.0285 | −0.3001 | 8.7% |
| **`empty`** *(control: no trigger)* | +0.0342 | +0.0346 | **+0.0004** | **101.1%** |

**13 of 13 triggers collapse, with no exception. A mean of 9.6% of apparent learnability
survives conditioning on the trigger (range 6.2–15.1%), while the control retains 101.1%.**

So roughly **90% of every trigger-defined label's learnability is the model recovering the
trigger from its own input columns** — and the consistency is the striking part: the same
~90% holds across mechanisms as different as momentum breakouts, a Bollinger condition,
four indicator-vote schemes, an ADX filter and a DWT-smoothed geometric valley.

`votes_tech` is the purest case in the study: the **second-highest unconditional learnability
of any trigger (+0.5906)** and the **worst economic value on the board** (−0.003605, now
`worse_than_null`). A label that trains beautifully and selects nothing. Tell the model the trigger already fired and essentially nothing predictable remains
— +0.0489, against +0.0342 for predicting the outcome at a *random* bar. The `empty` control
is what licenses the reading: with no trigger to recover, conditioning changes nothing
(−0.0008), so the drop is not an artefact of the smaller conditional sample or of the
different base rate.

**Not a linearity artefact.** An LSTM scout on the same conditional problem scored
**−0.0044** over 6 seeds — a non-linear model does no better than the ridge, and no better
than chance. This repository's standing finding that *an unlearnable target is a DATA problem,
not a model problem* applies: the information is not in these columns.

---

## The leaderboard, and a verdict on the verdicts

| trigger | verdict | base rate | EV (fillable) | 95% CI vs null | cond MCC |
|---|---|---|---|---|---|
| `breakout_gbb` | selection_value | 12.49% | +0.000692 | [+0.001284, +0.002675] | +0.0468 |
| `breakout_bb` | selection_value | 6.08% | +0.001546 | [+0.002032, +0.003650] | +0.0526 |
| `breakout_vol` | selection_value | 2.86% | +0.000582 | [+0.000930, +0.002783] | +0.0400 |
| `breakout` | selection_value | 6.45% | +0.000807 | [+0.001361, +0.002802] | +0.0384 |
| `votes_consensus` | **decoration** | 23.56% | −0.001183 | [−0.000453, +0.000633] | +0.0516 |
| `gbb` | indeterminate | 17.87% | −0.002009 | [−0.001468, +0.000004] | +0.0489 |
| `votes_tech` / `ind2` / `ind3` / `ind4` | indeterminate | 19.9–51.4% | −0.0031…−0.0036 | all below zero | +0.034…+0.045 |
| `adx`, `geometry`, `breakout_squeeze` | indeterminate | 2.3–61.2% | −0.0005…−0.0015 | span zero | +0.029…+0.038 |
| `gbb_stale`, `random_matched` *(controls)* | indeterminate | 17.7–17.9% | −0.0014 / −0.0009 | span zero | n/a |
| `empty` | (null) | 100% | −0.001272 | — | +0.0350 |
| `gbb_tight` | (diagnostic, FR-015b) | 0.07% | +0.030219 | [+0.019277, +0.044113] | n/a |
| `trends`, `bands` | UNSCOREABLE | — | — | — | — |

**Both controls land `indeterminate`, which is the board passing its own liveness check**
(`random_matched` at a matched base rate, `gbb_stale` on a deliberately stale signal). A board
that promoted either would have been reporting its own noise.

### The economic-value column does not survive contact with an arm

Four rows scored `selection_value`, all in the breakout family. Two triggers have since been
**traded alone** — no model, guards open, same exit — and the column is **0 for 2**:

| column *(scored before either arm ran)* | `breakout_vol` | `gbb` | signs |
|---|---|---|---|
| board: fillable, per-pair non-overlap | +0.000582 ✗ | −0.002009 ✗ | **0/2** |
| concurrency-aware, UNFILTERED, equal-weighted | −0.001548 ✓ | +0.001508 ✓ | **2/2** |
| stake-weighted by `__outcome_fillable_fraction` | +0.000279 ✗ | −0.000921 ✗ | **0/2** |
| **realised** (15 pooled folds each) | **−0.001512** | **+0.000155** | |
| **wallet** | **−63.19pp** | **+2.64pp** | |

Read as intended — take the top row and trade it — **the board picked the arm that lost 63pp
over the one that was flat.** The cause is population, not signal: the board scores only
*fillable* bars, while `custom_stake_amount` **caps** the stake on a thin candle rather than
rejecting the trade (`min(proposed, fillable)`, same `QUOTE_VOLUME_HEADROOM_MULT` the table
uses), so a strategy also enters bars the board never scored — and mean fillable fraction is
only **0.18–0.34** for every trigger.

A trap worth recording: stake-weighting *agrees* with the board (ρ = +0.890 over 17 triggers),
and that agreement was briefly read as validating it. Both columns up-weight liquid bars, so
**their agreement is a shared defect, not corroboration.** Two wrong columns agreeing look
exactly like two right ones; only an arm broke the tie.

---

## What this implies downstream

1. **Trigger discovery is not the next line of work.** The feature was framed so that
   `selection_value` would open it. The four rows that scored `selection_value` did so on a
   population no strategy enters, and the one tested lost 63pp. Nothing here licenses a search
   for better triggers using this column as the screen.
2. **The model is ornamental on trigger-defined labels, and that is now measured rather than
   inferred.** It corroborates, by a different route, the standing findings *guard ablation:
   model adds NOTHING* and *gate/mechanism move P&L, not model* (7 interventions, 2 families,
   every win gate-side, every null model-side). What 021 adds is the mechanism: the model's
   skill is recovering the entry rule, and conditioned on the rule there is ~nothing left.
3. **`gbb` trigger-only is a null, not a winner.** +2.64pp over 3.68 yr-equivalent is
   ~+0.7pp/yr — **below the harness's 3–4pp/yr floor** — with 7/15 folds positive. The robust
   claim is the *ordering* (`gbb` ≫ `breakout_vol`), because −63.19pp sits far outside that
   floor. `gbb`'s own sign is unresolved.
4. **Concurrency is payoff-density specific.** It is immaterial for a sparse trigger
   (`breakout_vol`, 2.86% of bars, ρ = +0.912 against the board on the same population) and
   decisive for a dense one (`gbb`, 17.87%, where `profit_pct` +1,035 against wallet +2.64pp
   *is* the capital constraint). Do not generalise a concurrency result across densities.
5. **`votes_consensus` is the clean example of explanation (b)**: EV indistinguishable from
   the null, second-highest conditional learnability on the board. A label that trains well
   and selects nothing.

---

## Caveats that must travel with these numbers

### The probe UNDERSTATES learnability for trigger-defined labels (FR-015a, research R8)

The fixed probe is ridge-regularised logistic regression, and **ridge cannot express a logical
AND**: it scored **0.533 against the `gbb` trigger rule's own 0.673**. Every trigger-defined
label is a conjunction, so the probe is biased *down* on exactly the labels this feature
scores, and the conditional MCCs above are **not ceilings**.

This does not affect any verdict — both sides of every comparison use the same probe, and the
`empty` control is measured with it too — but a reader could otherwise take +0.0489 as proof
that nothing is learnable within `gbb`. The honest statement is narrower: *nothing is
learnable within `gbb` by a linear model on these 24 indicator columns at `seq_len` 16*, with
an LSTM scout at −0.0044 as the only evidence against the architecture excuse.

### Economic value is a LOOKAHEAD BOUND, never a return (FR-017)

Every EV figure from the outcome table is computed with knowledge of the forward path. That is
legitimate for label construction and **illegitimate as a return claim**. The table's own
`report()` now says so, names the population it scores, and carries the 0/2 record above.

**Any P&L claim in this feature comes from the pooled harness in `evaluation/`**, which
resolves roughly **3–4pp/yr and not 1–2pp**. Both the −63.19pp and the +2.64pp above are
harness numbers; no EV figure is quoted as a return anywhere.

### Classifier metrics are context, never an acceptance input (FR-018, FR-015a)

The conditional-MCC column is diagnostic. It licenses **no** promotion, and it is never
combined with the economic-value column into a single score (FR-016). This feature's central
result is itself a classifier metric — the +0.5927 → +0.0489 drop — and it is offered as an
*explanation* of measured returns, not as evidence that anything should be deployed. Per
`regime/ORACLE_GATE.md` rule 1, the decision to trade anything rests on the harness alone.

### `trends` and `bands` could not be scored, which is itself a finding

`trends` needs four regime-state columns absent from the feature frame; `bands` needs eight
band columns of which six are absent. Both are declared UNSCOREABLE rather than ported at
reduced fidelity — a partial port would have produced a number that looked comparable and was
not. (An earlier `bands` port silently dropped 6 of 8 columns and was caught before it
reached the board.)

### A quarter to over a third of simulated exits fill AFTER a gap

`gapfill_share` runs **24.3%–37.7%** across every row. Those are exits where price gapped
through the stop level and the fill is taken at the next open, which the table models but
which is the least reliable part of any simulated exit.

It also partly co-varies with the column it sits beside: **Spearman(EV_fillable,
gapfill_share) = −0.475, p = 0.054** over 17 rows. The breakout family that tops the board
fires on less gap-prone bars (27–31%) than the vote family at the bottom (35–37%). So some of
what the EV column ranks is **how gap-prone a trigger's bars are** — an execution-realism
property rather than a selection property. `geometry` is the counter-example that keeps the
association from being stronger (lowest gapfill of any real trigger at 25.6%, EV −0.001526),
so this is a contributing confound, not the whole story, and at p = 0.054 on n = 17 it is
suggestive rather than established.

Recorded because it is the second mechanism found by which this column ranks something other
than selection value, the first being the fillable-population error above. Both point the same
way: the column is sensitive to execution properties of a trigger's bars, and a leaderboard
built on it ranks those properties alongside — or instead of — the thing it names.

---

# SIX trigger-only arms (2026-10-09) — the decisive section

Every trigger that mattered has now been traded alone: all four `selection_value` rows, plus
`geometry` (which the spec named as never tested) and `gbb`. One base class, one exit
(`exit_config_hash 5a18305d1fafa83e`), guards open, no model, same fold plan — so the arms
differ from each other in the trigger and nothing else.

| trigger | board | conc/ALL | stake-wt | **REALISED** | **wallet** | pos | trades |
|---|---|---|---|---|---|---|---|
| `breakout_bb` | +0.001546 | −0.001710 | +0.001266 | **−0.000628** | **−21.29pp** | 4/15 | 7,007 |
| `breakout` | +0.000807 | −0.001196 | +0.000623 | **−0.001608** | **−70.55pp** | 4/15 | 9,354 |
| `breakout_gbb` | +0.000692 | −0.001497 | +0.000488 | **−0.000607** | **−25.68pp** | 5/15 | 8,106 |
| `breakout_vol` | +0.000582 | −0.001548 | +0.000279 | **−0.001512** | **−63.19pp** | 6/15 | 8,290 |
| `geometry` | −0.001526 | −0.001380 | −0.001558 | **−0.001118** | **−32.90pp** | 4/15 | 4,781 |
| `gbb` | −0.002009 | +0.001508 | −0.000921 | **+0.000155** | **+2.64pp** | 7/15 | 8,150 |

## Finding 1 — NO trigger is tradeable alone

Five of six lost **21 to 71 percentage points**. The sixth, `gbb`, returned +2.64pp over 3.68
yr-equivalent — ~+0.7pp/yr, **below the harness's 3–4pp/yr floor**, 7/15 folds — so it is a
null, not a winner. **There is no trigger here that trades.**

This closes Deliverable 3's open item: `geometry`, the DWT-smoothed zero-lag valley condition
implemented but never tested, **loses 32.90pp**. It is tested now.

## Finding 2 — the board's economic-value column has no predictive validity

**All four `selection_value` rows lost money**, between −21.29pp and −70.55pp. The column is
**1 of 6 on sign** — and the one it got right was `geometry`, which it had scored
`indeterminate`. Rank correlation with realised per-trade EV is **−0.371 (p = 0.47)**:
negative in point estimate, not significant at n = 6, and certainly not positive.

The stake-weighted variant is **also 1 of 6** (ρ = −0.200, p = 0.70), confirming that its
earlier agreement with the board (ρ = +0.890) was two columns sharing one defect rather than
two columns being right.

## Finding 3 — the column that is 6/6 on sign still cannot RANK

`conc/ALL` called every sign correctly, including uniquely identifying `gbb` as the only
positive trigger — a column that simply said "all negative" would have scored 5/6, so this is
a real hit, not an artefact of the base rate. **But it does not order them: ρ = +0.143
(p = 0.79).** It ranks `breakout` second-best; `breakout` is in fact the worst. So the honest
summary is that it separates sign and nothing finer, and no column on this board ranks
triggers by realised return.

## Finding 4 — the conjunction composes

`breakout_gbb` (−0.000607) sits between its components, `breakout` (−0.001608) and `gbb`
(+0.000155). Adding the gbb condition to a breakout moves the result toward gbb rather than
producing anything the parts do not have.

## THE CAVEAT THAT BOUNDS ALL OF THIS — guards were OPEN

Every arm above ran `entry_enable_guards: False`. The incumbent's +39.98pp / 15-of-15 runs
**with** the guard chain and a per-fold model threshold at **1.4735% of bars**; these arms
trade 2.3–17.9% of bars with no guards and no model. **So these results do not refute the
incumbent** — they show that the *trigger alone, unguarded* is not tradeable, which is a
different and narrower claim.

It also sharpens where the incumbent's return must come from. The trigger alone is flat
(`gbb`, +2.64pp). Within the trigger almost nothing is linearly learnable (conditional MCC
+0.0489, LSTM −0.0044). Yet `gbb` + guards + a thresholded model earns +39.98pp. By
elimination the mechanism is **selectivity — the guards and the threshold cutting 14.19% of
bars down to 1.47%** — not the trigger and not general learnability. That is consistent with
the standing findings *guards bind, not the z-gate* (the guard chain rejects ~97%) and
*gate/mechanism move P&L, not model*.

**The missing arm, and the obvious next experiment**: trigger + guards, NO model. Same
scaffold, `entry_enable_guards: True`. If that recovers a large share of +39.98pp, the model
is ornamental and the guards are the strategy. If it does not, the model's threshold is doing
real work that conditional MCC over the whole trigger population fails to see — which is
plausible, because precision in the extreme tail is not MCC, and this repository has already
measured that break-even precision is only ~0.25.

---

# T035 gate review — accepted, with four fixes applied (2026-10-09)

The review was delegated rather than self-performed, because the point of an independent
gate is independence from whoever built the board. Its four checks: base rates **PASS**;
gap-fill **FAIL on the fillable half as vacuous** and unverifiable at trade level; block
bootstrap **PASS** with a provenance gap; controls **informative**, with `gbb_stale` doing
the real work. Four defects it found are now fixed.

## FIX 1 — `selection_value` is withdrawn as a verdict label

Renamed to **`beats_null_fillable`**. The statistic is sound — it survives the signal-bar-leak
fix, calendar blocks and block length 960 — but the *word* implied tradeability, and all four
rows carrying it lost 21–71pp. The label now says exactly what the test establishes: this
trigger beats the `empty` null over fillable, non-overlapping bars at unit stake with
unlimited pairs. The same table scores all four of those rows **negative over all bars**, so
the result rests on one filter choice.

## FIX 2 — FR-014b AMENDED: the learnability bar, and whose SE it uses

Two defects in one branch:

* the SE was the **trigger's own** fold SE, never combined with the reference's — so it was
  not "the SE on that difference" the contract names, understating it by up to √2. Now
  combined in quadrature (independent probe fits).
* the bar was **one SE** — a ~68% test against the economic-value column's 95%, making
  `decoration` strictly easier to earn. Now **1.96 SE**, matched.

**This changes a published verdict.** `votes_consensus` clears the old bar at z = 1.35 as one
of 13 tests, where ~2 false positives are expected, while the whole conditional-MCC column
spans 0.03–0.05. It moves from `decoration` to `indeterminate`. **So the board's one example
of explanation (b) by verdict is withdrawn** — the evidence for decoration now rests entirely
on the conditional-MCC collapse (13 of 13 triggers, ~90% of learnability lost), which is much stronger than any single
row's verdict and is unaffected by this amendment.

## FIX 3 — a new `worse_than_null` verdict

`indeterminate` was absorbing rows whose interval lies **wholly below zero** —
`votes_tech`/`ind2`/`ind3`/`ind4` — filing them with the controls. "Not shown to beat the
null" and "shown to be worse than the null" are different findings. The branch sits *after*
the learnability test, so a trigger that is unprofitable but genuinely learnable still reads
`decoration`, which is the more informative statement about it.

## FIX 4 — two provenance fields that asserted what they never measured

`regime/run_arm.py` recorded `"guards": args.guards`, the **CLI flag**, default `"on"`. So all
90 trigger-only arm records read `guards: on` while `TriggerOnly.py` hardcodes
`entry_enable_guards: False`. **The effective setting was OFF and the record said ON** — an
auditor reading records alone would conclude the arms tested the trigger *plus* the guard
chain, the exact opposite of what they tested. Replaced by `guards_requested` (the flag) and
`entry_enable_guards_written` (what was written, or `None` meaning the class default applies,
which this script deliberately does not import rather than guess).

Worse, and found in the same pass: **`"volume_filter_active": True` was a literal.** Nothing
checked it and it could not be false — and **I cited it as evidence** earlier in this feature
when refuting the "the volume filter was off" hypothesis. That conclusion survives, because it
was independently established from `custom_stake_amount`'s own code, but the evidence I quoted
for it was worthless. Replaced by a prose provenance string that says where the check lives
and that this script does not measure it.

Also removed: `research/outcomes/verdicts.json`, a pre-leak-fix artefact still on disk
contradicting the current board (it had `geometry` and `breakout_squeeze` as selection_value
with ~5× larger deltas). Recoverable from commit b648536.

## STILL OUTSTANDING — the 90 existing arm records

The generator is fixed; **the 90 already-written records still carry the false
`guards: on`**. They have not been rewritten, because they are a faithful record of what that
run emitted and silently editing committed result files is its own provenance problem. Anyone
reading `regime/arms/wf_*_f*.json` from before 2026-10-09 must treat the `guards` field as
meaningless and consult the strategy class.

## WHERE THE REVIEW SAYS I OVERREACHED, and I accept it

Finding 2's "the EV column has **no predictive validity**" is over-claimed at n = 6. The four
rows are not independent conditions — `breakout` is `breakout_vol`'s base, and `breakout_bb`
and `breakout_gbb` share `bb_width > 0.035` — so 4/4 wrong is roughly **two** effective
failures, and a Spearman at n = 6 resolves nothing in either direction. The defensible claim
is the **population** one, which needs no significance test: the verdict population is not
enterable by a capital-constrained strategy, and the population that is loses. I was not too
harsh on the column as a trading screen — FR-017 already forbade that use — but I was too
harsh on it as a measurement.

---

# Deliverable 3's missing half: what happened to the other label types

Deliverable 3 asked to express the `LabelMethod` family's conditions as triggers "**and report
which do not and why**". The first half was done — 15 triggers registered, 2 declared
unscoreable. The second half was not, which is why it looked as though two thirds of the
label family had silently dropped out of the study.

`Framework/TrainingSignals.py` holds **51 `labels_*` functions, 29 of them buy-side.** All 29
are accounted for below, and the accounting is enforced by
`research/outcomes/tests/test_labelmethod_inventory.py`, which fails if a new labeler appears
unclassified (proven to fail on a planted one, then restored).

## A — expressed as a trigger and scored: 14

`gbb`, `geometry`, `breakout`, `breakout_vol`, `breakout_squeeze`, `breakout_bb`,
`breakout_gbb`, `breakout_consensus`, `technical_indicators`, `indicators2/3/4`, plus
`trends` and `bands`, both **UNSCOREABLE** — the columns their conditions need are absent
from the feature frame, and declaring that beats porting at reduced fidelity (an earlier
`bands` port silently dropped 6 of 8 columns and produced a number that looked comparable
and was not).

## B — same trigger, different outcome test: 3

`gbb_path` → `gbb`, `gbb_fwdmae` → `gbb`, `breakout_tb` → `breakout`. Not missing: their
entry condition is already on the board and the only thing that differs is the forward test
this design replaces. `labels_gbb_fwdmae` is the clearest case — its body is literally
`labels_gbb(...) & labels_forward_return_mae_cap(...)`.

## C — no entry condition at all, pure forward/outcome tests: 11 → ALL become `empty`

`forward_return_mae_cap`, `triple_barrier`, `quantile_future_return`, `mfe_mae_ratio`,
`risk_adjusted_future_return`, `future_sharpe`, `future_sortino`, `future_expectancy`,
`multi_horizon_vote`, `optimal_signals`, `local_extrema`.

**This is the study's blind spot, and it is structural rather than an omission.** Each of
these is a different answer to *"what counts as a good outcome"* — a forward MAE cap, a
triple barrier, a forward Sharpe, Sortino, expectancy, a return quantile, an MFE/MAE ratio, a
vote across horizons, the oracle, a forward-window minimum. **None carries an entry
condition.** The design replaces every outcome with one shared definition derived from the
real exit, so under it these eleven are not merely unscored — **they are the same label**: the
no-trigger case. No further work on this instrument will separate them, because the thing
that distinguished them is exactly what the design removed.

Note the awkwardness this creates for the feature's own premise.
`labels_forward_return_mae_cap` **is** `forward_mae`, half of the motivating comparison
(`gbb` +39.98pp against `forward_mae` −0.83pp). Under this design that comparison becomes
"a trigger versus no trigger", which the board does reproduce — `empty` scores the worst
economic value on it, −0.001272 — but **by construction rather than by measurement.**

## D — a condition that reads forward, so not a trigger: 1

`local_min_followthrough`: `close[t] == min(close[t−3 : t+4])` is **centred**, so it peeks
3 bars ahead. A trigger must be computable at bar `t` from bar `t`'s information. This is
perfectly legitimate in a *label* — labels may look ahead — it simply is not expressible as
the entry rule this design varies.

## What this means for the instrument's reach, stated plainly

**This study assesses ENTRY conditions cheaply, and is blind to outcome-definition
differences by design.**

So the honest scope of every conclusion in this report is: *of the labels that differ in
their entry condition*, none is differentially learnable once you condition on the trigger
(13 of 13 triggers, ~90% of learnability lost), and none trades on its own (six arms, −70.55pp to
+2.64pp). About the eleven labels that differ only in their outcome test, this study says one
thing and one thing only — they are the no-trigger case — and anyone wanting them separated
needs a different instrument, one that varies the outcome definition and holds the trigger
fixed. That is the exact mirror of this feature, and it is not what was built.

---

# The tail-precision test: does selectivity rescue these labels? NO

Conditional MCC is a population average, and a model's threshold trades the extreme tail. A
little signal concentrated in the top 1% can pay while leaving MCC at zero, so the collapse
finding alone does not establish that these labels are useless. This is that test, run over
all three windows because a single window cannot distinguish an edge from a regime.

Method: fit the conditional probe (triggered bars only, outcome as the label, so the trigger
is not in the label), rank its test-set scores, measure precision in the top 10/5/1%.
3 seeds x 3 windows per trigger.

## Two corrections to things stated earlier in this feature

**Break-even precision here is ~0.58, NOT the ~0.25 carried in repo memory.** Computed per
trigger from its own payoff: `p* = -E[loss] / (E[win] - E[loss])`. Average loss is ~1.4x
average win (take-profit 2.2% against stops of -4%/-5%), so a ~58% hit rate is required. The
~0.25 figure belongs to a different label/exit configuration and does not transfer. It was
cited from memory before being computed, which is the error.

**The probe's tail is not a neutral instrument, and its bias is REGIME-DEPENDENT.** The
`empty` control — no trigger at all — has raw tail lift:

    window A  -0.040      window B  +0.112      window C  +0.117

So in A the probe's top-scored bars are *worse* than average (z = -5.60), and in B and C they
are substantially *better* (z ~ +10). Every trigger's lift must therefore be read against its
own window's control, and raw lifts are meaningless.

## The result: control-adjusted lift at top 1%

| trigger | A | B | C | |
|---|---|---|---|---|
| `breakout_gbb` | **+0.119** | **−0.088** | +0.034 | mixed |
| `geometry` | +0.109 | +0.077 | +0.022 | +++ |
| `breakout` | +0.088 | +0.040 | +0.005 | +++ |
| `breakout_bb` | +0.070 | +0.017 | +0.084 | +++ |
| `breakout_squeeze` | +0.064 | −0.060 | −0.086 | mixed |
| `votes_consensus` | +0.047 | −0.026 | −0.018 | mixed |
| `breakout_vol` | +0.039 | +0.044 | −0.058 | mixed |
| `votes_ind4` | −0.004 | +0.139 | +0.018 | mixed |
| `gbb` | −0.018 | **+0.172** | +0.068 | mixed |
| `adx` | −0.025 | +0.010 | +0.009 | mixed |
| `votes_tech` | −0.031 | **+0.220** | +0.041 | mixed |
| `votes_ind3` | −0.038 | **+0.215** | +0.071 | mixed |
| `votes_ind2` | −0.060 | +0.108 | −0.002 | mixed |

    rank correlation, control-adjusted lift:
        A vs B   rho -0.648   (p = 0.017)   <- SIGNIFICANTLY ANTI-correlated
        A vs C   rho -0.071   (p = 0.817)
        B vs C   rho +0.445   (p = 0.128)

    sign-unstable across the three windows: 10 / 13
    positive in all three: breakout, breakout_bb, geometry
        -> but 13 triggers x (1/8) = 1.6 expected by chance; 3 observed; P(>=3) = 0.216.
           NOT significant. Three-of-thirteen is what chance predicts.

**No window pair shows a significantly positive rank correlation, and A vs B is significantly
NEGATIVE.** A trigger's tail behaviour in one window actively mispredicts it in a temporally
distant one.

## RETRACTION

On window A alone I reported that the breakout family carried tail signal and that
`breakout_gbb` was the one trigger clearing both the raw and control-adjusted bar (+0.119,
z +3.40). **In window B it is −0.088 at z = −4.90.** That reading was a regime ghost and is
withdrawn. It is precisely the failure mode this repository's persistence rule exists to
catch, and I reported the single-window result before running the check.

## So: are they all bad labels?

**As labels, they are equivalent, and none carries stable individually-exploitable signal
under this exit.** Three independent measurements agree:

1. conditional learnability collapses ~90% for 13 of 13 (the ~9.6% retained is trigger
   recovery, with the `empty` control at 101%);
2. no trigger trades alone — six harness arms, −70.55pp to +2.64pp;
3. tail precision does not persist — no positive rank correlation between any window pair,
   10/13 sign-unstable, and the 3 stable ones are chance.

"Bad" should be read as **indistinguishable and not individually exploitable**, not as
worthless — two things cut against the stronger reading:

* **`gbb` + model + threshold earns +39.98pp in 15/15 folds.** 15-of-15 is not
  regime-dependent, so something real is there that none of these measurements captures.
* **The `empty` control itself lifts +0.112 / +0.117 in B and C at z ~ +10.** A model fitted
  with NO trigger can rank outcomes in the tail in two of three windows. So outcome
  predictability exists — it simply is not trigger-specific, which is why no label is better
  than another.

Those two together suggest a hypothesis this study cannot test: **the incumbent re-selects its
threshold per fold, on each fold's own training window.** A model that re-tunes every fold can
ride a signal whose sign flips between regimes; a probe with one fixed split cannot see it.
If that is right, the instability measured above is not evidence against the incumbent — it is
the reason the incumbent needs per-fold selection to work, and it is the thing to test next.

---

## CORRECTION (2026-10-10, found while implementing feature 022)

This report states in two places that `entry_enable_guards: False` disabled the `volume > 0`
data-sanity check, and explains the identical `trades_hash` by `confirm_trade_entry`'s own
headroom check acting as "a complete backstop".

**Both claims are wrong.** `volume > 0` was never disabled. Commit `e7cf8cf` (2026-09-08,
"volume+rvol mandatory") moved `volume > 0` and `rvol` OUTSIDE the `entry_enable_guards`
block in `populate_entry_trend`; only the five quality thresholds sit behind the flag.
BaseStrategy's `buy_params` comment still asserts the old behaviour and is stale — I read the
comment rather than the executable path, and cited it as current fact.

**What changes**: nothing measured. The six trigger-only arms are unaffected, and the
identical `trades_hash` is now *better* explained — the condition was present in both runs.
**What changes is the reasoning**: the "defect" was a convention mismatch, not a lost
tradeability check, and the 34.26% zero-volume figure, while true of the dataset, had no
bearing on the flag.

---

# FOLLOW-UP OUTCOME (T039): feature 022 ran, and it closed the guard thread

`followup-perfold-guard-tuning.md` proposed per-fold guard tuning as the explanation for a
model-free guard arm earning ~+5.1pp/yr. **Feature 022 ran it and the thread is closed.**

## The premise it was built on was an artefact

The +18.76pp that motivated the follow-up is **97.8% in-sample**. Its guard values were
hyperopted on `20240511-20260501` (`strategy_NNNC_CGP_MLX_2026-05-31_15-03-45.fthypt`, 6-of-6
exact match), a window covering 9 of the 15 folds and 53.5% of all test-window days. Inside that
window: +18.34pp over 8 folds. Outside it: **+0.42pp over 6 folds** — +0.07pp/fold against
arm A's +0.18pp, i.e. **tuned guards are not better than open guards out of sample.**

So the figure quoted in this report as "+18.76pp (in-sample — 97.8% from folds inside its own tuning window; see 022/provenance.md), 12/15 folds, ~+5.1pp/yr, above the floor"
must be read with that status attached. It was never a resolved result: 198 trades over 15 folds
is roughly half the 378 the harness needs, and it cleared the 25-trade floor in only 5 of 15.

## Adaptivity failed at both granularities

| scheme | result |
|---|---|
| per-fold tuning | **`indeterminate`** — 9/15 folds powered, below the pre-registered 10 |
| two-state (volatility regime, pre-registered, leave-one-out) | **degenerates to a constant**, and that constant loses to a fixed value (paired +0.0110, CI [+0.0002, +0.0217]) |

The fifth adaptive scheme to lose to a static value in this project. The remaining guard
question is *which constant*, not whether to switch — and the robust/maximin setting
(`atr_pct` open-to-near-open, `adx` open, `rvol` ~1.0) is the only configuration measurable in
all 15 folds, with a worst fold 5× milder than the mean-maximising one.

## What this does to feature 021's open puzzle

021 ended with: *`gbb` + model + per-fold threshold earns +39.98pp in 15/15 folds and nothing
measured explains it.* The hypothesis offered was that the signal is real but regime-dependent
and so only exploitable adaptively.

**022 removes that hypothesis.** Adaptivity on the guard space is not available — not merely
unprofitable, but not even trackable or differentiable. And the incumbent's +39.98pp is **not**
in-sample contaminated: its per-fold thresholds were verified as 15 distinct values selected on
each fold's own training window.

So the puzzle survives both features and has narrowed: it is **model-side, not guard-side**.
What remains unexplained is specifically how a thresholded classifier selecting ~1.47% of bars
earns +39.98pp at 15/15 when (i) the trigger alone is flat, (ii) conditional learnability within
the trigger collapses ~90% for 13 of 13 triggers, (iii) tail precision inverts between windows,
and (iv) the guard chain offers no tunable or adaptive edge. That is the one live thread out of
both features.
