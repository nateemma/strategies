# Feature 021 — Label/exit coordination: verdicts and what they imply

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

| trigger | unconditional MCC | **conditional** MCC | drop |
|---|---|---|---|
| `gbb` | +0.5927 | **+0.0489** | −0.5438 |
| `breakout_gbb` | +0.5622 | +0.0468 | −0.5154 |
| `breakout` | +0.5440 | +0.0384 | −0.5056 |
| `adx` | +0.3883 | +0.0380 | −0.3503 |
| `geometry` | +0.3286 | +0.0370 | −0.2917 |
| `empty` *(control: no trigger)* | +0.0342 | +0.0350 | **−0.0008** |

**~92% of `gbb`'s apparent learnability is the model recovering the trigger from its own
inputs.** Tell the model the trigger already fired and essentially nothing predictable remains
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
