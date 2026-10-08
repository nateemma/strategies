# Contract: the trigger registry and leaderboard

**FR-010 through FR-016.**

## Commands

```
python -m research.outcomes.triggers --list
python -m research.outcomes.leaderboard --build [--triggers a,b,c]
python -m research.outcomes.leaderboard --show
```

## Registry

A trigger registers a name, a callable `(dataframe) -> bool series`, its required columns,
and the `LabelMethod` it derives from (or `native`).

**Required members** (FR-013): `gbb` (`guard_metric < 0.3 AND bb_width > 0.035`), `empty`
(every bar — the forward_mae case and the verdict null), `geometry` (the DWT-smoothed zero-lag
valley condition that exists today and has never been tested).

**Constraints**: no trigger may reference a `__outcome_` column; none may be pair-specific
(Constitution X); `required_columns` is asserted present before evaluation.

## Labels

`label = trigger AND __outcome_positive`. No trigger supplies its own outcome test (FR-011).

## Leaderboard

One row per trigger, per `TriggerScore` in data-model.md. The two columns — economic value
with no model, and probe learnability — are reported separately and **MUST NOT** be combined
into a single score (FR-016). Learnability is diagnostic only and never licenses promotion
(FR-015a, FR-018).

## Verdict rule (FR-014b) — total, and against the `empty` trigger

- `selection_value` — economic value exceeds `empty`'s by more than the CI on the difference
- `decoration` — it does not, while learnability exceeds `empty`'s by more than the SE on that
  difference
- `indeterminate` — neither holds

Uncertainties per research R9: a **block** bootstrap for economic value, because adjacent
bars' forward windows overlap heavily and an i.i.d. bootstrap would understate the interval;
seed/window spread for learnability.

## Reporting obligations

- Non-reducible `LabelMethod`s listed with the reason (FR-012). Per research R7 these are the
  ones whose condition is purely a forward-path property — `mfe_mae_ratio`, `future_sharpe`,
  `future_sortino`, `future_expectancy`, `risk_adj_future`. They are competing *outcome*
  definitions, not triggers, and this design supersedes them.
- The probe's known bias stated plainly: ridge cannot express a logical AND, so it
  understates learnability for trigger-defined labels. The verdict rule is unaffected because
  both sides of the comparison use the same probe, but absolute numbers are not ceilings.
- Any P&L claim must come from `evaluation/` and its ~3–4pp/yr floor (FR-017). The table's
  economic value is a lookahead bound, not a realisable return.
