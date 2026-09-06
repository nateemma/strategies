# Contract — the NNMT arm classes

Three new strategy classes in `NNMT/`, following the `NNNC_Oracle` /
`NNNC_NoModel` pattern verified to transplant (research R3).

## Common requirements

1. **Each carries its own `<Strategy>.json`.** NNMT has none today, so without
   this an arm silently tracks whatever the shared defaults become — the exact
   mechanism that produced this feature.
2. **Each overrides `get_model_path`** to alias the control's weights, unless it
   is deliberately training its own. Omitting this makes a renamed strategy
   train inside its own backtest, fitting the evaluation window and returning
   different numbers with no error.
3. **Guards-off is expressed by neutralising the six thresholds**, never by
   disabling the guard flag — that flag also removes the `volume > 0`
   data-sanity check and the entire exit-side guard block.

## `NNMT_Oracle`

Replaces the model's entry opinion with the training label. Overrides
`get_entry_conditions` only, which bypasses the multi-task predict path, the
task filters and the forward-peak-regressor filter — correct, since it measures
what a perfect ENTRY signal is worth, not what the task machinery does.

Sets `predict_buy` and `predict_sell` so the exit path continues to work.

**Always reported as a lookahead ceiling. Never promotable.**

## `NNMT_NoModel`

Accepts every bar, so the guards alone select. Must call the parent's
`get_entry_conditions` first — on NNNC, skipping it meant `process_predictions`
never ran and the exit side raised a missing-column error.

## `NNMT_<variant>`

One per screened configuration. Changes exactly one thing against the control.
Where the changed parameter is a class attribute (label thresholds), the variant
is a subclass; where it is a hyperopt parameter (take-profit, thresholds, guard
values), the variant is expressed through the arm runner's flags and needs no
new class. **This does not weaken Common Requirement 1**: a flag-expressed arm
runs against the CONTROL's pinned `.json` with explicit overrides on top, so
nothing falls back to shared class defaults, which is what FR-022 exists to
prevent. A flag-expressed arm is only admissible while the control's `.json`
is the one in force.

**A subclass that redefines `buy_params`/`sell_params` MUST merge from the
parent** (`{**parent, **overrides}`); a fresh literal silently drops every
inherited key.

## Reporting

Every arm is judged on the acceptance window only. The other two windows are
reported and labelled. No classifier metric is an acceptance input. Any arm used
as evidence must come from a successful run, and any arm whose trades match its
control is inconclusive rather than refuted.
