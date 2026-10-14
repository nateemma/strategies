# Contract C2: the width-relative sizer arm (NNNC)

## `NNNC/_width_relative.py` — `WidthRelativeMixin`

- `POOLED_LABEL`: the NNNC spec (data-model).
- `TRAIN_ROW_FILTER = (("%wr_keep", "gt", 0.5),)`: entry- or exit-eligible bars with a valid
  outcome.
- `MODEL_KEY_ATTRIBUTES = ("TRAIN_ROW_FILTER", "POOLED_LABEL")`.
- `HELD_PAIR_ADD = None`.
- It does NOT change entries. The width book does that.

## `NNNC/_width_sizer.py` — `WidthSizerMixin` (placed before `WidthBookMixin` in the bases)

- `prediction_threshold`: `DecimalParameter(0.0, 1.0, decimals=4, default=0.5, space="buy",
  load=True)`.
- `AGREE_SHARE = 0.5`, `STAKE_WEIGHTS = (1.0, 0.5)`.
- **Entries:** unchanged from `WidthBookMixin`. The entry tag gets `:agree` or `:disagree` from the
  same row's `predict_buy` (written by the model's prediction pass in `get_entry_conditions`).
- **`custom_stake_amount`:**
  1. read agree from `entry_tag`, which freqtrade passes in (`:agree` / `:disagree`);
  2. `w = STAKE_WEIGHTS[0 if agree else 1] / (s + STAKE_WEIGHTS[1] * (1 − s))`;
  3. return `min(super().custom_stake_amount(... proposed_stake=proposed × w ...), max_stake)`. The
     parent applies the liquidity cap.

  The intended stake (`proposed × w`) is not stored on the trade. The instrument recomputes it from
  the tag and compares it with the realised `stake_amount`.
- **Exits:** untouched. Both control snapshots set `enable_exit_signal: false`, and no exit path reads
  the model's predictions, so exits do not depend on the model.

## The exit-on variant

The same classes and weights, with `enable_exit_signal: True` in the generated `.json`; the exit
gate is unchanged (rvol > 2.0, guard_metric > −0.1). The instrument counts `model_exit ` tags per
fold, and the exit-gate-passing bars inside open trades. A fold with gate-passing bars and zero
model exits fails liveness.

## Generated classes

`research/width_relative/build_arms.py` writes:
- `NNNC/NNNC_WR_ARO.py` and `NNNC/NNNC_WR_GBR.py`, with their `.json` files copied from the
  controls' snapshots;
- the plans for both variants: `plan_wr_<m>_sz.json` (exits off) and `plan_wr_<m>_szx.json` (exits
  on). They share the model key, so the weights are trained once;
- the model dirs are new, and `WIDTH_CUTS` come from `width_twin.json`.

## Checks

- **Flat-stake twin (exit-off arm):** with `STAKE_WEIGHTS = (1.0, 1.0)` and the width-relative weights loaded, the
  arm's `trades_hash` on one fold equals the `w_<m>` record's. Entries, exits and the stake path are
  then proven identical to the width book apart from the weights. This is a spot check on f00 per
  model, before the arm runs.
- **Tag-to-agree:** every `:agree` trade's signal bar has `buy_argmax` and `p_buy` > threshold in
  the signal tables, and every `:disagree` trade's does not. 0 mismatches.
- **Thresholds:** the loaded threshold equals the plan's to 4 decimals (`ft_bot_start()`). An
  existing class still loads a 2-decimal threshold (the parameter is not shared).
