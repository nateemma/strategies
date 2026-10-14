# Data Model: Width-Relative Label (033)

## PooledLabelSpec (strategy-declared data; `POOLED_LABEL`)

| field | type | NNNC value | rule |
|---|---|---|---|
| `outcome` | str | `"path"` | the only outcome kind in this feature (`TrainingSignals.path_outcome`) |
| `target` | float | 0.02 (`MIN_BUY_GAIN_THRESHOLD`) | > 0 |
| `stop` | float | −0.05 (gbb_path `stop_frac`) | < 0 |
| `horizon` | int | 48 (`HORIZON`) | ≥ 1 bars |
| `reference` | str | `"bb_width"` | a column present in every raw frame |
| `eligible` | tuple of (column, op, value) | the gbb rule | the 032 `row_filter_mask` spec format |
| `exit_eligible` | tuple or None | guard_metric > 0.3 AND bb_width > 0.035 | None means a BUY-only label; must be disjoint from `eligible` (checked on the data: an overlap raises) |
| `exit_loss`, `exit_take_profit` | float | 0.02 (`MIN_SELL_LOSS_THRESHOLD`), 0.022 | the sell-path outcome (`path_outcome(side="sell")`) |
| `buckets` | int | 10 | ≥ 2; equal-count quantiles over eligible, valid bars pooled across pairs |
| `share` | float | 0.5 | in (0, 1); the top share of each bucket is BUY |
| `min_bucket` | int | 500 | a smaller bucket is merged into its neighbour (reported) |

Absent (`None`) means no relabelling: byte-identical.

**Validation:** a bad field raises `ValueError` before training. A missing column raises
`KeyError`.

## PathOutcome (per bar)

| field | type | meaning |
|---|---|---|
| `outcome` | float or NaN | buy side: `target` (first hit), `stop` (stop first), the horizon close return, or NaN past the frame end; sell side: `−exit_loss` (fall first), `+exit_take_profit`, the horizon return, or NaN |
| `bars_to_exit` | int or −1 | bars to the first hit, or to the stop; `horizon` for a horizon close; −1 if NaN |

## WidthBucket (per fold, in the relabel report)

| field | meaning |
|---|---|
| `lo`, `hi` | reference-column edges (inclusive lo, exclusive hi; the last bucket is closed) |
| `n` | eligible valid bars |
| `side` | `buy` or `sell` |
| `pos_share` | the BUY (buy side) or SELL (sell side) share; target 0.5 |
| `median_outcome` | the bucket's median outcome |
| `merged` | true if merged because of `min_bucket` |

## RelabelReport (`width_relative.json`, beside the weights)

The fields are:
- `spec`;
- `buckets` (a list of WidthBucket);
- `eligible_valid`;
- `invalid`;
- `buy`;
- `hold`;
- `spearman_label_reference`;
- `pairs`.

It is written once per trained fold, next to 032's `row_filter.json`.

## Label (per eligible, valid bar)

The rule is BUY if the bar is in its bucket's top `share`, ranked by the key
`(outcome desc, tie-break, position)`. The tie-break is `bars_to_exit` ascending for target hits
and descending for stops. Otherwise the bar is HOLD. On exit-eligible valid bars, the rule is SELL if
the bar is in its exit-side bucket's bottom `share` (key: outcome asc, then a fall reached sooner
ranks lower and a take-profit reached later ranks higher). Otherwise it is HOLD. Bars outside both rules are
untouched, and the row filter excludes them.

## Sizer arms (per fold classes `NNNC_WR_<M>_SZ_fNN` and `NNNC_WR_<M>_SZX_fNN`)

| attribute | value |
|---|---|
| bases | `WidthSizerMixin, WidthBookMixin, WidthRelativeMixin, <control base>` |
| `WIDTH_CUTS` | from `research/model_calibration/width_twin.json` (the previous fold's cut) |
| `AGREE_SHARE` (`s`) | 0.5 |
| `STAKE_WEIGHTS` | agree 1.0, disagree 0.5; scaled by `1 / (s + 0.5 (1 − s))` |
| `prediction_threshold` | DecimalParameter, 4 decimals, from the fold plan's thresholds |
| `HELD_PAIR_ADD` | None |
| `enable_exit_signal` (in `.json`) | False for the exit-off arm `NNNC_WR_<M>_SZ`; True for the exit-on arm `NNNC_WR_<M>_SZX` |
| `MODEL_KEY_IGNORED` | the WidthBook names plus `AGREE_SHARE` and `STAKE_WEIGHTS` |

Trade tag: the family tag plus `:agree` or `:disagree`.

## Stage verdict (`verdicts.json`, per model)

The same shape as 032's `judge.py` output, plus:
- `S0.label` (per-fold bucket shares and Spearman);
- `S0.stakes` (per-fold count of short-stake trades and their share);
- `S1.gap_new` and `S1.gap_current` per half.

Decision is one of: `PASS (PROVISIONAL)` | `FAIL` | `INCONCLUSIVE`.
