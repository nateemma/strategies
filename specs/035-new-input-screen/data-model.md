# Data Model: New-Input Screen (035)

## Input (declared in `research/new_inputs/inputs.py`, hashed into the pre-registration)

| field | type | rule |
|---|---|---|
| tag | str | unique, `^[A-Z]{2}_[A-Z]{2,3}$` (R6 table) |
| family | enum | order_flow, positioning, market_state, basis |
| source | enum | global_spot, us_spot, um_metrics, um_premium, binanceus_ohlcv |
| raw | callable name | a builder function in `build.py`; trailing windows only |
| windows | dict[str, int] | declared once; never tuned |
| norm | enum | `z` (trailing 2880, min_periods 960, clip ±5, /5) or `c` (2s − 1) |
| missing_flag | bool | True for positioning only (clarification Q3); the flag is 1 when the value is undefined (raw absent OR in warm-up) |
| live_source | str | the R11 record; informational only |

Column names in the cache and the model: `ni_<tag lower>` (for example `ni_of_tbg`), and
`ni_<tag lower>_missing` for flags. Every value lies in [−1, 1], or is NaN before the final
fill. Flags are {0, 1}.

## Source cache (`user_data/data/external/`, not committed)

- `<source>/<SYM>/<file>.zip`, exactly as downloaded, plus a sibling `.CHECKSUM`.
- `manifest.json`: per file the url, sha256, bytes, fetched_at and a verified flag.
- State: absent → downloaded → verified. A file whose checksum fails is deleted and refetched
  once, then reported as a coverage gap.

## Input cache (`user_data/data/new_inputs/<PAIR>.parquet`, not committed)

- Columns: `date` (UTC, 15-minute open stamp, unique and sorted) and every `ni_*` column.
- `build_meta.json`: the builder code sha, input-table sha, source-manifest sha and build time.
  The mixin refuses a cache whose input-table sha differs from the class's declared one.

## Screen row (`research/new_inputs/screen.json`)

Per input:
- coverage: bars and trades, per pair and fold;
- spearman_width and spearman_pbuy;
- partial correlation: pooled, plus per-fold values;
- sign_folds: the number of folds matching the pooled sign;
- split gap: pooled and per fold;
- status: PASS, FAIL or INCONCLUSIVE, with a reason.

## Survivor model

- Class `NNNC_NI_<TAG>`: `NEW_INPUTS = ("<TAG>",)`, include_list = control's + columns,
  pre_normalized_columns = control's + columns, plus a sizer twin `NNNC_NI_<TAG>_SZ`.
- Plans `evaluation/folds/plan_ni_<tag>.json` and `plan_ni_<tag>_sz.json`, whose arm tags are
  `ni_<tag>_sz_fNN`.
- The model key differs from the control's on every fold. The control's key, scaler and weights
  are unchanged.

## Verdict (`research/new_inputs/verdicts.json`)

Per input: stage reached (S0, S1, S2 or S3), decision, reason and evidence. For the winner, the
judge_one records against `cs_aro` and `w_aro`, plus the report row for `wf_aro_g24`. Decisions are
PASS (PROVISIONAL), FAIL or INCONCLUSIVE.
