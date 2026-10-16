# Data Model: 037 Indicator Screen

## HarvestManifest (`research/indicator_screen/manifest.json`)

| Field | Type | Rule |
|---|---|---|
| code_sha | str | sha of discover/overrides/harvest sources plus SimpleStrategies sources; a cache whose sidecar sha differs is rebuilt |
| classes | list[ClassEntry] | every in-scope class (R2) |
| excluded | list[{class, reason}] | reason ∈ {trigger_arm, import_error, signal_error, constant, all_missing, lookahead} |
| columns | list[ColumnEntry] | every harvested column |
| pairs | list[str] | the config_wf whitelist plus "BTC/USDT" |
| range | {start, end, warmup_bars} | 2021-01-01 → 2026-09-07, 1,000 |

**ClassEntry**: `{class, base: "SimpleStrategy"|"EhlersBase"|..., override: null|"ehlers_triggers"|"force_trend_mode"|"ichimoku_switches", params: {name: value}}`

**ColumnEntry**:
- `name`: `<source>:<column>` for values, `raw_buy:<source>[:variant]` / `raw_sell:...` for signals;
  `btc:` prefix when joined from BTC;
- `source`: class name, "base", or "aro" (the ARO control's 24 model inputs, R1);
- `kind` ∈ {value, raw_buy, raw_sell, candidate};
- `causality` ∈ {PASS, FAIL, UNTESTED};
- `first_mismatch` (optional);
- `excluded_reason` (optional).

**Rules:**
- A value column that is identical to a base column (same values on the whole range) is recorded as
  an alias, not duplicated.
- Every raw signal is 0/1 int8.
- NaN during warm-up is kept, never filled.

## Harvest frame (cache; one feather per pair)

- `date` (UTC), OHLCV, every ColumnEntry with `causality = PASS` (float32 values, int8 signals);
- `base:gbb_candidate` (int8; from the gbb trigger's own populate step, R5);
- `label:buy`, `label:sell` (int8; NaN for the last 48 bars).

Sidecar `<PAIR>.json`: `{code_sha, columns, rows, first_date, last_date}`.

## Window

`{id: "E0"|"f00".."f14", role: "era"|"test", half: "e0"|"early"|"late", start, end,
train_start, train_end}`. E0 has no train span.

`purge(target)`: 48 bars for labels, N for regime targets. Model training rows must satisfy
`date < test_start − purge`.

## Target

| id | Definition | Values | Cut points |
|---|---|---|---|
| label:buy / label:sell | ARO gbb_path labels | {0, 1} | — |
| A1:N | pooled mean outcome-table profit of candidates opened in (t, t+N] | top / bottom tercile → 1/0; middle → dropped | fold training window |
| A2:N | pooled mean w_aro trade profit_ratio opened in (t, t+N] | as A1 | early test windows (pre-registered exception) |
| B-btc:N, B-pair:N | forward log return | as A1 | fold training window |
| C-btc:N, C-pair:N | forward std of 15m log returns | as A1 | fold training window |

Here N ∈ {48, 288, 672}. A bar with no forward observations is "not computable". Baselines use
the same quantity over [t−N, t). For A1/A2, only trades that closed before t are counted.

## WindowScore

`{column, target, population: "all"|"candidate", window, n_rows, n_pos, auc, direction, strength,
mi_norm, mi_edges_from: "train"|"self", lift (signals only), fire_rate (signals only),
strat_auc (candidate only), computable: bool, reason (optional)}`

## ScreenResult (one per column × target × population)

`{column, target, population, median_strength, median_direction, same_direction_share,
halves: {e0, early, late}: {median_auc, median_strength, n_windows}, median_mi_norm, median_lift,
median_strat_auc, width_sort_auc, redundancy_group, rho_guard, rho_width, baseline_auc (regime only),
dedup: bool (market-wide target × btc: column, R11),
candidate_flags: {beats_width_both_halves, residual_both_halves, beats_baseline_both_halves,
e0_agrees: bool|null}}`

A candidate flag counts only when `e0_agrees` is true; `null` (not scorable, e.g. A2) is reported as
"distant era not tested".

## PairResult

`{a, b, side: "buy"|"sell", target, population: "candidate"|"all", early: {median_lift, n_windows},
lift_a_early, lift_b_early, selected: bool, late: {median_lift, n_windows} (survivors only),
e0_lift, e0_lift_a, e0_lift_b (survivors only), confirmed: "yes"|"early+late only"|"no"}`

## ModelResult (one per fold × target × model)

`{fold, target, population, model: "tree"|"hgb", inputs: "all"|"restricted", n_train, n_test,
subsample: 1|4, auc, width_sort_auc, aro_auc (optional), rules (tree only): [str],
top_splits: [column], computable: bool, reason (optional, e.g. "A2: no prior test windows" for f00)}`

Every model candidate carries `persistence: "distant era not tested"`.

## Preregistration (`preregistration.json`)

`{min_rows, min_fires, max_fire_rate: 0.95, instrument: {top_k: 5, columns: [guard_metric,
bb_width]}, pairs: {min_windows: 6, uplift: 1.10, min_lift: 1.20, top_n: 20}, models: {...R10
settings}, regime: {horizons: [48, 288, 672], tercile: true, a2_cuts: "early_test",
populations: ["all", "candidate"], model_subsample: 4, a2_model_train: "prior_test_windows",
dedup_market_wide: true}, e0_required: {single: "same_direction", pairs: "lift_ge_max_component"},
instrument_gate_halves: ["e0", "early"],
residual_grid: 5, mi_bins: 20, redundancy_cut: 0.8, sample_rows: 200000, seed: 0}`

The late lock (R15) refuses any f08–f14 score unless this file is committed and has no
uncommitted change.
