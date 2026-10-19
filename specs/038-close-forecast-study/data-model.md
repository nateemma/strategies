# Data Model: 038 Close-Price Forecast Study

## Horizon

Each horizon N is one of {1, 2, 4, 8, 16, 32, 48, 96} candles of 15 minutes (FR-002).

| Field | Meaning |
|---|---|
| `n` | Candles ahead |
| `typical_move` | Median of the absolute N-candle log return, on traded candles (per pair and pooled) |
| `cost_rt[s]` | Round-trip cost at slippage s ∈ {0, 2.5, 5, 10} bps per side: `2 × (0.0010 + s/1e4)` |
| `move_to_cost[s]` | `typical_move / cost_rt[s]`. A value below 1 means the horizon is untradeable by construction (Story 2, AS3) |

## Candle frame (per pair, built once)

This frame is built from `binanceus/<PAIR>-15m.feather`, gap-filled.

| Column | Meaning |
|---|---|
| `date`, `open`, `high`, `low`, `close`, `volume` | Candle data |
| `lc` | `ln(close)` |
| `has_data` | The candle exists in the raw file (not gap-filled) |
| `fold` | `f00`–`f14` when the candle is in a test window, `fwd` for the forward window, otherwise empty |
| `half` | `early` (f00–f07), `late` (f08–f14) or `fwd` |
| `target[n]` | `lc[t+n] − lc[t]`. NaN where `t+n` is past the data |
| `trade_ret[n]` | `ln(open[t+1+n] / open[t+1])`. NaN where it is not available |
| `eligible[n]` | `volume[t+1] > 0 and volume[t+1+n] > 0 and has_data` on both candles (R3) |
| `data_ok` | The 037 cache's warm-data flag (`meta:data_ok`) up to 2026-09-07, `has_data` after that. Learned tiers require it, so their feature sets stay paired |

**Validation**:
- `target` and `trade_ret` for the last n candles are NaN, never 0.
- Every `fold` window matches `plan_cs_aro.json` exactly.

## Method

| Field | Meaning |
|---|---|
| `id` | For example `zero`, `hist_mean`, `ar`, `holt`, `ses`, `expsm`, `ets`, `ss_expsm`, `arima`, `theta`, `lgb_L`, `lgb_LA`, `lgb_LW`, `lgb_null`, `lgb_L_s2` |
| `tier` | 0 (baseline), 1 (statistical), 2 (learned), 3 (learned + decomposition) |
| `stride` | 1 for tiers 0, 2 and 3; 16 for tier 1 (R5) |
| `inputs` | A short description of the inputs |
| `params` | Fixed hyperparameters, copied into the pre-registration |

## Forecast record (per method; stored as one feather per method under `user_data/data/forecast_038/`)

| Column | Meaning |
|---|---|
| `pair`, `date`, `fold`, `half` | Where and when |
| `f[n]` | Forecast log return for each n. NaN if the window failed |
| `failed` | Bool. True when the forecaster raised `ForecastFailed` |
| `skipped_gap` | Bool. True when the input window's last candle is gap-filled, or more than half of its candles are. Forecasts are NaN, and the row is not counted as a failure |

**Validation**:
- Rows exist only at scored candles: those in a fold or the forward window, on the method's stride.
- The share of failures per n is reported, and a share above 5% excludes the method at that n (R9).

## Trade (derived; per method, n and pair)

| Field | Meaning |
|---|---|
| `open_date`, `close_date` | Entry at `open[t+1]`, exit at `open[t+1+n]` |
| `gross` | `trade_ret[n]` at the signal candle |
| `net[s]` | `gross − cost_rt[s]` |
| `fold`, `half` | Taken from the signal candle |

**Rules**:
- **Signal**: `f[n] > 0` and `f[n] ≥` the causal 90th-percentile threshold of the previous 30 days of
  this pair, method and n (long only).
- **One position per pair**: a signal is only taken if the candle is not inside an open trade.
- **Eligibility**: `eligible[n]` must hold.

## Cell score (per method, n and half; also per pair)

| Field | Meaning |
|---|---|
| `n_trades` | Number of trades |
| `mean_gross`, `mean_net[s]` | Mean gross return, and mean net return at each slippage level |
| `rand_gross` | Mean `trade_ret[n]` over all eligible candles of the same pairs and half (random entry) |
| `edge_vs_rand` | `mean_gross − rand_gross` |
| `net_lb95` | One-sided 95% lower bound of `mean_net[5bps]`, from a block bootstrap over calendar weeks |
| `r2_oos`, `ic_median`, `hit_rate` | Descriptive metrics, reported on all candles and on traded candles only (R10) |
| `short_mean_net[s]` | The mirror of the cost check on short signals (descriptive) |
| `fail_share` | Share of failed windows |
| `net_per_pair_year` | `Σ net[5bps] / pair-years in the half` |

## Verdict (per method and n)

| Field | Meaning |
|---|---|
| `verdict` | `PASS (PROVISIONAL)`, `FAIL` or `INCONCLUSIVE` |
| `reasons` | The conditions from R9 that decided it |
| `numbers` | The cell scores for both halves |

The verdict file also holds the study summary:
- `most_viable_n`, or null;
- `break_even_n`, or null;
- `null_passes`: the count of null-control cells that pass.

## Pre-registration (`research/forecast/preregistration.json`, committed before any non-baseline result)

It fixes:
- horizons, pairs, folds, halves and the forward window;
- costs and the slippage level used for the verdict;
- the selection quantile and its lookback;
- the bootstrap settings;
- the minimum trades and the maximum failure share;
- every method's params;
- the decision logic text (R9).

The verdict tool refuses to run when any of these holds:
- the file is uncommitted, or has uncommitted changes;
- any result file is older than the file's last commit;
- the instrument checks (SC-001, SC-002) have not passed.

## State transitions

```text
library fixed (tests green) → baselines scored → instrument checks pass → pre-registration committed
→ tier 1 → tier 2 (null control first) → tier 3 → verdicts → report
```

If an instrument check fails, the pipeline stops and no later result is read (SC-002).
