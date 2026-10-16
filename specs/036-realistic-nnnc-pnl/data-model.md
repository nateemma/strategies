# Data Model: Realistic P&L for NNNC (036)

**Where things live:**
- **Recorded live data (never committed):** `user_data/data/live_nnnc/<run_id>/`, where `run_id` =
  the live fold id (`live1`, `live2`, …).
- **Committed study outputs:** `research/live_pnl/`.

## LiveModelRecord (`research/live_pnl/live_models.json`, committed)

| field | meaning / rule |
|---|---|
| run_id | `live1`, `live2`, …; unique |
| strategy_class | generated class, e.g. `NNNC_LIVE_live1`; the bot runs `<class>_REC` |
| model_key | feature-030 key hash; MUST equal the `_REC` class's resolved key |
| train_timerange | `20210101-<cut>`; `<cut>` ≤ first live day |
| threshold | from `pick_threshold.py --target-rate 0.014735` on the training window |
| target_rate | 0.014735 |
| params_sha | of the params snapshot used |
| started / retired | UTC timestamps; `retired` is set when the next run_id starts |

**State:**
1. planned → trained (weights present for the key) → picked (threshold recorded) → live → retired.
2. Only one record is `live` at a time.
3. Open trades opened under a retired run stay attributed to it.

## OrderRecord (`orders-YYYY-MM-DD.jsonl`)

| field | meaning / rule |
|---|---|
| key | (`entry`, trade_id, order_id), or (`entry_intent`, pair, candle) before an id exists; unique per file |
| run_id, pair, signal_candle | candle open time (UTC) of the signal |
| signal_close, signal_quote_volume | as the live bot saw them (for parity cause (a)) |
| requested_stake | stake before any cap (`proposed_stake`) |
| granted_stake | after the cap; ≤ requested |
| rate, amount, order_type, order_time | |
| depth_ref | key of the DepthSnapshot taken at this order, or null with `depth_missing = true` |

## FillRecord (`fills-YYYY-MM-DD.jsonl`)

`trade_id`, `order_id`, `side` (entry/exit/stoploss/adjust), `fill_time`, `fill_price`, `filled`,
and `cancelled`/`timeout` flags. It is unique on (trade_id, order_id).

## ExitRecord (`exits-YYYY-MM-DD.jsonl`)

`trade_id`, `pair`, `exit_reason`, `rate`, `time`, `depth_ref`.

## DepthSnapshot (`depth-YYYY-MM-DD.jsonl.gz`)

| field | rule |
|---|---|
| pair, time, trigger | trigger ∈ {order, exit, schedule} |
| bids, asks | up to 20 [price, size] levels each; sorted; prices > 0 |
| mid | (best bid + best ask) / 2 |
| depth_missing | true if the request failed or timed out; then bids/asks are empty |

Scheduled snapshots are unique per (pair, candle).

## SignalRecord (`signals-YYYY-MM-DD.jsonl`)

One per pair per closed candle: `pair`, `candle`, `close`, `quote_volume`, `p_buy`, `threshold`,
`guards_pass`, `enter_long`, `run_id`. It is unique per (pair, candle). It feeds parity (R4) and
drift (R11).

## ParityReport (`research/live_pnl/parity/<day>.json`)

- the day and run_id;
- the counts: `matched`, `live_only`, `backtest_only`;
- a list of differences: {pair, candle, side, cause ∈ {data_revision, gap_fill, timing,
  wallet_state, UNEXPLAINED}, evidence};
- `unexplained` (a count).

A day with `unexplained > 0` blocks P&L reads until it is resolved or recorded as a defect.

## CostModel (`research/live_pnl/cost_model.json`, committed; fitted artefact)

| field | rule |
|---|---|
| fitted_on | first and last recorded day used for fitting; holdout = the last 25% of days |
| tiers | tier → list of pairs (terciles of median top-20 quote depth) |
| curves | tier × side → monotone points (participation p, cost bps); p = stake ÷ rolling 24 h quote volume |
| fallback | per-pair constant curves (live period only) |
| choice | `participation` or `per_pair`, by held-out error, rule declared before fitting |
| holdout_error | per pair: median absolute error in bps |
| excluded_pairs | outside tolerance (SC-003) |

## RestatedResult (`research/live_pnl/restated_<arm>.json`)

- per fold: n, gross P&L, cost P&L, net P&L (USDT and wallet %);
- per half (early f00–f07, late f08–f14) and the total;
- a flag `extrapolated` if the cost model's choice is `per_pair`.

## CapacityPoint (`research/live_pnl/capacity.json`)

- `arm` (cap_aro_ref or cap_aro_b<x>) and `budget_bps` (null for the reference);
- the restated net per fold and per half;
- the judge_one verdict against the reference;
- the risk verdict;
- trades;
- the median realised/requested stake;
- a `liveness` flag (whether the stake distribution differs from the reference).

## GateStatus (`research/live_pnl/gate.json`)

`days`, `orders_per_pair`, `open` (bool), and `thin_pairs`. It is open when days ≥ 28 and every
pair has ≥ 5 orders, or when days ≥ 84 with the thin pairs listed.
