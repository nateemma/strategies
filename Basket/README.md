and now for something completely different...

# Basket — slow portfolio-rebalancing strategies

A family of **portfolio-rebalancing** strategies for Freqtrade. These do *not*
trade in and out on signals — they **hold a fixed basket of coins and
periodically nudge them back toward target weights**, plus a stablecoin cash
bucket. Think automated portfolio management, not fast trading.

Targets **Freqtrade 2026.4-dev** (INTERFACE_VERSION 3).

Note that these strategies are _very_ different from most other strategies, and it gets a little complicated implementing these in freqtade. For example, the normal hyperopt loss functions and backtests do not work well because there are no trades beyond the initial entries. This resulted in the need for a custom hyperopt loss function (WalletCalmarHyperOptLoss) and scripts to help evaluate (walk_forward.sh). Also, these strategies only work over long time periods (months-years) - they are not get-rich-quick strategies. They should perform (slughtly) better than market conditions, but cannot perform well in bear markets (but they do move to cash in such cases).

---

## 1. The concept

- **The basket** is the config whitelist. `max_open_trades` = number of coins.
- **The cash bucket** is your stablecoin balance — just undeployed stake, never
  modelled as a position.
- Each coin has a **target weight**; the strategy rebalances toward it.

It is deliberately **slow and deliberate** — three throttles, all configurable:

1. **Band-triggered.** Only act when a coin drifts more than `rebalance_band`
   (default 0.05) from its target. Small drifts are ignored.
2. **Cadence-gated.** Check a coin at most once per `rebalance_interval_hours`
   (default 24h), never every candle.
3. **BB-timed.** When a rebalance *is* triggered, the Bollinger mid-band decides
   *when* it may execute — **add/enter only below mid-BB, trim only above** (buy
   dips, sell rips). The BB is an execution-timing filter, **not** a second
   signal; the drift band is the trigger.

The basket may fill **asynchronously** and sit in cash above target while it
waits for below-mid-BB entries. That's intended — there is no "enter anyway"
fallback.

### Why it's built this way

- Inherits Freqtrade `IStrategy` **directly**, not the repo's ML `BaseStrategy`.
  A basket is *held and maintained*, so ROI, trailing stop, stoploss and exit
  signals are all **disabled**, not inherited and fought.
- Rebalancing happens in `adjust_trade_position` (position adjustment), not
  entries/exits. **A rebalance is not a new trade** — there is one trade per
  coin, adjusted (trimmed/topped-up) over its life. *This has big consequences
  for hyperopt — see §5.*
- Sizing is **mark-to-market**: each coin is sized to its target weight of the
  *current* portfolio value (`custom_stake_amount` / `_portfolio_value`), not
  Freqtrade's default sizing.

**Every variant is just one method — `get_target_weight()`** — the rule for what
each coin's weight should be. Everything above is shared in `BasketStrategy`.

---

## 2. The variants

| File | Rule for target weight | Character |
|---|---|---|
| `ConstantMixBasket` | fixed equal weights | **mean-reverting** — trims winners, buys losers |
| `CppiBasket` | exposure ∝ cushion above a floor | **trend (absolute)** — protects a capital floor |
| `InverseVolBasket` | `w ∝ 1/vol` | **risk-parity** — equal risk, not equal dollars |
| `MomentumBasket` | `w ∝ excess trailing return`, gated | **trend (relative)** — rides winners, cash in downtrends |
| `VolTargetBasket` | equal-weight × exposure to hit a vol target | **risk-managed** — cuts exposure when vol spikes |
| `MinVarianceBasket` | `w ∝ Σ⁻¹·1` (long-only) | **diversified** — low-vol *and* low-correlation |
| `BlendBasket` | `exposure × selection` (both pickable) | **composition** — mix a risk rule with a selection rule |

**When each tends to win:**

- **ConstantMix** — choppy / range-bound markets (it harvests the oscillation).
  Underperforms in a sustained trend (keeps trimming the winner). The
  `runaway_trend_override` toggle (default off) lets a runaway coin be trimmed
  past a hard ceiling regardless of the BB gate.
- **CPPI** — sustained trends (rising value → bigger cushion → more exposure)
  *and* drawdowns (falls toward the floor → exposure cut toward zero, preserving
  capital). Underperforms in chop (whipsaws make it buy high / sell low).
- **InverseVol** — when you want lower drawdown than equal-weight without a view;
  overweights the calm coins. Ignores correlations.
- **Momentum** — persistent cross-sectional trends. "Dual momentum": a coin only
  qualifies if its trailing return clears `mom_threshold` (else its share goes to
  cash), so it de-risks in broad downturns. Whipsaws in choppy markets.
- **VolTarget** — keeps *portfolio* volatility near a target; steadier ride,
  automatically de-risks in turbulence. Won't outperform in calm bulls (it stays
  fully deployed but no more).
- **MinVariance** — the most "portfolio-theoretic"; leans into low-vol,
  low-correlation coins. Most **estimation-sensitive** (crypto covariance is
  noisy) — treat its results with extra out-of-sample suspicion.

The cross-sectional variants (InverseVol / Momentum / MinVariance) compute a
coin's weight from the *whole* basket, so they gather every coin's trailing stat
via the base `_cross_section` / `_return_matrix` helpers (cached per candle,
causal).

**Cash handling differs by variant.** ConstantMix / InverseVol / Momentum /
MinVariance treat `cash_target_weight` as a fixed reserve and split the rest.
CPPI and VolTarget use **dynamic cash** — a residual of the cushion / exposure —
so they fix `cash_target_weight` and don't optimise it.

### BlendBasket — compose exposure × selection

A basket allocation is really **two** decisions, and `BlendBasket` factors them
apart so you can pick each independently:

```
target_weight(coin) = exposure_fraction × selection_weight(coin)
```

- **`exposure_mode`** — *how much* to deploy (rest is cash): `cppi` (cushion
  above a floor), `voltarget` (scale to a target vol), or `fixed`.
- **`selection_mode`** — *which coins* get it (weights sum to 1): `momentum`,
  `inverse_vol`, `min_variance`, or `equal`.

So `cppi × momentum` = "ride the strongest coins but cut exposure to defend a
floor" — the best-of-both a hard regime switch reaches for, but **smooth** (no
discrete flips) and with fewer thresholds. This is the recommended way to get
"right tool per regime" behaviour. It's **still long-only** — no combo makes a
falling market positive; the win is better risk-adjusted return across a cycle.

Caveat: it exposes a **large hyperopt space** (both switches + every mode's
params, ~16 knobs). Most efficient is to **fix `exposure_mode`/`selection_mode`**
to a combo you like and tune the rest, or run more epochs.

### Profit-skim overlay (income)

A base-class flag — `profit_skim_enable` (default **off**) — that stacks on **any**
variant. When on, a fraction (`skim_fraction`) of every new equity high is
ratcheted into a **reserved cash bucket that is never redeployed** — chips off
the table. It only grows (one-way).

- **Purpose: income / gain-protection**, not extra return. In LIVE the reserved
  amount is what you'd periodically **withdraw as income**; in backtest freqtrade
  can't withdraw, so it sits as protected idle cash (earning 0, not compounding),
  which means backtest *total return* understates the income use-case.
- **Trade-off:** in a clean monotonic bull it caps upside. But in *volatile*
  markets it can even *help* return by pulling capital out of a give-back churn
  (observed: `ConstantMix + skim` beat plain ConstantMix in a volatile bull —
  it stopped recycling gains into falling losers). It always locks in realized
  gains and cuts the "gave it all back" problem.
- **Stacks on CPPI** for "skim gains on the way up + defend a floor on the way
  down" — the natural income config.
- Knobs: `profit_skim_enable`, `skim_fraction`.

---

## 3. Configuration

Use the dedicated **`config/config_basket.json`** (separate from the ML
strategies' `config.json`, which forces a 15m timeframe — see Performance).

```jsonc
{
  "max_open_trades": 5,            // == number of coins in the whitelist
  "stake_currency": "USDT",        // the cash-bucket currency
  "stake_amount": "unlimited",     // let custom_stake_amount size to target weight
  "tradable_balance_ratio": 1.0,   // cash buffer is enforced by sizing, not FT
  "timeframe": "4h",
  "exchange": { "pair_whitelist": ["XRP/USDT","SOL/USDT","LINK/USDT","AVAX/USDT","DOT/USDT"] }
}
```

`custom_stake_amount()` enforces the cash buffer, so `tradable_balance_ratio`
can stay 1.0 and `stake_amount` should be `"unlimited"` so sizing isn't capped.

### Performance: timeframe & basket size

`adjust_trade_position` runs **once per open trade, per candle**, and Freqtrade
deep-copies the `Trade` object on every call — so backtest/hyperopt cost scales
with `open_trades × candles`, not with the weight math. Therefore:

- **Use 4h, not 15m.** The scheme rebalances ~daily; fine candles just multiply
  overhead. 15m→4h was ~15× faster here with identical behaviour.
- **~5 pairs, not 10.** Fewer coins is proportionally faster *and* rebalances
  more meaningfully — the drift band is absolute (0.05), so at 5 coins
  (target ~16%) a coin breaches with a ~31% relative move vs ~62% at 10 coins.
  Crypto is highly correlated, so 6–10 coins add little diversification but many
  dust-sized positions.

---

## 4. Tunables

All controlling variables are **hyperopt parameters** in the `buy` space, so one
`--spaces buy` pass optimises the whole scheme. Defaults are slow-scheme values;
a hyperopt writes optimised values to `<Strategy>.json`, which then **overrides**
the defaults on later backtests (delete that file to revert).

- **Base (all variants):** `cash_target_weight`, `bb_period`, `use_bb_gate`,
  `entry_band`, `rebalance_band`, `rebalance_interval_hours`,
  `profit_skim_enable`, `skim_fraction` (last two = income overlay, off by
  default). `entry_band` (default 0.0) requires price to be that fraction BELOW
  mid-BB to buy (entry + top-ups) — 0.0 is the loose `close < mid`; raise it
  for a more selective dip-buy (fills slower). Walk-forward it before trusting
  a value — its effect is non-monotonic.
  (`target_weight_per_coin` is a plain override — `None` = equal split. Only the
  BB *mid* band is used, so there is no `bb_std` knob.)
- **ConstantMix:** `runaway_trend_override`, `hard_drift_ceiling`.
- **CPPI:** `cppi_floor_mode` (`ratchet`/`fixed`), `cppi_floor_ratio`, `cppi_multiplier`.
- **InverseVol:** `vol_lookback`.
- **Momentum:** `mom_lookback`, `mom_threshold`.
- **VolTarget:** `vol_lookback`, `target_vol` (annualised).
- **MinVariance:** `cov_lookback`, `cov_shrinkage`.
- **Blend:** `exposure_mode`, `selection_mode`, plus the params of whichever
  modes are active (`cppi_*`, `target_vol`, `fixed_exposure`, `vol_lookback`,
  `mom_*`, `cov_*`).

---

## 5. Hyperopt — use the custom loss function

**The loss function is the crux.** A basket makes one trade per coin, all
closing at the final force-exit, so every *trade-based* loss is degenerate:

- ❌ `Sharpe` / `Sortino` (and `*Daily`) — std of ~N trade returns / resample by
  `close_date`, which is all on the last day.
- ❌ `Calmar` — the *closed-trade* drawdown is a phantom ~0, so the ratio explodes
  and hyperopt chases a drawdown that doesn't exist. (Observed: a config the
  built-in Calmar reported at 0.26% drawdown actually had a **9.4%** real
  drawdown.)

Use **`WalletCalmarHyperOptLoss`** (in `hyperopts/`). It reconstructs the
**mark-to-market equity curve** (cash + open positions valued at current price)
from the trades and price data, and optimises *its* Calmar — the real portfolio
risk-adjusted return. (Freqtrade only captures the true wallet curve in backtest
runmode — `_capture_wallet` early-returns during hyperopt — hence the
reconstruction. It tracked the true wallet Calmar closely in testing, e.g. 1.73
vs 1.85.)

_Note_: since I can't put new files in user_data/hyperopts, please execute the copy command shown below. You only need to do this once per repo update

```zsh
# one-time: make the loss available to freqtrade
cp user_data/strategies/hyperopts/WalletCalmarHyperOptLoss.py user_data/hyperopts/

FT="user_data/strategies/config/config_basket.json"
P="user_data/strategies/Basket"

freqtrade hyperopt -c $FT --strategy-path $P --strategy CppiBasket \
  --hyperopt-loss WalletCalmarHyperOptLoss --spaces buy --epochs 300 \
  --min-trades 1 --timerange=20240101-20250101
```

- **Keep `--min-trades 1`** — rebalancing is not a trade, so trade count = number
  of coins (~5). A higher min-trades filters everything out.
- **Add `--analyze-per-epoch`** only if you want `bb_period` to vary — it's used
  in `populate_indicators()`, which Freqtrade computes once per run otherwise.
  All the other (callback) params vary every epoch regardless.
- Everything is one `buy` space — no buy/sell split (there's no directional signal).

---

## 6. Interpreting results

**Read the daily-wallet-balance metrics, not the trade-based ones.** In the
backtest report, use these rows:

- `Total profit %`
- `Calmar (daily wallet balance)`
- `Sharpe / Sortino (daily wallet balance)`
- `Max % of account underwater (balance)`  ← the **real** mark-to-market drawdown

Ignore the plain (non-"balance") `Max % of account underwater`, `Calmar`,
`Sharpe` — those are computed from closed trades and are **phantoms** for a
basket (trades close only at the end).

**Always confirm a hyperopt winner with a plain backtest.** The hyperopt loss
uses a *reconstructed* equity curve (it holds each trade's final size for its
whole life, so it approximates when there's heavy rebalancing). The plain
backtest's "(daily wallet balance)" rows are Freqtrade's exact numbers — treat
those as ground truth.

**Trade count ≈ number of coins is normal**, not a bug. Do not raise
`--min-trades` to "get more trades."

**Expect regime dependence, and judge accordingly.** These are directional,
long-only crypto baskets — in a broad bear market most will lose; the question is
*relative* behaviour:

- Judge **CPPI / VolTarget** primarily on **drawdown** (their job is protection),
  not absolute return. In a walk-forward here, CPPI lost only −7% while the
  market fell −51% — that's success for CPPI even though the return was negative.
- Judge **Momentum / MinVariance** on downside-capture + return.
- Judge **ConstantMix / InverseVol** on choppy/sideways windows where mean
  reversion pays.

**Compare variants on the same data:**

```zsh
freqtrade backtesting -c $FT --strategy-path $P --timerange=20240101-20250101 \
  --strategy-list ConstantMixBasket CppiBasket InverseVolBasket \
                  MomentumBasket VolTargetBasket MinVarianceBasket
```

**Walk-forward, don't trust a single window.** Freqtrade has no built-in
walk-forward, so tune on one window and *test on the next, untouched* one:

```zsh
# tune on IS ...
freqtrade hyperopt -c $FT --strategy-path $P --strategy CppiBasket \
  --hyperopt-loss WalletCalmarHyperOptLoss --spaces buy --epochs 300 \
  --min-trades 1 --timerange=20240610-20250610
# ... then backtest the SAME (now-saved) params on the held-out OOS window
freqtrade backtesting -c $FT --strategy-path $P --strategy CppiBasket \
  --timerange=20250610-20260531
```

If the OOS wallet-Calmar/drawdown holds up, the tuning generalised; if it
collapses, it was regime-fit. (Be aware a bull IS + bear OOS split will always
look like a "failure" on return even when the strategy behaved correctly.)

### Raw band-rebalance vs BB-gated

To measure what the BB timing gate contributes, flip it off with a one-line
subclass and compare:

```python
# Basket/ConstantMixBasket_Raw.py
from ConstantMixBasket import ConstantMixBasket
class ConstantMixBasket_Raw(ConstantMixBasket):
    use_bb_gate = False   # pure drift band, no BB timing
```

Gated should trade less and at better prices; raw shows the drift band alone.

---

## 7. No-lookahead discipline

All indicators are **causal** by construction:

- The mid-BB is a trailing SMA (`qtpylib.bollinger_bands`) — row `t` uses only
  rows `≤ t`.
- Callbacks read `dp.get_analyzed_dataframe(...).iloc[-1]`; in backtest that frame
  is truncated to the current candle, so `.iloc[-1]` is the latest **closed**
  bar, never a future one. (Flagged `# LOOKAHEAD:` in the base.)
- Portfolio value / CPPI cushion / cross-sectional stats come from present state
  and trailing windows only; the CPPI high-water mark ratchets on realised value.

`lookahead-analysis` can't run on these (no sell signals → nothing to compare).
Use **`recursive-analysis`** instead — it verifies indicator causality without
needing trades:

```zsh
zsh user_data/strategies/scripts/recursive_check.sh Basket ConstantMixBasket
# or: freqtrade recursive-analysis -c $FT --strategy-path $P --strategy ConstantMixBasket
```
