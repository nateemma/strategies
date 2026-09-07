# US exchange venues for this repository — survey 2026-09-06

Measured, not recalled: freqtrade's own `list-exchanges` capability table, the
`_ft_has` overrides in `freqtrade/exchange/*.py`, and live ccxt market/ticker
data. Fee figures are ccxt's BASE tier; all venues tier down with 30-day volume.
Volumes are a SINGLE 24h snapshot and should be re-sampled before being acted on.

## What this repo needs from a venue

Derived from `config/config.json` and the measured strategy behaviour:

| requirement | why |
|---|---|
| `fetchL2OrderBook` | `entry_pricing`/`exit_pricing` both set `use_order_book: true`, `order_book_top: 1` |
| `createMarketOrder` | `order_types` uses market for exit, stoploss, emergency_exit, force_exit |
| OHLCV history | training window is 20220101-20250831 at 15m across 11 pairs |
| **ZEC listed** | ZEC is **63 of 176 W1 trades (36%)**, the largest single pair, and the universe study found the gbb edge is structurally the high-vol alt |
| low fees | turnover is **9.7x the wallet per year** (97,340 notional on a 10,000 book) |
| depth | `custom_stake_amount` CAPS stake at 1/10 candle quote volume, so thin books shrink stakes rather than rejecting trades |

## The survey

| venue | ft tier | blocking gaps | ZEC | coverage | quote | 24h depth | fees base | est. W1 net |
|---|---|---|---|---|---|---|---|---|
| **Binance.US** (current) | Supported | none | yes | 11/11 | USDT | $4.4M | 0.10 / 0.10 | **+10.72%** |
| **Kraken** | Supported | `ohlcv_has_history: False` | yes | 11/11 | USD | **$227M** | 0.25 / 0.40 | +4.88 → +7.80% |
| Bitstamp | not supported | no `watchOHLCV`, no `fetchOrders` | yes | 11/11 | USD | $58.7M | 0.40 / 0.40 | +4.88% |
| Crypto.com | not supported | 300-candle limit | **NO** | 10/11 | USDT | $40.3M | 0.25 / 0.50 | out |
| Gemini | not supported | **no `createMarketOrder`** | yes | 10/11 (no NEAR) | USD | — | 0.20 / 0.40 | out |
| CEX.IO | not supported | no `fetchMyTrades` | **NO** | 10/11 | USDT | — | — | out |
| Coinbase Advanced | not supported | no `fetchL2OrderBook`, no `watchOHLCV` | — | — | USD/USDC | — | — | out |

Every other Supported-tier venue (Binance global, Bybit, Gate, HTX, OKX,
Bitget, BingX, BitMart) geoblocks US retail. Hyperliquid is Supported and cheap
but is a non-custodial perps DEX restricting US persons — a different asset
class, not a drop-in.

## Depth, per pair (24h quote volume)

| pair | Kraken | Binance.US | Crypto.com | Bitstamp |
|---|---|---|---|---|
| ZEC | 103.8M | 2.03M | — | 1.82M |
| XRP | 36.1M | 0.90M | 8.64M | 16.27M |
| SOL | 44.0M | 0.88M | 27.73M | 31.79M |
| LINK | 13.8M | 0.24M | 0.75M | 1.31M |
| NEAR | 10.5M | 0.08M | 0.44M | 1.51M |
| AAVE | 3.15M | **0.008M** | 0.78M | 1.55M |
| DOT | 1.54M | **0.007M** | 0.44M | 0.70M |
| **total** | **227M** | **4.36M** | 40.3M | 58.7M |

Binance.US shows $7.8k/day on AAVE and $6.5k/day on DOT. At those volumes the
1/10-candle cap binds at roughly **$8 per entry**, against the $227 median stake
those pairs actually achieved over W1 — i.e. current Binance.US depth is far
thinner than the backtest window's average.

## Fee arithmetic (W1 control, 176 trades, 97,340 notional on 10,000)

| venue / tier | round-trip cost | W1 net |
|---|---|---|
| Binance.US 0.10% | 1.95pp | +10.72% |
| Kraken maker 0.25% | 4.87pp | +7.80% |
| Kraken taker 0.26% (≥$50k/30d) | 5.06pp | +7.60% |
| Kraken / Bitstamp taker 0.40% | 7.79pp | +4.88% |
| Crypto.com taker 0.50% | 9.73pp | +2.93% |

`price_side: "other"` with `order_book_top: 1` CROSSES the spread, so limit
entries fill as taker. Capturing a maker rate would require passive entries —
a real change to fill assumptions needing re-validation (though the entry-timing
study found fill luck runs ~0.21x median spread, so it may survive).

The 0.26% Kraken tier needs $50k/30d ≈ $600k/yr notional ≈ a ~$60k book.

## Verdict

**No US venue has both low fees and deep books.**

- Binance.US has the fees but not the books.
- Kraken has the books (52x) but not the fees; base-tier taker costs more than
  half the W1 return, and it additionally has NO OHLCV history — backtest data
  requires `--dl-trades` or Kraken's quarterly trades CSV zips in
  `user_data/data/kraken/trades_csv`, plus a USDT->USD quote migration that
  invalidates every pinned-window baseline in `regime/`.
- Bitstamp is the only alternative preserving normal OHLCV download, but is
  dominated by Kraken on depth, fees and feature support.

**The crossover is book size**, and it is a measurable question rather than a
judgement: run the existing capacity machinery (`QUOTE_VOLUME_HEADROOM_MULT`,
`regime/liquidity.py`) against the fee table above. Revisit at 005 T026, which
already measures the capacity curve.

Do not migrate mid-feature: a quote-currency change would invalidate the
pinned windows every current result depends on.
