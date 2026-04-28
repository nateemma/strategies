# config — Exchange Configurations

Static config files used for backtesting, hyperopt, dry-run, and live
trading.  One set per exchange.

API keys and secrets do **not** belong here — keep those in a separate
config at the project root that's not committed to git.

## Naming convention

```
config_<exchange>[_<purpose>].json
```

| Suffix | Purpose |
|---|---|
| _(none)_ | Main backtest/hyperopt config (static pairlist) |
| `_short` | Futures / short-trading config (`trading_mode: futures`) |
| `_download` | Download-only config — may use VolumePairlist |
| `_train` | Long-range config for training NN models |
| `_leveraged` | Leveraged-trading config |
| `_top` | Top-N pairlist (fewer pairs, faster iteration) |

## Main files

Currently configured for BinanceUS:

| File | Purpose |
|---|---|
| `config_binanceus.json` | Default backtest / hyperopt |
| `config_binanceus_download.json` | Download data (broader pair list) |
| `config_binanceus_short.json` | Futures / short trading |
| `config_binanceus_leveraged.json` | Leveraged trading |
| `config_binanceus_train.json` | Long-range model training |
| `config_binanceus_top.json` | Top-N pair list (faster iteration) |
| `config.json` | Generic / fallback |

## Adding a new exchange

1. Copy `config_binanceus.json` to `config_<newexchange>.json`.
2. Update `exchange.name`, `pair_whitelist`, and any
   exchange-specific fields.
3. Keep the file checked in — but again, no API keys.

## Pairlist generation

`VolumePairlist` doesn't work for backtesting / hyperopt.  Generate a
static pairlist with `freqtrade test-pairlist` and paste the output
into the config's `pair_whitelist`.  See top-level `README.md` for
details.
