# hyperopts — Custom Hyperopt Loss Functions

Hyperopt minimises a loss function — freqtrade ships with several
built-ins (`ShortTradeDurHyperOptLoss`, `OnlyProfitHyperOptLoss`, etc.)
but they tend to overfit to total-profit metrics and recommend
parameters that don't survive contact with live trading.  The losses
in this directory weight other things — expectancy, win rate, trade
duration — so the recommendations are more robust.

To use any of these, **copy** the file into
`<freqtrade>/user_data/hyperopts/` (freqtrade looks there, not here).

## Main files

| Loss function | Optimises primarily on |
|---|---|
| `ExpectancyHyperOptLoss.py` | Expectancy (projected profit per trade).  General-purpose recommendation. |
| `OnlyExpectancyHyperOptLoss.py` | Pure expectancy, ignoring everything else. |
| `WeightedProfitHyperOptLoss.py` | Total profit, with secondary metrics. |
| `QuickProfitHyperOptLoss.py` | Profit weighted toward short trade durations. |
| `WinHyperOptLoss.py` | Win/Loss ratio. |
| `MarketHyperOptLoss.py` | Win/Loss adjusted for market conditions (trending vs sideways). |
| `MedianProfitHyperOptLoss.py` | Median profit per trade — robust to outliers. |
| `PEDHyperOptLoss.py` | Equally-weighted Profit, Expectancy, Duration. |

## Conventions

All loss functions in this directory:

* Subclass `freqtrade.optimize.hyperopt_loss_interface.IHyperOptLoss`.
* Implement `hyperopt_loss_function(...) -> float` where lower is better.
* Apply a minimum-trade-count filter (otherwise hyperopt finds
  pathological no-trade solutions).
* Apply a minimum win/loss filter.

## Usage

```zsh
zsh user_data/strategies/scripts/hyp_strat.sh \
    -l ExpectancyHyperOptLoss \
    binanceus NNNC_CGP
```

Or pass the class name to freqtrade directly via `--hyperopt-loss`.
