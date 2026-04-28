# Debug — Diagnostic Strategies

Strategies that don't trade.  Each one runs as a normal freqtrade
backtest but its purpose is to print, plot, or analyse something — not
to generate buy/sell signals.  Naming convention: every file starts
with `Debug`.

## Main files

| File | What it does |
|---|---|
| `DebugDfAnalyse.py` | Analyses the dataframe produced by `DataframePopulator` — column ranges, correlations, redundancy.  Useful when adding or removing indicators. |
| `DebugCheckRedundancy.py` | Pairwise correlation check across indicators.  Helps trim columns that carry the same information (which confuses the NN models). |
| `DebugEvaluateCtabGan.py` | Loads a saved CTAB-GAN+ model and runs the built-in `evaluate_with_dataframes` quality metrics on a sample.  Diagnostic for GAN training. |
| `DebugEvaluateMTCtabGan.py` | Same idea for the multi-task CTAB-GAN+ variant. |
| `DebugRegimeIndicator.py` | Plots the `regime` indicator (bear/sideways/bull) over time so you can sanity-check it visually. |
| `DebugTradingType.py` | Prints / plots the training labels (`%train_buy`, `%train_sell`) to verify they make sense for the chosen `TRAINING_TYPE`. |
| `DebugNNStrategy.py` / `DebugNNMTStrategy.py` | Lightweight wrappers that load a trained NN model and dump diagnostics (probabilities, per-pair stats) without making trades. |

## Running

Same as any other strategy:

```zsh
zsh user_data/strategies/scripts/test_strat.sh -n 30 Debug DebugDfAnalyse
```

The output is printed to stdout (or saved as plots in `user_data/plot/`
where applicable).  Most Debug strategies are short — read the source
to see what they actually do.
