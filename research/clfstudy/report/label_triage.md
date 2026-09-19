| method | horizon | SELL | HOLD | BUY | signal rate | persistence MCC | vol AUROC | dir AUROC | error |
|---|---|---|---|---|---|---|---|---|---|
| forward_mae (0) | 24 | 15.7% | 61.5% | 22.8% | 38.5% | 0.817 | 0.609 | 0.602 | |
| forward_mae (0) | 48 | 21.1% | 47.1% | 31.8% | 52.9% | 0.848 | 0.527 | 0.627 | |
| triple_barrier (1) | 24 | 23.3% | 54.5% | 22.2% | 45.5% | 0.815 | 0.672 | 0.463 | |
| triple_barrier (1) | 48 | 35.5% | 32.0% | 32.5% | 68.0% | 0.839 | 0.661 | 0.472 | |
| quantile_future (2) | 24 | 18.1% | 65.0% | 17.0% | 35.0% | 0.832 | 0.642 | 0.481 | |
| quantile_future (2) | 48 | 18.2% | 65.0% | 16.8% | 34.9% | 0.885 | 0.640 | 0.502 | |
| mfe_mae_ratio (3) | 24 | 16.3% | 83.7% | 0.0% | 16.3% | 0.656 | 0.623 | nan | |
| mfe_mae_ratio (3) | 48 | 25.2% | 74.8% | 0.0% | 25.2% | 0.717 | 0.582 | nan | |
| local_min_followthrough (4) | 24 | 7.7% | 84.8% | 7.5% | 15.2% | 0.436 | 0.633 | 0.293 | |
| local_min_followthrough (4) | 48 | 11.0% | 78.7% | 10.3% | 21.3% | 0.428 | 0.599 | 0.311 | |
| risk_adj_future (5) | 24 | 1.2% | 98.8% | 0.0% | 1.2% | 0.645 | 0.479 | nan | |
| risk_adj_future (5) | 48 | 1.2% | 98.8% | 0.0% | 1.2% | 0.653 | 0.476 | nan | |
| multi_horizon_vote (6) | 24 | 24.4% | 75.6% | 0.0% | 24.4% | 0.826 | 0.603 | nan | |
| multi_horizon_vote (6) | 48 | 28.9% | 71.1% | 0.0% | 28.9% | 0.844 | 0.585 | nan | |
| optimal_signals (9) | 24 | 26.8% | 48.6% | 24.6% | 51.4% | 0.827 | 0.678 | 0.473 | |
| optimal_signals (9) | 48 | 38.6% | 26.8% | 34.6% | 73.1% | 0.861 | 0.678 | 0.484 | |
| future_sharpe (10) | 24 | 14.9% | 71.0% | 14.0% | 28.9% | 0.801 | 0.636 | 0.455 | |
| future_sharpe (10) | 48 | 20.7% | 60.2% | 19.2% | 39.8% | 0.841 | 0.601 | 0.467 | |
| future_sortino (11) | 24 | 5.6% | 89.5% | 4.9% | 10.4% | 0.578 | 0.620 | 0.515 | |
| future_sortino (11) | 48 | 11.8% | 77.9% | 10.3% | 22.1% | 0.684 | 0.581 | 0.513 | |
| future_expectancy (12) | 24 | 12.1% | 76.4% | 11.6% | 23.6% | 0.780 | 0.644 | 0.452 | |
| future_expectancy (12) | 48 | 16.9% | 67.3% | 15.8% | 32.7% | 0.828 | 0.615 | 0.460 | |
| local_extrema (13) | 24 | 9.5% | 80.8% | 9.6% | 19.2% | 0.588 | 0.597 | 0.419 | |
| local_extrema (13) | 48 | 9.7% | 80.7% | 9.6% | 19.3% | 0.617 | 0.554 | 0.414 | |
| geometry (14) | 24 | 4.7% | 86.6% | 8.7% | 13.3% | 0.717 | 0.598 | 0.243 | |
| geometry (14) | 48 | 3.6% | 90.3% | 6.1% | 9.6% | 0.729 | 0.574 | 0.143 | |

Legend:
- persistence MCC: matthews_corrcoef(y[1:], y[:-1]) on pooled test rows (per pair, boundaries
  never crossed between pairs or windows) -- what a PERSISTENCE oracle (predict the previous
  bar's label) would score, computed with NO training. High values (>= 0.7) mean the label has
  strong overlapping-horizon autocorrelation, which inflates any model's apparent MCC on the
  same label (adjacent rows share most of their label-horizon window).
- vol AUROC: AUROC of a trailing volatility feature (rolling std of log close-to-close returns,
  48 bars) against `label != HOLD`. Near 0.5 is good (the label is not just a volatility
  detector); >= 0.60 flags a labeler whose signal rate is substantially explained by recent
  realised volatility alone.
- dir AUROC: AUROC of a trailing momentum feature (log(close[t]/close[t-16])) against
  `label == BUY`, restricted to non-HOLD rows. Far from 0.5 (either direction) means a trivial
  linear momentum feature already separates BUY from SELL -- the labeler may not need a model.

