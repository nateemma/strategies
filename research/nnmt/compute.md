# 040 compute and instrument notes

## Training time (T011)

| Arm | Fold | Train window | Training | Evaluation | Trades |
|---|---|---|---|---|---|
| mt (NNMT_MLX, LSTM, 24 indicators) | f00 | 2021-01-01 → 2023-01-01 (2 y) | 23.1 min | 16 s | 40 |

Later folds train on longer anchored windows (up to ~5.7 y), so an NNMT fold is expected to take
~25–60 min. A 15-fold arm then takes ~10 h, and the ~135 planned MLX fold trainings take several days
of sequential compute.

## Fold boundary (Principle XI check, T011) — FINDING, harness-wide

freqtrade trims a timerange with `date <= stop` (freqtrade/data/converter/converter.py:195). So a fold
trained on `20210101-20230101` includes the 2023-01-01 00:00 candle, the FIRST bar of its own test
window. The last 48 training bars' forward labels (gbb_path H = 48) can therefore see one test-window
bar (15 minutes of a 90-day window).

- This is NOT NNMT-specific: every walk-forward arm since feature 007 (NNNC, cs_aro, w_aro, NNPredict,
  SimpleTrigger) trains the same way, so all comparisons in this study stay like-for-like.
- The exposure is one bar per fold, and the label at most reads that bar's close. It is not a purge of
  the label horizon as the constitution (XI) requires.
- Not fixed here: changing the harness alone for NNMT would break comparability with every recorded
  comparator. The fix (end `train_timerange` one bar, or H bars, before `test_start`) is a harness
  decision for the user, and it would require re-running the comparators.
