# 032 follow-up 1: run-matched CO arms, declared reading (EXPLORATORY)

Written and committed BEFORE any `co_aro_rm` / `co_gbr_rm` record exists.

**Arms:** `co_aro_rm` and `co_gbr_rm`. Same CO weights as 032. Each fold's threshold comes from
`run_match.py`: the 0.01-grid threshold at which the CO model's entry runs on the training window
come closest to the control's runs at the control's own threshold.

**Comparators:** the control (`wf_aro_g24` / `fs_gbr_gr`) and the width book (`w_aro` / `w_gbr`).

**Reading, per model:**
- The run-matched arm *passes the screen* if both conditions hold:
  - its pp summed over f00–f14 is at least the control's AND at least the width book's;
  - its late pp (f08–f14) is at least the control's AND at least the width book's.
- Also reported, but not part of the rule:
  - `judge_one(arm, control)`;
  - trades, and pp per trade, by half;
  - entry runs on the training window against the control's.

**Status:** EXPLORATORY, not a finding. 032's early half was read in S2, and the f10–f14 totals were
printed by its judge, so this cannot be a confirmatory test. A pass would need confirmation on unread
data: the 2022 window or the folds after 2026-09-07.

**Threshold rounding:** `prediction_threshold` is a `DecimalParameter(decimals=2)`, so every threshold
`pick_threshold.py` writes is rounded when it is loaded. This holds for 032's arms and for every
picked-threshold control. `run_match.py` therefore searches the 0.01 grid only.

## Amendment (2026-10-14, before any backtest or P&L)

The first run matched over the WHOLE training window, and it could not reach the target. On data it
was trained on, the control fires about 13 times more often than it does out of sample. On ARO f00
it produced 14,603 runs (about 9% of bars), while the CO model peaks at about 10,000 runs at
threshold 0.49. Every ARO fold therefore picked 0.49. That chain was stopped during matching, before
any `co_*_rm` record existed.

The matching now uses only the validation tail: the last 20% of each pair's training frame
(`TRAIN_DATA_SPLIT` 0.8). Neither model trains on it; it is used only for early stopping. The arms,
comparators and reading above are unchanged.
