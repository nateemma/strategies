# 031 T015 smoke backtest

`hp_smoke` = `NNNC_HP_ARO_DCA` (NNNC_A_Raw_Open + `HELD_PAIR_ADD = {}`) on ARO f00 (early half,
allowed before the pre-registration; listed as exempt). The plan is `plan_hp_aro_dca.json`
restricted to f00, with the control `wf_aro_g24`'s 24 h snapshot, threshold and model dir.
"model already present", so nothing trained.

| | control `wf_aro_g24` f00 | `hp_smoke` f00 |
|---|---|---|
| trades | 49 | 49 |
| trades_hash | 6e32009f6dd2bc9c | d374c193acec9b0c |
| adds | – | 10 (10 trades) |

`hp_counters` (per `adjust_trade_position` call):
- adds 10;
- refused `no_signal` 1966, `max_adds` 499, `too_young` 102, `thin` 10;
- `row_misaligned` 0. The dataprovider's last row is the signal candle on every call.

`smoke_check.py` → `smoke.json` checked all 10 adds against the archive and the ARO f00 signal
table:
- every add's signal candle (fill − 15 min) has `enter_long == 1`;
- every add is ≥ 1 h after the last fill (1.0 h to 16.75 h);
- every add stake is ≤ 900 and ≤ the liquidity cap (4 adds were cap-bound, e.g. SOL 216.09 = cap);
- the exported `open_rate` equals the amount-weighted average of the fills, to the pair's price
  precision (relative difference ≤ 1.5e-4: freqtrade rounds the exported rate).

Result: **PASS**.
