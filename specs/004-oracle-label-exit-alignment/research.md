# Phase 0 Research — 004 Oracle-First Label/Exit Alignment

All items resolved. Nothing carried into Phase 1 as NEEDS CLARIFICATION.

---

## R1. Where the diagnostic data lives — RESOLVED, no re-runs needed

**Decision**: read the existing backtest archives. No arm needs re-running for
the US1 diagnosis.

Every backtest writes a zip to `user_data/backtest_results/`, and
`.last_result.json` names the newest. Inside, the main JSON carries
`strategy.<name>.trades`, a complete per-trade list. Verified fields:

    amount, close_date, close_rate, close_timestamp, enter_tag, EXIT_REASON,
    FEE_CLOSE, FEE_OPEN, funding_fees, initial_stop_loss_abs,
    initial_stop_loss_ratio, is_open, is_short, leverage, MAX_RATE, MIN_RATE,
    max_stake_amount, open_date, open_rate, open_timestamp, orders, PAIR,
    profit_abs, profit_ratio, stake_amount, stop_loss_abs, stop_loss_ratio,
    TRADE_DURATION, weekday

This single source answers FR-001..FR-003 (exit reason, count, P&L, duration),
FR-018/FR-019 (fees), FR-020 (pair mix) **and** FR-004/FR-005 — because
`max_rate` and `min_rate` give each trade's maximum favourable and adverse
excursion directly, so stop-before-payoff needs no path re-computation from
candles at all.

**Alternatives considered**: recomputing excursions from OHLCV via
`regime/tail_path.py`. Rejected — the archive already holds the realised
excursion for the actual trade, which is strictly better evidence than a
recomputation over a nominal window, and costs nothing.

**Consequence for the plan**: the diagnosis is a reader over data already on
disk. It is minutes of work, not a measurement campaign.

---

## R2. `trailing_stop_loss` does NOT mean a trailing stop — CRITICAL

**Decision**: the exit-reason reader MUST map freqtrade's raw reasons onto the
mechanism that actually fired, and must document the mapping.

`Framework/BaseStrategy.py:319` sets `trailing_stop = False`, and
`trailing_stop_positive`/`_offset` are unreachable while it is False. Yet
`trailing_stop_loss` is the most common exit reason in the archives.

The explanation is that `use_custom_stoploss` is True and `BaseNNStrategy` sets
`use_atr_adaptive_stoploss = True`. freqtrade reports `stop_loss` when the
INITIAL stop is hit and `trailing_stop_loss` when a stop that has since MOVED is
hit. Both are the ATR adaptive band; neither is the native trailing stop.

**Without this mapping the diagnosis would attribute 40% of trades to a
mechanism that is switched off**, and would then "fix" a parameter
(`trailing_stop_positive`) already recorded as inert. Recording it here so the
task text can carry it.

---

## R3. Feasibility probe — the diagnostic works and already points somewhere

Run on one existing archive (`NNNC_Oracle`, guards-off, W3, 1588 trades). This
is a PROBE to size the work, not the US1 deliverable: it covers one arm on one
window, and W3 is not the acceptance window.

    exit_reason                n     sum P&L  mean P&L  mean hrs
    trailing_stop_loss       641     -9452.6    -14.75      11.2
    stop_loss                  3      -393.7   -131.23       3.3
    metric_overbought          1         3.3      3.28      11.2
    unclog_1d                 18         9.1      0.51      27.9
    roi                       19       446.4     23.49       5.4
    unclog_12h               428      1847.6      4.32      13.8
    take_profit              478      6371.3     13.33       5.7
    TOTAL                   1588     -1168.7

Coverage 1588/1588 — nothing unattributed, so SC-002 is achievable exactly.

Two things this already suggests, both to be established properly in US1 across
W1 and both arms:

1. **The ATR stop is the candidate dominant mechanism** — 641 trades, 40% of the
   book, mean −14.75, and it is the only large negative contributor.
2. **Cost may account for the entire loss.** A first-order fee estimate for this
   arm is ≈1468 against a net result of −1169 on a 10000 wallet. If that holds,
   the oracle's guards-off book is roughly break-even to positive GROSS and the
   loss is substantially the cost of trading 1588 times. FR-018/FR-019 exist
   precisely for this and must be settled before the label is blamed.

**This does not pre-empt US1**: one arm, one window, an approximate fee model,
and the acceptance window not yet examined.

---

## R4. Reader design — separate module, do NOT extend `run_arm.summarize()`

**Decision**: add a standalone reader; leave `regime/run_arm.py` untouched
except where already patched.

`run_arm.py` is shared infrastructure behind 480+ existing arm records. Its
`summarize()` output shape is what every one of those records was written with.
Extending it risks changing that shape, and a comparison against a historical
arm would silently straddle two formats.

**Alternatives considered**: (a) extend `summarize()` — rejected, regression risk
across 480 records for no benefit; (b) re-run arms with a richer exporter —
rejected, the data already exists.

**Consequence**: the reader takes an archive path (or resolves the newest) and
emits the breakdown. It is read-only over `user_data/backtest_results/`.

---

## R5. Associating an arm with its archive

**Decision**: the reader must be given the archive explicitly, or the arm must be
re-run immediately before reading.

`run_arm.py` records metrics but NOT which archive produced them, and
`latest_result()` returns the newest archive globally. For the six arms already
run, the archives can be matched by strategy name and trade count, both of which
are in the archive — `n_trades` in the arm record is the join key, and the
strategy name disambiguates. Where that is ambiguous, re-running the arm is
cheap and makes the newest archive unambiguous.

**Recorded risk**: matching by trade count is a heuristic. The reader must print
the archive path and the trade count it used so any mismatch is visible.

---

## R6. Varying the label and the stop requires SUBCLASSES, not `--set`

**Decision**: label and ATR-stop variants are per-variant strategy subclasses,
each with its own params file.

- Label knobs (`TRAINING_TYPE`, `MIN_BUY_GAIN_THRESHOLD`,
  `MIN_SELL_LOSS_THRESHOLD`, `HORIZON`) are CLASS attributes from
  `Framework/TrainingConfig`, not hyperopt parameters. Not reachable via `--set`.
  Precedent: `NNNC/NNNC_Brk_L1.py` (feature 003).
- ATR stop knobs are class attributes on `NNNC/NNNCStrategy.py`
  (`atr_stoploss_multiplier = 4.0`, `atr_stoploss_floor = -0.06`,
  `atr_stoploss_cap = -0.04`). Also not reachable via `--set`. Note `--stoploss`
  is INERT for NN strategies.
- Exit knobs that ARE reachable via `--set-sell`: `cexit_take_profit`,
  `cexit_max_days`, `enable_exit_signal`, `cexit_enable_profit_checks`,
  `stoploss_grace_hours`, `stoploss_grace_level`. `--roi` reaches the ROI table.
- The hold window is neither: it needs `NNNC/_hold_window.py`'s
  `HoldWindowMixin`, which is a method override.

**Consequence**: exit-side variants split into two classes — cheap ones via
`--set-sell`/`--roi` on the existing `NNNC_Oracle`, and ones needing a subclass.
Cheapest-first ordering (FR-013) should prefer the former.

---

## R7. Guards-off must neutralise thresholds, not disable the block

**Decision**: `entry_rvol_threshold=-1.0`, `entry_atr_pct=-1.0`,
`entry_adx_threshold=-1.0`, `entry_bb_width_threshold=-1.0`,
`entry_guard_threshold=5.0`, `entry_close_norm_threshold=5.0`. Never
`--guards off`.

`volume > 0` sits inside the same all-or-nothing `entry_enable_guards` block, so
`--guards off` also removes a data-sanity check. Verified that out-of-range
values are HONOURED, not clamped (`rvol=20` → 0 trades; `rvol=5` → 3 trades), and
that `guard_metric` and `close_norm` are both bounded in [−1, 1] so 5.0 is a
complete neutralisation.

Measured difference is small but real: all-six-neutralised is −14.46% vs
−14.65% for `--guards off`, same 1970 trades. Use the neutralised form for
consistency.

---

## R8. The missing W2 arms

**Decision**: fill W2 for `oracle_nog` and `gpool` as part of the baseline, before
any variant is screened.

FR-014 requires every variant on all three windows, and the baseline the
variants are compared against must be equally complete or the comparison is
asymmetric. These are two backtests, no training.

---

## R9. The gate's pass condition

**Decision**: a variant PASSES the oracle gate when, on W1, its guards-off oracle
return is **positive** AND it holds at least **200 trades**.

- *Positive, not merely improved*: the feature's premise is that until perfect
  prediction can make money without the guards there is no edge to learn. An
  improvement from −18.89% to −12% is progress worth recording but does not
  license training.
- *200 trades*: guards-on arms hold ~95, and the ~1pp power floor at that count
  is already established. A guards-off variant that collapses toward the
  guards-on count has re-imposed a gate rather than fixed the label, which
  Edge Case 4 explicitly warns against. 200 is roughly twice the guards-on count
  — low enough not to force a particular solution, high enough that a variant
  cannot pass by trading almost nothing.
- *Inconclusive*: trades identical to the variant's control (absorbed), or a
  failed/stale arm record.

**Recorded caveat**: 200 is a judgement, not a measurement. It is stated here so
it can be argued with rather than discovered later inside a task.

---

## R10. In-sample status for arms that use no trained weights

**Decision**: W1 remains the sole acceptance window for every arm, oracle or
trained. W2/W3 are reported for all arms and labelled.

For ORACLE arms the in-sample objection is weaker but not absent: no weights are
fitted, so nothing memorises the window, but the label PARAMETERS
(`HORIZON`, thresholds, method) were themselves chosen partly on data covering
W2/W3. So oracle arms may use all three windows as corroboration of a
mechanism, while acceptance stays on W1 — which also keeps oracle and trained
arms judged on the same footing, as FR-022 requires.

**Alternatives considered**: treating all three windows as equally admissible for
oracle arms. Rejected — it would let an oracle variant be accepted on evidence a
trained arm could not use, and the two are compared to each other.

---

## R11. Cost model

**Decision**: compute cost per trade from the archive's own `fee_open`/
`fee_close` and stake, and report every guards-off result gross and net.

`fee` is not set in `config/config.json`, so the exchange default applies and is
recorded per trade in the archive. `stake_amount` is 900 with
`max_open_trades` 10 on a 10000 wallet, so trade count drives cost almost
linearly. Comparing a 95-trade arm with a 1760-trade arm without this is
meaningless.

---

## Summary of decisions

| id | decision |
|---|---|
| R1 | Diagnosis reads existing archives; no re-runs. `max_rate`/`min_rate` give excursions directly |
| R2 | `trailing_stop_loss` = ATR adaptive stop, NOT native trailing. Mapping is mandatory |
| R3 | Probe confirms feasibility, 100% coverage; ATR stop and cost are the leading candidates |
| R4 | Separate reader; do not extend `run_arm.summarize()` (480+ records depend on its shape) |
| R5 | Archive must be named explicitly or the arm re-run; reader prints what it used |
| R6 | Label and ATR-stop variants need subclasses; take-profit/max-days/ROI reachable via flags |
| R7 | Guards-off by neutralising six thresholds, keeping `volume > 0` |
| R8 | Fill the missing W2 baselines before screening variants |
| R9 | Gate passes on W1 guards-off return > 0 AND >= 200 trades |
| R10 | W1 is the sole acceptance window for all arms; oracle arms may corroborate on W2/W3 |
| R11 | Cost from the archive's own fee fields; report gross and net |
