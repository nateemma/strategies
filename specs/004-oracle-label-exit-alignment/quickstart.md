# Quickstart — 004 Oracle-First Label/Exit Alignment

Validation scenarios proving the feature end to end. Run from the repository
root `/Users/philprice95/projects/freqtrade`. Nothing here trains a model.

## Window treatment (applies to every scenario)

W1 `20250901-20260831` is the SOLE acceptance window. W2 `20240601-20250531` and
W3 `20230101-20231231` fall inside the model's training window. For ORACLE arms
no weights are fitted, so those windows may CORROBORATE a mechanism — but they
never confer acceptance, because the label parameters were themselves chosen
partly on data covering them (R10, FR-022; Constitution Principle II
PASS-WITH-CAVEAT). No scenario below treats W2 or W3 as out-of-sample evidence.

Arms are run SEQUENTIALLY, never concurrently (FR-025).

## Prerequisites

- Existing arms: `oracle`, `oracle_nog`, `gctl`, `nomodel`, `gpool`, `gqoff` in
  `regime/arms/`
- Existing hosts: `NNNC/NNNC_Oracle.py` + `.json`, `NNNC/NNNC_NoModel.py` + `.json`
- Backtest archives present in `user_data/backtest_results/`

Guards-off is always expressed by neutralising thresholds (R7), never
`--guards off`:

    NEUT="--set entry_rvol_threshold=-1.0 --set entry_atr_pct=-1.0 \
          --set entry_adx_threshold=-1.0 --set entry_bb_width_threshold=-1.0 \
          --set entry_guard_threshold=5.0 --set entry_close_norm_threshold=5.0"

## Scenario 1 — Complete the baseline (Phase 1)

    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/run_arm.py --strategy NNNC_Oracle \
      --window W2 --tag oracle_nog $NEUT

Repeat for `NNNC_NoModel` / `gpool`.

**Expected**: `returncode` 0, `n_trades` of the order of the W1/W3 counts
(1500–6000), and a `wallet_profit_pct` recorded. A record carrying an `error`
field means the run failed — it is not evidence (FR-026).

## Scenario 2 — The diagnosis (Phase 2, the hard gate)

    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/trade_forensics.py \
      --match-trades 1760 --strategy NNNC_Oracle --section all

**Expected**: coverage exactly 100%; the archive path, strategy and trade count
printed; the mechanism mapping table shown with `trailing_stop_loss` reported as
the ATR adaptive stop; mechanisms ordered with the largest P&L destroyer first;
every row carrying its trade count.

**Gate**: the dominant mechanism must be named, with its share of destroyed P&L
and its trade count, and the cost question answered — does the guards-off loss
survive removal of fees — before any parameter is varied.

## Scenario 3 — The gate applied to a worked example (Phase 3)

    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/run_arm.py --strategy NNNC_Oracle \
      --window W1 --tag ora_tp040 $NEUT --set-sell cexit_take_profit=0.040

    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/tail/compare.py ora_tp040 oracle_nog

**`compare.py` does NOT emit the gate verdict.** It emits its own four-way
classification — `inconclusive_absorbed` / `improved` / `worsened` / `unchanged`
— using `POWER_THRESHOLD_PP = 1.0`, a floor calibrated for ~97-trade arms. On a
1760-trade guards-off arm it would report `improved` for a +2pp change that is
still deeply negative, which is exactly the misreading to avoid. Use it here for
its liveness comparison (trade hashes), its trade counts and its in-sample
labelling — **not** its verdict line.

**Expected**: from `compare.py`, the liveness rows and per-window figures. The
GATE verdict is then applied by hand (or by the rule recorded in
`regime/ORACLE_GATE.md`): PASS requires W1 guards-off return **> 0** AND at least
**200 trades**. A variant whose trades are identical to `oracle_nog` is
inconclusive, not refuted. A variant that improves but stays negative is a
`fail` that records progress and does not license training.

## Scenario 4 — Separation of contributions (Phase 4)

Run a label-side variant and an exit-side variant independently before any
combined variant, and report their contributions separately (FR-015).

**Expected**: three arms whose relationship is arithmetically valid — the
combined arm is compared against arms that differ from it in exactly one
respect, never against an arm differing in two.

## Scenario 5 — Hygiene at the end

    git status --short Framework/     # must be clean (FR-024, SC-008)
    grep -rn "ceiling" regime/ORACLE_GATE.md   # lookahead labelling present

**Expected**: no shared framework file modified; every oracle figure in the
findings labelled a lookahead ceiling; no oracle configuration promoted.

## Success criteria coverage

| SC | validated by |
|---|---|
| SC-001, SC-002 | Scenario 2 |
| SC-003 | Scenario 3 |
| SC-004, SC-005 | Scenarios 3 and 4 |
| SC-006 | Scenario 2 (costs section) |
| SC-007 | Phase 5 conclusion |
| SC-008 | Scenario 5 |
| SC-009 | Scenarios 3 and 5 |
| SC-010 | Phase 5 conclusion |
