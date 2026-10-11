# Quickstart: Trigger-Family Exit Presets (024)

Run everything from `user_data/strategies`. Backtests need `set -a; source ~/.freqtrade_env; set +a`.
Run one backtest chain at a time. Every result is PROVISIONAL (gate 2).

## 0. Neutrality of the framework edit (research R9; C1)

```bash
../../.venv/bin/python -m pytest -q Framework/tests evaluation/tests SimpleTrigger/tests research/outcomes/tests
```

Then re-run `st_gbb` f00 and arm D's `wf_aro` f00 through `run_arm`, and check the 023 regression
arms `st_bbo` f00 and `stl_bo` f00.

**Expected:** each `trades_hash` equals its recorded `regime/arms/*_F00.json` value. Also run
`python -m evaluation.dump_strategy --strategy MomentumRegimeBasket15mFast --config
config/config_mom_dryrun_15m.json` before and after the edit; the output must be identical (SC-004).

## 1. Family classification (SC-006)

```bash
../../.venv/bin/python -m research.outcomes.triggers --list
```

**Expected:** every trigger shows a family, and registering a trigger without one fails the
registry test.

## 2. Family arms reproduce 023 (User Story 1, SC-001)

Two configs on `PRESET = "ARM_D_FAMILY"`: `SimpleTrigger_FamGbb` (`gbb`) and
`SimpleTrigger_FamBgo` (`breakout_gbb`). For each one:

```bash
../../.venv/bin/python -m evaluation.foldplan derive --from evaluation/folds/plan_NNNC_A_Raw_Open.json \
    --family SimpleTrigger --base SimpleTrigger_FamGbb --tag st_fam_gbb --out evaluation/folds/plan_st_fam_gbb.json
../../.venv/bin/python -m evaluation.run_folds --plan evaluation/folds/plan_st_fam_gbb.json
```

**Expected:**
- `st_fam_gbb` equals `st_gbb`, and `st_fam_bgo` equals `st_bgo_mx`, by `trades_hash` in 15/15 folds.
- `untagged_exits` = 0.
- The `unclog_12h` exit count of `st_fam_bgo` equals `st_bgo_mx`'s, which proves the momentum
  unclog setting is live.

## 3. Combined arm, sides disabled (User Story 2, SC-002)

`SimpleTrigger_FamCombined`: `TRIGGERS = ("gbb", "breakout_gbb")`, `PRESET = "ARM_D_FAMILY"`.
Two variants set `DISABLED_TRIGGERS = ("breakout_gbb",)` and `("gbb",)`.

**Expected:** the variants reproduce `st_gbb` and `st_bgo_mx` in 15/15 folds.

## 4. Combined arm, both sides (User Story 2)

```bash
../../.venv/bin/python -m evaluation.run_folds --plan evaluation/folds/plan_st_fam_combo.json
../../.venv/bin/python -m evaluation.pool --plan evaluation/folds/plan_st_fam_combo.json --out <scratch>/pooled_st_fam_combo.json
../../.venv/bin/python -m evaluation.contention --combined <scratch>/pooled_st_fam_combo.json \
    --single reversion=<scratch>/pooled_st_gbb.json --single momentum=<scratch>/pooled_st_bgo_mx.json \
    --config config/config_wf.json --out evaluation/contention_st_fam_combo.json
PYTHONHASHSEED=0 ../../.venv/bin/python -m evaluation.compare_pooled --arm <combo> --control <st_gbb>     # and vs st_bgo_mx
../../.venv/bin/python -m evaluation.halves --arm <combo> --control <st_gbb>                              # and vs st_bgo_mx
```

**Expected:** a report giving per-family trades, P&L and contention by half, paired verdicts
against both single-family arms, the sum-of-singles reference, and the per-fold correlation
between the families. Every figure is labelled PROVISIONAL.

## 5. Fingerprint (SC-003)

Run `pytest evaluation/tests/test_fingerprint.py`.

**Expected:** arm D's sha is still `303456c1fa322b4c`. Changing one family value, the priority,
or the disabled set changes the sha.
