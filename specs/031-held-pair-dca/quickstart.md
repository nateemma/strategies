# Quickstart / Validation: Held-Pair Position Add (DCA)

All commands are run from `user_data/strategies` unless noted. Before any run:
`set -a; source ~/.freqtrade_env; set +a`. Run one backtest chain at a time.

## 1. Switch off: nothing changes

```bash
../../.venv/bin/python -m pytest Framework/tests utils/tests SimpleTrigger/tests research/exit_coupling/tests evaluation -q
../../.venv/bin/python -m pytest Framework/tests/test_exit_policy_single_source.py -q
```

Then, from the repo root, re-run the reference arms and compare their per-fold `trades_hash` with
their records (15/15 each): `wf_aro`, `wf_aro_g24`, `fs_gbr_gr`, `fs_gfm_gr`, `fs_mlx_gr`, `st_bgo_mx`
and `st_prom_c2`. Each runs under a `_chk` tag, with model-key and fingerprint checks.

Expected:
- every hash is equal;
- `framework_sha` is unchanged for every recorded class;
- no `ModelMissing`, and no training.

## 2. Switch on: unit behaviour

`utils/tests/test_held_pair_add.py` covers:
- each AddDecision refusal;
- the stake rule;
- the clock restart (grace, unclog, max hold, minimum hold);
- the ATR take-profit from the add's signal bar;
- the stop anchored to the averaged price;
- the stop invariants holding through an add;
- an invalid declaration raising at import.

A one-pair, one-fold backtest smoke test asserts:
- at least one trade with two filled entry orders;
- add fill date = signal candle + 1 bar;
- `open_rate` = the amount-weighted average.

## 3. Fidelity against the screen (before the pre-registration)

```bash
PYTHONPATH=user_data/strategies .venv/bin/python -m evaluation.run_folds --plan user_data/strategies/evaluation/folds/plan_hp_aro_dca.json --folds f00-f07
cd user_data/strategies && NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.fidelity hp_aro_dca --folds f00-f07
```

Expected:
- at least 95% of backtest adds match screen adds (pair, candle);
- add-candle exits are counted;
- the per-fold pp gap is reported.

## 4. Pre-register, then run the judged arms

Commit `research/held_pair/preregistration.json` and `.md`. Then run, in one sequential chain:
- `hp_aro_dca` f08–f14;
- `hp_gbr_dca`, `hp_gfm_dca` and `hp_mlx_dca` on f00–f14;
- `st_e22_hp_rev` and `st_e22_hp_mom` on e00–e02;
- `st_hp_rev` and `st_hp_mom` on f00–f14.

```bash
PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.judge
```

Expected:
- `verdicts.json` has one decision per book;
- each book's exposure, max drawdown and leverage control are present;
- every record is newer than the pre-registration commit.

## 5. Report

`research/held_pair/report.md`:
- screen prediction vs arm result vs leverage control, per book;
- evidence labels;
- the switch stays off by default.
