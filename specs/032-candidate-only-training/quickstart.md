# Quickstart / Validation: Candidate-Only Training (032)

All commands run from `user_data/strategies` unless noted. Load the environment first:
`set -a; source ~/.freqtrade_env; set +a`.

## 1. Switch-off parity and unit checks (no training)

```sh
../../.venv/bin/python -m pytest Framework/tests utils/tests SimpleTrigger/tests \
  research/exit_coupling/tests research/held_pair/tests evaluation -q -p no:cacheprovider
```

Expected: everything passes, including:
- the new row-filter and prepare_training_data parity tests;
- the model-key audit;
- the pinned model keys and fingerprints (arm D `ff03db982af38fa6`, add off at 6.5 h
  `303456c1fa322b4c`).

## 2. Build arms and commit the pre-registration (before any training)

```sh
PYTHONPATH=. ../../.venv/bin/python -m research.candidate_training.build_arms
git add <named files>; git commit   # preregistration.json committed BEFORE step 3
```

## 3. Train, pick thresholds, backtest (one model at a time)

```sh
PYTHONPATH=. ../../.venv/bin/python -m research.candidate_training.train --model ARO   # ~3 h
cd ../.. && PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/evaluation/pick_threshold.py \
  --plan user_data/strategies/evaluation/folds/plan_co_aro.json --target-rate 0.014735 \
  --out user_data/strategies/evaluation/folds/thresholds_co_aro.json
# write the map into plan_co_aro.json "thresholds", then:
PYTHONPATH=user_data/strategies .venv/bin/python -m evaluation.run_folds \
  --plan user_data/strategies/evaluation/folds/plan_co_aro.json      # evaluates only
```

Expected:
- every fold's `row_filter.json` shows a kept share of about 20%, a BUY share of about 50% and
  SELL 0;
- each fold's achieved rate is about 1.47%;
- 15/15 folds are evaluated with trades.

Then repeat for GBR.

## 4. Stages

```sh
PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.signals --models CO_ARO,CO_GBR
PYTHONPATH=. ../../.venv/bin/python -m research.candidate_training.auc
PYTHONPATH=. ../../.venv/bin/python -m research.candidate_training.judge
```

Expected:
- `verdicts.json` has S1/S2/S3 per model, in order, with later stages marked `not_reached`
  after a failure;
- simulator fidelity is ≥ 95%;
- entry violations are 0.
