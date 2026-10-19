# Quickstart: 039 NNPredict Improvement Study

Run these from the freqtrade root, after `set -a; source ~/.freqtrade_env; set +a`.
`PY=.venv/bin/python`. Run one chain at a time and check after each arm that both bots are still up
(see `holt_chain.sh` for the pattern).

## 1. Fixes and tests

```sh
cd user_data/strategies && ../../.venv/bin/python -m pytest NNPredict/tests Framework/tests/test_model_key.py utils/tests -q
```

**Expected**: all pass, including the new model-key, artifact, inputs-causality and framework-identity
tests.

## 2. Model stage (after committing preregistration_models.json)

```sh
for arm in nnp_ridge nnp_ridge_sa nnp_lstm_raw nnp_lstm_ind nnp_lstm_ret; do
  PYTHONPATH=user_data/strategies $PY user_data/strategies/evaluation/run_folds.py \
      --plan user_data/strategies/evaluation/folds/plan_$arm.json
done
cd user_data/strategies && PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.judge models
```

**Expected**: `verdicts_models.json` and the R7 selection.

## 3. Framework stage (after committing preregistration_framework.json)

```sh
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.build_arms framework   # <M>_fw0, _g24, _hpa, _famm, _gate
# run_folds for each, fw0 first; then:
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.judge framework
```

**Expected**: identity holds on 15/15 folds before any feature verdict is read.

## 4. Sizer stage (after committing preregistration_sizer.json)

```sh
PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.signals --models NNP
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.screen
# only if the screen passes:
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.build_arms sizer
# run_folds plan_nnp_cs; then:
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.judge sizer
```
