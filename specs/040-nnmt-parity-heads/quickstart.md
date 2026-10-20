# Quickstart: 040 NNMT Parity and Head Study

Run these from the freqtrade root, after `set -a; source ~/.freqtrade_env; set +a`.
`PY=.venv/bin/python`.
- Run ONE chain at a time (MLX training or HGB fits). Check after each arm that both bots are still
  up.
- `S=user_data/strategies`.

## 1. Code and tests

```sh
cd $S && ../../.venv/bin/python -m pytest NNMT/tests research/nnmt/tests Framework/tests/test_model_key.py -q
```

**Expected**: all pass (model key, identity wiring, `LSTM_RAW` shapes, head capture, transition
causality).

## 2. Model stage (after committing `preregistration_models.json`)

```sh
cd $S && PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.build_arms models
cd ../.. && PYTHONPATH=$S $PY $S/evaluation/run_folds.py --plan $S/evaluation/folds/plan_mt.json --folds f00   # time f00 first
# then: the rest of plan_mt, then the port lands, then plan_mt_fw0 (no training: same key)
cd $S && PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.judge identity
# then plan_mt_op (no training), and the four variants (training), then:
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.judge models
```

**Expected**: identity 15/15, the model-decided shares, and `verdicts_models.json` with the R7
selection.

## 3. Framework stage and sizer (after committing `preregistration_framework.json` and `preregistration_screens.json`)

```sh
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.build_arms framework
# run_folds each (only _tr trains); then:
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.judge framework
PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.signals --models <M>
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.screen sizer
```

**Expected**: four feature verdicts, the twin's verdict, and the sizer screen (an arm only if it
passes).

## 4. Heads

```sh
# run_folds plan_<M>_ah (trains); signals --models <M>_ah; then:
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.head_screen
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.build_arms heads   # passing heads only
```

**Expected**: `screens_heads.json` with a pass or fail per head, and arms only for passes.

## 5. Transition (after committing `preregistration_transition.json`)

```sh
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.transition labels
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.transition causality   # must exit 0
for f in f00 f01 ...; do PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.transition screen --fold $f; done
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.transition judge
PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.build_arms transition  # passing cells only
```

**Expected**: base rates per fold, a truncation PASS, a pass or fail per cell, and arms only for
passing cells.

## 6. Report

`research/nnmt/report.md`: one table of every arm's per-half result against its comparator and
against `cs_aro`, `w_aro`, `st_g24_rev` and `wf_aro_g24`, then the three FR-015 verdicts.
