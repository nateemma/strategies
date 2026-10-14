# Contract C2: research/label_screen command line

All commands run from `user_data/strategies` with `PYTHONPATH=. ../../.venv/bin/python -m`.
Every long-running command refuses to start while another chain runs.

| command | does | writes | refuses when |
|---|---|---|---|
| `research.label_screen.build_arms [--model ARO]` | generate classes and `sz`/`es` plans from the control plan | `NNNC/NNNC_LS_*`, `evaluation/folds/plan_ls_*` | — |
| `research.label_screen.audit [--folds f00..]` | S0a label audit, no model | `audit.json` | — |
| `research.label_screen.train --tag T [--folds ...] [--seed N]` | train via the 032 driver | weights at model-key paths | another chain runs; candidate failed S0a |
| `research.nnnc_framework.signals --models LS_T` | test-window signal tables | `signals/LS_T_fNN.parquet` | missing weights (strict) |
| `research.label_screen.pick --tag T --fold fNN --role sz|es [--verify]` | one fold's threshold (sz: 0.5 share; es: rate 0.014735) | the plan's `thresholds` | more than one fold requested |
| `research.label_screen.instrument` | S0 per candidate | `instrument.json` | pre-registration uncommitted |
| `research.label_screen.separation` | S1, early half only | `separation.json` | — (no `--late` option exists) |
| `research.label_screen.survivors` | apply the rule | `survivors.json` | S1 incomplete |
| `evaluation.run_folds --plan plan_ls_T_R --folds f00..f07` | S2 backtests | arm records | — |
| `research.label_screen.judge [--seed-repeat 2]` | staged verdicts | `verdicts.json` | pre-registration dirty/uncommitted; a record older than its commit |
