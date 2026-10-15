# Contract: 035 command-line tools

All run from `user_data/strategies` with `PYTHONPATH=.`, except `run_folds` (repo root,
`PYTHONPATH=user_data/strategies`). Each tool writes its JSON in `research/new_inputs/`, and
refuses (exit 2) when its precondition is unmet.

| tool | does | precondition / refusal |
|---|---|---|
| `fetch --source {global_spot,us_spot,um_metrics,um_premium} [--pairs ...]` | resumable download into `user_data/data/external/`, checksum verified, manifest updated | network; never runs while a trainer or backtest chain is running |
| `build [--pairs ...]` | writes `user_data/data/new_inputs/<PAIR>.parquet` and `build_meta.json` | sources verified in the manifest |
| `causality` | R12 truncation test, all inputs; writes `causality.json` | a build is present |
| `coverage` | per input, pair and fold bar coverage; writes `coverage.json` (S0) | a build is present |
| `screen` | S1: reproduction gate, then every input's row; writes `screen.json` | pre-registration committed and clean; causality all PASS (an input failing causality is INCONCLUSIVE, not scored) |
| `survivors` | applies FR-010 and writes `survivors.json` | screen.json present |
| `build_arms` | generates `NNNC_NI_<TAG>[_SZ]` classes, `.json` params copied from the control, and plans | survivors.json committed |
| `train --tag T [--folds] [--seed]` | the 032 driver, early folds by default; late folds only for the S2 winner | another chain running; tag not a survivor |
| `pick --plan plan_ni_<tag>_sz --fold fNN [--verify]` | the control-sizer picker with a plan argument | one fold per process |
| `separation` | S2: the gap per survivor on the NI tables vs 0.1004 | survivors trained, picks verified |
| `instrument` | S3 checks per half (control-sizer instrument) | S3 records exist |
| `judge` | staged verdicts; judge_one vs `cs_aro` and `w_aro` | pre-registration committed and clean; no `ni_*` record older than its commit |

Invocation: `python -m research.new_inputs.<tool>`.
