# CLI Contract: `research/clfstudy`

All commands run from the repo root `user_data/strategies` with the venv interpreter:
`PYTHONPATH=. /Users/philprice95/projects/freqtrade/.venv/bin/python -m research.clfstudy.<cmd>`.
Every command refuses to run outside the `.venv` (guard on `sys.prefix`).

| command | arguments | effect |
|---|---|---|
| `splits build` | `--set v1 [--out splits/v1.json]` | builds and validates the split set; prints per-window row counts and excluded pairs |
| `run` | `--config <yaml or inline json> --window A\|B\|C\|all --seed N [--save-weights] [--dry-run]` | one experiment; appends one row per window; exit 0 on `ok/collapsed/class_absent`, 1 on `failed` |
| `phase` | `--phase N [--only <config_id,...>] [--retry-failed] [--max-runs K]` | runs every (config, window, seed) in `configs/phase_N.yaml` not already in results; prints skipped/ran/failed counts |
| `leaderboard` | `[--phase N] [--out report/]` | writes `results.csv`, `results.parquet`, `leaderboard_<phase>.csv|md`, and plots |
| `gate` | `--phase N [--margin-floor 0.02] [--std-bound 0.03]` | applies FR-028 with the adaptive margin; writes `gates/phase_N.{json,md}`; prints the proposed viable set |
| `importance` | `--config-id ID --window W [--group]` | permutation importance on a saved model; writes `report/importance_<id>_<W>.csv` |
| `report` | | assembles `report/report.md` from leaderboard, gates and importance outputs |
| `calibrate` | `--arch LSTM` | one run per window at the study's train settings; prints minutes/run and the projected phase totals |

Exit codes: 0 success, 1 run failure, 2 invalid config/split, 3 wrong interpreter.

Resumability guarantee: `phase` never re-trains a run key already present with status
`ok`, `collapsed` or `class_absent`; a `failed` key is retried only with `--retry-failed`.
