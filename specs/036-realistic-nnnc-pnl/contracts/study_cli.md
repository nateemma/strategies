# Contract: study tools (`research/live_pnl/`)

Run them from `user_data/strategies` with `PYTHONPATH=.` and `../../.venv/bin/python -m
research.live_pnl.<tool>`, unless noted.

| tool | does | refuses when |
|---|---|---|
| `live_fold --cut YYYYMMDD [--run-id liveN]` | writes `plan_live_aro.json` (fold liveN), generates the class, trains (one trainer; the driver's busy check), picks the threshold (target 0.014735), writes the `_REC` class, appends a LiveModelRecord | another trainer is running; the cut is after the last downloaded candle; liveN already exists |
| `scripts/live_nnnc.sh start\|stop\|status` (zsh, repo root) | starts or stops the dry-run bot for the current `live` record, using its own config, port 8081, database, log and pid file | port taken; database path equals the momentum bot's; already running; key/weights mismatch (strict mode) |
| `parity --day YYYY-MM-DD` | the R4 report for one day | the day has no signal log |
| `gate` | writes `gate.json` | — |
| `cost_model --fit` | fits the R5 model from recorded depth, holdout = the last 25% of days, writes `cost_model.json` | gate closed; the fitting rule is not committed |
| `restate --arm TAG [--folds …]` | RestatedResult for an arm | no cost model; arm records missing |
| `capacity` | judges the capacity arms (R8) into `capacity.json` | gate closed; pre-registration uncommitted; any arm INERT (reported); `model_check` bias flag open (no result offered for adoption) |
| `model_check` | the R13 comparison of recorded orders against cost-model predictions, writes `model_check.json` | no cost model |
| `drift` | weekly live BUY rate against target (R11) | — |

Every tool writes JSON with a `label` field ("PROVISIONAL …"), and never writes into
`user_data/data/live_nnnc/` except through the recorder.
