# CLI Contract: 037 Indicator Screen

All commands run from `user_data/strategies` as `PYTHONPATH=. ../../.venv/bin/python -m
research.indicator_screen.<module> ...`. Exit codes: 0 success, 1 refused (stated reason), 2 usage error.

## discover

`discover [--json]`: prints in-scope classes and exclusions (R2). It writes nothing.

## harvest

`harvest --pair <PAIR|BTC/USDT|all> [--force]`:
- builds or refreshes `user_data/data/indicator_screen/<PAIR>.feather` plus its sidecar;
- updates `manifest.json`;
- a cache is reused only when its sidecar `code_sha` matches;
- `--pair all` runs the pairs sequentially in one process;
- refuses if a training or backtest chain is running (the `ps` check used by
  `research/live_pnl/parity.py`).

`harvest --time-one <PAIR>`: harvests one pair into a scratch path and prints the wall time per
class (the cost gate, R13).

## causality

`causality [--pairs SOL,LTC,AAVE]`:
- runs the R12 truncation test for every column;
- writes `causality.json` and marks FAIL columns in `manifest.json`;
- harvest caches written before the causality run must be rebuilt (the sidecar records the
  causality sha).

## screen_single

`screen_single --targets labels|regime [--horizons 48,288,672] [--halves e0,early[,late]]`:
- writes `single_labels.json` / `single_regime.json`;
- `labels` first runs the instrument check. If it fails, it writes `{"instrument": "FAIL", ...}`
  and exits 1 without scoring other columns;
- `late` is refused unless the pre-registration lock passes (R15).

## screen_pairs

`screen_pairs --target <id> [--select | --confirm]`:
- `--select` uses the early windows (and E0, descriptively) and writes the survivors;
- `--confirm` scores the survivors on the late windows (lock required);
- writes `pairs_<target>.json`.

## screen_models

`screen_models --target <id> --fold fNN`:
- one fold per process;
- writes `models/<target>/<fold>.json`;
- late folds require the lock.

`screen_models --merge --target <id>`: aggregates the per-fold files into `models_<target>.json`.

## report

`report`:
- reads every output and writes `report.md`;
- includes the SC-004 lists (a)–(e), every exclusion, and the instrument check;
- labels every result as a SCREEN (FR-025).

## AllIndicators (strategy)

`freqtrade plot-dataframe -s AllIndicators -c user_data/strategies/config/config_wf.json -p SOL/USDT
--indicators2 "<cols>"`. It never enters or exits.
