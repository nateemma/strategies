# Contract: 038 command-line entry points

Run every command from `user_data/strategies` as
`PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.<module> …`.

Each command:
- writes its outputs atomically (to a temporary file, then a rename);
- exits with 0 on success, 2 on a refusal and 1 on an error;
- prints one summary line per cell.

| Module | Arguments | Reads | Writes | Refuses when |
|---|---|---|---|---|
| `frames` | `[--pair P]` | feathers, `plan_cs_aro.json` | `forecast_038/frames/<PAIR>.feather` | the fold windows do not match the plan |
| `baselines` | none | frames | `forecast_038/pred/zero.feather`, `hist_mean.feather`; `research/forecast/baselines.json` (horizon table, cost line, baseline scores) | — |
| `instrument` | none | baselines, the tests' exit status | `research/forecast/instrument.json` | — (it reports PASS/FAIL per check) |
| `stat_tier` | `[--model M] [--workers 4]` | frames | `forecast_038/pred/<model>.feather` | the pre-registration is not committed |
| `learned_tier` | `--set {L,LA,LW,null} [--seed 1]` | frames, the 037 cache (LA) | `forecast_038/pred/lgb_<set>[_s2].feather` | the pre-registration is not committed; the LW inputs failed truncation |
| `truncation` | none | frames | `research/forecast/truncation.json` | — |
| `score` | `[--method M]` | predictions, frames | `research/forecast/scores.json` | — |
| `verdict` | none | scores, `instrument.json`, pre-registration | `research/forecast/verdicts.json` | see data-model § Pre-registration |
| `report` | none | verdicts, scores, baselines | `research/forecast/report.md` | — |

## Library contract (utils/Forecasters.py, after the fixes)

- `forecast(data, steps)` returns an array whose last `steps` elements are the forecasts for 1…`steps`
  candles ahead, in order. Element `[-1]` is exactly `steps` ahead.
- `forecast` raises `Forecasters.ForecastFailed` when the underlying fit fails. It never returns
  zero-filled output in place of a forecast.
- `FFT_EXTRAPOLATION.forecast` raises `Forecasters.ForecastUnavailable(reason)`.
- With detrending on, the trend that is restored comes from the window being forecast.
