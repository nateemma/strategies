# Quickstart: 038 Close-Price Forecast Study

Run these from `user_data/strategies`, with `source ~/.freqtrade_env` loaded. Set
`P="PYTHONPATH=.:evaluation ../../.venv/bin/python"`.

## 0. Explore (no verdicts)

```sh
cd utils && ../../../.venv/bin/python test_forecasters.py   # plots on the restored test_data.npy
```

**Expected**: a plot window opens with no exception.

## 1. Library fixes (Story 1)

```sh
../../.venv/bin/python -m pytest utils/tests/test_forecaster_horizons.py -q
```

**Expected**: every known-answer test passes:
- each statistical forecaster's line forecast is exact at h = 1, 4 and 96;
- ARIMA's failure raises `ForecastFailed`;
- FFT_EXTRAPOLATION raises `ForecastUnavailable`.

## 2. Frames and baselines (Story 2)

```sh
eval $P -m research.forecast.frames
eval $P -m research.forecast.baselines
eval $P -m research.forecast.instrument
```

**Expected**:
- `baselines.json` has a horizon table for 10 pairs × 8 N.
- `instrument.json` shows `zero_skill == 0.0` exactly, and less than 1% of eligible candles unscored
  for reasons other than gaps.

## 3. Pre-register (before any model result)

```sh
git add research/forecast/preregistration.json && git commit -m "038: pre-register pass rule"
```

## 4. Tiers (Stories 3–5), one at a time

```sh
eval $P -m research.forecast.stat_tier --workers 4        # about 20 min
eval $P -m research.forecast.learned_tier --set null       # null control first
eval $P -m research.forecast.learned_tier --set L
eval $P -m research.forecast.learned_tier --set LA
eval $P -m research.forecast.truncation && eval $P -m research.forecast.learned_tier --set LW
```

## 5. Score, verdict, report (Story 6)

```sh
eval $P -m research.forecast.score
eval $P -m research.forecast.verdict
eval $P -m research.forecast.report
```

**Expected**: `verdicts.json` has one verdict per (method, N), plus `most_viable_n`, `break_even_n` and
`null_passes`. Any learned PASS then needs a seed-2 rerun:

```sh
eval $P -m research.forecast.learned_tier --set <S> --seed 2
```

After it, run `score` and `verdict` again.

## Harness tests

```sh
../../.venv/bin/python -m pytest research/forecast/tests -q
```
