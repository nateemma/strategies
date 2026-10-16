# Quickstart: 037 Indicator Screen

**Prerequisites:**
- Run from `user_data/strategies` with `set -a; source ~/.freqtrade_env; set +a`.
- No training or backtest chain is running.
- The live bots are untouched throughout.

```sh
PY="../../.venv/bin/python"; export PYTHONPATH=.
M=research.indicator_screen
```

1. **Tests.** Run `$PY -m pytest research/indicator_screen/tests -q -p no:cacheprovider`.
   Expect all to pass.
2. **Scope.** Run `$PY -m $M.discover`. It should list about 67 classes in scope; every exclusion has
   a reason.
3. **Cost gate.** Run `$PY -m $M.harvest --time-one SOL/USDT`. Check the per-class wall time. The
   total must allow 11 frames in a reasonable session.
4. **Lookahead.** Run `$PY -m $M.causality`. Every FAIL column is named in `causality.json`; Ehlers
   DPO and Ichimoku chikou are the likely suspects.
5. **Harvest.** Run `$PY -m $M.harvest --pair all`. This writes 11 feathers to
   `user_data/data/indicator_screen/`.
6. **Instrument check and early labels.** Run
   `$PY -m $M.screen_single --targets labels --halves e0,early`.
   - Expect guard_metric and bb_width in the top 5 for `label:buy` on all bars (E0 + early gate).
   - If they are not, stop: the output says INSTRUMENT FAIL.
7. **Early regime screens and pair selection.**
   - `$PY -m $M.screen_single --targets regime --halves e0,early`
   - `$PY -m $M.screen_pairs --target label:buy --select` (and `--target label:sell`)
8. **Pre-registration.** Commit `research/indicator_screen/preregistration.json`.
9. **Late windows and models.**
   - `$PY -m $M.screen_single --targets labels --halves late`
   - `$PY -m $M.screen_single --targets regime --halves late`
   - `$PY -m $M.screen_pairs --target label:buy --confirm` (and `label:sell`)
   - `for f in f00 ... f14; do $PY -m $M.screen_models --target label:buy --fold $f; done`, then
     `$PY -m $M.screen_models --merge --target label:buy`. Do the same for the regime targets.
10. **Report.** Run `$PY -m $M.report`, then read `research/indicator_screen/report.md`.
    - It should answer SC-004 (a)–(e).
    - A rerun of steps 6–10 should reproduce it byte for byte (SC-007).

**Plot.** `freqtrade plot-dataframe -s AllIndicators -c user_data/strategies/config/config_wf.json
-p SOL/USDT --timerange 20250101-20250201 --indicators2 "raw_buy:CCI cci:CCI"`
