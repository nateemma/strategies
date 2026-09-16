# Quickstart / Validation Guide: signal-context-timing

All commands run from `user_data/strategies`, **sequentially**, under
`/Users/philprice95/projects/freqtrade/.venv/bin/python`.

```bash
PY=/Users/philprice95/projects/freqtrade/.venv/bin/python
$PY -c "import sys; print(sys.prefix)"   # must end in freqtrade/.venv
```

## 0. Rescue the side-cars (do this FIRST)

```bash
$PY -m research.signalctx.signals --preserve
```
Copies **only** `/tmp/lf/es_sink` (30 side-cars, **366 MB**) to durable, **gitignored** storage.
Without it a `/tmp` clear turns a free study into ~44 minutes of backtests.

**The 2.1 GB panel cache at `/tmp/lf/es_panels` is deliberately NOT copied** — measured
2026-09-16, it rebuilds from side-cars in 3.4 s per fold (~1.7 min for all 30). Preserving it
would add 2.1 GB to the repository to save under two minutes.

## 1. Unit tests

```bash
$PY -m pytest research/labelfree/tests/test_signalctx_*.py -q
```

| test | proves |
|---|---|
| `test_reconstruction_fidelity` | recorded trades appear as Signals (≥0.95) |
| `test_runs_collapse_to_events` | 139 bars are 120 events, not 139 |
| `test_prev_signal_won_is_null_while_open` | the likeliest leak is closed |
| `test_path_reconciles_with_archive` | unrealised P&L reconstruction is right |

## 2. Signal census — no backtest

```bash
$PY -m research.signalctx.signals --fold all --census
```
Expected: per fold, signal events vs admitted trades vs masked, with the fidelity rate. f08
reference: **120 events, 71 admitted, 49 masked (41%)**, fidelity 0.986.

**This is a gate, not a report.** Any fold below 0.95 fidelity is unusable, and if several fail
the instrument is dead and Phases 3-7 are wasted. It costs seconds — run it before building
anything on top.

## 3. US1 — the confound gate (the cheapest kill)

```bash
$PY -m research.signalctx.confound --fold all
```
Expected: CMH statistic and p, the logistic coefficient, and **two effect sizes** —
`effect_winrate_pp` (reported) and `effect_return_pp_yr` (gated against 7.633). Only the second
may meet the floor; comparing the first to it is a units error the implementation must refuse.
`proxy_for_drawdown` stops the repeat-signal arm and writes `exploratory: true` into
`out/confound.json`, which `sweep.py` and `report.py` read from disk.

## 4. US3 — first-signal context sweep

```bash
$PY -m research.signalctx.sweep --fold all
```
Four pre-declared hypotheses at **α = 0.0125**, with their windows frozen (1-day trailing high;
6-hour burst gap). Both effect sizes beside every p; anything whose **return consequence** is
inside 7.633 pp/yr is "not resolved".

## 5. US4 — masked-signal outcomes (the fenced instrument)

```bash
$PY -m research.signalctx.masked --fold all
```
Runs stacked backtests. **Every figure is labelled not reproducible in dry/live**, and none may
reach a P&L comparison.

## 6. US2 — exit-side rule, redundancy first

```bash
$PY -m research.signalctx.arms --rule tighten_on_repeat --check-overlap
```
Reports overlap with `stoploss_grace` and the ATR stop **before** any return figure.
**≥ 0.70 is inertness**, not a result. The rule is stop-tightening by a frozen factor of 0.5.

## 7. US2/US5 — evaluate (15 backtests per arm, no retraining)

```bash
$PY -m research.signalctx.arms --rule tighten_on_repeat --fold all
```
Asserts weight reuse and byte-identical entries before running. The `exit_on_repeat` sensitivity
runs **only if the primary arm clears the verdict** — 15 backtests for something that cannot
produce a verdict are not spent speculatively.

## 8. Report

```bash
$PY -m research.signalctx.report
```

---

## End-to-end expectation

The pre-registered expectation is that **US1 kills the repeat-signal arm** — "received a repeat
signal" most likely restates "is underwater". That is a complete, reportable result reached with
zero backtests, after which US3 continues as EXPLORATORY.
