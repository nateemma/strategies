# Quickstart: labelfree-es-policy-search

How to run the feature end to end and what each step must produce before the next may start.
All commands run from `user_data/strategies`, sequentially.

## Prerequisites

- `~/.freqtrade_env` sourced explicitly (it is interactive-shell only):
  `zsh -c 'source ~/.freqtrade_env && <command>'`
- the 15-fold plan and the archived 378 trades present under `evaluation/`
- `research/labelfree/` importable; `admissibility.py`, `trials.py`, `sim.py`, `stats.py` unchanged

## Step 0 — regenerate the per-bar probability side-cars

```
python -m research.labelfree.es.signals --folds f00..f14 --which both
```

**Expect**: 15 folds x 2 ranges, ~12 minutes, 9-10 pairs each. Without these the entry guards
cannot be recomputed and the search space collapses to exits only.

## Step 1 — the space round-trips

```
python -m research.labelfree.es.space --show
python -m research.labelfree.es.space --check
```

**Expect**: 10 dimensions with incumbent values matching `NNNC/NNNC_MLX.json`
(`cexit_take_profit` 0.024, `stoploss_grace_hours` 6.5, `entry_rvol_threshold` 2.0) and
`Framework/BaseStrategy.py` (`atr_stoploss_multiplier` 2.5). `--check` exits 0.

## Step 2 — the surrogate reproduces the incumbent's own trades

```
pytest tests/test_es_exits.py -x
```

**Expect**: 12 passed. The decisive test is `test_incumbent_exit_mix_matches_a_real_backtest`,
which replays the incumbent over f07's training range and compares the exit-reason counts
against a **real freqtrade backtest** (take_profit 211, trailing_stop_loss 176, model_exit 36,
stop_loss 2, total 478). It skips if the f07 side-car is absent, so run Step 0 first.

This is the check that found every real defect in the surrogate — four bugs, each on
`take_profit` or the stop. None of them failed any other test in the file, because every other
test is self-consistent with the surrogate's own logic.

## Step 3 — THE BLOCKING GATE

```
zsh -c 'source ~/.freqtrade_env && python -m research.labelfree.es.calibrate --n 40 --folds f05..f14 --out calibration.json'
```

Takes ~25 minutes (40 freqtrade backtests at ~35 s on cached data; the ~175 s figure in the plan was a full walk-forward arm run, not a single fold backtest). **Expect**: a reported `spearman_rho` and agreement on the winner.
**Measured 2026-09-14**: rho **+0.9766**, tau +0.8952, agree-on-winner True, over 21 comparable
candidates in 24 minutes. Note ~half of uniformly drawn candidates come back inadmissible
(19 of 40) — roughly half this space describes a policy that barely trades.

**If this fails, the feature stops here** and the result report says so. Everything downstream
refuses to run without a passing `calibration.json`, because an uncalibrated surrogate means the
entire search budget optimises a fiction.

## Step 4 — which dimensions can move anything

```
python -m research.labelfree.es.liveness --folds f05..f14 --out liveness.json
```

**Expect**: a span in pp per dimension scored against the pre-registered thresholds
(`inert` < 0.10 pp/yr, `weak` < 3.35, else `live`). The census (research.md R2) predicts the ATR
and take-profit dimensions are live; any dimension reported `inert` is carried into the report
rather than searched. `hold_window_hours` sits on its lower bound by construction — the mechanism
can only widen the window — and that is reported as a known one-sided bound, not as a space drawn
too narrowly.

## Step 5 — search and control, identical budgets

```
python -m research.labelfree.es.drive --folds f00..f14 --budget 1600 --seed 7
```

**Expect**: 48,015 evaluations in the sidecar (48,000 search + 15 single test evaluations), the
two methods' counts equal per fold, and each fold's test range opened exactly once.
**Measured**: 66 minutes, 9,869 of 48,015 evaluations inadmissible (20.6%).

## Step 6 — prove the test ranges were not leaked

```
python -m research.labelfree.es.history --audit
```

**Expect**: exit 0, with all four invariants reported. **Measured**: `problems: []`,
15 folds, 15 test evaluations, 0 abandoned-unscored, shared summary row matching — one test evaluation per fold, each later
than every train evaluation for that fold, every evaluation written before it was scored, and the
shared trial log's summary row matching the sidecar's line count and sha256.

## Step 7 — the report

```
python -m research.labelfree.es.report --out research/label-free-study/briefs/results/05-2026-09-13.md
```

**Expect** the protocol §5 verdict — pooled return vs the control against the 3.35 pp/yr floor
with a bootstrap interval, per-fold positivity (≥10/15), W1/W2/W3 persistence, and the four
baselines against the +7.86 pp/yr bar — and then, with PBO as the headline and the evaluation
count beside it:

- `PBO > 0.2` → **KILL**, and the measured PBO *is* the finding (clarification Q3)
- `|delta vs incumbent| < 4.93 pp` → **NOT RESOLVED**, never "improvement"
- CMA-ES not beating the random control → **KILL**, recorded as the optimiser fitting noise
- a parameter on a bound → reported as a space drawn too narrowly

The report is written in every case, including a kill.

## What a pass actually requires

Three conditions together, none of which the prior evidence suggests: beat the random control,
sit more than 4.93 pp from the incumbent, and clear the 3.35 pp/yr floor — with PBO ≤ 0.2.
A quantified negative is the expected deliverable, and is how briefs 01–04 delivered their value.
