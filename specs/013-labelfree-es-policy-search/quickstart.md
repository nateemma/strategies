# Quickstart: labelfree-es-policy-search

How to run the feature end to end and what each step must produce before the next may start.
All commands run from `user_data/strategies`, sequentially.

## Prerequisites

- `~/.freqtrade_env` sourced explicitly (it is interactive-shell only):
  `zsh -c 'source ~/.freqtrade_env && <command>'`
- the 15-fold plan and the archived 378 trades present under `regime/wf/`
- `research/labelfree/` importable; `admissibility.py`, `trials.py`, `sim.py`, `stats.py` unchanged

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

**Expect**: green. The decisive test replays the incumbent parameters over a fold and compares
the surrogate's exit-reason mix against the archive's — `take_profit` ≈ 52%, the ATR stop ≈ 26%.
A surrogate that cannot reproduce the incumbent's own exit mix cannot rank anything.

## Step 3 — THE BLOCKING GATE

```
zsh -c 'source ~/.freqtrade_env && python -m research.labelfree.es.calibrate --n 40 --folds f05..f14 --out calibration.json'
```

Takes ≈2 hours (40 freqtrade backtests at ~175 s). **Expect**: a reported `spearman_rho` and an
empty `win_disagreements`.

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
for f in f00 f01 f02 f03 f04 f05 f06 f07 f08 f09 f10 f11 f12 f13 f14; do
  python -m research.labelfree.es.run --fold $f --method cmaes  --budget 1600 --seed 7
  python -m research.labelfree.es.run --fold $f --method random --budget 1600 --seed 7
done
```

**Expect**: 48,000 evaluations in the sidecar, and the two methods' counts equal per fold. Each
fold's test range is opened exactly once, after that fold's search ends.

## Step 6 — prove the test ranges were not leaked

```
python -m research.labelfree.es.history --audit
```

**Expect**: exit 0, with all four invariants reported — one test evaluation per fold, each later
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
