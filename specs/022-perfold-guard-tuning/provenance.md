# US1 — Provenance verdict

**Required by FR-001 to exist before any arm runs. Completed 2026-10-10, no compute.**

## Verdict

| field | value |
|---|---|
| `parameter_set` | BaseStrategy's pre-2026-09-06 tuned guard block |
| `tuning_window` | **2024-05-11 → 2026-05-01** (720 days) |
| `overlaps_folds` | **YES — 9 of 15 folds, 53.5% of all test-window days** |
| **`arm_b_status`** | **`in_sample`** |

## Evidence

The six values (`entry_atr_pct` 0.013, `entry_adx_threshold` 10.0,
`entry_bb_width_threshold` 0.094, `entry_close_norm_threshold` 0.6,
`entry_guard_threshold` 0.1, `entry_rvol_threshold` 1.9) were traced to an exact **6-of-6**
match in a committed hyperopt record:

```
user_data/hyperopt_results/strategy_NNNC_CGP_MLX_2026-05-31_15-03-45.fthypt
  is_best: True   loss: -26.636   total_trades: 109
  timerange: 20240511-20260501        backtest_days: 720
```

Fourteen records across 2026-05-31 match ≥4 of the 6; the file above is the best-epoch 6/6.
The values reached `Framework/BaseStrategy.py` and were later commented out by
`543d198 2026-09-06 feat: promote the model-driven operating point to the FRAMEWORK base params`.

## Fold overlap

| | |
|---|---|
| folds entirely outside the tuning window | f00–f04, f14 (6) |
| folds overlapping it | **f05–f13 (9)**, seven of them fully (90/90 days) |
| test-window days inside the tuning window | **720 / 1345 = 53.5%** |

## THE DECISIVE SPLIT — arm B's return is 97.8% in-sample

Arm B's +18.76pp, decomposed by whether the fold lies inside the window its guard values were
tuned on:

| | folds | wallet | trades |
|---|---|---|---|
| **inside** the tuning window | 8 | **+18.34pp** | 155 |
| **outside** the tuning window | 6 | **+0.42pp** | 43 |

**97.8% of arm B's return comes from folds inside its own tuning window**, and the three
largest contributors — f07 (+5.10), f11 (+4.83), f08 (+3.29), together 70% of the total — are
**all three** inside it.

Out of sample, globally tuned guards return **+0.42pp across 6 folds**. Arm A (guards open, no
model) returns +2.64pp across 15. Per fold that is **+0.07pp against +0.18pp** — the tuned
guards are not better than open guards out of sample; if anything slightly worse, and both are
far below the 3.350 pp/yr floor.

## Consequence for this feature

**The premise is gone.** Feature 022 exists because arm B appeared to show that guards alone
earn ~+5.1pp/yr, which needed explaining and suggested per-fold tuning as the mechanism. That
number is an artefact of the guard values having been hyperopted on 53.5% of the evaluation
window. **There is no longer a +18.76pp to explain.**

What survives:

- **C vs B is void**, as the spec anticipated (FR-002): B is not an out-of-sample result, so
  "does per-fold tuning hold up where global tuning may be contaminated" has no clean reading.
- **C vs A is intact** and still interpretable — it was always the comparison carrying the
  feature.
- **US2's trackability gate is unaffected** and remains the licensing question for arm C. It is
  now the *only* reason to continue, and the prior against it is stronger than when the feature
  was written: the out-of-sample evidence for *any* guard tuning is +0.42pp over 6 folds.

Also noted: arm B's f10 produced **no wallet figure at all** (the zero-trade fold), and it lies
inside the tuning window.

## Arm D's configuration, verified from artefacts (FR-003)

Read from `NNNC/NNNC_ARO_f{00..14}.json`, not from prose — feature 021 twice trusted a
provenance field that asserted rather than measured.

- `entry_enable_guards: True` with **all six thresholds neutralised** (`adx`/`atr_pct`/
  `bb_width`/`rvol` at −1.0; `close_norm` and `guard_metric` at 5.0, more permissive still than
  the base's 1.1). **Guards-open confirmed.**
- `prediction_threshold` takes **15 distinct values** across the 15 folds (0.7337 … 0.7832), so
  the incumbent's per-fold selection is real and not a constant.

The spec's premise for arm D holds exactly as stated.
