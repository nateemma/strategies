# 032 candidate-only training: report

Pre-registered in `preregistration.json` / `.md` (commit 95df1cc9, before any `NNNC_CO_*` model or
`co_*` record existed). Verdicts: `verdicts.json` (`judge.py`). Run 2026-10-13/14.

## Decision

| model | instrument | S1 (within-candidate AUC) | S2 (early f00–f07) | S3 (late) | decision |
|---|---|---|---|---|---|
| ARO | pass | PASS (+0.008 / +0.006) | **FAIL** (−10.6 vs width, −10.0 vs control) | not reached | **FAIL at S2** |
| GBR | pass | PASS (+0.031 / +0.031) | **FAIL** (−13.5 vs width, −14.8 vs control) | not reached | **FAIL at S2** |

Negative result: training on rule candidates only made the model rank candidates slightly better
(S1), but the walk-forward books earned less than both the current model and the model-free width
book in the early half. As pre-registered, S3 was not run, and no seed-2 repeat was triggered: both
margins are far over the 1 pp small-margin rule, and they are failures, not narrow passes. Nothing
is adopted or promoted; the arms stay as declared classes (`NNNC_CO_ARO`, `NNNC_CO_GBR`).

## Instrument gates (`instrument.json`; entries only, read before any P&L)

| | CO_ARO | CO_GBR |
|---|---|---|
| simulator fidelity recall / precision | 0.9987 / 0.9948 (766 recorded, 769 simulated) | 0.9975 / 0.9975 (1,219 / 1,219) |
| entry violations (signal bar fails the gbb rule) | 0 | 0 |
| SELL rows kept in training | 0 (all 15 folds) | 0 (all 15 folds) |
| folds with zero trades | 0 | 0 |
| training rows kept | 27.5% (f00) → 16–18% (f05–f14) | similar |
| threshold achieved rate on the training window | 1.4735–1.4736% (target 1.4735%), all folds | same |

Thresholds: ARO 0.529–0.556, GBR 0.539–0.552 (`evaluation/folds/thresholds_co_*_log.txt`).

## S1: within-candidate AUC (`auc.json`)

Identical eligible, labelled bars for both models (early 100,833 bars, late 104,239).

| model | half | current | retrained | diff | 95% CI (info only) |
|---|---|---|---|---|---|
| ARO | early | 0.6177 | 0.6260 | +0.0083 | [0.0017, 0.0153] |
| ARO | late | 0.6009 | 0.6063 | +0.0055 | [0.0010, 0.0103] |
| GBR | early | 0.5910 | 0.6222 | +0.0312 | [0.0232, 0.0384] |
| GBR | late | 0.5832 | 0.6145 | +0.0313 | [0.0236, 0.0394] |

The mechanism worked in the narrow sense: freed of re-learning the rule, GBR's ranking of
candidates improved by about 0.03 AUC, ARO's barely. Both remain low (0.61–0.63).

## S2: early half, pp summed over f00–f07 (`verdicts.json`)

| | arm | control | width book |
|---|---|---|---|
| ARO | 25.23 | 35.28 (`wf_aro_g24`) | 35.85 (`w_aro`) |
| GBR | 23.19 | 37.94 (`fs_gbr_gr`) | 36.72 (`w_gbr`) |

Thirds (context only; computed by the judge alongside S2): ARO arm 12.8 / 12.8 / 10.1 vs control
15.8 / 26.3 / 7.9 vs width 20.2 / 21.0 / 10.3; GBR arm 14.9 / 12.1 / 4.9 vs control 30.6 / 13.9 /
8.9 vs width 17.9 / 28.7 / 7.0. Note: t3 (f10–f14) lies in the late half, so the judge prints those
late-fold totals even when S2 fails. They play no part in the decision and are reported only
because they were printed.

## Descriptive: why the books earn less (not part of the verdict)

Early-half trades and pp per trade:

| | trades | pp/trade |
|---|---|---|
| co_aro | 372 | 0.068 |
| wf_aro_g24 | 607 | 0.058 |
| w_aro | 411 | 0.087 |
| co_gbr | 633 | 0.037 |
| fs_gbr_gr | 1,089 | 0.035 |
| w_gbr | 493 | 0.074 |

The arms take about 40% fewer trades than the controls at about the same, or slightly better, pp per
trade. Most of the shortfall is trade count, not trade quality. The width book still earns the
most per trade.

The causes differ by model. Here is the share of early test bars with an entry signal (signal
tables, f00–f07 mean):

| ARO | CO_ARO | GBR | CO_GBR |
|---|---|---|---|
| 0.715% | 0.519% | 0.736% | 0.737% |

- **ARO:** the retrained model signals on about 27% fewer bars. The threshold hits the target rate
  on the training window, but the scores sit lower out of sample.
- **GBR:** the signal rate is unchanged, yet the arm takes 42% fewer trades. Its signals must
  cluster more: runs of consecutive bars on a pair that is already held cannot open trades. Not
  verified further.

Fix-oriented follow-ups (exploratory, not run):

1. Match the arm's trade count, not its bar rate, to the control's. For ARO, re-pick the threshold
   on recent training data. For GBR, reduce the clustering, e.g. by counting only the first signal
   bar of each run. Then compare per trade.
2. Use the retrained GBR model (the one whose S1 gain is real) as the sizer/filter on the width
   book (as in `research/model_calibration/width_sizing.py`), rather than as the entry source.
3. A width-relative label (out of scope for 032; noted in the spec), so the model learns what
   width does not already explain.

## Provenance

- Arm records: `regime/arms/co_aro_*`, `co_gbr_*`. Pooled results:
  `evaluation/folds/results_co_aro_seed1.json`, `results_co_gbr_seed1.json`.
- Plans with thresholds: `evaluation/folds/plan_co_aro.json`, `plan_co_gbr.json`.
- The first chain run was killed by the host for low memory during the ARO threshold pick, after all
  15 ARO folds had trained. It was restarted with one pick process per fold; the training was
  skipped as already done, and no record was written before the restart.
