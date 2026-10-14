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

## Follow-up 2: the CO model as a sizer on the width book (`co_sizing.py` → `co_sizing.json`)

EXPLORATORY simulator screen. The rule was declared in the script before the run. Both models size
the IDENTICAL width-book trades; the current model's numbers reproduce `width_sizing.json`.

| | half | agree pp/trade (n) | disagree pp/trade (n) | gap | agree_0.5 pp per capital |
|---|---|---|---|---|---|
| ARO current | early | 0.146 (73) | 0.074 (339) | 0.072 | 39.4 |
| ARO CO | early | 0.106 (110) | 0.080 (302) | 0.026 | 37.4 |
| ARO current | late | 0.096 (118) | 0.011 (401) | 0.085 | 21.9 |
| ARO CO | late | 0.090 (126) | 0.011 (393) | 0.079 | 21.6 |
| GBR current | early | 0.114 (178) | 0.051 (316) | 0.063 | 41.8 |
| GBR CO | early | 0.105 (169) | 0.058 (325) | 0.047 | 40.4 |
| GBR current | late | 0.082 (175) | 0.010 (261) | 0.072 | 22.5 |
| GBR CO | late | 0.057 (155) | 0.030 (281) | 0.027 | 19.1 |

**Result: CO is NOT the better sizer, for either model.** Its gap and its pp per capital are lower in
both halves. Rank weighting (reported only) is mixed: CO is higher for ARO late, GBR early and GBR
late.

The CO scores are LESS width-correlated than the current models'. Spearman of p_buy with bb_width
on eligible bars, early / late:

| ARO | CO_ARO | GBR | CO_GBR |
|---|---|---|---|
| 0.44 / 0.37 | 0.31 / 0.25 | 0.58 / 0.58 | 0.30 / 0.27 |

So the CO model's S1 gain comes from something other than width, but it does not separate the
widest candidates, which are the only ones the width book trades.

## Harness note: picked thresholds are rounded to 2 decimals

`prediction_threshold` is `DecimalParameter(0.2, 0.9, decimals=2)`. Any threshold
`pick_threshold.py` writes into a fold's `.json` is rounded when it is loaded (CO_ARO f00: 0.5342
traded as 0.53; control `wf_aro_g24` f00: 0.7337 traded as 0.73). This holds for every
picked-threshold arm and control, not just 032.

The achieved entry rate therefore differs from the target by up to half a 0.01 step of the score
distribution. That can be large: one 0.01 step changes CO_ARO f04's training-tail runs from 133 to
327. Nothing was changed. Raising `decimals` in Framework/BaseStrategy.py would change the trades
of every picked-threshold record when re-run, so that is the user's decision.

## Follow-up 1: run-matched thresholds (`run_match.py` → `run_match_results.json`)

EXPLORATORY. The reading was declared in `run_match_rule.md` before any record existed. Each fold's
threshold is the 0.01-grid value whose CO entry runs on the training window's validation tail come
closest to the control's. Two amendments were committed before any P&L:
- matching on the validation tail only;
- loading fold params with `ft_bot_start()`. Without it, buy params stayed at their defaults, and
  the threshold-0 pass leaked into the control.

The threshold-0 equivalence was verified on ARO f00 with 0 mismatched bars.

pp, trades and pp/trade over all 15 folds / early / late:

| arm | all | early | late |
|---|---|---|---|
| co_aro_rm | 38.5 (1,251; 0.031) | 33.0 (652) | 5.5 (599) |
| co_aro (032) | 35.7 (766; 0.047) | 25.2 (372) | 10.5 (394) |
| wf_aro_g24 (control) | 50.0 (1,176; 0.043) | 35.3 (607) | 14.7 (569) |
| w_aro (width) | 51.4 (930; 0.055) | 35.9 (411) | 15.6 (519) |
| co_gbr_rm | 36.4 (1,379; 0.026) | 29.4 (822) | 7.0 (557) |
| co_gbr (032) | 31.9 (1,219; 0.026) | 23.2 (633) | 8.7 (586) |
| fs_gbr_gr (control) | 53.5 (2,021; 0.026) | 37.9 (1,089) | 15.5 (932) |
| w_gbr (width) | 53.7 (928; 0.058) | 36.7 (493) | 17.0 (435) |

**Result: FAILS the declared screen for both models.** `judge_one` vs the control is FAIL for both
(late gain −9.1 / −8.5 pp).

- **ARO:** matching the trade count worked (1,251 vs 1,176) and closed most of the early gap. But
  the extra trades are poor: pp per trade fell from 0.047 to 0.031, below the control's 0.043. Late,
  the arm is worse than 032's.
- **GBR:** the 0.01 grid is too coarse to reach the control's count (1,379 vs 2,021), and the arm
  trails in both halves.
- **Conclusion:** the trade-count shortfall was not what held the CO books back. Lowering the
  threshold to recover the count admits worse trades.
