# 039 NNPredict improvement study: report

PROVISIONAL: one era (2023-26), 15 walk-forward folds, halves early f00-f07 / late f08-f14 as the
persistence test. Every rule was committed before its result (`preregistration_*.json`; the
amendments say what had been seen first). Records are local (`regime/arms/`); verdicts are in
`verdicts_{op,threshold,framework}.json`, `screen.json` and `deciles_*.json`.

## Bottom line

- **NNPredict loses money on its own.** All eight model configurations lose over the 15 folds (−12.7
  to −74.1 pp). None beats the Ridge baseline on the paired test.
- **The NNNC framework helps a lot, but not enough.** It ports with byte-identical trades. Gating the
  model by the gbb trigger takes the selected model from −12.7 to **+29.7 pp**: IMPROVED, p = 0.003,
  better in both halves. That still trails the width book (`w_aro` 51.4), `cs_aro` (59.9) and the gbb
  reversion book (`st_g24_rev` 71.5).
- **The predicted move doesn't work as a width-book sizer.** The screen fails in both directions.
- **Why:** the predictions rank 1–2 h outcomes inverted, and 8 h outcomes the right way but weakly.
  The best decile is 7–12 bp short of the 30 bp round trip. This matches 038.

## 1. Fixes (US1)

| Defect | Fix | Test |
|---|---|---|
| Post-model gate settings were part of the model key: a gate change demanded new weights | `MODEL_KEY_IGNORED` lists them | `NNPredict/tests/test_model_key_classes.py` |
| Ridge `.joblib` failed the fold runner's artifact check | `WEIGHT_SUFFIXES` includes `.joblib` | `evaluation/test_run_folds_joblib.py` |
| README described the old quantile gate | rewritten | — |
| `add_additional_indicators` appended to the class-level `include_list`, changing later instances' model keys in one process (ModelMissing in the signal tool) | instance copies | the 8-fold signal run |

## 2. Models (US2/US3), at the operating point

The as-is Ridge (`nnp_ridge`) made only 152 trades in 15 folds: its absolute floor sat about 5σ out
on compressed predictions, and the guards removed the rest. The operating-point amendment
(`preregistration_op.json`) opened the guards, as in ARO, and used the existing percentile floor.
Both settings act after the model, so the weights are reused.

| Arm | Change | Trades | Total pp | Δ early | Δ late | Verdict vs `nnp_ridge_op` |
|---|---|---|---|---|---|---|
| `nnp_ridge_op` | baseline | 15,181 | −25.64 | | | (beside `w_aro`: −42.2 / −34.9) |
| `nnp_ridge_sa_op` | stop-aware target | 15,131 | −30.53 | −0.42 | −4.46 | FAIL |
| `nnp_lstm_raw_op` | LSTM, raw8 + gains | 12,568 | −47.67 | −14.26 | −7.77 | FAIL |
| `nnp_lstm_ind_op` | LSTM, indicators + raw8 | 10,936 | **−12.73** | +15.61 | −2.70 | FAIL (no worsened half) |
| `nnp_lstm_ret_op` | LSTM, raw8 + lagged returns | 11,901 | −23.97 | −4.59 | +6.26 | FAIL |
| `nnp_ridge_flip_op` | sign flipped | 11,993 | −74.12 | −22.87 | −25.60 | FAIL |
| `nnp_ridge_exc_op` | excursion target | 13,058 | −25.07 | +4.78 | −4.21 | FAIL |
| `nnp_ridge_h24_op` | horizon 24 (6 h) | 12,116 | −26.29 | +18.98 | −19.63 | FAIL |

The selection rule (R7) was applied over all eight: highest total with no worsened half. It selects
**`nnp_lstm_ind_op`**. The last three rows are the threshold amendment
(`preregistration_threshold.json`), which tests the user's three options after the decile check.

### Decile check (early half, diagnostic, `deciles.py`)

The table gives Spearman(decile, mean return) and the top decile's edge over random entry; net is
after 30 bp.

| Model | 1 h ranking | 2 h ranking | 8 h ranking | 8 h top decile edge / net | 8 h gated entries edge |
|---|---|---|---|---|---|
| Ridge | −0.73 | −0.62 | +0.22 | +8.8 / −8.5 bp | +6.8 bp |
| Ridge excursion | −0.22 | +0.35 | +0.61 | +5.3 / −12.0 bp | +6.9 bp |
| Ridge H24 | −0.86 | −0.42 | +0.16 | +10.4 / −6.9 bp | +8.7 bp |

The prediction marks short-term reversal (inverted at 1–2 h) inside a weak longer move (right way
at 8 h). Flipping the sign therefore hurts at the hold that matters, and the flip arm is the worst
arm. No decile clears costs at any horizon, so no threshold can rescue the standalone entry.

## 3. Framework (US4), on `nnp_lstm_ind_op` (weights reused, key `50ccbec98c1fbdc4`)

Identity: `fw0` (`EntryFrameworkMixin`, everything off) reproduces `nnp_lstm_ind_op`'s trades on
**15/15** folds.

| Arm | Trades | Total pp | Δ early | Δ late | Paired | Decision |
|---|---|---|---|---|---|---|
| `fw0` | 10,936 | −12.73 | | | | identity |
| `_g24` 24 h grace | 7,374 | −0.02 | +11.93 | +0.78 | INCONCLUSIVE (p 0.44) | PASS → candidate default |
| `_hpa` in-profit add | 10,822 | −12.09 | +1.59 | −0.96 | INCONCLUSIVE | FAIL |
| `_famm` momentum exits | 10,142 | −28.80 | −11.77 | −4.30 | INCONCLUSIVE (p 0.07) | FAIL |
| `_gate` model AND gbb | 4,913 | **+29.75** | +18.95 | +23.53 | **IMPROVED (p 0.003)** | PASS → candidate default |

Every feature arm changed trades on 15/15 folds (live).

The gate arm beside the references (reported, no rule):

| Reference | Total pp | Δ early | Δ late | Paired |
|---|---|---|---|---|
| `st_gbb` (trigger alone, its own exits) | 11.04 | −2.07 | +20.78 | UNCHANGED |
| `st_g24_mom` | 34.85 | −7.30 | +2.20 | UNCHANGED |
| `w_aro` | 51.44 | −7.60 | −14.10 | UNCHANGED |
| `cs_aro` | 59.95 | −12.45 | −17.75 | WORSENED |
| `st_g24_rev` (gbb reversion, 24 h grace) | 71.50 | −35.31 | −6.44 | WORSENED |

## 4. Sizer (US5)

The screen used the 412 early width-book trades, scored by `nnp_lstm_ind_op`'s predicted move:

| Direction | gap_share50 | Partial ρ \| bb_width | Sign folds | Result |
|---|---|---|---|---|
| up | 0.028 (bar 0.05) | +0.079 | 6/8 | fail |
| flip | −0.027 | −0.079 | 6/8 | fail |

Correlation with `cs_aro`'s p_buy is ρ = 0.25. Per the pre-registration there is no sizer backtest;
the late half stays unread.

## Recommendation (FR-012)

1. **Keep the framework port** on `NNPredictStrategy` (done, everything off, identical trades). This
   makes every NNNC feature declarable on NNPredict.
2. **Candidate defaults for the user to decide:** the gbb trigger gate (`ENTRY_MODE =
   "trigger_and_model"`, `TRIGGER_NAME = "gbb"`) and the 24 h grace, on `nnp_lstm_ind_op`. The gate
   is the only strong result (IMPROVED, both halves).
3. **Not for deployment:** gated, NNPredict still trails the width book and the gbb reversion book.
   NNNC's `cs_aro` remains the reference.
4. **Open, not run:**
   - gate + 24 h grace combined (each passed alone; not pre-registered as a pair);
   - the gated model vs `st_g24_rev` with the same exits (the reference comparison above mixes exit
     stacks);
   - a target aligned with the 8 h ranking (excursion at H = 32), the one place the deciles point the
     right way.

## Limitations

- One era, so PROVISIONAL.
- The shared 6-column `main_scaler.pkl` is an existing exposure, identical across arms.
- The model selection saw both halves.
- The flip arm was chosen after seeing the early-half inversion.
- The decile tool has no H = 24, so it uses H = 32.
- One seed. The two PASSes gained ≥ 1 pp, so no seed-2 rerun was required.
- The binance.us outage (2026-10-19 20:18–20:29) failed some LSTM folds; they were re-run before
  being judged.
