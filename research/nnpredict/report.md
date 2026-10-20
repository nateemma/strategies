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
  reversion book (`st_g24_rev` 71.5). On `st_g24_rev`'s exact exits, adding the model to the gbb
  trigger loses 20.9 pp (follow-ups, section 5): the gain is the trigger's, not the model's.
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
   "trigger_and_model"`, `TRIGGER_NAME = "gbb"`) and the 24 h grace, on `nnp_lstm_ind_op`. Each
   passed alone; combined they FAIL vs the gate alone (section 5), so they are alternatives, not a pair. The gate
   is the only strong result (IMPROVED, both halves).
3. **Not for deployment:** gated, NNPredict still trails the width book and the gbb reversion book.
   NNNC's `cs_aro` remains the reference.
4. The three open ideas were run as follow-ups (section 5): the gate stays the only candidate
   default; on identical exits the model's filter costs P&L vs the gbb trigger alone; excursion H=32
   is the best standalone target but still loses.

## 5. Follow-ups (`preregistration_followups.json`, `verdicts_followups.json`)

Run on the user's request ("do all 3") after the framework verdicts; the pre-registration was
committed before any run (8ec08733).

| Arm | Change | Trades | Total pp | vs | Δ early | Δ late | Paired | Verdict |
|---|---|---|---|---|---|---|---|---|
| `_gate_g24` | gate + 24 h grace | 3,563 | 29.97 | `_gate` 29.75 (4,913) | +1.44 | −1.22 | UNCHANGED (p 0.99) | FAIL |
| `_revx` | `st_g24_rev`'s framework, trigger entry only | — | — | `st_g24_rev` | | | identical trades 15/15 | identity holds |
| `_revg` | the same framework, model AND gbb | 1,351 | 50.57 | `st_g24_rev` 71.50 (2,518) | −17.82 | −3.12 | UNCHANGED (p 0.12) | FAIL |
| `nnp_ridge_exc32_op` | Ridge, excursion target, H = 32 | 9,131 | −10.99 | `nnp_ridge_op` −25.64 | +13.00 | +1.66 | INCONCLUSIVE_ARM_LOSES | PASS |

- **Gate + 24 h grace:** the grace adds nothing on top of the gate (+0.2 pp, late half negative). Beside
  `fw0` it is IMPROVED in both halves, but so is the gate alone, so the gate alone stays the candidate.
- **Gated model vs the trigger on identical exits:** the copied framework reproduces `st_g24_rev`
  exactly, so the only difference is the model. The model keeps 54% of the trigger's trades and loses
  20.9 pp, worse in both halves (not significant). The filter removes profitable reversion entries;
  the gate arm's +42 pp in section 3 came from the trigger, not the model.
- **Excursion H = 32:** the best standalone NNPredict result in the study (also beats excursion H = 4 by
  +14.1 pp, both halves), but it still loses money and the paired test says the arm loses. Deciles
  (early half): it ranks 8 h outcomes almost perfectly (Spearman +0.95, top–bottom 16.9 bp) but the
  top decile nets −6.0 bp after 30 bp; its gated entries carry +14.1 bp edge over random at 8 h. No
  model selection was re-run (pre-registered).

## Limitations

- One era, so PROVISIONAL.
- The shared 6-column `main_scaler.pkl` is an existing exposure, identical across arms.
- The model selection saw both halves.
- The flip arm was chosen after seeing the early-half inversion.
- The decile tool has no H = 24, so it uses H = 32.
- One seed. The two PASSes gained ≥ 1 pp, so no seed-2 rerun was required.
- The binance.us outage (2026-10-19 20:18–20:29) failed some LSTM folds; they were re-run before
  being judged.
