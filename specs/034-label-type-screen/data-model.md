# Data Model: Label-Type Screen (034)

## CandidateLabel (`research/label_screen/candidates.py`)

| field | type | rule |
|---|---|---|
| tag | str | one of CONTROL, GFM, FMAE, TB, MMR, FEXP, QF, LMF, BOG; unique |
| method | int | a `LabelMethod` id (27, 28, 0, 1, 3, 12, 2, 4, 24) |
| label_params | dict | declared extras (research R2); `{}` if none; never edited after the pre-registration commit |
| label_params_sell | dict or None | None = mirror label_params |
| reuse | str or None | existing plan whose weights are reused (`wf_aro_g24`, `sw_gfm_ctl_b030`); None = trained |
| roles | tuple | ("sz", "es") for all except BOG ("sz",) and CONTROL (none: comparator) |
| order | int | training order in the chain |

## Preregistration (`preregistration.json`)

- candidates (the table above, as data), architecture "ARO", folds early f00–f07 / late f08–f14;
- S0a floor: kept BUY share ≥ 0.01 of training rows in every early fold;
- S1: agree_share 0.5; references `{pre_registered: 0.072, control_at_share: <computed>}`;
  max_degenerate_folds 2; survivors_max 3; tie-break AUC;
- S2: comparators `{width: "w_aro", control: "wf_aro_g24"}`; entry_rate 0.014735;
  sizer stakes 1.333 / 0.667; max_trade_mismatch_folds 2; capital band ±10%;
  small_margin_pp 1.0; seed_repeat 2;
- S3: judge_one settings copied from 032 (min_control_trades 200); GBR comparators `fs_gbr_gr`,
  `w_gbr`;
- confirmation_route; exempt_records (smoke records set aside, by name).

## LabelAudit (`audit.json`, S0a; per candidate × early fold)

buy_share, sell_share, hold_share (after peak filter and conflict resolution), raw_overlap (BUY and
SELL on the same bar before resolution), param_liveness `{param: changed_labels_count}`,
feature_columns_equal_control (bool), status PASS | INCONCLUSIVE (reason).

## Instrument (`instrument.json`, S0; per candidate × fold)

weights_present, model_key (distinct from every other candidate and the control),
spearman_label_width_eligible (buy, sell), threshold_sz (4 dp), threshold_es (survivors only),
loaded_threshold_matches (bool).

## ScreenRow (`separation.json`, S1; per candidate)

early `{agree: {n, pp_per_trade}, disagree: {...}, gap, auc, degenerate_folds}`,
references `{pre_registered, control_at_share}`, S1_pass, rank.

## Arm (plans `plan_ls_<tag>_<role>.json`; records `regime/arms/ls_<tag>_<role>_fNN_FNN.json`)

role sz: entries identical to `w_aro` (≤ 2 folds differ), capital ratio, realised agree share,
stake ratio, wallet shortfalls, tag mismatches. role es: entry-signal rate vs 0.014735 (±5%),
trade count vs control.

## Verdict (`verdicts.json`)

per candidate: S0a, S0, S1 stage results; per arm: S2 (early pp, margins, seed-2), S3 (late, judge),
decision PASS (PROVISIONAL) | FAIL | INCONCLUSIVE, stopped_at, reason; GBR repeat if reached.

## State transitions

candidate: declared → audited (S0a) → trained → instrumented (S0) → screened (S1) → survivor? →
arms backtested (S2) → winner? → late-trained and judged (S3) → GBR repeat. A failed stage ends the
path; later stages are recorded `not_reached`.
