# Control sizer: pre-registration

Committed before any `cs_aro` threshold is picked or arm record exists. The judge refuses any
`cs_aro` record older than this file's commit. Machine-readable: `preregistration.json`.

**Question:** inside the width book, the current ARO model ranks trades well. The simulator shows
this in both halves, and it was the best ranker in 034's S1. Does that ranking survive real
execution as return?

**Arm `cs_aro`:** `NNNC_CS_ARO`, the same class as 034's `NNNC_LS_CONTROL_SZ`.
- The width book's entries, with the control's own weights. The model keys equal `wf_aro_g24`'s on
  all 15 folds.
- Stakes are 1.333× when the model agrees and 0.667× when it disagrees. Agree means p_buy is above
  the fold's 0.5-share threshold, with no argmax condition.
- The held-pair add is off, as in the width book.
- **Thresholds:** one per fold. Each is taken from the first bar of each width-book run in the
  fold's validation tail (training data), at 4 decimals, and every fold is verified.

**Comparators:** the width book `w_aro` and the control `wf_aro_g24` (24 h grace, add-off records).

**Instrument (per half; a failure is INCONCLUSIVE and no P&L is read):**
- the trade set equals `w_aro`'s on (pair, open, close), with at most 2 folds differing;
- 0 entry violations and 0 tag mismatches;
- thresholds load at 4 decimals;
- mean capital is within ±10% of the width book's;
- no fold has zero trades.

**PASS (PROVISIONAL):** `judge_one` must pass against BOTH `w_aro` and `wf_aro_g24`. That means
early and late gains above 0, and neither the paired verdict nor the risk verdict WORSENED. The
rule maps are 034's.
- If any half gain is under 1 pp, the result is INCONCLUSIVE until a seed-2 repeat.

## What has already been seen (this is NOT a blind late-half test)

- **findings §5 (simulator):** ARO's own-decision split of the width-book trades.
  - Early: agree 0.146 pp/trade vs disagree 0.074.
  - Late: agree 0.096 vs disagree 0.011.
  - A 1.0/0.5 sizing gave early 39.4 / late 21.9 pp per unit capital, against unsized 35.7 / 15.6.
- **034 S1 (simulator, early only):** the 0.5-share split has a gap of 0.100.
- **Backtest P&L** of both comparators on all 15 folds.
- **Not yet seen:** no backtest of a capital-neutral sizer on these weights. This study tests
  whether the simulated ranking survives the wallet, the liquidity cap, `max_open_trades` and
  capital-neutral stakes.
- A pass is PROVISIONAL. The confirmation route is post-2026-09-07 folds or the dry-run.
