# 032 pre-registration: candidate-only training

Committed BEFORE any `NNNC_CO_*` model is trained and before any `co_aro` / `co_gbr` record exists.
The judge refuses any arm record older than this file's commit.

**Machine-readable rules:** `preregistration.json` (stages S1–S3, small-margin rule, decision,
confirmation route). Summary:

1. **S1 (mechanism, read first).** The within-candidate AUC of the retrained model minus the
   current model's, on identical eligible bars and labels, must be > 0 in both halves. The current
   reference values are ARO 0.618 / 0.601 and GBR 0.591 / 0.583 (early / late).
2. **S2 (early f00–f07, read only after S1 passes).** The arm's early pp must beat both the width
   book's (`w_aro` / `w_gbr`) and the control's (`wf_aro_g24` / `fs_gbr_gr`).
3. **S3 (late f08–f14, confirmatory, read only after S2 passes).**
   - `judge_one(arm, control)` must be PASS;
   - the arm's late pp must beat the width book's;
   - `judge_one(arm, width)` is reported only.
4. **Small margins.** A stage that passes by under 1 pp is INCONCLUSIVE until a paired seed-2
   repeat.
5. **Decision.** PASS (PROVISIONAL), FAIL (naming the stage) or INCONCLUSIVE. No outcome is a
   finding: there is no distant-era window (constitution II). A pass names its confirmation route:
   the folds after 2026-09-07, or dry-run data.

Threshold: `pick_threshold.py` at rate 0.014735 on each training window, with ineligible bars
scored 0.

Committed: 95df1cc9 (2026-10-13). Records predating this commit are refused.

## T015 smoke (ARO f00, after the pre-registration commit)

- Training rows: 352,431 rows, of which 97,032 were kept (27.5%). Class counts SELL / HOLD / BUY
  were 0 / 44,510 / 52,522, so BUY is 54%.
- Validation: 13.0% of rows were kept.
- The weights were written at the new key `lstmraw_gbbpath_g0.02_h48__4d7815a2ebcf8cfd`. The
  existing `NNNC_ARO_f00` weights and manifest are unchanged (sha checked).
- The threshold is 0.5342, at an achieved rate of 1.4736% against the 1.4735% target.

All the T015 checks passed.
