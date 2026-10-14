# 033 pre-registration: width-relative label, sizer arms

Committed BEFORE any `NNNC_WR_*` model is trained, and before any `wr_*` arm record exists. The
judge refuses any arm record older than this file's commit.

**Machine-readable rules:** `preregistration.json`. Summary:

**The label (NNNC/_width_relative.py):**
- **BUY:** entry-eligible bars (guard_metric < 0.3, bb_width > 0.035) are ranked by the gbb_path
  trade outcome (+2% target / −5% stop / the 48-bar close) within 10 bb_width deciles. The top half
  is BUY.
- **SELL:** exit-eligible bars (guard_metric > 0.3, bb_width > 0.035) are ranked by the
  gbb_path_sell outcome (2% fall / 2.2% take-profit / the 48-bar close) within their own deciles.
  The bottom half is SELL.
- **Fitting:** deciles and cuts are fitted on the training rows only.

**The arms (per model, same weights):**
- **Exit-off sizer** (`wr_<m>_sz`): the width book's entries, with stakes of 1.333× when the model
  agrees and 0.667× when it disagrees. The model exits are off.
- **Exit-on sizer** (`wr_<m>_szx`): the same, with `enable_exit_signal` on and the exit gate
  unchanged.

**Threshold:** 4 decimals, set to a 50% agree share on the first bar of each width-book candidate
run in the fold's validation tail.

**Stages (each arm judged independently, in this order):**
1. **S0, instrument and label.**
   - Every fold, on the training rows, each side: every decile's labelled share is in
     [0.40, 0.60], and |Spearman(label, bb_width)| ≤ 0.05.
   - The kept rows, BUY and SELL are all > 0.
   - The exit-off arm reproduces the width book's trades_hash per fold. A differing fold is
     flagged; more than 2 of 15 fails S0.
   - Entry violations 0, tag-to-agree mismatches 0, and the thresholds load at 4 decimals.
   - Exit-on liveness: every fold with exit-gate-passing bars inside open trades has at least one
     model exit.
   - The stake shortfall is flagged above 5%. A zero-trade fold is INCONCLUSIVE.
2. **S1 (exit-off only, EARLY half only).** The width-relative model's agree − disagree pp per
   trade on the identical width-book trades must exceed the current model's (ARO 0.072, GBR 0.063).
   The late gap is not computed before S3.
3. **S2 (early, f00–f07).**
   - Exit-off: pp above both the width book's and the control's.
   - Exit-on: pp above both the exit-off arm's and the width book's.
4. **S3 (late, f08–f14).**
   - Exit-off: `judge_one` against the control is PASS, AND the late pp is above the width book's.
   - Exit-on: `judge_one` against the exit-off arm is PASS, AND the late pp is above the width
     book's.
   - `judge_one` against the width book is reported for both.

**Small margins:** a stage passing by less than 1 pp is INCONCLUSIVE until a paired seed-2 repeat.

**Outcomes:** PASS (PROVISIONAL), FAIL or INCONCLUSIVE. There is no distant era, so nothing is a
finding. Confirmation route: the folds after 2026-09-07, or dry-run.

Committed: 1612b603 (2026-10-14 10:03). Records predating this commit are refused.
