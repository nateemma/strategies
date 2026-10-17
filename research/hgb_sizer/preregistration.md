# HGB sizer: pre-registration

Committed before any HGB signal, threshold pick or `cs_hgb` arm record exists. The judge refuses
any `cs_hgb` record older than this file's commit. Machine-readable: `preregistration.json`.

**Question:** 037 found that a gradient-boosted model on the harvested indicator columns ranks
label:buy within gbb candidates better than ARO (AUC 0.62 / 0.625 vs 0.61 / 0.60, early / late).
Does that ranking survive real execution as a width-book sizer?

**Arm `cs_hgb` (`NNNC_CS_HGB`):** exactly `cs_aro` (control sizer) with one change: the agree
score is the fold's HGB p_buy instead of ARO's. Same entries (the width book), same ARO weights
(model keys equal cs_aro's), 1.333× agree / 0.667× disagree at a 0.5 agree share, held-pair add off.

**HGB:** 037's label:buy settings and columns (no `btc:`; includes the post-lock extras), fitted per
fold on candidate bars that end 48 bars before the fold's validation tail. The threshold is picked
on that tail (cs_aro's rule), which the HGB never fitted. Every fold is verified.

**Comparators:** the width book `w_aro`, the control `wf_aro_g24`, and the ARO sizer `cs_aro`.

**Instrument (per half; failure = INCONCLUSIVE, no P&L read):** cs_aro's checks, plus at most 1%
of width-book entries without an HGB score.

**PASS (PROVISIONAL):** `judge_one` passes against both `w_aro` and `wf_aro_g24` (cs_aro's bar).
A half gain under 1 pp makes a PASS INCONCLUSIVE.
**Secondary:** `judge_one(cs_hgb, cs_aro)` — whether it beats the ARO sizer. Replacing ARO as the
sizer would need both.

## Already seen (NOT a blind late half)
- 037's AUCs in both halves, including late (that model was fitted up to the fold start).
- cs_aro, w_aro and wf_aro_g24 backtests on all 15 folds; cs_aro's forward window (INCONSISTENT).
- Not seen: any split of the width-book trades by HGB score, simulated or backtested.

**Caveat:** live use would need 794 columns (643 indicator values and 151 raw signals) computed at runtime. That is not
assessed here.
