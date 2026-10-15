# 034 label-type screen: report

**Sources:**
- Pre-registered in `preregistration.json` / `.md`: commit 02fbad5f, with amendment 1 (b2edc66b:
  agree = p_buy above the fold's threshold, no BUY-argmax condition). The amendment was made at
  the T015 smoke, before any training chain, S1 table or arm record.
- Verdicts: `verdicts.json` (`judge.py`), from `audit.json`, `instrument.json`,
  `separation.json` and `survivors.json`.
- Run: 2026-10-14.

**Label: PROVISIONAL.** There is one era (2023–26), so no outcome is a finding. Nothing is adopted
or promoted.

## Decision

| candidate | label (id) | stopped at | decision | early gap (agree − disagree pp/trade) | AUC |
|---|---|---|---|---|---|
| GFM | gbb_fwdmae (28) | S1 | **FAIL** | 0.0146 | 0.516 |
| FMAE | forward_mae (0) | S1 | **FAIL** | −0.0257 | 0.410 |
| TB | triple_barrier (1) | S1 | **FAIL** | −0.0175 | 0.418 |
| MMR | mfe_mae_ratio (3) | S1 | **FAIL** | 0.0168 | 0.463 |
| FEXP | future_expectancy (12) | S1 | **FAIL** | −0.0184 | 0.473 |
| QF | quantile_future (2) | S1 | **FAIL** | −0.0152 | 0.447 |
| LMF | local_min_followthrough (4) | S1 | **FAIL** | 0.0273 | 0.474 |
| BOG | breakout_gbb (24) | S1 | **FAIL** | 0.0447 | 0.494 |
| *control, own decisions* | gbb_path (27) | reference | — | **0.0719** (pre-registered 0.072) | 0.583 |
| *control at the 0.5 share* | gbb_path (27) | reference | — | **0.1004** | 0.583 |

The pass rule was a gap above both references (0.072 and 0.1004), with at most 2 of 8 folds
degenerate. No candidate came close. The best was BOG at 0.045, under half the control at share.
There were no survivors (`survivors.json` is empty), so S2 and S3 did not run. **No backtest P&L
was read for any 034 arm, and the late half (f08–f14) stays unread.**

**Negative result:**
- None of the seven other label types teaches ARO anything that ranks width-book trades better
  than gbb_path does.
- Every candidate's AUC of p_buy against pp > 0 is at or **below** 0.5 (0.41–0.52). For the
  control it is 0.583.
- The outcome-only labels (FMAE, TB, FEXP, QF) have negative gaps: inside the width book, the
  trades they favour do slightly worse than the ones they reject. This is descriptive only, and no
  rule was pre-registered on it. FMAE's AUC of 0.41 on 412 trades is roughly 3 standard errors
  below 0.5.
- The S0a audit (below) shows these labels are uncorrelated or slightly negatively correlated with
  bb_width on rule-eligible bars. That fits the 033 result: taking width out of the target leaves
  little the model can rank on inside a book already sorted by width.

## S0a: labels (`audit.json`; recorded in the pre-registration)

All seven trained candidates passed: kept BUY above 1% on every early fold, every declared extra
live, and indicator columns equal to the control's. See the table in `preregistration.md`.

## S0: instrument (`instrument.json`; no P&L read)

- Every candidate passed. Weights are present for f00–f07 under keys distinct from the control's
  and from each other's, and GFM's are its existing `sw_gfm_ctl_b030` weights.
- Sizer thresholds exist for every early fold at 4 decimals, and the classes declare 4 decimals.
- The sizer verify (T017): 72 fold × candidate checks with 0 tag mismatches. Each loaded
  threshold equals the picked one.

**Training:** early stopping fired after 35–58 epochs, against the 032 control's 33, with the
same 100-epoch cap and 20-epoch patience. So training ended normally, although each fold took
only about 2–3 min.

**Sizer thresholds** (0.5 agree share on the run-start bars of the validation tail):

| model | f00–f07 range | note |
|---|---|---|
| CONTROL (at share) | 0.723–0.770 | BUY argmax on 97–100% of run starts |
| GFM | 0.649–0.723 | |
| FMAE | 0.261–0.386 | balanced 3-class label; BUY argmax rare (amendment 1) |
| TB | 0.290–0.403 | |
| MMR | 0.343–0.431 | |
| FEXP | 0.449–0.471 | |
| QF | 0.367–0.411 | |
| LMF | 0.340–0.607 | |
| BOG | 0.008–0.019 | rare-positive label; p_buy small everywhere |

## S1: separation inside the width book, early half (`separation.json`)

- The width book was simulated once per fold from the control's tables with the 24 h grace: 412
  trades on f00–f07, the same trades for every model.
- **SC-002:** the control's own gap reproduces as 0.0719, inside the pre-registered 0.072 ± 0.002.
- No fold was degenerate for any candidate. The agree shares were 0.48–0.55, except BOG at 0.40.
- The control at the 0.5 share separates better than its own entry decisions (0.100 vs 0.072).
  The current model, used as a sizer at a balanced share, is the best ranker measured in 033 or
  034.

## SC-003: byte identity of the control (T028)

`wf_aro_g24` was re-run on f00 and f07 into scratch records
(`regime/arms/sc003_034_bytecheck_aro*`).

- **As the class stands now:** the trades differ. f00 has 48 trades vs 49 recorded, and f07 has
  195 vs 197.
  - The cause is not 034. The `wf_aro_g24` records were written at 09:35–09:37 on 2026-10-13,
    before the in-profit held-pair add became the NNNC default later that day (feature 031).
  - The re-runs show the add working: 4 adds on f00 and 13 on f07.
- **With `HELD_PAIR_ADD = None` declared** (scratch subclasses, since deleted): f00
  6e32009f6dd2bc9c and f07 b3fe1bf90ca9841d. Both are **byte-identical** to the records, and the
  model keys are equal apart from the class name.
- So 034 changed nothing for the control. The held-pair add default is the only drift.

## Harness lessons

1. **Agree rule for balanced labels (amendment 1).**
   - A three-class label with balanced classes (forward_mae, triple_barrier and the like) rarely
     makes BUY the argmax. FMAE f00 had BUY as argmax on 8.8% of eligible bars, against 98.7% for
     the control.
   - So any agree rule that requires the argmax cannot reach a 0.5 share. A sizer or split on
     such a model must rank by p_buy alone (`WidthSizerMixin.AGREE_RULE = "p_buy"`).
2. **Controls recorded before a default change.**
   - The `wf_aro_g24` records are add-off, but `NNNC_A_Raw_Open` is now add-on.
   - An entry-source arm (es role) inherits the add, while its control's records do not. Had S2
     run, the es comparison would have mixed two configurations.
   - Before any future es arm is compared with `wf_aro_g24`, either declare `HELD_PAIR_ADD = None`
     on the arm, or re-run the control with the add on.
   - Sizer arms are unaffected: the width book and the sizers are add-off, and the tests pin this.

## Fix-oriented follow-ups (not run)

1. **A sizer arm with the current model.** The control at share is the strongest ranker measured
   (gap 0.100), and 034 has already built it:
   - `NNNC_LS_CONTROL_SZ` uses the control's own weights with `AGREE_RULE = "p_buy"`.
   - Its f00–f07 thresholds are in `thresholds_control_share.json`.
   - What remains: write those thresholds into a plan, then run 8 early backtests (and later 7
     late ones) against `w_aro` and `wf_aro_g24`.
   - This is 033's follow-up 1. It should be pre-registered on the unread late half.
2. **Rank-weighted stakes** from the control's p_buy instead of a binary split (033 follow-up 2).
3. **A second information source rather than a second label.** Per the quant lessons, a target
   that the same inputs cannot learn better is a data limit, not a label limit. Candidates are
   inputs the width book does not see: cross-asset or BTC state, volume or order-flow proxies, and
   funding. Test them as features on the gbb_path control, screened with this same S1 instrument.

## Provenance

- **Weights:** 56 folds (7 candidates × 8 early folds, seed 1) under each candidate's own model
  key, written by `research.label_screen.train`. Training logs are in
  `/tmp/wf_train_NNNC_LS_<TAG>_ES_fNN.log` (not kept).
- **Signal tables:** `research/nnnc_framework/signals/LS_<TAG>_fNN.parquet` (git-ignored), with
  `meta.json` (tracked).
- **Sizer thresholds:** `evaluation/folds/plan_ls_<tag>_sz.json` (`thresholds` and
  `threshold_picks`); the control at share is in `thresholds_control_share.json`.
- **Stage outputs:** `audit.json`, `instrument.json`, `separation.json`, `survivors.json` and
  `verdicts.json` in this directory.
- **Chain:** `ls_chain.sh` (scratch), 18:21–23:32 on 2026-10-14, exit 0, no errors.
