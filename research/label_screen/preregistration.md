# 034 pre-registration: label-type screen for NNNC

Committed BEFORE any `NNNC_LS_*` model is trained and before any `ls_*` arm record exists. The
judge refuses any arm record older than this file's commit.

**Machine-readable rules:** `preregistration.json`. The candidate table is
`research/label_screen/candidates.py`, and its hash is recorded in the JSON.

**Question:** does a model trained on another label rank width-book trades better than the
gbb_path control (ARO), and does that ranking turn into return?

**Model:** ARO's architecture (NNNC_A_Raw_Open: LSTM_RAW, 24 indicators, seq_len 16), the 24 h
grace control `wf_aro_g24`. Only the label changes. The entry family is pinned to the control's
(reversion), so held-pair adds, exits and tags are the control's.

**Candidates:**

| tag | label (id) | declared extras | weights |
|---|---|---|---|
| CONTROL | gbb_path (27) | — | existing (`wf_aro_g24`) |
| GFM | gbb_fwdmae (28) | LABEL_MAX_DRAWDOWN 0.04 (GFM's own) | existing (`sw_gfm_ctl_b030`) |
| FMAE | forward_mae (0) | max_drawdown 0.04 | trained |
| TB | triple_barrier (1) | buy pt 0.02, sl 0.05; sell pt 0.05, sl 0.02 | trained |
| MMR | mfe_mae_ratio (3) | — | trained |
| FEXP | future_expectancy (12) | — | trained |
| QF | quantile_future (2) | — | trained |
| LMF | local_min_followthrough (4) | max_drawdown 0.04 | trained |
| BOG | breakout_gbb (24) | — (sizer role only) | trained |

Every candidate also receives the control's horizon 48 and ±2% thresholds. Parameters are never
tuned.

**Stages:**
1. **S0a, label audit (labels only, no model).** On every early fold's training window: kept BUY,
   SELL and HOLD shares; raw BUY–SELL overlap; Spearman(label, bb_width) on rule-eligible bars;
   indicator columns equal to the control's; on f00, each declared extra that differs from its
   labeler's default must change the labels. A candidate with kept BUY < 1% in any early fold, or
   a no-op extra, is INCONCLUSIVE and not trained. The outcome is recorded below and in the JSON.
2. **S0, instrument.** Weights present for f00–f07 under keys distinct from the control's and
   each other's; sizer thresholds for every early fold at 4 decimals; classes declare 4 decimals.
3. **S1, separation (early half f00–f07 only; no backtest).** On the identical width-book trades
   (033's simulator), agree − disagree pp per trade at a 0.5 agree share (threshold on the first
   bar of each width-book run in the validation tail; agree = p_buy > threshold, no BUY-argmax
   condition — amendment 1). PASS = gap above BOTH the pre-registered
   control gap 0.072 AND the control's own gap at the same 0.5 share, with at most 2 of 8 folds
   degenerate (agree share 0 or 1). AUC of p_buy against pp > 0 is reported.
4. **Survivors.** The S1 passes, top 3 by gap, ties by AUC. `survivors.json` is committed before
   any survivor backtest.
5. **S2 (early, f00–f07), per survivor and role.**
   - **sz:** the width book's entries, stakes 1.333× agree / 0.667× disagree.
   - **es:** the control's entry configuration with only the label changed (no trigger gate);
     threshold picked on the fold's training window at the control's entry rate 0.014735.
     breakout_gbb is not run in this role.
   - Instrument: sz trades equal the width book's on (pair, open, close) with ≤ 2 folds differing,
     0 entry violations, 0 tag mismatches, capital within ±10% of the width book; es entries all
     model decisions; thresholds loaded at 4 decimals. Zero-trade folds are INCONCLUSIVE.
   - PASS = early pp above the width book (`w_aro`) AND the control (`wf_aro_g24`). A minimum
     margin under 1 pp needs a paired seed-2 repeat (every margin > 0 on both seeds).
6. **S3 (late, f08–f14), the S2 winner only** (the passing arm with the largest minimum margin):
   judge_one(arm, control) PASS (rule and min_control_trades 200 copied from 032/033) AND late pp
   above the width book; small-margin rule as S2. On a PASS, GBR repeats it on all 15 folds
   (`fs_gbr_gr`, `w_gbr`), same rules.

**Decisions:** PASS (PROVISIONAL) | FAIL | INCONCLUSIVE. One era (2023–26), so nothing is
recorded as a finding and nothing is adopted. Confirmation route: post-2026-09-07 folds or
dry-run.

## S0a outcome (recorded before this commit)

All seven trained candidates PASS (`audit.json`, f00–f07, labels only). Ranges across the 8 early
folds of kept class shares on the training window, and Spearman(BUY, bb_width) on rule-eligible
bars:

| tag | status | kept BUY | kept SELL | raw BUY–SELL overlap | Spearman BUY~width |
|---|---|---|---|---|---|
| CONTROL | reference | 8.7–13.1% | 5.0–7.1% | 0 | +0.06..+0.08 |
| GFM | reference | 6.6–9.4% | 1.5–2.0% | 0 | −0.03..0.00 |
| FMAE | PASS | 26.9–33.3% | 28.1–35.5% | 0 | −0.04..−0.01 |
| TB | PASS | 24.5–29.8% | 35.6–46.7% | ≤ 10.8% (sells override) | −0.06..−0.04 |
| MMR | PASS | 25.3–31.5% | 26.9–34.0% | 0 | 0.00..+0.02 |
| FEXP | PASS | 13.7–17.3% | 13.8–18.0% | 0 | +0.10..+0.11 |
| QF | PASS | 15.6–17.9% | 17.8–20.0% | 0 | +0.04..+0.05 |
| LMF | PASS | 7.8–8.9% | 6.9–8.0% | 0 | −0.02..−0.01 |
| BOG | PASS | 5.6–8.2% | 1.0–1.3% | ≤ 0.2% | −0.01..0.00 |

Every declared extra that differs from its labeler's default changed the f00 labels; TB's sell
`sl` 0.02 equals its default (recorded). Indicator columns equal the control's for every candidate.

**Amendments made before this commit:**
- the entry family is pinned to reversion on every candidate (feature 029's label-driven family
  would untag the outcome-only labels' entries and make breakout_gbb momentum, which turns the
  held-pair add off);
- mfe_mae_ratio: a mirrored sell variant (`labels_mfe_mae_ratio_sell`); before it the sell
  accessor fell back to the BUY rule;
- mfe_mae_ratio: the adverse excursion is clamped at 0 on both sides (unclamped, the cleanest
  winners were excluded). MMR was re-audited on all 8 folds after both fixes.

## Amendments after the commit (before any training chain, S1 table or arm record)

1. **2026-10-14, at the T015 smoke (FMAE f00): agree = p_buy > threshold, no BUY-argmax
   condition**, for the sizer (`WidthSizerMixin.AGREE_RULE = "p_buy"` on every 034 sizer class,
   CONTROL_SZ included), the S1 split (candidates and the control at share) and the S2 tag check.
   Why: on f00 rule-eligible bars BUY was the argmax on 8.8% for FMAE (control 98.7%, GFM 98.2%),
   median p_buy 0.33, so the 0.5 agree share was unreachable (threshold 0.0, achieved share
   0.0215): every balanced three-class label would have been S1-degenerate by construction. The
   pre-registered 0.072 reference (the control's own entry decisions) and the es role's entry rule
   (BUY argmax AND p_buy > threshold, as the control) are unchanged. No P&L had been read.
