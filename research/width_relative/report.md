# 033 width-relative label: report

**Sources:**
- Pre-registered in `preregistration.json` / `.md`: commit 1612b603, with amendments ee2e8eb6
  (trade identity) and 530dede3 (judge_one rule). Both amendments were made before any P&L was
  read.
- Verdicts: `verdicts.json` (`judge.py --seed-repeat 2`).
- Run: 2026-10-14.

**Label: PROVISIONAL.** There is one era (2023–26), so no outcome is a finding. Nothing is adopted
or promoted.

## Decision

| model | arm | stopped at | decision | evidence |
|---|---|---|---|---|
| ARO | exit-off sizer (`wr_aro_sz`) | S1 | **FAIL** | early agree − disagree gap 0.043 vs the current model's 0.072 |
| GBR | exit-off sizer (`wr_gbr_sz`) | S1 | **FAIL** | gap 0.005 vs 0.063 |
| ARO | exit-on sizer (`wr_aro_szx`) | S2 | **FAIL** | early pp 36.59 vs exit-off 37.03 (−0.45); vs width 35.85 (+0.73) |
| GBR | exit-on sizer (`wr_gbr_szx`) | S2 | **FAIL** | seed 1: +0.74 vs exit-off (under 1 pp, so a repeat) and +3.29 vs width; seed 2: −0.09 vs exit-off (41.58 vs 41.67) |

No arm reached S3, so no late-half (f08–f14) P&L was read for any arm.

**Negative result:**
- The width-relative label does what it was built to do: every width bucket is about 50%
  positive, and the label is uncorrelated with width.
- But a model trained on it separates width-book trades **worse** than the current model:
  - ARO's gap is 0.043, under the current model's 0.072;
  - GBR's gap is 0.005, against 0.063.
- Model exits on top of the sizer add nothing measurable in the early half: −0.45 pp on ARO, and
  +0.74 then −0.09 pp across GBR's two seeds.

## S0: label and instrument (`instrument.json`; no P&L read)

**Label, all 15 folds, both models:**
- every training-row bucket share is 0.4999–0.5001 on both sides;
- Spearman(label, bb_width) is 0.0015–0.0020 (BUY side) and −0.0002 to 0.0010 (SELL side);
- the training classes are about 19% SELL, 50% HOLD and 31% BUY of kept rows;
- validation-row BUY shares are 0.37–0.43 on f00 (reported, not bounded).

On the BUY side the bucket median outcome is the +2% target in every bucket. So BUY means "hit the
target faster than the median hit".

| check | ARO exit-off | ARO exit-on | GBR exit-off | GBR exit-on |
|---|---|---|---|---|
| entry violations | 0 | 0 | 0 | 0 |
| tag-to-agree mismatches | 0 | 0 | 0 | 0 |
| thresholds loaded at 4 dp | yes | yes | yes | yes |
| trade identity with width book | 15/15 | n/a | 14/15 (f04: 2 trades differ) | n/a |
| model exits (folds with gate bars) | n/a | 211 (all folds live) | n/a | 231 (all folds live) |
| realised agree share among trades | 0.452 | 0.453 | 0.633 | 0.635 |
| capital vs width book (fold mean) | 0.961 | 0.966 | 1.047 | 1.053 |
| realised agree / disagree stake | 1.74 | 1.74 | 2.01 | 2.01 |
| wallet-short (uncapped matched trades) | 16/273 | 16/273 | 31/278 | 31/277 |

**Stakes:**
- The width book's own stakes are mostly below 900. Sampled on ARO folds f00, f08 and f11, its
  mean stake is 458–626, and 44–77% of its trades are under 900. The causes are BaseStrategy's
  liquidity cap and wallet availability.
- On those trades the cap also binds for the sizer, so sizing is effectively weaker than 2:1
  there.
- Wallet shortfalls on uncapped trades are 6–11%. They are flagged in 8 of 15 (ARO) and 12 of 15
  (GBR) folds, and reported, not an S0 condition.

**Thresholds:**
- picked on the first bar of each width-book candidate run in the validation tail;
- the achieved share is 0.49–0.51 on the training data;
- the realised test share is 0.45 (ARO) and 0.63 (GBR).

## S1: separation inside the width book, early half (`separation.json`)

Identical width-book trades, f00–f07. The current model's gaps reproduce the pre-registered values.

| model | width-relative agree pp/trade (n) | disagree (n) | gap | current gap |
|---|---|---|---|---|
| ARO | 0.109 (199) | 0.066 (213) | 0.043 | 0.072 |
| GBR | 0.076 (313) | 0.071 (181) | 0.005 | 0.063 |

The late-half gap was not computed: no exit-off arm reached S3.

## S2: early half, pp summed over f00–f07

| model | exit-on | exit-off (comparator) | width book | control (context) |
|---|---|---|---|---|
| ARO | 36.59 | 37.03 | 35.85 | 35.28 (`wf_aro_g24`) |
| GBR, seed 1 | 40.01 | 39.27 | 36.72 | 37.94 (`fs_gbr_gr`) |
| GBR, seed 2 | 41.58 | 41.67 | 36.72 | — |

The exit-off sizers' early pp were read only as the exit-on arms' comparator (pre-registration).
Their own verdicts stopped at S1. They are above the width book (ARO +1.18, GBR +2.55, GBR seed 2
+4.95) and above the controls. That is descriptive context, not a judged result.

**A note on capital:** GBR's sizer deployed about 5% more capital than the width book, ARO's about
4% less.

## Harness lessons (recorded in the pre-registration amendments and memory)

1. **Trade identity.** A different stake changes `profit_ratio` at the 1e-7 level through the
   rounding of the coin amount. So trade identity is checked on (pair, open_date, close_date), not
   on `trades_hash`.
2. **Stale fold classes.** `research/candidate_training/train.py` regenerates fold classes without
   thresholds. `run_folds` then skips folds that already have a record, so a smoke record survives
   with a stale class file. The f00 smoke record was set aside (`*.smoke.json`, identical
   trades_hash dbc49c69cc0f4795) and f00 was re-run after the pre-registration's latest commit.
3. **Liquidity cap.** Most of the width book's trades are below the 900 stake (liquidity cap and
   wallet), so a stake multiplier is partly absorbed. Any sizing study must measure realised stake ratios, not
   intended ones.

## Fix-oriented follow-ups (not run)

1. **A sizer arm with the CURRENT models.** No retraining: `WidthSizerMixin` on the controls' own
   weights. The current models separate better inside the width book (S1 reference gaps 0.072 and
   0.063), and the exit-off sizers with the weaker width-relative model already sat above the width
   book in the early half (context only). It is cheap: 2 × 15 backtests plus 30 single-fold picks.
   Pre-register it on the late half, which is still unread for sizer arms.
2. **Rank-weighted stakes** (0.5 + 0.5 × the p_buy percentile) instead of agree/disagree. Rank
   weighting beat unsized per unit capital on 4/4 models in the earlier exploratory screen
   (findings §5).
3. **Model exits under an open exit gate.** The controls' gate (rvol > 2.0) passes few bars: 211
   and 231 model exits over 15 folds. The gate study's +10.4 pp was at an open gate. Test exits
   with the gate opened, as a separate parameter space.

## Provenance

- **Arm records:** `regime/arms/wr_<m>_sz*_fNN_FNN.json`, and the seed 2 records `*_s2_*`.
- **Plans and pooled results:** `evaluation/folds/plan_wr_*`, `results_wr_*`.
- **Thresholds:** `thresholds_wr_<m>.json`, `thresholds_wr_<m>_log.txt` and
  `thresholds_wr_gbr_s2.json`.
- **Weights:** 30 seed-1 folds under keys `lstmraw_gbbpath_g0.02_h48__…` (one per fold and model),
  plus 8 seed-2 GBR folds. Each has `width_relative.json` and `row_filter.json` beside it.
