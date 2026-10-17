# Label / indicator study (037 + addenda): closing summary

Closed 2026-10-17 at the user's request. Everything below is PROVISIONAL (one era, 2023-26; late halves
not blind after the screens were read). Nothing is adopted or promoted.

## What was asked
Are there better trainable labels, or indicators correlated with a look-ahead label, to add to the
model; and can combinations (pairs, M-of-N, composites) make a better signal than gbb?

## What was found
1. **The money is in volatility.** On ARO's label and on the fixed-% triple barrier, volatility /
   band-width columns lead (atr_pct_roll, kc_upper/close); the width book already captures them
   (tb lift 1.36 / 1.62 / 1.62). The 037 HGB on all columns ranks candidates better than ARO (0.625 vs
   0.597 late) but as a sizer only matches it (cs_hgb 58.67 vs cs_aro 59.95).
2. **Direction is a small, persistent signal on traded bars**, about half as strong as the first
   screens said (32% of bars were zero-volume gap fills). A 7-column equal-weight "dip" composite
   (price below its trend and band top, still falling) gives AUC 0.535 (all) / 0.552 (candidate) late
   on the volatility-neutral label, where ARO has none (0.50 / 0.54).
3. **As entry triggers, nothing beat gbb in both halves**: KST∧ZScore and CCI∧COPP split halves; the
   CMO pairs fired on zero-volume bars; the 4-of-8 vote lost (−15.3 vs gbb 11.0), with ATR exits too.
4. **As sizers, everything beat the width book; nothing beat cs_aro in both halves.**

| sizer (all 15 folds) | total % | vs width early / late | vs cs_aro early / late |
|---|---|---|---|
| cs_aro (ARO p_buy) | 59.95 | +4.85 / +3.66 | – |
| cs_hgb (037 HGB) | 58.67 | | −3.46 / +2.18 |
| cs_cmp (dip composite) | 60.92 | +2.73 / +6.75 | −2.12 / +3.09 (late IMPROVED p 0.012) |
| cs_mix (ARO + composite) | 61.07 | +4.27 / +5.36 (both IMPROVED) | −0.58 / +1.70 (Sharpe IMPROVED) |

## Open items carried forward
- Forward (blind) check of cs_cmp and cs_mix on 2026-09-07 → 2026-10-12 (needs the 7 composite
  columns computed past the 037 data end).
- ARO retrained without zero-volume rows (TRAIN_ROW_FILTER volume > 0): 24% of ARO's BUY positives
  sit on bars no strategy can trade.

Sources: report.md, vol/report_vol.md, report_tb.md / report_tbatr.md, research/{hgb_sizer,
pair_triggers, mofn_trigger, cmp_sizer, mix_sizer}/report.md.
