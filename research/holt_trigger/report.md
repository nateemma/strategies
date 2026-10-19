# 038 vote C (multi-Holt) as SimpleTrigger triggers: report

**Sources:** `preregistration.json` (commit 7f247bcc, before any backtest); `verdicts.json`. Run
2026-10-19. **Label: PROVISIONAL.** Vote C was chosen with both halves in view, so neither half is
blind. Nothing is adopted.

## Decision: not a candidate (both arms FAIL both verdicts)

| arm | trades (early / late) | alone pp vs comparator | gain early / late | paired (whole) | beside width book: total / early / late (w_aro 51.44 / 35.85 / 15.59) | paired |
|---|---|---|---|---|---|---|
| `st_holt_up` (vs `st_g24_mom`) | 1854 (1136 / 718) | 7.01 vs 34.85 | −18.46 / −9.37 → FAIL | UNCHANGED (p 0.22) | 58.46 / 52.94 / 5.52 → FAIL (late below w_aro) | UNCHANGED (+7.0 pp, p 0.57) |
| `st_holt_fade` (vs `st_fam_gbb`) | 1870 (997 / 873) | −7.45 vs 11.04 | −26.81 / +8.32 → FAIL | INCONCLUSIVE_ARM_LOSES (p 0.32); early WORSENED (p 0.042) | 41.60 / 36.97 / 4.63 → FAIL | UNCHANGED (−9.8 pp, p 0.23) |

**Instrument:** PASS for both. Causality PASS (re-measured for every trigger with the forecast cache off;
no other verdict changed). Every fold has records, and neither half has zero trades.

## Reading

- **Up (momentum):** the Holt vote is a much weaker momentum entry than `breakout_gbb`. With identical
  exits it makes 7 pp against 34.85, and it loses in both halves. Beside the width book it adds
  +17 pp early but costs −10 pp late. This is the same late-half decay as everywhere in 038, and in
  the forward window. The 10–30 bp timing edge measured in 038 doesn't survive the framework's exit
  stack (trailing stop and ROI dominate its exits) as an entry on its own.
- **Fade (reversion):** worse than gbb overall (−7.45 against 11.04), driven by the early half (WORSENED,
  p 0.042). The late half beats gbb by +8.3 pp, which matches the recent reversal the plot showed, but
  it's one half (INCONCLUSIVE_ARM_LOSES) and the rule needs both.
- **Together with 038:** no standalone use of the Holt trend signal passes, whether as a fixed-hold
  forecast, a vote, a momentum trigger or a fade trigger. The remaining open idea is the one already
  noted for the return to NNNC/NNMT: the trend state as a filter or sizer inside an existing book.

**Housekeeping:** both plans originally shared one `run_id`. Only the unused per-run summary file
(`results_<run_id>_seed1.json`) was overwritten; the judge reads per-arm records. The plans now carry
distinct ids.
