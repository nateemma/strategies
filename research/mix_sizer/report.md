# Mixed sizer (cs_mix): report

**Sources:** `preregistration.json` (commit d0f966b6, before any backtest); `instrument.json`;
`verdicts.json`; `vs_cs_cmp.json` (descriptive). Run 2026-10-17.
**Label: PROVISIONAL**; one era, late half not blind.

Score = mean of the ARO p_buy and 037 composite percentiles, ranked on the fold's validation-tail
width-book run starts; agree at the 0.5-share threshold picked on the same tail.

## Decision: PASS (PROVISIONAL) vs the width book and the control; FAIL vs cs_aro

| vs | arm % | comparator % | early / late gain pp | paired (whole) | halves | Sharpe / maxDD |
|---|---|---|---|---|---|---|
| width book `w_aro` | 61.07 | 51.44 | +4.27 / +5.36 → PASS | IMPROVED (p 0.0002) | IMPROVED (p 0.003) / IMPROVED (p 0.010) | UNCHANGED / UNCHANGED |
| control `wf_aro_g24` | 61.07 | 49.97 | +4.84 / +6.27 → PASS | UNCHANGED (p 0.40) | | UNCHANGED / UNCHANGED |
| ARO sizer `cs_aro` (secondary) | 61.07 | 59.95 | −0.58 / +1.70 → FAIL | UNCHANGED (p 0.45) | UNCHANGED (p 0.56) / UNCHANGED (p 0.10) | **IMPROVED** / UNCHANGED |
| composite sizer `cs_cmp` (descriptive) | 61.07 | 60.92 | +1.54 / −1.39 | | | |

- Instrument PASS both halves (0 entry violations, 0 tag mismatches, capital 1.01 / 0.98).
- The mix halves cs_cmp's early shortfall vs cs_aro (−2.12 → −0.58) and keeps part of its late gain
  (+3.09 → +1.70). It is the only sizer that improves the width book significantly in BOTH halves,
  and its whole-period Sharpe beats cs_aro's; but the pre-registered bar vs cs_aro needs a gain in
  both halves, so it does not replace cs_aro.
