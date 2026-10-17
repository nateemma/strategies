# ARO volume-filter retrain (cs_arov): report

**Sources:** `preregistration.json` (commit 9ca3fda4, before training); `instrument.json`; `verdicts.json`;
`auc.json`; per-fold `row_filter.json` beside the weights. Run 2026-10-17, seed 1.
**Label: PROVISIONAL.** Nothing is adopted.

## Decision: PASS (PROVISIONAL) vs the width book and the control; FAIL on the question (vs cs_aro)

| vs | arm % | comparator % | early / late gain pp | paired (whole) | halves |
|---|---|---|---|---|---|
| width book `w_aro` | 57.30 | 51.44 | +3.61 / +2.25 → PASS | IMPROVED (p 0.026) | IMPROVED (p 0.011) / UNCHANGED (p 0.30) |
| control `wf_aro_g24` | 57.30 | 49.97 | +4.18 / +3.16 → PASS | UNCHANGED (p 0.57) | |
| **current ARO sizer `cs_aro`** | 57.30 | 59.95 | **−1.24 / −1.41 → FAIL** | UNCHANGED (p 0.19) | UNCHANGED (p 0.20) / UNCHANGED (p 0.48) |

- Retrained weights rank tradeable gbb candidates slightly WORSE on ARO's own label: AUC 0.597 vs 0.603
  (early), 0.590 vs 0.599 (late) (`auc.json`).
- The filter kept 64-66% of training windows and 54-76% of validation windows per fold.
- Instrument PASS: 0 tag mismatches against the retrained model's own table, capital 1.01 / 1.03,
  trade set = width book except f04.
- Seed 1 only; both half losses are under 1.5 pp and not significant, inside the seed-noise range seen
  before, so "no better" is the finding, not "worse".
- The filter drops windows whose LAST bar has zero volume; the windows' earlier bars, the features'
  look-backs and the label path still include gap-filled bars.
