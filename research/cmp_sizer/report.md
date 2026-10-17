# Composite sizer (cs_cmp): report

**Sources:** `preregistration.json` (commit c4bf7aa2, before any run; pick-wrapper fix 3ec4e897, before
any cs_cmp backtest); `instrument.json`; `verdicts.json`; `vs_cs_hgb.json` (descriptive). Run 2026-10-17.
**Label: PROVISIONAL**; one era, late half not blind (the composite was screened on all halves).

## Decision: PASS (PROVISIONAL) vs the width book and the control; FAIL vs cs_aro

| vs | arm % | comparator % | early / late gain pp | paired (whole) | Sharpe / maxDD |
|---|---|---|---|---|---|
| width book `w_aro` | 60.92 | 51.44 | +2.73 / +6.75 → PASS | IMPROVED (p 0.0002) | UNCHANGED / UNCHANGED |
| control `wf_aro_g24` | 60.92 | 49.97 | +3.30 / +7.66 → PASS | UNCHANGED (p 0.40) | UNCHANGED / UNCHANGED |
| ARO sizer `cs_aro` (secondary) | 60.92 | 59.95 | −2.12 / +3.09 → FAIL | UNCHANGED (p 0.64); early UNCHANGED p 0.20, **late IMPROVED p 0.012** | UNCHANGED / UNCHANGED |
| HGB sizer `cs_hgb` (descriptive) | 60.92 | 58.67 | +1.33 / +0.92 | UNCHANGED (p 0.24) | |

- Instrument PASS both halves: 0 entry violations, 0 tag mismatches, thresholds loaded, capital
  ratio 1.05 / 1.01, trade set equals the width book's except f04, no entry without a score.
- Highest total of the sizers so far, but against cs_aro it repeats cs_hgb's pattern: behind early,
  ahead late. The pre-registered bar needs both halves, so the composite does not replace ARO.
- The composite needs 7 columns at runtime (5 SimpleStrategies columns + 2 returns), no model.
- Instrument incident (fixed before any backtest): the first chain's pick wrapper patched
  `hgb_sizer.pick.PLAN`, but `pick()` / `verify()` bind the plan as a default argument, so the
  picks ran on plan_cs_hgb (identical thresholds gave it away). Aborted at f12; plan_cs_cmp reverted;
  cs_hgb's fold classes re-verified (thresholds equal its plan). The wrapper now passes the plan.
