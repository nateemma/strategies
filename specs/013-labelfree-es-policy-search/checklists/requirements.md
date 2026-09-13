# Specification Quality Checklist: labelfree-es-policy-search

**Purpose**: Validate specification completeness and quality before planning
**Created**: 2026-09-13
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Validation notes (2026-09-13)

**The brief's stated evaluator cannot evaluate the brief's search space.** It says "using the
vectorised simulator from brief 1" and "all vectorised in MLX", but `sim.replay()` is documented
as replaying a **FIXED** entry/exit series and `net_returns` is exposure-path economics with no
stop, grace or minimum-hold logic — while every parameter being searched (`stoploss_grace_hours`,
the ATR multiplier, `cexit_*`, the guard thresholds) *determines* the trade series. A backtest
per evaluation costs ~175 s measured, so the brief's 24,000 evaluations would take ~1,167 hours.
Resolved by building a calibrated exit surrogate, with the calibration as a **blocking gate**
(SC-017, SC-017a). This is the third consecutive brief whose named mechanism did not exist:
brief 03's signals export could not carry the prediction, brief 04's comparator was never
persisted, and now this.

**The trial log does not scale to this brief.** It holds 1,158 lines for the 505 trials of four
completed features and is committed to git; brief 05 implies up to 48,000 evaluations. Resolved
with a per-feature sidecar and one summary row in the shared log — and SC-018 explicitly
requires the **audit to verify both files**, since an audit reading only the summary would
certify a record it never examined.

**The brief and the protocol disagreed about PBO**, the brief saying "expect it to be high —
that is the measurement" and §5 making > 0.2 a failure. Resolved as both: the candidate is
killed, the measured value is the finding. §5 is preserved rather than relaxed, which the brief
requires ("verbatim or stricter").

**Two measured numbers govern this feature and are written into Context rather than left in the
brief's prose**: the noise floor is 7.633 pp and an entire rvol parameter sweep spans 4.93 pp —
so the search space's total range of effect is *smaller than the uncertainty on measuring it*.
SC-010 makes that a kill, and the Assumptions state plainly that a quantified negative is the
expected outcome.

**SC-003's bar was corrected against measurement.** "Beats all four baselines" reads as beating
zero; measured over folds f05–f14 the production control is the only profitable baseline at
**+7.86 pp/yr**, so the gate is materially harder than it appears. Recorded inline.

**Three requirements come from this session's defects rather than the brief**: FR-003/SC-017a
(agreement measured, not assumed — after a "faithful restatement" needed a bit-for-bit gate in
feature 012); FR-004/FR-005/SC-015 (abstention and losing-to-cash cannot be reported as
improvement — the admissibility gate, added after four historical comparisons failed it); and
FR-011/SC-011 (the record must let an auditor *verify* the single test evaluation rather than
take it on trust).
