# Pre-implementation Analysis: 017-signal-context-timing

**Date**: 2026-09-16 · **Run**: `/speckit-analyze` before `/speckit-implement` (SC-016, T004)

## Pass 1 — findings (8, all remediated)

| ID | Category | Severity | Summary | Resolution |
|----|----------|----------|---------|------------|
| **C1** | Correctness | **CRITICAL** | **Units error that inverted the feature's central gate.** `effect_size_pp` was compared against **7.633 pp**, but the confound test measures a **win-rate** difference while 7.633 pp is an **annualised-return** floor. The motivating effect is 14.8 win-rate points — it would have **passed** the gate while carrying no return information. Principle III's defence was denominated in the wrong currency. | FR-007 now requires **two** effect sizes: `effect_winrate_pp` (reported, never gated) and `effect_return_pp_yr = Δ mean profit_ratio × 9.0 × trades/yr` (the only quantity the floor applies to). `verdict()` must **raise** rather than silently gate on the wrong one; `test_winrate_effect_cannot_reach_the_return_floor` (T015a) enforces it |
| **C2** | Process | **CRITICAL** | **T001 would have copied 2.5 GB into the git repo** — `es_sink` 366 MB plus `es_panels` **2.1 GB** — and T003 gitignored only `out/`. | Measured: the panel cache **rebuilds in 3.4 s/fold (~1.7 min for all 30)**, so it is never copied. T001 now takes only `es_sink`, to a path it gitignores in the same task |
| **A1** | Ambiguity | HIGH | The T040 KILL gate fired on **"large overlap"** with no threshold — the same defect caught as A3 in feature 016 and fixed there, repeated here. | Pre-registered: overlap = share of triggers where the existing stack fired anyway **within 8 bars**; **≥ 0.70 is inert** |
| **A2** | Ambiguity | HIGH | *"Tighten the stop"* left the magnitude undefined — an unbounded free parameter, inviting the joint search Principle VIII forbids. | FR-019a freezes the factor at **0.5**; an outright exit is a labelled sensitivity that cannot produce the verdict |
| **A3** | Ambiguity | MEDIUM | "Within fee rounding" was an undefined tolerance. | `RECONCILE_TOL = 0.002` in `profit_ratio` units |
| **A4** | Underspecification | MEDIUM | Two of four hypotheses carried undefined windows — `drawdown_from_high` had no lookback, `burst_position` no gap threshold — i.e. silent free parameters inside a test claiming exactly four pre-declared hypotheses. | Frozen: **96-bar (1-day)** trailing high; **24-bar (6h)** burst gap, chosen to exceed the 3–10h median hold so a burst cannot span a completed trade |
| **G1** | Coverage | MEDIUM | Fidelity was validated on **f08 only**. Nothing checked the other fourteen before the build proceeded — yet it is the cheapest possible kill. | **T011a**: all-fold fidelity gate immediately after T006, seconds to run |
| **U1** | Underspecification | MEDIUM | The EXPLORATORY flag's mechanism was undefined — an un-persisted flag is how a label quietly fails to appear on output. | Written to `out/confound.json`; downstream stages **read it from disk**, never inherit it in memory |

## The one worth reading twice

**C1 fails open, which is the dangerous direction.** Feature 016's analogous defect — a kill gate
keyed on undefined redundancy — would have refused to fire. This one would have fired *approvingly*
on the wrong quantity: 14.8 > 7.633, gate cleared, feature proceeds to spend 45 backtests on an
effect that may be worth a fraction of a percentage point of return.

The two currencies are easy to conflate because both are called "percentage points". A win-rate
shift and an annualised return are not comparable quantities, and the study's entire
Principle III defence rested on comparing them. The contract now refuses the comparison rather
than trusting a reader to notice.

## Pass 2 — verification after remediation

| check | result |
|---|---|
| Functional requirements | 24 (FR-001…FR-023 + FR-019a), all referenced in tasks |
| Success criteria | 20, all referenced in tasks |
| Tasks | **59** (56 + 3 inserted), unique, base IDs contiguous T001–T056 |
| Story-label / phase integrity | 0 mismatches |
| Undefined gate triggers | **0** — every kill gate fires on a number |
| Constitution | I–X hold; III and IX carry the binding constraints, both now numeric |

**Critical issues: 0. Ready for `/speckit-implement`.**

## Pre-registered expectation, unchanged

The repeat-signal arm most likely dies at **T023** — "received a repeat signal" restating "is
underwater" — reached with **zero backtests**. Second most likely, it survives and proves inert
at **T039** against the stop stack it was measured through. A result clearing both and then the
7.633 pp/yr floor would be the first adoptable finding in this line.
