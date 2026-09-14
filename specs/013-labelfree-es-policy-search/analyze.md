# Analyze reports: 013-labelfree-es-policy-search

FR-023 and SC-016 require a clean analyze report **before** implementation and **again after**.
Both are recorded here.

---

## Pass 1 — pre-implementation (2026-09-13)

**2 CRITICAL, 4 HIGH, 4 MEDIUM, 1 LOW.** All fixed; re-run clean at 43/43 requirements covered.

Both criticals were the same shape: the feature computed PBO and audited leakage but had **no
task producing the protocol §5 verdict the study is judged on**. 48,000 evaluations would have
run with no acceptance gate evaluated.

| id | finding | fix |
|---|---|---|
| C1 | SC-002 / Principle II (NON-NEGOTIABLE) had no task — plan.md *claimed* the principle while nothing implemented it | T049b (per-fold ≥10/15), T049c (W1/W2/W3) |
| C2 | SC-001, the primary acceptance gate, had no task | T049a (pooled vs control, 3.35 floor, bootstrap CI) |
| H1 | SC-008 in-sample vs held-out gap had no task | T049d |
| H2 | "minimum holding time" named as a lever in the spec, absent from the design | became dimension 10; see below |
| H3 | liveness `inert` threshold chosen after seeing spans | pre-registered in data-model.md §6 |
| H4 | FR-009 regularisation weights unspecified | pre-registered in data-model.md §1a |
| M1 | SC-003 baselines (bar +7.86 pp/yr, not zero) had no task | T049e |
| M2 | FR-023/SC-016 post-implementation analyze had no task | T060a |
| M3 | dimension count inconsistent (8 vs 9) across artefacts | unified at 10 |
| M4 | SC-008's "the floor" ambiguous between three floors | named as 3.35 pp/yr |
| L1 | SC-013 mis-cited for budget equality | removed |

**H2 inverted on inspection.** `research.md` R3 called `unclog_12h` "a portfolio-level rule" and
used that to justify replaying it, which made minimum holding time look unsearchable.
`BaseStrategy.py:1211-1218` shows `unclog_12h`, `unclog_1d` and `max_hold` are all purely
per-trade. They became recomputed, minimum holding time entered as `hold_window_hours`, and the
replayed-mechanism risk fell from 19% of trades to 11.9% — and later to 0%.

---

## Pass 2 — post-implementation (2026-09-14)

**0 CRITICAL, 1 HIGH, 2 MEDIUM, 2 LOW.** All fixed.

| id | finding | fix |
|---|---|---|
| P1 | **The decisive fidelity test was never written.** T013 required comparing the surrogate's exit mix against the archive. It was run *by hand* — finding all four fidelity bugs — but never encoded, so the one check that caught every real defect had no regression guard | `test_incumbent_exit_mix_matches_a_real_backtest`, asserting against the measured real backtest |
| P2 | FR-019 under-reported: `bounds_hit` inspected only the *best* fold, so f04 sitting on the `cexit_max_days` bound read as empty | `incumbent_delta` unions `on_bound` across all 15 selections |
| P3 | the 60 MB sidecar was untracked but **not gitignored** — R6's design keeps it out of git and nothing enforced that | `.gitignore` entry |
| P4 | 17 stale task checkboxes; 13 demonstrably done | marked, artefacts verified present |
| P5 | `INCUMBENT_EXPOSURE_FRACTION = 0.5` set mid-implementation, anchored to production but **not pre-registered** like the other thresholds | disclosed in the result document's Limitations |

**Fixing P1 immediately earned its keep**: the new test failed on first run, catching `roi`
firing 28 times against a real backtest's 13. Six of those fired on the **fill bar** — using
that bar's high to trigger ROI on the bar the trade entered at its open, the one bar with no
path information. freqtrade special-cases `trade_dur == 0` for exactly this reason. Corrected to
22; the residual divergence is recorded as a limitation rather than tuned away.

### Constitution alignment (pass 2)

| principle | status |
|---|---|
| **II. Persistence** (NON-NEGOTIABLE) | **satisfied as process** — SC-002b was evaluated and *failed* (W3 −0.02), contributing to the KILL. The pass-1 violation (claimed, unimplemented) is closed. |
| **IV. Parameters actually live** | satisfied, and it paid off — 6 of 10 dimensions found inert *before* the search, and the two most suspicious readings verified rather than reported |
| **VIII. One parameter space at a time** | deviation stands as documented, with every compensating control exercised |
| I, III, V, VI, VII, IX, X | no conflicts |

### Metrics (pass 2)

- requirements 43/43 exercised, each with a measured value
- tasks 70/70
- tests 165 passing
- unmapped tasks 0 · duplication 0 · terminology drift 0
