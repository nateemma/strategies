# Feature Specification: labelfree-jump-model-gate

**Feature Branch**: `016-labelfree-jump-model-gate`

**Created**: 2026-09-15

**Status**: Draft — clarifications pending

**Input**: User description: "Feature: labelfree-jump-model-gate. This is research/label-free-study brief 08 … Implement research/labelfree/jump_gate.py using an online (filtered) sparse jump model refit every 90 days on an expanding window, with the smoothed-state leak test run first and reported. Evaluate as a hard entry gate on the incumbent NNNC_MLX through the pooled walk-forward, reporting drawdown metrics with bootstrap CIs alongside return."

## Context

This is the study's **last brief and an explicit negative control**. `regime/VERDICT.md` already
tested eight regime instruments — BTC vol, variance ratios, Hurst, autocorrelation, trend,
breadth, correlation — and recorded **VERDICT: FAIL**: no instrument beat the better fixed choice
on two temporally distant windows.

The brief exists because two things were not tested, and both are cheap to close out. A jump
model differs *mechanically* from those instruments: it fits regimes with an explicit penalty on
switching, which cuts turnover roughly threefold versus an HMM and halves max drawdown on equity
indices in the source study. And the honest expectation is recorded up front: **the gate is
expected to buy drawdown, not return.**

### What Phase C's own evidence shows, beyond its verdict

Parsing `regime/VERDICT.md`'s evidence table — all 32 rows, measured 2026-09-15 — **18 rows
(56%) scored `edge_pct` exactly 0 because `n_taken == n_marginal`: the gate admitted *every*
marginal trade.** Only one row in the entire table has `beats_bar True`.

```
btc_vol            W3->W1  46/46  edge=0.0      asset_trend        W3->W1  46/46  edge=0.0
btc_vol            W2->W3  28/28  edge=0.0      asset_trend        W2->W3  28/28  edge=0.0
btc_vol            W3->W2  58/58  edge=0.0      asset_trend        W3->W2  58/58  edge=0.0
asset_vr4          W3->W1  46/46  edge=0.0      market_breadth     W1->W3  28/28  edge=0.0
asset_vr4          W3->W2  58/58  edge=0.0      market_breadth     W2->W3  28/28  edge=0.0
asset_hurst        W1->W3  28/28  edge=0.0      market_correlation W1->W3  28/28  edge=0.0
asset_hurst        W3->W1  46/46  edge=0.0      market_correlation W3->W1  46/46  edge=0.0
asset_hurst        W3->W2  58/58  edge=0.0      market_correlation W3->W2  58/58  edge=0.0
asset_autocorr     W3->W1  46/46  edge=0.0      asset_autocorr     W3->W2  58/58  edge=0.0
```

**More than half of Phase C's evidence is not a failure of regime detection. It is gates that
never fired.** A jump-model gate can fail the same way and would look identical to a clean null —
the eleventh instance of the inert-parameter pattern this repository has recorded. So this feature
must demonstrate its gate *actually gates* before any verdict it produces means anything.

### The leak test comes first, and its result is a published number

The brief is unusual in requiring a **deliberate leak** to be run and reported: the same gate
with *smoothed* (Viterbi) states rather than online filtered ones. The smoothed version sees the
whole series and should look materially better (**FR-009** fixes the threshold). **That difference is the size of the leak being
avoided**, and it belongs beside the honest number rather than in a footnote.

This study has learned the hard way why that matters. Brief 07's forward-return target started
one bar early, and the resulting leak did not merely inflate a number — it **inverted a
conclusion** and survived two other controls. A leak whose magnitude is measured up front is one
that cannot quietly explain a result later.

### What the study has established

| brief | verdict |
|---|---|
| 01 direct policy optimisation | NOT RESOLVED |
| 02 label warm-start | NOT RESOLVED — 92% of the advantage was turnover |
| 03 meta-labelling | KILL — nested AUC 0.4978 |
| 04 RL in a pooled env | KILL — pooled reward IQM −0.1301 |
| 05 gradient-free policy search | KILL — PBO 0.7128 over 48,015 evaluations |
| 06 generative stress harness | harness ACCEPTED — 38.6% of re-orderings lose |
| 07 self-supervised features | KILL — the encoder loses information in 10/10 folds |

**No brief produced an adoptable candidate**, which is why this one gates the **incumbent**
`NNNC_MLX` rather than "whichever policy wins briefs 1–3".

## Clarifications

### Session 2026-09-15

- Q: The `jumpmodels` package is not installed — implement the sparse jump model in-repo, or add
  the dependency? → A: **Both.** The in-repo implementation is primary and carries the verdict;
  `jumpmodels` is installed and run as an independent cross-check. The dry-run resolves to
  `jumpmodels 0.1.1`, a pure-Python wheel whose every dependency (numpy, pandas, scipy,
  scikit-learn, matplotlib) is already satisfied in `freqtrade/.venv`, so the cross-check adds no
  new transitive packages.
- Q: Where does the 90-day refit boundary sit relative to a fold's test range? → A: **Refits
  continue through the test range**, each using only data strictly before its own refit point.
  This is what a deployed system does; the gate adapts within the fold and every fit remains
  causal.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A gate that demonstrably gates (Priority: P1) 🎯 MVP

A researcher can show the gate **changes which trades happen** before any performance number is
read from it.

**Why this priority**: Phase C's table contains instruments whose `edge_pct` was exactly 0
because they admitted every trade. A gate that never fires produces a clean-looking null that
means nothing, and this study has recorded that pattern ten times.

**Acceptance Scenarios**:

1. **Given** the gate, **When** it runs over a fold, **Then** the count of entries it blocks is
   reported, and a gate blocking none is declared **inert, not neutral**.
2. **Given** an inert gate, **When** results are reported, **Then** no return or drawdown
   comparison is drawn from it.
3. **Given** the gate, **When** its blocking rate is reported, **Then** the rate is given per
   fold, not only pooled — a gate that fires in one fold and never elsewhere is not a gate.

---

### User Story 2 - The leak is measured, not merely avoided (Priority: P2)

A researcher runs the gate with **smoothed** states, confirms it clears the FR-009 threshold, and
publishes that gap as the size of the leak the online version avoids.

**Why this priority**: the brief mandates it, and it is run **first** so the honest number is
never read without it.

**Acceptance Scenarios**:

1. **Given** both variants, **When** they are compared, **Then** the smoothed result is reported
   beside the online one with the difference stated as the leak's magnitude.
2. **Given** a smoothed variant that does **not** clear the FR-009 threshold, **When** that is observed,
   **Then** it is treated as evidence the online implementation is itself leaking, and the
   feature stops until resolved.

---

### User Story 3 - Regimes inferred causally (Priority: P3)

A researcher can verify that a bar's regime label uses no information after that bar.

**Acceptance Scenarios**:

1. **Given** a bar's state, **When** it is computed, **Then** it depends on no later bar,
   demonstrated by a deletion test rather than asserted.
2. **Given** a refit, **When** it occurs, **Then** it uses only data before the refit point.
3. **Given** successive refits, **When** states are relabelled, **Then** they are mapped by mean
   volatility so labels do not swap between refits.
4. **Given** the in-repo model and the `jumpmodels` package run on the same data, **When** their
   state sequences are compared under matched weights and centroids, **Then** they agree at
   ≥ 0.95, and the independently-fitted agreement is reported beside it with its cause named.

---

### User Story 4 - Drawdown, with intervals (Priority: P4)

A researcher sees max drawdown and time-underwater **with bootstrap intervals**, beside return,
for the gated and ungated incumbent.

**Acceptance Scenarios**:

1. **Given** results, **When** reported, **Then** return, max drawdown and time-underwater each
   carry a bootstrap interval.
2. **Given** a drawdown improvement, **When** it is claimed, **Then** its interval excludes zero
   in at least 10 of 15 folds.
3. **Given** any result, **When** it is framed, **Then** it is a **deployment** modification, not
   an alpha claim.

### Edge Cases

- **The gate blocks everything.** Zero trades is not risk reduction; it is abstention, and the
  study's shared admissibility gate must refuse it.
- **The gate fires only in one regime era.** A gate active in 2 of 15 folds cannot support a
  drawdown claim across the pooled set.
- **Return falls and drawdown falls.** The brief accepts only if return is unchanged *within the
  floor*; a gate that buys drawdown by giving up return is a different trade-off and must be
  reported as such, not as success.
- **State labels swap between refits.** "High vol" becoming "low vol" silently inverts the gate.
- **Smoothed and online agree closely** (gap below 7.633 pp). That is a red flag, not a reassurance — see US2.
- **The two implementations disagree end-to-end.** Measured: they reach different local optima
  of the same objective — ours weights the pair's own volatility, theirs weights BTC's, at an
  identical L1 budget. Characterise it; do not average the two into a range, and do not treat a
  tie-break between equally valid optima as a correctness failure.
- **The weight solve is bimodal between own-vol and BTC-vol.** Those are different *economic*
  gates — idiosyncratic versus market-wide regime — so which optimum is selected changes what
  the gate means, and the report MUST say which one produced the verdict.

## Requirements *(mandatory)*

**Gate activity**

- **FR-001**: The number of entries blocked MUST be reported, per fold and pooled.
- **FR-002**: A gate blocking no entries MUST be declared **inert**, and no performance
  comparison may be drawn from it.
- **FR-003**: A gate blocking *all* entries MUST be refused through the study's shared
  admissibility gate, not reported as risk reduction.

**Causality**

- **FR-004**: Regime inference MUST be **online (filtered)** — a bar's state uses no later bar,
  demonstrated by a deletion test.
- **FR-005**: Refits MUST continue through a fold's test range on an expanding window, each
  using only data strictly preceding its own refit point. No refit window may include a bar at or
  after the refit point, demonstrated by test rather than asserted.
- **FR-006**: States MUST be mapped by mean volatility after each refit so labels do not swap.

**The leak test**

- **FR-007**: The smoothed-state variant MUST be run **first** and its result published beside
  the online one.
- **FR-008**: The difference MUST be reported as the **magnitude of the leak avoided**.
- **FR-009**: "Materially better" is defined as the smoothed arm exceeding the online arm's
  pooled return by at least the pinned noise floor, **7.633 pp**
  (`evaluation.bootstrap.BASELINE_HALF_WIDTH_PP`). If the smoothed variant does not clear that
  margin, the online path MUST be treated as suspect and the feature stopped until resolved.

**Evaluation**

- **FR-010**: The gate MUST be evaluated as a **hard entry gate on the incumbent** through the
  pooled walk-forward.
- **FR-011**: Return, max drawdown and time-underwater MUST each carry a bootstrap interval.
- **FR-012**: The comparison MUST include the ungated incumbent, the four mandatory baselines of
  `00-evaluation-protocol.md` §3, and **Phase C's best fixed choice**. That choice is defined at
  `regime/phase_c_trackability.py:177` as `max(take-every-marginal-trade, the consistent rule)`,
  which in this feature's terms is **`max(ungated incumbent, cash)`** — so it is computed from
  the existing baseline and requires no new run.
- **FR-013**: Any accepted result MUST be framed as a **deployment** modification, never an alpha
  claim.

**Two implementations**

- **FR-014**: The in-repo sparse jump model MUST be the primary implementation and the one the
  verdict is read from.
- **FR-015**: The `jumpmodels` package MUST be run as an independent cross-check, and the
  agreement between the two state sequences reported as a number.
- **FR-016**: The cross-check MUST separate a **correctness** question from a **tie-breaking**
  one, because only the first is a property an implementation can be wrong about.
  - **FR-016a (hard gate).** Given the *same* feature weights and centroids, the two
    implementations' state sequences MUST agree at **≥ 0.95**. This tests the inference
    recursion. Below it, the feature stops.
  - **FR-016b (advisory).** End-to-end agreement, each implementation fitting independently,
    MUST be reported and its cause characterised — but it is **not** a stop. Measured
    2026-09-15: the two optimisers reach different local optima of the *same* objective at the
    *same* L1 budget, one loading the pair's own volatility and the other BTC's. That is
    multi-modality in the weight solve, not a defect in either implementation, and a threshold
    on it would be a threshold on a coin flip.
- **FR-017**: The cross-check MUST use the same interpreter as the primary run, verified by
  `sys.executable`, and its inference path MUST be audited to confirm it is filtered rather than
  smoothed — the package's default is not assumed.

**Seed robustness**

- **FR-022**: The jump-model fit is stochastic through its k-means initialisation, so it MUST be
  run across **at least 5 seeds** per `00-evaluation-protocol.md` §4. The reported result MUST be
  the interquartile mean across seeds with a stratified-bootstrap interval, never a single seed.
- **FR-023**: State assignments MUST agree across seeds at a rate **≥ 0.95**. Below that, the
  regime structure is an initialisation artifact and the feature stops.

**Redundancy against the incumbent (Principle IX)**

- **FR-024**: Before any return or drawdown number is read, the overlap between bars the jump gate
  blocks and bars the incumbent's `rvol ≥ 2.0` guard already rejects MUST be measured and
  reported as `novel_block_rate`. A `novel_block_rate < 0.20` is verdict
  `redundant_with_rvol`: the gate is inert **because rvol got there first**, reported as a defect
  under Principle IV and explicitly not as "regime gating does not help".

**Process**

- **FR-018**: All runs MUST execute sequentially.
- **FR-019**: Only files this feature creates or modifies MAY be staged.
- **FR-020**: A result is admissible only with a specification, plan, task list and a clean
  analyze report produced before implementation and again after.
- **FR-021**: The result report MUST be written whatever the outcome.

### Key Entities

- **Regime state**: a per-bar label inferred causally from past data only.
- **Gate**: the rule turning a state into an entry permission.
- **Leak reference**: the smoothed-state variant, run to measure what is being avoided.
- **Incumbent**: the ungated production `NNNC_MLX`, the comparison baseline.

## Success Criteria *(mandatory)*

### Gate validity (checked before any performance claim)

- **SC-001**: The gate blocks a non-zero, non-total share of entries, reported per fold.
- **SC-002**: Causality is demonstrated by a deletion test, not asserted.
- **SC-003**: The smoothed-state leak is measured and published beside the honest number.
- **SC-004**: State labels are stable across refits, verified by their volatility ordering.
- **SC-004a**: Under matched weights and centroids the two implementations agree at **≥ 0.95**
  (FR-016a). The independent end-to-end agreement is reported alongside, with its cause named
  (FR-016b).

### Protocol §5 (return)

- **SC-005**: Pooled walk-forward return at least the control plus 3.35 pp/yr with a bootstrap
  interval excluding zero — **or**, per the brief, unchanged *within* the floor if the drawdown
  criterion below is met.
- **SC-006**: Positive in at least 10 of 15 folds and in each of W1/W2/W3.
- **SC-007**: Beats all four mandatory baselines net of full costs.
- **SC-008**: Overfitting probability at most 0.2 with deflated risk-adjusted return above zero.
- **SC-009**: The return delta remains positive at 1.5× fees.
- **SC-010**: A separate process reproduces the accepted result from its saved configuration.

### The brief's own acceptance (deployment, not alpha)

- **SC-011**: Return is unchanged within the floor — defined as
  `|Δ annualised return| < 3.350 pp/yr` — **and** max drawdown falls with a bootstrap
  interval excluding zero in at least **10 of 15 folds**. Anything else is a KILL.

- **SC-017**: Across ≥5 seeds, state assignments agree at ≥ 0.95 and the gate's per-fold block
  rate is reported as an interquartile mean with a stratified-bootstrap interval. A block rate
  that varies more across seeds than between the gated and ungated arms is an initialisation
  artifact and a **KILL**.
- **SC-018**: The in-sample (W2) advantage does **not** exceed the W1 advantage by more than the
  floor, computed with `research/labelfree/stats.py::overfitting_differential()`. Exceeding it is
  overfitting: stop and record (`00-evaluation-protocol.md` §5 kill list).
- **SC-019**: `novel_block_rate ≥ 0.20` — the gate blocks a set meaningfully distinct from what
  the incumbent's rvol guard already rejects.
- **SC-020**: The gate is `active` in at least **10 of 15 folds**. A gate firing in fewer cannot
  support a pooled drawdown claim, however large the pooled effect looks.

### Kill gates

- **SC-012**: An inert gate is a **KILL**, recorded as "the gate never fired" and explicitly not
  as "regime gating does not help".
- **SC-013**: A gate that buys drawdown by surrendering return beyond the floor is a **KILL**,
  reported as a trade-off rather than a success.
- **SC-014**: If the smoothed variant does not exceed the online one by ≥ 7.633 pp (FR-009), the
  feature **stops** — the online path is suspected of leaking.

### Process gates

- **SC-015**: The feature directory holds a specification, plan, task list and a clean analyze
  report produced before implementation and again after.
- **SC-016**: The result report is written whatever the outcome, per protocol §6.

## Assumptions

- Fold geometry, purge and embargo come from `evaluation/foldplan.py` via
  `research/labelfree/folds.py`, unchanged.
- The pooled walk-forward harness (`evaluation/`) is used as-is; its pinned constants are not
  recomputed.
- `jumpmodels 0.1.1` installs into `freqtrade/.venv` without pulling new transitive dependencies
  (verified by dry-run, 2026-09-15). The interpreter is `freqtrade/.venv/bin/python`, never a
  pyenv shim.
- **K = 2 states** (bull/bear), matching the source study, is the pre-registered primary
  specification. A `K = 3` run may be reported only as a labelled sensitivity and cannot produce
  the verdict.
- A negative result is the expected deliverable, consistent with briefs 01–05 and 07, and with
  `regime/VERDICT.md`'s own FAIL.

## Non-Goals

- The **soft** use the brief also mentions — regime probability as an input feature to a
  brief-1 policy. The user's scope is a hard gate; brief 01 produced no policy to condition.
- Re-testing Phase C's eight instruments.
- Any alpha claim. This brief's best case is a deployment modification.
- The adjacent approaches listed in `briefs/09-excluded-and-why.md`.
