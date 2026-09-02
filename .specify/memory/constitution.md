<!--
SYNC IMPACT REPORT
==================
Version change: (unratified template) → 1.0.0
Bump rationale: MAJOR/initial ratification. The prior file was the unmodified Spec Kit
scaffold with zero project content, so this is the first ratified constitution rather
than an amendment to an existing one.

Modified principles: none (no prior principles existed).

Added sections:
  - Core Principles I–X (10 principles; the resolved scaffold provides 5 slots, expanded
    per explicit instruction)
      I.    Evaluate in the Deployment Regime
      II.   Persistence Across Temporally Distant Eras (NON-NEGOTIABLE)
      III.  Learnability Is Not Edge
      IV.   Verify Parameters Are Actually Live
      V.    Lookahead-Clean Is Not Realistic
      VI.   Diagnose Before Tuning
      VII.  Validate Cheaply; Test Trackability Before Building Adaptivity
      VIII. One Parameter Space at a Time
      IX.   Orthogonality Earns a Factor Its Place
      X.    Structural Fixes, Not Per-Pair Overrides
  - Repository and Code Constraints (scaffold slot SECTION_2)
  - Experiment Workflow and Quality Gates (scaffold slot SECTION_3)
  - Governance

Removed sections: none.

Follow-up TODOs: none. All placeholder tokens resolved; no deferred items.
-->

# Freqtrade Strategy Research Constitution

This constitution governs systematic-trading research in this repository. Every principle
below was derived from a specific, recorded failure — each states the rule, then the
evidence that produced it. Principles constrain what counts as a finding, not merely how
code is written.

## Core Principles

### I. Evaluate in the Deployment Regime

Regime-dependent strategies MUST be judged in the regime they will actually be deployed
into. The primary verdict is the current-regime result; full-history results are context
only and MUST NOT be quoted as the headline outcome.

Full-history averages blend across regimes that may be structurally extinct, silently
inflating every result. The opposite failure — overfitting a short recent window — MUST be
guarded by a persistence split across at least two temporally distant windows, per
Principle II. Neither guard is optional; a result that survives only one of them is not a
finding.

### II. Persistence Across Temporally Distant Eras (NON-NEGOTIABLE)

Two adjacent walk-forward windows share a regime and are NOT independent evidence. Any
claim that an effect persists MUST be supported by at least one genuinely distant era.

A pattern observed in two adjacent windows is one observation, not two. Recorded failure: a
day-of-week P&L skew held across two adjacent windows and looked robust; in an earlier era
it reversed sign completely, and the filter built on it removed 26 of 27 trades in that
era.

Corollary — three points do not make a line. Before calling a small-N pattern a trend, get
more observations; rolling windows are usually cheap. Recorded failure: three per-window
hyperopt winners were read as a monotonic trend in recency; 1681 rolling observations
showed no trend at all.

### III. Learnability Is Not Edge

A more learnable target is not a better trade. Arms MUST be accepted or rejected on
realistic, cost-aware and stop-aware backtest P&L and risk-adjusted return. Model-fit
metrics — MCC, correlation, R-squared, validation loss — MUST NOT decide acceptance.

Learnability is a necessary-but-not-sufficient screen: it may be used to eliminate a
candidate, never to accept one. Confirmed at least five independent times here, including a
breakout label at MCC 0.77 that lost money at every timeframe and exit tested, and a
P&L-weighted loss that raised validation MCC monotonically across 6 of 6 runs while P&L
stayed indistinguishable from noise.

### IV. Verify Parameters Are Actually Live

A parameter the framework confirms it loaded MAY still do nothing. Before any null result
is read as a finding, the parameter MUST be proven to have changed behavior by measuring
something downstream of it — an entry-candidate pass rate, a trade count, a byte-difference
in results.

An arm whose downstream measurement is unchanged is INERT, and MUST be reported as a defect
rather than as evidence about the hypothesis. At least five instances are recorded: trailing
stop parameters and the static stoploss are inert whenever a custom stoploss path owns the
stop; hyperopt parameters go inert via stale strategy JSON files, per-epoch analysis flags,
and cache keys. In each case the framework logged the value as applied.

### V. Lookahead-Clean Is Not Realistic

Implausibly strong backtest results MUST be presumed to be execution-side bias until proven
otherwise. Every evaluation MUST run under realistic execution constraints — volume,
liquidity, and spread.

Fills at displayed prices that could not happen live, such as a stake exceeding available
volume, inflate out-of-sample returns even when lookahead analysis is clean. Recorded: a
funding-carry prototype's roughly 73% return was phantom, and a cross-sectional reversion
study's ~92% edge vanished entirely when restricted to liquid names.

Separately, framework-provided lookahead analysis is NOT sufficient evidence. Lookahead MUST
be measured end to end, by comparing output computed from truncated history against output
computed from full history over the overlapping region.

### VI. Diagnose Before Tuning

When a system misbehaves, diagnostics that show what it is actually doing — per-class
statistics, per-step state, pass rates — MUST be added before any hyperparameter is touched.
Localize the failure first, then make one targeted change.

Speculative hyperparameter bumping wastes cycles and frequently regresses. A tuning change
proposed without a diagnostic that localizes the fault is out of order.

### VII. Validate Cheaply; Test Trackability Before Building Adaptivity

Every expensive build MUST be gated behind a minimal feasibility measurement capable of
killing or confirming the hypothesis in an afternoon.

For any proposed adaptive or regime-conditional parameter this gate is specific and
mandatory: build the rolling-optimum series and test whether a trailing-best forecast beats
a CONSTANT. A parameter whose optimum merely VARIES is not a candidate for adaptivity —
variation is necessary but nowhere near sufficient, and predictability is the missing
requirement. Four adaptive schemes have now lost to static values in this project.

A regime-to-parameter mapping MUST NOT be fitted on a handful of window observations.

### VIII. One Parameter Space at a Time

When the sample is thin, search MUST proceed one parameter space at a time: tune one side,
lock it, then tune the other. Joint search over many degrees of freedom overfits the
specific backtest.

Leading overfit signals — one risk-adjusted metric falling while another rises — MUST be
watched during the search and reported, not discovered afterward.

### IX. Orthogonality Earns a Factor Its Place

In a multi-factor book, a new factor earns inclusion only if it is roughly uncorrelated with
what is already there. Return correlation against the existing factors MUST be checked
FIRST, before performance is evaluated.

A factor correlated with the dominant axis is that axis in disguise; adding it dilutes the
book while appearing to diversify it.

### X. Structural Fixes, Not Per-Pair Overrides

Strategies MUST NOT contain pair-specific code or per-pair parameter overrides. Fixes MUST
be structural and apply uniformly across the universe.

If a change works only for one pair, it is overfit, not a fix. This applies equally to
thresholds tuned until a single problem pair behaves.

## Repository and Code Constraints

**Repository boundary.** The outer freqtrade checkout is upstream code and MUST NOT receive
commits. Only the nested `user_data/strategies` repository is ours, and only it is
committed to.

**Commit workflow.** Commit directly to master. This is a solo repository with no
pull-request workflow, and branching has been explicitly waived.

**Staging.** Stage only the specific files a change touched. Blanket staging (`git add -A`,
`git add .`) is prohibited: repository files are edited concurrently by the user, and an
unexpected staged file is their work in progress, not part of the change.

**Framework must stay strategy-agnostic.** The `Framework` directory MUST NOT contain
strategy-specific code. Strategy specifics are registered at runtime from the strategy
itself.

**Parameter dictionaries MUST merge from the parent.** A strategy that redefines a parameter
dictionary as a fresh literal silently drops every inherited key. Recorded failure: this
caused a strategy to run with its stop-loss grace disabled, undetected. Parameter
dictionaries MUST be merged from the parent, never redefined wholesale.

## Experiment Workflow and Quality Gates

An experiment is not complete, and its result MUST NOT be recorded as a finding, until all
of the following hold:

1. **Pinned time range.** Any multi-day A/B MUST pin its time range explicitly, so arms
   measured on different days remain comparable.
2. **Distant-era window.** At least three windows, at least one temporally distant from the
   others (Principle II).
3. **Realistic execution.** Volume and liquidity constraints active (Principle V).
4. **Liveness proof.** A downstream measurement showing the changed parameter actually
   altered behavior (Principle IV).
5. **Paired seed-robustness.** Any accepted delta under one percentage point MUST be
   confirmed under paired repeated runs that isolate run-to-run variance.
6. **Trade count reported.** A zero-trade arm is inconclusive, never a zero-return result.
7. **Scalers regenerated.** Any change to the feature include list REQUIRES regenerating the
   feature scalers before the comparison is run; otherwise the comparison is made against
   stale normalization and is void.

A negative result that satisfies these gates is a successful outcome and MUST be recorded
with the same weight as a positive one, including the evidence that produced it.

## Governance

This constitution supersedes other practices and conventions in this repository. Where a
plan, spec, or prior convention conflicts with a principle here, the principle wins.

**Compliance review.** Every specification and implementation plan MUST be checked against
these principles before work begins. A proposal that violates a principle MUST either be
revised or carry an explicit, written justification naming the principle and the reason the
exception is warranted. Silent exceptions are not permitted.

**Amendment procedure.** Amendments MUST be made by updating this file, with the change
recorded in the Sync Impact Report at its top and the version and amendment date updated in
the same commit. Principles are added or changed only from evidence — a recorded measurement
or a documented failure — never from expectation alone.

**Versioning policy.** Semantic versioning applies to this document. MAJOR for backward
incompatible governance changes, principle removals, or redefinitions that invalidate prior
work. MINOR for a new principle or materially expanded guidance. PATCH for clarifications,
wording, and non-semantic refinements.

**Runtime guidance.** Day-to-day conventions that are not governance live in `AGENT_GUIDE.md`
and in project memory. Those are subordinate to this document and MUST NOT contradict it.

**Version**: 1.0.0 | **Ratified**: 2026-09-02 | **Last Amended**: 2026-09-02
