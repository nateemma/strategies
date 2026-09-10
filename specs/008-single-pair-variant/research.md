# Phase 0 Research: Single-Pair (LINK) Variant

All RESOLVED. Tasks act on these; they are not re-derived.

## R1 — Pair dominance ROTATES. DONE.

Measured 2026-09-10 from the `nnnc_v2` control arms:

    pair    W1 %P&L   W2 %P&L   W3 %P&L
    ZEC       44.9      7.0       2.8
    XRP       23.6     38.3      -3.1
    NEAR       5.0      6.9      27.5
    LINK      11.6      9.5      26.4
    SUI        4.0     21.6       3.2
    AVAX         -     -2.8      21.9

Top-3: W1 ZEC/XRP/LINK · W2 XRP/SUI/DOT · W3 NEAR/LINK/AVAX. **"The edge is ZEC" is a W1
observation, not a property of the strategy.** This is a finding in its own right and must appear
in the conclusion (FR-012, SC-005).

## R2 — LINK chosen over ZEC. DONE (decision).

LINK is the only pair contributing materially in all three windows and never collapsing:

    window  trades  per-trade          win    %P&L
    W1           8  +2.270% (sd 1.47)  100%   11.6
    W2          26  +0.454% (sd 3.67)   73%    9.5
    W3           5  +2.500% (sd 0.98)  100%   26.4

ZEC rejected: 45% of W1 but 2.8% of W3, so a ZEC model tests a regime artifact. XRP rejected: 38%
of W2 but NEGATIVE in W3. LINK is nobody's top pair in W1 or W2, which is what makes it a test of
"is single-pair modelling viable" rather than "was ZEC special last year".

## R3 — Power: returns cannot resolve; expectancy can. DONE.

W1's floor is 7.633pp measured on 99 trades. LINK contributes **8**. Total-return comparison is
meaningless and `compare.py`'s MIN_DIFF_TRADES=20 will refuse a CI.

But per-trade expectancy at n=8, sd 1.47 gives SE ≈ **0.52pp**, so +2.270% is resolvable as
positive. **The feature's power lives in per-trade expectancy and trade selection, not totals.**
This is why US2/US3 are P1 and US4 is P3.

## R4 — Isolation requires a class AND a full params file. DONE.

`get_model_path` keys off `self.__class__.__name__`, so `NNNC_MLX_LINK(NNNC_MLX)` with an empty
body gets its own `saved_data/`. It MUST also get a **full** copy of `NNNC_MLX.json`:

- The tuned gate lives in the FILE and is NOT inherited; the class `buy_params` it merges from have
  every guard OPEN. Five strategies were found running completely ungated this way on 2026-09-09.
- A **buy-only** file silently reverts the exit stack to BaseStrategy defaults — measured at
  **−2.4pp** the same day.
- `strategy_name` must equal the class name or freqtrade aborts with `Invalid parameter file
  provided`.

## R5 — Verify, do not assume. DONE.

Two checks, both because the corresponding failure has occurred:
- The ungated-gate warning (added 2026-09-09) must be ABSENT from every run. Its presence means the
  params file is not reaching the strategy and every number is void (FR-004).
- The model artefact must exist. `test_strat.sh` now returns meaningful exit codes (9238524) but
  the artefact check is retained because it has caught real failures (FR-005).

## R6 — Training fires on one pair. DONE.

`TrainingEngine.maybe_train` triggers when `pair_count == len(whitelist)`, which is 1 here, so a
single-pair whitelist trains normally. Cross-pair aggregation becomes degenerate but is not an
error. Contrast the 11-pair walk-forward, where SUI's late listing meant `pair_count` never reached
11 and training silently never fired — the same mechanism, opposite direction.

## R7 — Analysis must be pair-restricted. DONE.

Both arms' trades are filtered to LINK before any statistic. A whole-portfolio comparison is
meaningless: the variant has no other pairs. No existing tool does this — `compare.py` compares
whole arms and `trade_forensics.py` aggregates by exit reason — hence one new module,
`regime/pair_restricted.py`, which FEEDS `compare.py`'s bootstrap rather than reimplementing it.

## R8 — The control is not retrained. DONE.

Existing `nnnc_v2` arms are the control. Retraining would change two things at once.

## R9 — LINK's history starts 2022-01-14. DONE.

Later than the 2021-01-01 span the 11-pair walk-forward used. Any fold plan must exclude folds
whose TEST range predates it (FR-013); folds whose TRAINING range extends earlier simply have less
history and are acceptable if reported.

## R10 — Principle X constrains the outcome, not the experiment. DONE.

Building a per-pair model to answer a measurement question is not the per-pair *override* Principle
X prohibits. But a LINK-only win must NOT be read as licence to ship per-pair models; it would mean
cross-pair training dilutes, and the structural response is a change to how ONE model trains. The
conclusion must say so.
