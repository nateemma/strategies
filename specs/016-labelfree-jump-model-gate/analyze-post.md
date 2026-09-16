# Post-implementation Analysis: 016-labelfree-jump-model-gate

**Date**: 2026-09-15 · **Run**: `/speckit-analyze` after `/speckit-implement` (SC-015, T054)
**Pre-implementation report**: [`analyze.md`](./analyze.md) — 14 findings, all remediated

## Outcome

**KILL at T032**, the gate-activity checkpoint, before any backtest. Verdict artefact
`research/labelfree/jump/out/verdict.json`; §6 report
`research/label-free-study/briefs/results/08-2026-09-15.md`.

| gate | requirement | measured | |
|---|---|---|---|
| SC-020 | active in ≥ 10 of 15 folds | 13/15 | PASS |
| SC-019 | `novel_block_rate` ≥ 0.20 | 1/15 folds, mean 0.119 | **FAIL** |
| FR-023 | seed agreement ≥ 0.95 | mean 0.916, 4/15 folds | **FAIL** |
| SC-017 | seed spread < effect | spread 0.115 vs block rate 0.104 | **FAIL** |

## Findings from this pass

| ID | Category | Severity | Summary | Status |
|----|----------|----------|---------|--------|
| **P1** | Process | **HIGH** | **`research/labelfree/jump_gate.py` — the file the brief names by path — did not exist, while T001 was marked `[X]` for creating it.** Third occurrence of this exact pattern in the study (013's T045, 015's T009). | FIXED — written, with the kill-blocked subcommands refusing explicitly rather than returning silence |
| **P2** | Coverage | **HIGH** | The deletion test ran against the in-memory function, never the **written artefact**, which research R7 required by name. `refit.write()` was called from nowhere. | FIXED — `test_written_artefact_is_causal` + `test_writer_refuses_nothing_and_roundtrips` |
| **P3** | Coverage | MEDIUM | No trial reached the log the DSR is computed over — the exact gap G2 raised pre-implementation, still open after implementation. | FIXED — 75 trials registered and **abandoned** (never "completed": no P&L was measured, and recording one for a run that did not happen is the audit failure `pre_register` refuses at write time). Audit **PASSES**, 580 trials counting toward deflation |
| **P4** | Reporting | MEDIUM | `guard_bound` was one-sided and reported "did not bind" while calibration had missed its target by 5× on the low side. | FIXED mid-implementation |
| **P6** | Correctness | **HIGH** | **The redundancy comparator used the wrong `rvol`.** The incumbent has two quantities with that name — the `rvol` column (`ewm(span=10)`, what the entry guard tests) and the stop's inline `rolling(20).mean()`. `redundancy.py` used the latter while claiming the former. They disagree on 5.5% of bars. | FIXED — `which=entry_guard\|stop`, defaulting to the entry guard. Re-measured: novel rate FALLS to 0.10-0.13, so the error was conservative and the KILL is unchanged |
| **P5** | Clarity | LOW | Phases 5 and 6 sat as bare unchecked boxes, indistinguishable from forgotten work. | FIXED — both phases carry a NOT RUN banner naming T032 as the cause |

**P1 is the one that matters beyond this feature.** Three features in a row have now marked a
task `[X]` while its artefact did not exist, and in each case the task was one nothing
downstream immediately consumed — 013's block-returns writer, 015's decisive test, and here an
entry point that only the quickstart would have exercised. The common shape is a task whose
absence is invisible until something else reaches for it. A `/speckit-implement` completion
check that greps tasks.md for file paths and asserts they exist would have caught all three in
seconds.

## Verification

| check | result |
|---|---|
| Tasks | **43/62 complete**, 17 deliberately NOT RUN under T032, 2 open (T054 this report, T055 staging) |
| Tests | **227 pass** (201 pre-existing + 26 new), no regressions |
| Requirements | FR-001…FR-024 and SC-001…SC-020 all referenced; those gated by T032 recorded as unevaluated with cause |
| Reproduction (SC-010) | **REPRODUCED** — `research/labelfree/jump/reproduce.py` re-derives KILL from `activity.json` in a fresh process, reading the measurements and re-applying the rules rather than reading `verdict.json` |
| Trial-log audit (§6) | **PASS** |
| Constitution | I–X hold; VII and IX carry their written justifications, and IX is the principle the result turned on |

## What the constitution caught

- **Principle IV** is why this feature has an answer rather than a number. `gate.require_active`
  makes an inert gate raise instead of reporting performance, and two folds blocked nothing at
  all.
- **Principle IX** predicted the outcome in the plan, before any code: a model fitted on
  volatility would likely be the incumbent's rvol guard in disguise. 14 of 15 folds.
- **Protocol §4's seed requirement** — added only because the pre-implementation pass caught
  its absence (C1) — supplied two of the three failing gates. Without it this would have run
  30 backtests and reported a drawdown number from a gate whose blocked set is decided by the
  random seed.

## Residual limitations, recorded not fixed

1. **λ was calibrated on fold 0 and frozen.** With 8 of 12 grid points degenerate, the
   admissible band is narrow and a different fold-0 window could select a different λ. The
   verdict rests on a gate that is *redundant*, which is a property of the feature set rather
   than of λ, so this is unlikely to be load-bearing — but it is untested.
2. **The fit is subsampled to hourly bars** (`FIT_STRIDE = 4`); inference is at full 15-minute
   resolution. Defensible for an object averaging ~2 months, never validated against an
   unsubsampled fit.
3. **The sweep ran single-threaded** — ~45 independent fits per fold executed in sequence. The
   DP recursion is inherently sequential, but the pairs and seeds are not; this cost roughly an
   8× speedup and no accuracy.
4. **`K = 3` was never run.** Pre-registered as a sensitivity that cannot produce the verdict,
   and the kill arrived before it was reached.

**Critical issues: 0. The feature is complete as a negative control.**
