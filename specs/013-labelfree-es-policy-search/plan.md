# Implementation Plan: labelfree-es-policy-search

**Branch**: `013-labelfree-es-policy-search` | **Date**: 2026-09-13 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/013-labelfree-es-policy-search/spec.md`

## Summary

Brief 05 asks whether the exit and guard levers gradient descent cannot reach — ATR stop
multiplier, `stoploss_grace_*`, guard thresholds, minimum holding time — *jointly* matter when
searched together rather than one at a time.

The exit-reason census (research.md R2) says the space is aimed at the right mechanism and
sharpens it: two mechanisms decide the ledger and they **oppose** each other — `cexit_take_profit`
at +179.7% of P&L and the ATR + grace stop stack at −120.2%. `cexit_take_profit` is therefore
**added** to the search space; the brief's own list omits it.

The feature is built in four layers:

1. an **exit surrogate** that replays a fold's exposure path under a candidate's exit stack, and
   a **calibration gate** against real freqtrade backtests that blocks everything downstream;
2. a **liveness probe over the search space itself**, which this study has paid for skipping
   before — four parameters configured, logged, believed active and reaching nothing;
3. a **CMA-ES search** and a **uniform random control at an identical evaluation budget**, with
   every evaluation written to a per-feature sidecar before it is scored;
4. **PBO/CSCV over the complete history** as the headline number, a single test-range evaluation
   per fold, and a result report written whether the verdict is accept or kill.

The expected deliverable is a quantified negative. Clarification Q3 makes PBO > 0.2 kill the
candidate while the measured PBO is the finding, and clarification Q1 makes surrogate
calibration a blocking gate rather than a validation step.

## Technical Context

**Language/Version**: Python 3.12.13

**Primary Dependencies**: numpy, pandas, scipy, freqtrade 2026.9-dev (backtest arbiter), **`cma`
4.5.0** — installed into `freqtrade/.venv` on 2026-09-13 and smoke-tested (converges to the
sphere optimum at ~6e-08). The reference implementation is used rather than a hand-rolled one, so
a negative result cannot be blamed on the optimiser.

**Interpreter**: `freqtrade/.venv/bin/python`. Note `~/.pyenv/shims/python3` is a *different*
3.12.13 with `cma` but **without pandas or scipy** — every command in this feature must use the
venv interpreter explicitly.

**Storage**:
- `research/labelfree/es/history.jsonl` — the full evaluation history (sidecar, SC-018)
- `research/labelfree/trials.jsonl` — one summary row per search run (shared log)
- `research/labelfree/es/panel/` — the keyed feature/price cache, reused from feature 012

**Testing**: pytest, sequential. Deterministic seeding on every stochastic path.

**Target Platform**: macOS / Apple Silicon, single process.

**Project Type**: research library under `research/labelfree/`.

**Performance Goals**: one surrogate evaluation ≤ 50 ms on a fold's training range, so a
1,600-evaluation budget per fold per method completes in ≲ 3 minutes. Measured contrast: one
freqtrade backtest is ~175 s, which is why the surrogate exists.

**Constraints**:
- Every run sequential (FR-021).
- Stage only files this feature touches (FR-022).
- The outer freqtrade checkout receives no commits.
- The test range of a fold is opened exactly once, after that fold's search concludes (FR-010).

**Scale/Scope**: 15 anchored folds; a budget of 1,600 evaluations per method per fold is the
planned default (48,000 evaluations total across both methods and all folds).

## Constitution Check

| Principle | Status | How this plan satisfies it |
|---|---|---|
| **I. Evaluate in the deployment regime** | PASS | Pooled walk-forward over the 15 anchored folds is the primary verdict; single-window W1/W2/W3 is context (SC-002). |
| **II. Persistence across distant eras** (NON-NEGOTIABLE) | PASS | SC-002 requires positive in each of W1/W2/W3 and in ≥10/15 folds. A fold-concentrated win fails. |
| **III. Learnability is not edge** | PASS | The objective is cost-aware net return, never a fit statistic. No ML metric appears in any gate. |
| **IV. Verify parameters are actually live** | **GATE ADDED** | The exit-reason census (research.md R2) confirms the ATR + grace stack governs 26.2% of exits and the **entire loss side**, so the space is aimed correctly — but four parameters in this study have already been configured, logged, believed active and reaching nothing. A liveness probe over every search dimension runs *before* the search, and a dimension that cannot move the objective is reported inert rather than searched. |
| **V. Lookahead-clean is not realistic** | PASS | The surrogate inherits brief 1's cost model; SC-005 re-tests the delta at 1.5× fees; purge/embargo via `apply_leakage_control`. |
| **VI. Diagnose before tuning** | PASS | The census and the liveness probe both precede any search. |
| **VII. Validate cheaply; trackability before adaptivity** | PASS | The calibration gate is the cheap validation, and it blocks. The liveness probe is cheaper still and runs first. |
| **VIII. One parameter space at a time** | **DEVIATION — justified** | See Complexity Tracking. |
| **IX. Orthogonality earns a factor its place** | PASS | The liveness probe reports each dimension's independent effect; a dimension that moves nothing is not carried. |
| **X. Structural fixes, not per-pair overrides** | PASS | One parameter set across all pairs. No per-pair branch exists anywhere in the design. |

**Post-design re-evaluation**: unchanged. The single deviation is Principle VIII and it is the
feature's stated hypothesis, recorded below with its compensating controls.

## Project Structure

### Documentation (this feature)

```
specs/013-labelfree-es-policy-search/
├── spec.md
├── plan.md               # this file
├── research.md           # Phase 0 — 8 decisions, incl. the exit-reason census
├── data-model.md         # Phase 1
├── contracts/
│   └── es-cli.md         # Phase 1 — command surface
├── quickstart.md         # Phase 1
├── checklists/requirements.md
└── tasks.md              # /speckit-tasks
```

### Source Code (repository root)

```
research/labelfree/
├── es/
│   ├── __init__.py
│   ├── space.py          # the parameter space: bounds, encode/decode, incumbent
│   ├── exits.py          # THE EXIT SURROGATE — the feature's core risk
│   ├── calibrate.py      # the blocking gate: surrogate vs freqtrade backtests
│   ├── liveness.py       # per-dimension liveness probe over the search space
│   ├── cmaes.py          # thin wrapper over `cma` (ask/tell, bounds, seeding)
│   ├── random_search.py  # the identical-budget control
│   ├── objective.py      # cost-aware score + incumbent/turnover regularisation
│   ├── history.py        # the sidecar log; write-before-score
│   ├── run.py            # per-fold driver: search, then ONE test evaluation
│   └── report.py         # PBO/CSCV, deflation, the result report
├── admissibility.py      # REUSED unchanged — the shared gate
├── trials.py             # REUSED — one summary row per run
└── sim.py, stats.py      # REUSED — cost model, PBO, bootstrap

tests/
├── test_es_exits.py
├── test_es_space.py
├── test_es_cmaes.py
└── test_es_history.py
```

**Reuse is deliberate**: `admissibility.compare` is the gate that FR-004 and FR-005 describe and
it already exists; `stats.pbo_cscv` already implements CSCV; `trials.block_returns` already
supplies the time axis SC-004 needs. This feature writes the surrogate, the search and the
per-dimension probe, and nothing else.

## Phases

| Phase | Content | Gate to pass before continuing |
|---|---|---|
| **0** | Research — the exit-reason census, the surrogate's scope, the calibration design | `research.md` written |
| **1** | Design — data model, CLI contract, quickstart | this document |
| **2** | The surrogate + its unit tests | tests green |
| **3** | **Calibration gate** — 40 candidates scored both ways | **BLOCKING**: rank agreement measured and the win-disagreement check clean (SC-017a) |
| **4** | **Liveness probe** over all 8 dimensions | inert dimensions identified and reported |
| **5** | CMA-ES + random control, both at an identical budget | evaluation counts recorded and equal (SC-013 → FR-007) |
| **6** | One test evaluation per fold | audit shows exactly one (SC-011) |
| **7** | PBO/CSCV, deflation, result report | report written regardless of verdict (FR-020) |

Phase 3 blocks phases 4–7. If the surrogate cannot be calibrated, the feature stops there and
reports that, rather than spending 48,000 evaluations on an unvalidated evaluator.

## Complexity Tracking

| Deviation | Why it is needed | Why the simpler alternative is rejected |
|---|---|---|
| **Principle VIII — one parameter space at a time** | The feature's hypothesis *is* that these levers matter jointly and have only ever been swept one at a time. Honouring VIII here would refuse to test the brief. | The simpler alternative — sweep each lever alone — is precisely what the study has already done, and it produced the 4.93 pp span that this brief cites as its own kill criterion. Compensating controls: the random control at an identical budget separates "the optimiser found structure" from "we looked 1,600 times"; PBO over the full history quantifies the multiple-testing cost that VIII exists to avoid; the 4.93 pp band makes a move smaller than one sweep's span a non-result; and the regularisation in FR-009 makes distance from the incumbent something the search must earn. |
