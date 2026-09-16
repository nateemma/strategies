# Implementation Plan: signal-context-timing

**Branch**: `017-signal-context-timing` | **Date**: 2026-09-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/017-signal-context-timing/spec.md`

## Summary

Test whether a repeat entry signal is **information about an open position** (US1, US2), and
whether **first**-signal context predicts signal quality (US3) — then evaluate any surviving
rule through the pooled walk-forward (US5).

Four measurements taken during planning decide the whole shape of this feature:

1. **The signal set can be reconstructed offline, in under a second per fold.**
   `research/labelfree/es/panel.py::entries()` already reimplements
   `BaseStrategy.populate_entry_trend` (lines 886-957), and all 30 probability side-cars
   (15 train + 15 test) already exist. Loading f08's test panel takes **0.8s**.
2. **The reconstruction is faithful: 70 of 71** recorded f08 trades have their entry bar in the
   reconstructed signal set. The one miss is SOL/USDT at `2024-12-21 16:15`, the **first bar of
   the test range** — a warm-up boundary artefact, not a logic error.
3. **Masked signals are directly countable, not inferred.** f08's test range holds **139 signal
   bars / 120 distinct signal events** against **71 admitted trades**: **49 masked events,
   41%**. This replaces the earlier `--eps`-derived estimate of "~43%".
4. **`--eps` is therefore needed only for masked-signal OUTCOMES**, never for identifying them.
   That shrinks the blast radius of the one dangerous instrument in this feature.

The practical consequence: **US1, US3 and most of US4 need no backtest at all.** The expensive
half is US2 and US5, and only if the cheap half survives.

## Technical Context

**Language/Version**: Python 3.12.13 at `/Users/philprice95/projects/freqtrade/.venv/bin/python`.
Guarded by `sys.prefix`, never `Path(sys.executable).resolve()` — the venv's `bin/python` is a
symlink to the pyenv binary, so resolving it lands outside the venv and rejects the correct
interpreter (recorded 2026-09-15).

**Primary Dependencies**: numpy, pandas, scipy, statsmodels, scikit-learn — all present. **No
MLX and no torch**: this feature trains nothing. Every model output it consumes already exists
as a probability side-car.

**Storage**: probability side-cars at `/tmp/lf/es_sink` (30 present), panel cache at
`/tmp/lf/es_panels`. **Both are in `/tmp` and are the feature's single largest fragility** — see
Complexity Tracking.

**Testing**: pytest, `research/labelfree/tests/test_signalctx_*.py`.

**Project Type**: research module inside `user_data/strategies`.

**Performance Goals**: not a throughput feature. The binding budget is **backtest count**:
0 for US1/US3, 15 per arm for US2/US5, plus one `--eps` run per fold for US4. No retraining —
position of every model output is fixed.

**Constraints**: all runs sequential; the pooled harness and its pinned constants
(`BOOTSTRAP_SEED = 0`, floor 3.350 pp/yr, noise floor 7.633 pp) are used as-is.

**Scale/Scope**: 15 folds, ~10 pairs, 15-minute bars, 378 pooled incumbent trades, ~1,800
reconstructed signal events across the study.

## Constitution Check

*GATE: evaluated before Phase 0, re-evaluated after Phase 1.*

| Principle | Status | How this plan satisfies it |
|---|---|---|
| **I. Deployment regime** | PASS | Verdict from the pooled walk-forward across W1/W2/W3. |
| **II. Persistence (NON-NEGOTIABLE)** | PASS | 15 anchored folds, three distant windows; SC-006 requires the sign in each. No claim rests on one fold — including the motivating f08 measurement, which is explicitly a planning input and not evidence. |
| **III. Learnability ≠ edge** | **PASS, and it is the central discipline here** | The analogue of MCC in this feature is the **p-value**. FR-007 requires effect size beside every p, and SC-015 makes anything inside 7.633 pp "not resolved" regardless of significance. A p=0.005 finding that moves 0.3 pp is not a trade. |
| **IV. Parameters actually live** | PASS | FR-014, SC-014, SC-020: a rule that changes no trades is inert and reported as a defect. |
| **V. Lookahead ≠ realistic** | PASS | FR-004's truncation test on every context feature; volume/liquidity unchanged from the incumbent. |
| **VI. Diagnose before tuning** | **PASS — the feature IS a diagnosis** | US1 exists solely to localise whether the observed effect is real or a proxy, before anything is built. |
| **VII. Validate cheaply** | PASS | US1 and US3 cost zero backtests. The expensive half runs only if the cheap half survives. |
| **VIII. One parameter space** | PASS | The exit rule has one decision (what to do on a repeat signal). The first-signal sweep has four features, handled by the pre-declared correction of FR-006 rather than by searching. |
| **IX. Orthogonality** | **PASS — twice** | Entry side FR-011/012 against the guard chain; exit side FR-022/SC-019 against stoploss-grace and the ATR stop. |
| **X. Structural, not per-pair** | PASS | One rule across the universe; no per-pair thresholds. |

### Principle III — why it is the binding constraint, not a formality

This feature's natural failure mode is not a wrong answer, it is a **true but useless** one. A
stratified test on ~440 trades can resolve a win-rate shift of a few points at p < 0.01 while
the P&L consequence sits far inside the 7.633 pp noise floor. This repository has recorded that
shape five times — most sharply a P&L-weighted loss that raised validation MCC monotonically
across 6 of 6 runs while P&L stayed indistinguishable from noise.

So the plan orders the questions as **is it real → is it big → is it novel → does it survive the
harness**, and no stage may be skipped because an earlier one was significant.

### Principle IX — two different redundancy risks, not one

- **Entry side**: `project_guard_ablation_model_adds_nothing` found the guard chain, not the
  model, selects entries. A first-signal context filter may be another guard in disguise.
- **Exit side**: `stoploss_grace` and the ATR-adaptive stop already react to adverse moves. A
  repeat-signal trigger that fires where they would have fired anyway is inert — and this is the
  more likely of the two, because the motivating effect (36.5% vs 21.7% trailing-stop exits) is
  itself measured *through* that stack.

## Project Structure

### Documentation (this feature)

```text
specs/017-signal-context-timing/
├── plan.md              # This file
├── research.md          # Phase 0 — R1..R9
├── data-model.md        # Phase 1 — entities and invariants
├── quickstart.md        # Phase 1 — runnable validation sequence
├── contracts/
│   └── signalctx.md     # Phase 1 — module interface contract
├── checklists/
│   └── requirements.md  # from /speckit-specify, 16/16
└── tasks.md             # /speckit-tasks output — NOT created here
```

### Source Code (repository root = `user_data/strategies`)

```text
research/signalctx/
├── __init__.py          # pinned thresholds + interpreter guard
├── signals.py           # offline signal reconstruction + fidelity check vs recorded trades
├── context.py           # causal context features at signal time
├── paths.py             # per-bar unrealised P&L for an open trade
├── confound.py          # US1 — stratified + regression test against unrealised P&L
├── sweep.py             # US3 — first-signal context, pre-declared correction
├── masked.py            # US4 — --eps as a measurement instrument, fenced
├── arms.py              # US2/US5 — fold arms reusing weights (pattern from research/posadj)
└── report.py            # §6 report + trial-log registration

NNNC/
└── _signal_context.py   # the exit-side mixin (pattern: NNNC/_hold_window.py)

research/labelfree/tests/
├── test_signalctx_signals.py     # reconstruction fidelity, signal-event definition
├── test_signalctx_context.py     # causality by truncation
├── test_signalctx_confound.py    # the matched comparison, incl. a planted-null control
└── test_signalctx_paths.py       # unrealised P&L reconstruction
```

**Structure Decision**: a new top-level `research/signalctx/` package, because this is not part
of the label-free study (which closed at eight briefs) and should not inherit its directory.
Existing instruments are **reused, not reimplemented**: `research/labelfree/es/panel.py`
(signal reconstruction and panels), `research/labelfree/folds.py` (fold geometry),
`evaluation/` (pooled walk-forward, pinned constants), `research/labelfree/stats.py`
(bootstrap, PBO, DSR), `research/labelfree/trials.py` (trial log), and
`research/posadj/arms.py`'s weight-reuse pattern. The exit-side mixin lives in `NNNC/` following
`_hold_window.py` and `_position_adjust.py`, because a new mechanism is validated on the family
before any promotion to `Framework/`.

## Constitution Check — re-evaluated after Phase 1 design

No principle moved to FAIL. Three sharpened once the design was concrete:

- **Principle III** became the ordering rule for the whole feature rather than a caveat.
  `confound.verdict()` cannot return `survives` on an effect inside 7.633 pp however small its
  p-value — the constraint is in the contract, not in a reviewer's memory. This feature's
  natural failure is a **true but useless** result, and that is now structurally hard to report.
- **Principle IX** produced two separate checks, because the entry and exit sides carry
  different redundancy risks. The exit-side one is the more dangerous: the motivating effect
  (36.5% vs 21.7% trailing-stop exits) was itself measured **through** the stop stack it would
  have to be distinct from.
- **Principle IV** gained the null-case assertion from R9. Three liveness probes reported zero
  on working arms in one day, so `liveness()` now returns `n_triggered` **and** whether the arm
  is byte-identical to its control; zero triggers on a non-identical arm means the probe is
  broken, not that the rule is inert.

One design element is worth naming because it inverts the usual reading: **`prev_signal_won`
must be null while the previous trade is still open.** The obvious implementation joins a signal
to the previous signal on that pair and takes its outcome — which, at the moment our signal
fires, is not yet knowable. That is a leak that looks like a join.

**Workflow gates** (constitution §"Experiment Workflow and Quality Gates"): pinned ranges via
the fold plan (1); 15 folds across three distant windows (2); volume/liquidity unchanged from
the incumbent (3); liveness via `ExitRuleArm.n_triggered` plus the null-case check (4); paired
seed-robustness **not engaged and the reason is structural** — this feature trains nothing and
every arm reuses fixed fold weights, so the backtests are deterministic and there is no
run-to-run variance to isolate (5); trade counts per fold with zero-trade folds inconclusive
(6); scalers untouched, because no feature include list changes (7).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| `--enable-position-stacking` as a measurement instrument | Masked signals have **no outcome** in the control — they never became trades. Nothing else can give them one. | There is no alternative: the information does not exist in the baseline. Fenced by FR-008/009/010 so no figure derived from it can be reported as an edge. |
| A second signal path (offline reconstruction) alongside the production path | US1 and US3 need **every** signal, including the ~41% freqtrade discards and never records. | `es/signals.py` deliberately rejected offline *inference* to avoid a second model path — but this reuses its **production-generated** probabilities and only recomputes the **gate**, which `panel.entries()` already does and which is validated here at 70/71 against recorded trades. |
| Side-car dependency on `/tmp` | The 30 existing side-cars make US1/US3 free. | Regenerating them costs ~44 minutes of backtests. They must be **copied to durable storage as the first task**, because macOS may clear `/tmp` at any reboot and the feature would silently become expensive. |
