# Implementation Plan: SimpleTrigger Strategy Family

**Branch**: `023-simple-trigger-family` (work commits to master; branching waived) | **Date**: 2026-10-10 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/023-simple-trigger-family/spec.md`

## Summary

Add a no-model strategy family, `SimpleTrigger/`, whose entry is a registered, measured-causal
trigger fed through `BaseStrategy`'s unchanged guard chain. Its exits and stop come from a named,
complete **framework preset**. The default preset is arm D's resolved framework. A second preset,
`LEGACY_TRIGGERONLY`, exists only to reproduce the recorded 021/022 arms.

Around the family, the plan adds four pieces of instrumentation:
- a **framework fingerprint**, so two arms can be proven to share a framework;
- a **no-model fold-plan path** in the walk-forward harness;
- **liveness records**: entry-condition pass counts and exit-reason counts;
- three analysis tools: twin check/compare, component attribution, and the early/late halves
  split.

Coupled exits are a pure parameter mapping into the existing single exit definition. A
strategy-agnostic stop-invariant mixin (`utils/StopInvariants.py`), applied to SimpleTrigger and
the twin's equivalence arm, halts any run that produces a wrong-sign, over-loose or
wrong-direction stop. A matched-count selection test answers the twin question even when the
paired test lacks shared trades.

Every delta this feature measures is **provisional, not a finding**, under quality gate 2:
there's no distant era. Reports and memory carry that label.

Research changed the brief in two places, and the user should know about both before tasks are
generated:

- **R1.** The 021/022 no-model arms ran with a **static −5% stop**, not arm D's ATR-adaptive stop.
  No existing number isolates the model.
- **R2.** Arm D's quoted +39.98pp is the **pre-fix** number. The current-code reference is
  **1420 trades, +42.71pp, 14/15**.

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`; never conda, never `setup.sh -i`)

**Primary Dependencies**: freqtrade (upstream, not modified), pandas, numpy. The existing modules
reused are:
- `research.outcomes.triggers`
- `utils/ExitPolicy.py`
- `evaluation/{foldplan,run_folds,pool,bootstrap,compare_pooled,power}.py`
- `regime/run_arm.py`

**Storage**: JSON files, namely arm records (`regime/arms/`), fold plans (`evaluation/folds/`),
`research/outcomes/trigger_causality.json`, and attribution and twin reports (`evaluation/`).

**Testing**: pytest, in `SimpleTrigger/tests/`, `evaluation/tests/` (new) and
`research/outcomes/tests/`. Backtest-level acceptance uses `trades_hash` identity (quickstart).

**Target Platform**: macOS / Apple Silicon. Only `NNNC_TriggerGate` touches MLX, and only for
inference with existing arm D fold models.

**Project Type**: Research codebase: a strategy family plus CLI analysis tools.

**Performance Goals**: A no-model 15-fold run needs no training, so it is bounded by backtest
time. One chain at a time (project rule).

**Constraints**:
- Framework/ stays strategy-agnostic.
- No behaviour change to any existing arm. This is proven by `trades_hash` identity on a recorded
  NNNC fold (`wf_aro` f00) and a recorded trigger fold (`wf_gbbo` f00) after the shared-file edits.
- The live dry-run is unaffected.

**Scale/Scope**: About 19 triggers, 29 label methods, 15 folds, about 14 components, and 2 presets.

## Constitution Check

*GATE: checked before Phase 0, and re-checked after Phase 1 (below).*

| Principle | Status | How |
|---|---|---|
| I. Deployment regime | PASS | Results are reported per fold, plus the halves split. No full-history headline. |
| II. Distant era (NON-NEG.) | **JUSTIFIED EXCEPTION** | See Complexity Tracking. The early/late split (f00–f07 / f08–f14) is the persistence test, as in 022. Per gate 2, no US2/US3 delta is recorded as a finding; the report (T056) and memory (T057) label them PROVISIONAL. |
| III. Learnability ≠ edge | PASS | Every verdict is on walk-forward P&L. No model metrics are used. |
| IV. Liveness | PASS, strengthened | Every component arm carries a moved metric, and INERT is a defect. The causality checker has a planted FAIL control. The condition sink is proven neutral. |
| V. Lookahead-clean ≠ realistic | PASS | Trigger causality is measured end-to-end by truncation (R11). Volume and liquidity constraints stay on, and the `volume>0` and `rvol` gates are mandatory. |
| VI. Diagnose before tuning | PASS | This feature *is* the diagnostic layer. It does no tuning. |
| VII. Trackability before adaptivity | PASS | Regime detection is out of scope. Any later regime work must pass the trackability gate first. |
| VIII. One space at a time | PASS | No search. Component arms change one component each. |
| IX. Orthogonality | N/A | No factor book. |
| X. No per-pair code | PASS | The registry already enforces pair-invariance. The family adds none. |
| XI. No leakage | PASS | No new model fits. `NNNC_TriggerGate` reuses arm D's per-fold models and scalers as they are and discards their opinion. |
| Repo boundary / custom code location | PASS | All edits are under `user_data/strategies/`. |
| Framework strategy-agnostic | PASS | The only Framework/ edit is the generic condition sink (C4). |
| Params merge from parent | PASS | Presets merge as `{**parent, **preset, **overrides}`, with a test asserting no key is dropped. |
| Staging / commits | PASS | Named-file staging, committed to master. |
| Quality gates 1–7 | 1 ✓ pinned fold windows (derived plans). 2 see II: results are PROVISIONAL, not findings. 3 ✓. 4 ✓ liveness records. 5: no-model arms are deterministic; `NNNC_TriggerGate` uses frozen models. Any sub-1pp delta is still reported as unresolved (noise floor). 6 ✓ zero-trade folds are reported as such. 7 N/A, since the feature lists are unchanged. |

**Post-design re-check (after Phase 1):** there are no new violations.
- The design adds three shared-file edits: `BaseStrategy` (sink), `run_arm.py` (record fields) and
  `foldplan.py`/`run_folds.py`/`pool.py` (no-model path, exact tag match).
- Each one is gated by a byte-identity regression on recorded arms (quickstart §4, plus tasks).

## Project Structure

### Documentation (this feature)

```text
specs/023-simple-trigger-family/
├── spec.md
├── plan.md              # this file
├── research.md          # Phase 0: R1–R13
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1: validation scenarios 0–8
├── contracts/
│   └── interfaces.md    # C1–C5
├── checklists/requirements.md
└── tasks.md             # Phase 2 (/speckit-tasks)
```

### Source Code (`user_data/strategies/`)

```text
SimpleTrigger/                      # NEW family (sibling of NNNC; no SimpleStrategy inheritance)
├── SimpleTriggerStrategy.py        # base class: C1 guarantees G1–G6, stop-invariant wrapper
├── presets.py                      # ARM_D_OPEN (default), LEGACY_TRIGGERONLY; total, validated
├── components.py                   # Component table: neutral overrides + liveness metric
├── coupling.py                     # coupled-mode mapping: label targets → exit params (R8)
├── label_map.py                    # 29 LabelMethod → trigger | empty | unsupported (R10)
├── SimpleTrigger_<Trigger>.py      # one thin config per trigger used (ARM_D_OPEN)
├── SimpleTrigger_Legacy_<Arm>.py   # 7 reproduction configs (LEGACY_TRIGGERONLY)
└── tests/

NNNC/NNNC_TriggerGate.py            # NEW: equivalence arm. Trigger entry on arm D's real NNNC path
NNNC/NNNC_TriggerGate.json          # NEW: copy of NNNC_A_Raw_Open.json (strategy_name changed)

utils/StopInvariants.py             # NEW: StopInvariantMixin + StopInvariantError (FR-017)
utils/tests/test_stop_invariants.py

research/outcomes/
├── causality.py                    # NEW: truncation causality checker (R11)
├── trigger_causality.json          # NEW: measured verdicts
└── tests/test_causality.py

evaluation/
├── fingerprint.py                  # NEW: resolved-framework hash (R4)
├── twins.py                        # NEW: check / compare (US3)
├── attribution.py                  # NEW: per-component report (US2)
├── halves.py                       # NEW: early/late split (R13)
├── matched_count.py                # NEW: same-count random-draw selection test (R13)
├── dump_strategy.py                # NEW: class-attribute + JSON dump for the SC-008 dry-run diff
├── foldplan.py                     # EDIT: `derive` subcommand; no_model validate rule
├── run_folds.py                    # EDIT: no_model path; exact tag match
├── pool.py                         # EDIT: exact tag match
└── tests/                          # NEW

regime/run_arm.py                   # EDIT: C3 record fields
Framework/BaseStrategy.py           # EDIT: C4 generic entry-condition sink (off by default)
```

**Structure decision.** A new top-level family directory. `run_arm.strategy_group` already
resolves it by its leading family token, so no registry edit is needed. `SimpleStrategies/` and
`TriggerOnly` are left untouched (FR-022).

## Phasing (for /speckit-tasks)

Each phase gates the next one.

1. **Instrument before using it.**
   - Build the fingerprint, record fields, exact tag match, condition sink, no-model plan path
     (with a causality preflight), halves and the stop-invariant mixin.
   - Gate: byte-identity on `wf_aro` f00 and `wf_gbbo` f00.
2. **Causality.**
   - Build the checker and the planted control, then produce `trigger_causality.json`.
   - Resolve the `future_*` record conflict (R10) by reading the labelers.
3. **Family + legacy reproduction (US1, MVP).**
   - Build the base class, presets and the 7 legacy configs.
   - Gate: SC-001 (15/15 hash identity, all 7 arms).
4. **Twin (US3).**
   - Build `SimpleTrigger_Gbb` on `ARM_D_OPEN` and `NNNC_TriggerGate`.
   - Gate: fingerprint equality and SC-003. On failure, localise and fix the parity defect before
     going on.
   - Then `twins compare` against post-fix arm D, including the matched-count test.
5. **Attribution (US2).** Build the components table, the arms and the report. Expected: the
   R1-measured INERT stop components under the legacy preset.
6. **Coupled mode (US4).** Build the mapping. Gate: SC-007 identity.
7. **Report + memory.**
   - `report.md`, including the FR-007a merge follow-up.
   - Correct the arm D figure in project memory (R2).
   - File notes: `pick_guards.floor_for_window` is unwired, and `guard_trackability.py` is
     missing (R9, R13).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Principle II: no genuinely distant era | Binanceus history (2021→) is all selection or evaluation, and `data/binance/` is empty (recorded in 022). Every fold is already out-of-sample for arm D's anchored training. | A distant era doesn't exist to use. The early/late halves split is the strongest persistence test available, and the report must say so whenever it is used. |
| Shared-file edits in a "sibling" feature (BaseStrategy sink, run_arm, harness) | Liveness (IV) and twin proof (FR-013) need measurements that nothing records today. | Offline re-evaluation would duplicate guard and exit logic, the drift class 021 removed. Each edit is generic, off by default where it touches behaviour, and gated by byte-identity on recorded arms. |
| `LEGACY_TRIGGERONLY` preset alongside the default | SC-001 reproduction requires TriggerOnly's actual (static-stop) framework. | Reproducing under `ARM_D_OPEN` would fail by design (R1). Dropping SC-001 would leave the new family unvalidated against any recorded result. |
