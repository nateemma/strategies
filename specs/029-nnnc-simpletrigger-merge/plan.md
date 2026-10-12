# Implementation Plan: NNNC / SimpleTrigger Merge

**Branch**: `029-nnnc-simpletrigger-merge` (spec directory; commits go to master) | **Date**:
2026-10-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/029-nnnc-simpletrigger-merge/spec.md`

## Summary

`SimpleTriggerStrategy` currently holds the framework layer itself:
- trigger evaluation and family tags;
- 025 coexistence rule B;
- 026 regime gate and momentum pre-emption;
- family-exit override merging;
- liveness sinks;
- mixed in: `FamilyCoexistenceMixin`, `FamilyExitMixin`, `StopInvariantMixin`.

NNNC has none of it. The plan moves that layer into one strategy-agnostic mixin,
`utils/EntryFramework.EntryFrameworkMixin`, built around **entry sources**:
- **Triggers** are one source. **The model** is another. The `ENTRY_MODE` attribute decides how
  they combine.
- `SimpleTriggerStrategy` becomes presets + trigger source over the mixin.
- `NNNCStrategy` becomes the model source over the mixin, with `ENTRY_MODE = "model"`. That is
  today's entry: the model's conditions through BaseStrategy's unchanged guard chain.
- `"trigger_and_model"` is the default-off switch (FR-003).
- Model entries are tagged with their training label's family via a new label-family table (A′).
  Arm D's `gbb_path` maps to reversion.

The work proceeds in four phases:
1. **Baseline**, on unchanged NNNC code:
   - arm D, the twin, and an f00 determinism re-run;
   - the 8 SimpleTrigger reference arms, so every comparison is against a same-day, pre-change run;
   - model artefact hashes (the no-training check);
   - a stop-invariant probe in report-only mode;
   - the fingerprint of every loadable NNNC class.
2. **Refactor.**
3. **Verify:**
   - arm D and the twin match the baseline in 15/15 folds;
   - the twin still equals `st_gbb`;
   - 8 SimpleTrigger reference arms match their pre-change baseline;
   - every NNNC fingerprint is unchanged.
4. **Liveness** of each declared switch on an NNNC test arm.

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`), freqtrade backtesting via `regime/run_arm.py`.

**Primary Dependencies**:
- `SimpleTrigger/SimpleTriggerStrategy.py`, `SimpleTrigger/presets.py`;
- `utils/FamilyExits.py`, `utils/FamilyCoexistence.py` (or wherever `FamilyCoexistenceMixin`
  lives), `utils/StopInvariants.py`, `utils/RegimeState.py`;
- `Framework/BaseStrategy.py`, `Framework/BaseNNStrategy.py`, `Framework/TrainingSignals.py`
  (`LabelMethod`);
- `NNNC/NNNCStrategy.py`, `NNNC/NNNC_TriggerGate.py`;
- `evaluation/` (fingerprint, run_folds, foldplan).

**Storage**: fold plans (`evaluation/folds/`), arm records (`regime/arms/`), and baseline JSONs in
the spec directory.

**Testing**: pytest: `NNNC/tests`, `SimpleTrigger/tests`, `utils/tests`, `evaluation`,
`Framework/tests`, `research/outcomes/tests`.

**Target Platform**: macOS / Apple Silicon. Model folds load MLX weights; no training.

**Project Type**: framework refactor plus an equivalence harness.

**Performance Goals**:
- an arm D or twin fold takes about 10 s, so about 2.5 min per 15-fold arm;
- a SimpleTrigger arm takes about 3.3 min.

Totals:
- baseline: arm D + twin + f00 re-run + probe + 8 SimpleTrigger arms ≈ 35 min;
- after the change: arm D + twin + 8 SimpleTrigger arms ≈ 32 min;
- liveness: about 4 folds.

**Constraints**:
- no behaviour change;
- no training;
- `Framework/` gets no strategy-specific code;
- dicts merge from the parent;
- one backtest chain at a time;
- the live `BasketStrategy` bot is untouched (it is an `IStrategy`, outside this hierarchy).

**Scale/Scope**: 298 NNNC files (about 150 classes plus per-fold JSONs), 1 shared mixin, 1
label-family table.

## Constitution Check

*GATE: checked before Phase 0 and re-checked after Phase 1. Unchanged.*

| principle | status | how |
|---|---|---|
| I–III, V, IX | N/A | Equivalence work; no performance claim |
| IV. Parameters proven live | **PASS, central** | Each declared switch must change a decision on an NNNC test arm and be inert undeclared (US3) |
| VI. Diagnose before tuning | PASS | Nothing is tuned |
| VII. Validate cheaply | PASS | The baseline and fingerprint scan (minutes) gate the refactor; a stop-invariant violation stops the feature before merging |
| VIII. One space at a time | PASS | No parameter changes |
| X. No per-pair code | PASS | |
| XI. No leakage | PASS | Models are borrowed per fold, never retrained (`train_model` refusal reused) |
| Repo boundary | PASS | Only `user_data/strategies` |
| **Framework strategy-agnostic** | **PASS, central** | The shared mixin lives in `utils/`, as `FamilyExits`/`StopInvariants` do. It names families and entry sources, never NNNC or SimpleTrigger. `Framework/` is not edited. |
| **Parameter dicts merge** | PASS | `FAMILY_EXIT_OVERRIDES` keeps its MRO merge, now in the mixin |
| Quality gate 4 (liveness) | PASS | US3 |
| Quality gate 7 (scalers) | N/A | The feature list is unchanged |

## Project Structure

### Documentation (this feature)

```text
specs/029-nnnc-simpletrigger-merge/
├── spec.md, plan.md, research.md, data-model.md, quickstart.md, contracts/interfaces.md
├── baseline.md / baseline.json          # US1 pre-change: per-fold hashes, probe, determinism
├── fingerprints_before.json / _after.json
├── verification.md                      # US1/US2 after-change comparison
├── liveness.md                          # US3
└── report.md                            # US4
```

### Source Code

```text
utils/EntryFramework.py            # NEW: EntryFrameworkMixin (entry sources, tags, rule B,
                                   #   regime gate, preempt, family-exit override merge, sinks)
utils/LabelFamilies.py             # NEW: LabelMethod -> (family, reason) table (A′)
utils/StopInvariants.py            # + report-only mode (env FT_STOP_INVARIANTS=report)
utils/RegimeState.py               # + REGIME_CONTEXT_PAIR moved here from SimpleTrigger/presets
SimpleTrigger/SimpleTriggerStrategy.py   # thinned: presets + trigger source over the mixin
NNNC/NNNCStrategy.py               # + EntryFrameworkMixin, ENTRY_MODE="model", label family
NNNC/NNNC_TriggerGate.py           # drop its own StopInvariantMixin (C3 MRO conflict, R5)
evaluation/fingerprint.py          # label family / ENTRY_MODE hashed only when non-default
scripts/nnnc_fingerprints.py       # NEW: scan every NNNC class -> {class: sha | load error}
tests: utils/tests/test_entry_framework.py, utils/tests/test_label_families.py,
       NNNC/tests/test_nnnc_merge.py, utils/tests/test_stop_invariants.py (report mode)
evaluation/folds/plan_wf_aro_b029*.json, plan_wf_trgate_b029.json, plan_*_a029.json,
       plan_nnnc_si_probe.json, plan_nnnc_live_*.json
```

**Structure Decision**:
- The mixin goes in `utils/` because `Framework/` must stay strategy-agnostic and the existing
  shared mixins already live in `utils/`.
- SimpleTrigger keeps its presets: they mirror NNNC's operating point for a model-free arm, and
  NNNC does not need them.

## Complexity Tracking

| addition | why needed | simpler alternative rejected because |
|---|---|---|
| `ENTRY_MODE` enum (`triggers`, `model`, `trigger_and_model`) | One mixin, three entry semantics; FR-002/003 | Two mixins would duplicate tagging, the gate and the sinks, which is the drift this feature removes |
| Report-only stop-invariant mode | FR-004 needs a violation count on unchanged arm D without halting | Running arm D with halting on could stop the baseline at the first defect and report nothing else |
