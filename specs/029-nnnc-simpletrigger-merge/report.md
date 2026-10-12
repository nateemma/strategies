# Report: NNNC / SimpleTrigger Merge (029)

*An equivalence feature: every result is "identical" or "changed", not a performance claim.*

## SUMMARY

NNNC and SimpleTrigger now derive from one entry framework, `utils/EntryFramework.EntryFrameworkMixin`.
For NNNC the model is just another entry source. Nothing any existing strategy does has changed.

| criterion | result |
|---|---|
| **SC-001** arm D, twin and 8 SimpleTrigger arms reproduce their pre-change baseline | **15/15 each** (also 15/15 vs their records); twin = `st_gbb` 15/15 |
| **SC-002** NNNC fingerprints unchanged | **297/297** strategies; the 2 pre-existing load errors are unchanged |
| **SC-003** stop invariants on every NNNC arm; arm D baseline violations | every NNNC strategy carries `StopInvariantMixin`; **0 violations** in arm D (report-mode probe); arm D runs with the halting check |
| **SC-004** each declared switch live when declared, inert otherwise | C2 live (exits), regime gate live (175 rows refused), trigger-AND-model live (2 rows refused) but changes no arm D trade; all inert undeclared |
| **SC-005** no training | model artefacts byte-identical; 0 × `training on` |
| **SC-006** test suites | 885 passed; 3 failed are the pre-existing training-pipeline goldens |

Commits:

| commit | step |
|---|---|
| `6c978e8` | baseline |
| `5ef1b7d` | refactor |
| `6da3fbe` | verification |
| (this commit) | liveness and report |

**Correction of a long-standing figure.** "56.5% of arm D's entries are outside the gbb trigger"
(023) was a misreading. It was a trade-set overlap with the twin, which takes about 9,000 trades
against arm D's 1,400, so pair occupancy alone splits the two sets. Measured directly, **2 of
13,608** arm D model-signal rows (0.015%) are off-trigger. Arm D's model is selection *within* the
gbb trigger, as its `gbb_path` label implies.

## What changed

- **`utils/EntryFramework.py`** (new): the shared mixin. It holds trigger evaluation and tags,
  025 rule B, 026 pre-emption and regime gate, `FAMILY_EXIT_OVERRIDES` MRO merging and the
  liveness sinks, moved verbatim from `SimpleTriggerStrategy`. It also bundles
  `FamilyCoexistenceMixin`, `FamilyExitMixin` and `StopInvariantMixin`. New in it:
  - `ENTRY_MODE`: `triggers` (SimpleTrigger), `model` (NNNC), or `trigger_and_model` (a switch);
  - `MODEL_ENTRY_FAMILY` (AUTO = from the training label);
  - refusals at class creation for declarations that can't apply.

  It is strategy-agnostic, and `Framework/` is not edited.
- **`utils/LabelFamilies.py`** (new, A′): label → family table.
  - gbb, `gbb_path`, `gbb_fwdmae` and geometry → reversion; the breakout labels → momentum;
    outcome-only labels → none.
  - Every row was checked against its labeler, and agrees with the same-named trigger.
- **`SimpleTrigger/SimpleTriggerStrategy.py`**: 451 → 137 lines, now presets + trigger source
  over the mixin.
- **`NNNC/NNNCStrategy.py`**: `NNNCStrategy(EntryFrameworkMixin, BaseNNStrategy)` with
  `ENTRY_MODE = "model"`.
- **`NNNC/NNNC_TriggerGate.py`** (the twin):
  - its `StopInvariantMixin` is now inherited (listing it twice made the MRO impossible);
  - weights come via the new `NNNC/BorrowedModelMixin.py`;
  - its trigger is now `TWIN_TRIGGER`, so its record's `trigger` field reads null;
  - it discards the model-entry tags its `super()` call writes.
- **`utils/StopInvariants.py`**: a report-only mode (`FT_STOP_INVARIANTS=report`, own sink
  variable), read at every check.
- **`evaluation/fingerprint.py`**: `trigger_and_model` is hashed; model entry family is hashed when
  the family block exists. All pre-029 shas are unchanged.
- **`utils/RegimeState.py`**: owns `REGIME_CONTEXT_PAIR`, which `presets.py` re-exports.
- **`scripts/nnnc_fingerprints.py`** (new): scans and compares every NNNC class's fingerprint.
- **Tests:** 46 new (20 + 16 + 7 + 3) across `test_entry_framework`, `test_label_families`, `test_nnnc_merge` and
  the stop-invariant report mode.
  - Three SimpleTrigger tests were adjusted to the new structure: the refusal message now names
    the class, and the pre-emption parent stub moved with the code.
  - The single-source exit guard gained two allowlist entries (the merge test and the C2 liveness arm).

Record-level differences that leave every trade unchanged:
- arm D's entries are tagged `reversion:model` (was `model_entry `);
- the twin's `trigger` record field is null.

## Baseline (FR-004, [baseline.md](baseline.md))

The baseline was run before any NNNC or SimpleTrigger change, on the same day as the after-change
runs:
- arm D (15 folds) plus an f00 re-run;
- the twin;
- 8 SimpleTrigger reference arms;
- a stop-invariant probe: arm D's own entries and per-fold thresholds, report mode;
- model artefact hashes and an NNNC fingerprint scan.

**Stop gate: passed.**
- 0 violations;
- the probe = arm D 15/15;
- deterministic;
- the twin = `st_gbb` 15/15.

The probe's check was confirmed to be in the call path, so its 0 is not an inert reading.

## Verification ([verification.md](verification.md))

After the change, all 10 arms reproduce their baseline (and record) in 15/15 folds, and every
framework sha is unchanged. The twin still equals `st_gbb`. Fingerprint scan: 297 strategies,
0 changed.

## Liveness ([liveness.md](liveness.md))

Each switch was declared on arm D, on its own borrowed weights and thresholds:
- **C2 for reversion:** exits change (take-profit 23 → 19, ROI 1 → 2), 51 vs 52 trades on f00.
- **Regime gate:** 175 reversion-tagged signal rows refused; 31 vs 52 trades against its gate-off
  control. The 4096-bar startup alone changes arm D's trades, hence the dedicated control.
- **Trigger AND model:** it refuses exactly the off-trigger model signals. There are only 2 in 15
  folds, so trades are identical.

## The 6 NNNC variants that bypass the framework

These replace `get_entry_conditions` outright: the oracles (`NNNC_Oracle`, `NNNC_Ora_Prec`, the
three `NNNC_BrkOra*`), `NNNC_NoModel`/`TriggerOnlyMixin` and the twin. They keep exactly their
previous behaviour, and the declared switches don't apply to them. `NNNC_Breakout` calls
`super()`, so it reaches the framework and is tagged momentum by its `breakout_gbb` label.

## How to declare each switch on an NNNC arm

```python
class MyArm(NNNC_A_Raw_Open):
    FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit_atr_mult": 2.8970711921574543}}  # C2
    # startup_candle_count = 4096
    # REVERSION_REGIME_GATE = {"instrument": "volreg", "threshold": 1.3658023280926226, "off_when": "lt"}
    # ENTRY_MODE = "trigger_and_model"; TRIGGER_NAME = "gbb"
    # MODEL_ENTRY_FAMILY = None   # opt out of the label family (untagged, shared exit)
```

## Follow-ups

1. **Model naming by a resolved model fingerprint**, agreed as the next feature. It removes the
   class-name model paths, the self-training hazard and `BorrowedModelMixin`.
2. **C2 on arm D**, judged walk-forward (the liveness run is one fold, not a verdict).
3. **The regime gate on arm D**, judged walk-forward against the 4096-startup control.
4. **Retire the "arm D's out-of-trigger entries" investigation.** There are none to speak of.
   Trigger-AND-model is not a distinct arm for arm D; it would matter only for a model whose
   label doesn't encode the trigger's condition.
5. **`BaseNNStrategy.use_forward_peak_profit_label`** is declared and read nowhere: dead code.

## Tasks

- T001–T034 done. T017 and T014a belong to 028; 029 has no N/A tasks.
- T032: `AGENT_GUIDE.md` doesn't describe the NNNC/SimpleTrigger structure. The module docstrings
  were updated.
