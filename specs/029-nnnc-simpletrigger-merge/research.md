# Research: NNNC / SimpleTrigger Merge (029)

## R1 — Where the shared framework lives, and what moves

**Decision.** A new strategy-agnostic `utils/EntryFramework.EntryFrameworkMixin`. It holds
everything `SimpleTriggerStrategy` currently implements beyond presets:
- `get_entry_conditions` built on entry sources (R2);
- `_tags` (`<family>:<source>`, `+ov`);
- 025 rule B (`MOMENTUM_RECENCY_BARS`);
- 026 momentum pre-emption (`MOMENTUM_PREEMPT`, `custom_exit` hook);
- 026 regime gate (`REVERSION_REGIME_GATE`, `_regime_state`, `_apply_regime_gate`,
  `informative_pairs`, warm-up validation);
- `populate_entry_trend` tagging;
- the coexistence and gate liveness sinks;
- the `FAMILY_EXIT_OVERRIDES` MRO merge (R6).

It is composed with the existing mixins in this order:

```text
EntryFrameworkMixin, FamilyCoexistenceMixin, FamilyExitMixin, StopInvariantMixin
```

- `SimpleTriggerStrategy(EntryFrameworkMixin, BaseStrategy)`: keeps `_apply_preset`, the
  causality refusal and the trigger validation.
- `NNNCStrategy(EntryFrameworkMixin, BaseNNStrategy)`: `ENTRY_MODE = "model"`.

**Rationale.**
- `Framework/` must hold no strategy-specific code, and the existing shared mixins
  (`FamilyExits`, `FamilyCoexistence`, `StopInvariants`, `RegimeState`) already live in `utils/`.
- The moved code names families and triggers generically; it never names NNNC or SimpleTrigger.
- `_presets.REGIME_CONTEXT_PAIR` (BTC/USDT, used by the `btctrend` gate) moves to
  `utils/RegimeState.py`, so the mixin does not import SimpleTrigger.

**Alternatives rejected.**
- *NNNC inherits SimpleTriggerStrategy:* this drags in presets, the causality refusal and the
  trigger-required check, and makes the model family depend on the model-free one.
- *Copy the code into NNNCStrategy:* duplication, which is the drift this feature removes.

## R2 — Entry sources and `ENTRY_MODE`

| `ENTRY_MODE` | entry mask | tag on a signal row | used by |
|---|---|---|---|
| `"triggers"` | OR of the enabled triggers | `<trigger family>:<trigger>` (+`ov`), as today | SimpleTrigger (default there) |
| `"model"` | `super().get_entry_conditions(df)`, the model's prediction through the next class in the MRO (`BaseNNStrategy`), unchanged | `<label family>:model` if the label declares a family (R3), else unchanged (`"model_entry "`) | NNNC (default there) |
| `"trigger_and_model"` | model mask AND OR(triggers) | label family, as for `"model"` (spec edge case) | switch, off by default |

- **The guard chain is untouched.** `BaseStrategy.populate_entry_trend` still ANDs the mask with
  volume, rvol and the guards, in the same order. The mixin only supplies the first mask and the
  tags.
- **Rule B and the regime gate apply only after trigger families are known.**
  - In `"model"` mode they are refused at class creation if declared without triggers.
  - Exception: the regime gate keys on the entry's tag family, so a reversion-tagged model entry
    can be gated (US3, FR-008).
- **NNNC variants that override `get_entry_conditions`** (9 files: oracles, `NNNC_NoModel`,
  `TriggerOnlyMixin`, `NNNC_TriggerGate`, `NNNC_Breakout`):
  - those that call `super()` (`NNNC_Breakout`) reach the mixin and are tagged;
  - those that replace the mask outright keep exactly today's behaviour, with untagged
    `"model_entry "` rows, and cannot use the declared switches;
  - the report lists them.

## R3 — Label families (A′)

**Decision.**
- Add `utils/LabelFamilies.py`: `LABEL_FAMILY: dict[LabelMethod, tuple[family, reason]]`, only for
  labels whose positives encode an entry state. Every other label maps to none.
- The model's label is the class's resolved `TRAINING_TYPE`.
- `NNNCStrategy` resolves `MODEL_ENTRY_FAMILY` once per class, from `TRAINING_TYPE`, unless the
  class sets it explicitly. An explicit `None` means "untagged", deliberately.

| label (`LabelMethod`) | family | reason (verified against the labeler) |
|---|---|---|
| `gbb` (17), `gbb_path` (27), `gbb_fwdmae` (28) | reversion | positives require `guard_metric < 0.3` with a wide band: a weak band position (as the `gbb` trigger) |
| `breakout_gbb` (24), `breakout_bb` (26) | momentum | positives require a high band position (as the `breakout_gbb` and `breakout_bb` triggers) |
| `breakout` (20), `breakout_tb` (21), `breakout_vol` (22), `breakout_squeeze` (23), `breakout_consensus` (25) | momentum | positives require a close above the prior high (strength) |
| `geometry` (14) | reversion | a confirmed valley after a ≥ 1% low (as the `geometry` trigger) |
| everything else (`forward_mae`, `triple_barrier`, `quantile_future`, …) | none | outcome-only: a bounce and a breakout both qualify |

- **Pre-implementation check.** Each row is re-read against its labeler's condition before
  commit, and any row that doesn't hold is dropped to none.
- **Consistency test.** Where a trigger shares the name, its family must match.
- **Arm D:** `TRAINING_TYPE = gbb_path`, so reversion. Its `use_forward_peak_profit_label` flag is
  read nowhere, and is noted in the report.

## R4 — The baseline (FR-004), before any NNNC edit

| run | plan | purpose |
|---|---|---|
| arm D, f00–f14 | `plan_wf_aro_b029.json` (= `plan_NNNC_A_Raw_Open.json`, `arm_tag` `wf_aro_b029_fNN`) | per-fold `trades_hash` baseline; also compared with the old `wf_aro_*` records, any difference reported (spec edge case) |
| arm D f00 again | `plan_wf_aro_b029r.json` (f00, `wf_aro_b029r_f00`) | determinism: must equal the first run |
| twin, f00–f14 | `plan_wf_trgate_b029.json` (from `plan_wf_trgate.json`) | twin baseline; must equal the fresh `st_gbb` baseline (and is reported against the 10 Oct record) |
| SimpleTrigger reference arms | `plan_<arm>_b029.json` for the 8 arms of R8 (`st_gbb`, `st_bgo_mx`, `st_book_rev2`, `st_reg_volreg2`, `st_coex_win4`, `st_coex_cap3`, `st_pre_volreg2`, `st_prom_c2`) | same-day, pre-change baseline. Their records date from 10–12 Oct, and `BaseStrategy` has changed since (028), so the records are reported, never the pass condition |
| stop-invariant probe | `plan_nnnc_si_probe.json` (see below) with `FT_STOP_INVARIANTS=report` | violation count. Its trades must equal arm D's baseline, because the mixin returns the parent's value unchanged. |
| fingerprints | `scripts/nnnc_fingerprints.py` → `fingerprints_before.json` | `{class: sha}`, `{class: "LOAD_ERROR: …"}`, or `{stem: "NOT_A_STRATEGY"}` when the file defines no same-named strategy class (mixins, classifiers) |
| model artefacts | `sha256` of every file under `saved_data/NNNC_ARO_f*/` → `baseline.json` | the no-training check (SC-005), re-hashed after the change |

**The probe must enter exactly as arm D does.**
- A plan from `foldplan derive --model-source` carries no `thresholds` (`plan_wf_trgate.json` has
  none: the twin ignores the model), but arm D's entries depend on its per-fold prediction
  thresholds (0.73–0.78). `run_folds.generate_fold` writes them into each fold's JSON only from
  `plan["thresholds"]`.
- So `plan_nnnc_si_probe.json` is a **copy of `plan_NNNC_A_Raw_Open.json`** keeping `thresholds`
  and `params_snapshot`. It sets `base_strategy = "NNNC_SIProbe"`, `model_source = "NNNC_ARO"`,
  and per fold `strategy_class = "NNNC_SIP_fNN"` and `arm_tag = "nnnc_si_probe_fNN"`. `model_dir`
  stays `saved_data/NNNC_ARO_fNN`.
- With `model_source` set, `run_folds` refuses to train if the model is missing.
- A dry-run plus a check that each generated `NNNC_SIP_fNN.json` threshold equals
  `NNNC_ARO_fNN.json`'s precede the run.

**Generated files.** Re-running arm D and the twin regenerates the tracked per-fold
`NNNC_ARO_fNN.*` / `NNNC_TGT_fNN.*` files through `generate_fold`. `git diff --stat` on them must
be empty after the baseline; any difference is explained in `baseline.md` before going on.

**No-training evidence (SC-005).**
- No `run_folds` "training on" line appears in its output.
- Every `saved_data/NNNC_ARO_f*` artefact hash is unchanged.
- `TrainingEngine`'s own message is a `debug_print` that may be silent, so it isn't used.

- Run-to-run cost: about 10 s per model fold and about 3.3 min per SimpleTrigger arm. The whole
  baseline is about 35 min.
- **STOP rule (FR-006):** the feature stops before the refactor, and writes the defect up, if any
  of these hold:
  - the probe reports more than 0 violations;
  - the probe's trades differ from arm D's baseline (an instrument defect);
  - the f00 re-run differs;
  - the twin differs from the fresh `st_gbb` baseline.

## R5 — Class hierarchy (C3)

`NNNC_TriggerGate(StopInvariantMixin, NNNC_A_Raw_Open)` can't linearise once `NNNCStrategy`
already carries `StopInvariantMixin`: StopInvariantMixin must precede `NNNC_A_Raw_Open` yet appears
after it in that class's MRO, so Python raises `TypeError`. The twin therefore drops its own
mixin and inherits the same check from `NNNCStrategy`. A test asserts that every NNNC class's
MRO contains `StopInvariantMixin` exactly once, before `BaseStrategy`.

## R6 — Overrides that merge

- `FAMILY_EXIT_OVERRIDES` keeps its 025 G1 semantics: merged per family, per key, ancestors first.
  It moves from `_apply_preset` to the mixin's `__init_subclass__`.
- The base family sets are the preset's for SimpleTrigger (unchanged) and empty for NNNC. So an
  NNNC class has `FAMILY_EXITS == {}` unless it declares overrides.
- NNNC keeps its own parameter mechanism (class attributes plus params JSON). SimpleTrigger's
  `BUY/SELL/ATTR_OVERRIDES` stay a SimpleTrigger preset feature: NNNC is the source those presets
  mirror.

## R7 — Fingerprint neutrality

- The family block in `evaluation/fingerprint.py` is emitted only when family sets, several
  triggers, disabled triggers or coexistence are set. For an NNNC class none are, so it stays
  empty.
- New class attributes (`ENTRY_MODE`, `MODEL_ENTRY_FAMILY`, trigger fields left at None) are not
  in `ATTRIBUTE_MEMBERS`.
- `ENTRY_MODE` and `MODEL_ENTRY_FAMILY` are added to the family block only when the block is
  already non-empty, or when `ENTRY_MODE` is not the class family's default. Every current class
  therefore keeps its sha; FR-005 checks this over all classes.
- Moving code out of `SimpleTriggerStrategy` changes no resolved attribute, so the SimpleTrigger
  shas hold. The 028 pinned-sha test covers 10 of them.

## R8 — Verification after the change (FR-005, FR-005a)

- **Arm D and the twin:** new tags `wf_aro_a029`, `wf_trgate_a029`. 15/15 `trades_hash` equal to
  the baseline, and the twin equal to `st_gbb`.
- **SimpleTrigger, under fresh `_a029` tags, compared with their `_b029` baselines** (records
  reported alongside):

  | arm | what it exercises |
  |---|---|
  | `st_gbb` | twin partner |
  | `st_bgo_mx` | momentum family exits |
  | `st_book_rev2` | regime gate, reversion only |
  | `st_reg_volreg2` | gate, both families |
  | `st_coex_win4` | rule B |
  | `st_coex_cap3` | rule A |
  | `st_pre_volreg2` | pre-emption |
  | `st_prom_c2` | 028 switch |

  That is 8 arms at about 3.3 min each.
- **Fingerprints:** `fingerprints_after.json` must equal the before scan for every class that
  loaded before. Classes that failed before must fail identically.
- **Artefacts:** all `saved_data/NNNC_ARO_f*` hashes are unchanged (SC-005).

## R9 — Liveness of the declared switches (FR-008, US3)

Test classes under `NNNC/` (`NNNC_Live_*`, per fold, borrowing arm D's models), on f00:

| class | declares | liveness signal |
|---|---|---|
| `NNNC_Live_TrigGate` | `ENTRY_MODE="trigger_and_model"`, `TRIGGER_NAME="gbb"` | fewer entries than arm D; every entry on a gbb bar; dropped count in the sink |
| `NNNC_Live_C2` | `FAMILY_EXIT_OVERRIDES={"reversion": {"take_profit_atr_mult": 2.897…}}` | take-profit exits change vs arm D f00 (exit-reason counts, `trades_hash`) |
| `NNNC_Live_Gate` | `REVERSION_REGIME_GATE` = st_book_rev2's | gate liveness counts in the sink; entries in the "off" state dropped |

Undeclared, every one equals arm D, and `NNNC_ARO_f00` is the control.

C2 stands for the whole per-family-exit path: it is declared through `FAMILY_EXIT_OVERRIDES`,
resolved by `FamilyExitMixin`, and switched by the entry's tag, which is the same path any
family exit takes. A second family-exit liveness case would test nothing new.

## R10 — Tests

**New:**
- the mixin, with stub dataframes:
  - each `ENTRY_MODE`'s mask and tags;
  - the refusals (rule B or pre-emption without triggers in model mode; `trigger_and_model`
    without a trigger);
  - `FAMILY_EXIT_OVERRIDES` merging;
- the label families (table vs labelers, consistency with triggers, `gbb_path` → reversion);
- report-only stop invariants (no raise; the sink is written);
- MRO uniqueness over all NNNC classes;
- the 9 overriding variants keep their behaviour (static check: their `get_entry_conditions`
  bodies are unchanged, and those calling `super()` are listed).

**Existing suites:** must pass except the 3 known training-pipeline golden failures (028).
