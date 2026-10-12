# Data Model: NNNC / SimpleTrigger Merge (029)

## Class attributes (shared mixin)

| attribute | type | default | meaning |
|---|---|---|---|
| `ENTRY_MODE` | `"triggers" \| "model" \| "trigger_and_model"` | SimpleTrigger `"triggers"`; NNNC `"model"` | which entry sources make the first mask (R2) |
| `TRIGGER_NAME` / `TRIGGERS` / `DISABLED_TRIGGERS` | str / tuple / tuple | None / None / () | trigger source. Required for `"triggers"` and `"trigger_and_model"`; refused in `"model"` |
| `FAMILY_PRIORITY` | tuple | ("reversion", "momentum", "unassigned") | tag winner on overlap |
| `MOMENTUM_RECENCY_BARS`, `MOMENTUM_PREEMPT`, `REVERSION_REGIME_GATE` | int / bool / dict \| None | 0 / False / None | 025/026 switches, unchanged semantics |
| `FAMILY_EXIT_OVERRIDES` | dict | {} | merged along the MRO into `FAMILY_EXITS` |
| `MODEL_ENTRY_FAMILY` | str \| None \| `AUTO` | `AUTO` (NNNC) | `AUTO` resolves from `TRAINING_TYPE` via `LABEL_FAMILY`; an explicit None means untagged |

## Label family table (`utils/LabelFamilies.LABEL_FAMILY`)

`{LabelMethod: (family, reason)}`. Families are drawn from `research.outcomes.triggers.TRIGGER_FAMILIES`.
A label not in the table has no family.

## Stop-invariant mode

`FT_STOP_INVARIANTS` = `halt` (the default when unset) or `report`. In `report` mode a violation is
written to `$FT_STOP_INVARIANTS_SINK/<class>_<pair>_stopinv.json` as
`{pair, trade_id, time, check, message}`, and the parent's value is returned unchanged.

## Baseline (`baseline.json`)

```json
{"commit": "...", "arm_d": {"f00": "<trades_hash>", ...}, "arm_d_rerun_f00": "...",
 "twin": {...}, "old_records_arm_d": {...}, "old_vs_baseline_diff": [...],
 "probe": {"trades_equal_arm_d": true, "violations": 0, "by_check": {}},
 "twin_vs_st_gbb_baseline": {"f00": true, ...}, "twin_vs_st_gbb_record": {...},
 "simpletrigger": {"<arm>": {"f00": {"baseline": "...", "record": "..."}, ...}},
 "model_artifacts": {"saved_data/NNNC_ARO_f00/<file>": "<sha256>", ...},
 "generated_files_diff": []}
```

## Fingerprint scan (`fingerprints_before.json` / `_after.json`)

`{"<class>": "<sha16>" | "LOAD_ERROR: <ExcType>: <msg>" | "NOT_A_STRATEGY"}` for every file stem
in `NNNC/*.py`.

## Verification (`verification.md`)

- Per arm: per-fold `baseline/recorded hash`, `after hash`, `equal`.
- Fingerprint diff: added, removed, changed and error-changed classes. Only "none" passes.
