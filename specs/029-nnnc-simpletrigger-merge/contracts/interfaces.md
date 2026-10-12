# Interfaces: NNNC / SimpleTrigger Merge (029)

## I1 — `utils.EntryFramework.EntryFrameworkMixin`

```python
class EntryFrameworkMixin(FamilyCoexistenceMixin, FamilyExitMixin, StopInvariantMixin):
    ENTRY_MODE: str            # "triggers" | "model" | "trigger_and_model"
    def get_entry_conditions(self, dataframe) -> np.ndarray | pd.Series | None
    def populate_entry_trend(self, dataframe, metadata) -> DataFrame   # super() then tags
    def custom_exit(self, pair, trade, ...)                            # 026 preempt, else super()
    def informative_pairs(self)                                       # gate context pair, else []
    @classmethod
    def _resolve_family_exits(cls, base_sets: dict) -> dict            # 025 G1 MRO merge
```

- **Model source:** `super().get_entry_conditions(dataframe)`, the next class in the MRO.
- **Trigger source:** `research.outcomes.triggers.evaluate`.
- **Tag:**
  - `<family>:<trigger>[+ov]` for a trigger;
  - `<family>:model` for a model entry with a label family;
  - otherwise `BaseStrategy`'s `"model_entry "`.
- **Refusals at class creation** (`ValueError`):
  - `ENTRY_MODE` is not one of the three values;
  - `"trigger_and_model"` or `"triggers"` without a trigger;
  - `"model"` with `TRIGGER_NAME`/`TRIGGERS` set (ambiguous);
  - rule B or pre-emption in `"model"` mode;
  - the regime gate with `startup_candle_count` < 4096 (already enforced for SimpleTrigger).

## I2 — `utils.LabelFamilies`

```python
LABEL_FAMILY: dict[LabelMethod, tuple[str, str]]
def label_family(training_type) -> str | None
```

## I3 — `utils.StopInvariants`

```python
def _mode() -> str:   # read at EVERY check, never cached at import
    return os.environ.get("FT_STOP_INVARIANTS", "halt")   # "halt" | "report"
```

In report mode `_check_stop` records the violation (sink file) and returns. In halt mode the
behaviour is unchanged.

## I4 — `scripts/nnnc_fingerprints.py`

```text
python user_data/strategies/scripts/nnnc_fingerprints.py --out <json> [--compare <before.json>]
```

- Without `--compare`: writes `{class: sha | "LOAD_ERROR: …" | "NOT_A_STRATEGY"}`.
  `NOT_A_STRATEGY` marks a file stem with no same-named `IStrategy` subclass (mixins,
  classifiers), reported separately from real load errors.
- With `--compare`: also prints the diff, and exits 1 on any change.

## I5 — Plans

All are model plans with borrowed weights, never training.
- Arm D, the twin and the SimpleTrigger plans copy their source plan with only `arm_tag` (and the
  fold subset) changed.
- The probe and the liveness plans copy `plan_NNNC_A_Raw_Open.json`, **keeping its per-fold
  `thresholds`**: a `foldplan derive` plan carries none, and arm D's entries depend on them. They
  set `model_source="NNNC_ARO"` and `model_dir=saved_data/NNNC_ARO_fNN`. Missing models fail loudly (existing `train_model` refusal / `model_exists` check).
