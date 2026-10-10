# Interface Contracts: SimpleTrigger (023)

Paths are relative to `user_data/strategies/`. These are the interfaces other code and the
researcher depend on. The data shapes are in [data-model.md](../data-model.md).

## C1. Strategy class: `SimpleTriggerStrategy`

```python
# SimpleTrigger/SimpleTriggerStrategy.py
class SimpleTriggerStrategy(BaseStrategy):
    TRIGGER_NAME: str | None = None       # registry name; required
    PRESET: str = "ARM_D_OPEN"            # FrameworkPreset name
    EXIT_MODE: str = "decoupled"          # "decoupled" | "coupled"
    LABEL_METHOD: LabelMethod | None = None  # required iff EXIT_MODE == "coupled"
```

A configuration is a subclass that sets these and optionally merges overrides:

```python
class SimpleTrigger_Gbb(SimpleTriggerStrategy):
    TRIGGER_NAME = "gbb"
```

Guarantees:
- **G1**: the StrategyConfig has `model_type` NONE and `needs_training` False. No model file is read
  or written, and `saved_data/<cls>/` is never created.
- **G2**: entry is `triggers.evaluate(TRIGGER_NAME, df)` passed to `BaseStrategy.populate_entry_trend`'s
  guard chain unchanged. On an unset, unregistered, non-PASS or stale-causality trigger it enters
  **no trades** and prints one `*** SimpleTrigger:` line giving the reason. There is no fallback
  condition.
- **G3**: the framework attributes resolve to the named preset exactly. `buy_params`/`sell_params`
  are `{**parent, **preset, **overrides}`.
- **G4**: `custom_stoploss` returns `super()`'s value after the invariant check (research R9). A
  violation raises `StopInvariantError`.
- **G5**: `EXIT_MODE="coupled"` only changes parameter *values*, through
  `SimpleTrigger.coupling.resolve()`. There is no other code path.
- **G6**: no pair-specific behaviour.

## C2. Fold plan, no-model extension

```jsonc
{
  "family": "SimpleTrigger",
  "base_strategy": "SimpleTrigger_Gbb",
  "no_model": true,
  "folds": [ { "fold_id": "f00", "strategy_class": "SimpleTrigger_Gbb", "model_dir": null,
               "arm_tag": "st_gbb", ... } ]
}
```

- New subcommand `python -m evaluation.foldplan derive --from <plan.json> --family SimpleTrigger
  --base <cls> --tag <arm_tag> --out <plan.json>` copies the source plan's fold windows verbatim,
  so they are identical by construction, and emits the above. Rebuilding from dates is not offered
  for no-model plans, because `build()`'s `--data-end` default (2026-09-07) would have to be
  matched by hand.
- `foldplan.validate()`: duplicate `strategy_class` is allowed **iff** `no_model` is true and every
  `model_dir` is null.
- `run_folds.py`, when `no_model` is set:
  - skips `generate_fold`, `model_exists` and `train_fold`;
  - asserts G1 before the first evaluation;
  - ignores `thresholds` and refuses a plan whose `thresholds` is non-empty, rather than silently
    dropping it;
  - keeps everything else unchanged.
- `arm_record` and `pool.arm_record` resolve `{tag}_{label}.json` by exact name (research R6).

## C3. Arm record additions (`regime/run_arm.py`)

These are written for **every** arm, NNNC included:
- `framework_sha`, `framework_resolved` (from `evaluation.fingerprint.resolve(strategy_cls,
  params_json, config)`);
- `exit_reason_counts`;
- `entry_condition_counts` when `FT_ENTRY_CONDITION_SINK` is set;
- `no_model`, `preset`, `exit_mode`, `trigger`, `label_method`, `coupled_resolution`.

Existing fields, including `exit_config_hash` and `trades_hash`, are unchanged and computed the same
way.

## C4. Entry-condition sink (`Framework/BaseStrategy.py`, generic)

`populate_entry_trend` names each mask it ANDs (`signal`, `volume`, `rvol`, and the quality guards
when enabled). If env `FT_ENTRY_CONDITION_SINK=<dir>` is set, per-pair pass counts are written to
`<dir>/<strategy>_<pair>.json`. When unset, behaviour and output are byte-identical to HEAD. It
contains no strategy names or strategy-specific branches.

## C5. Command-line tools

```text
python -m research.outcomes.causality [--trigger NAME ...] [--pairs ...] [--write]
    -> trigger_causality.json rows; exit 1 if the planted control does not FAIL

python -m evaluation.fingerprint --strategy CLS [--config PATH]
    -> prints framework_sha + resolved dict

python -m evaluation.twins check  --nnnc-tag T --simple-tag S --equivalence-tag E
    -> exit 2 on fingerprint mismatch, 3 on any per-fold trades_hash mismatch (E vs S)
python -m evaluation.twins compare --nnnc-tag T --simple-tag S
    -> paired delta, CI, halves, shared-trade fraction; refuses unless `check` passed

python -m evaluation.attribution --full-tag T --components all|NAME,... [--preset P]
    -> attribution_<T>.json/.md; INERT rows flagged
python -m evaluation.halves --arm-tag A --ctl-tag C [--split f07]
```

Exit codes are non-zero on every refusal. The refusal reason is printed and also written to the
output file when one is requested.
