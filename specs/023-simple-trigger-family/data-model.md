# Data Model: SimpleTrigger Strategy Family (023)

These are the entities from the spec, made concrete. "Lives in" gives the repository location;
paths are relative to `user_data/strategies/`.

## Trigger (existing, unchanged)

A named boolean condition over closed bars. It lives in `research/outcomes/triggers.py`, which
holds the registry (19 entries), and `builtin_triggers.py`.

| field | type | rule |
|---|---|---|
| name | str | unique in the registry |
| evaluate | frame → bool Series | pure, aligned to the index, no NaN (`_validate_output`) |
| required_columns | tuple[str] | must not name an `__outcome_*` or `pair` column (enforced at registration) |
| source | str | the `LabelMethod` labeler it ports, `native`, or `CONTROL` |

## TriggerCausality (new)

One row per trigger, written by `research/outcomes/causality.py` to
`research/outcomes/trigger_causality.json`.

| field | type | rule |
|---|---|---|
| trigger | str | registry name |
| verdict | `PASS` \| `FAIL` \| `UNSCOREABLE` | `UNSCOREABLE` when required columns are absent from the populator |
| pairs, eras | list | ≥3 pairs × ≥3 eras, including one distant era (Principle II) |
| cut_points | int | prefixes per pair/era |
| warmup_bars | int | declared warm-up excluded from the comparison |
| first_mismatch | {pair, bar, full, truncated} \| null | set when the verdict is FAIL |
| checker_control | {trigger: `local_min_followthrough`, verdict: `FAIL`} | must be FAIL, or the file is invalid |
| code_sha | str | git sha of the populator, trigger and checker at measurement time |

**Rule**: SimpleTrigger accepts a trigger only if its row exists, is `PASS`, and its `code_sha`
matches the current files. A stale row counts as not PASS.

## LabelTriggerMapping (new)

One row per `LabelMethod` member (29), in `SimpleTrigger/label_map.py` as a literal table, with a
test asserting it covers exactly the enum.

| field | type | rule |
|---|---|---|
| label_method | LabelMethod | every member exactly once |
| trigger | str \| null | null iff unsupported |
| status | `supported` \| `empty` \| `unsupported` | `empty` = pure-outcome label; it is labelled as such in run records |
| reason | str | required when unsupported (lookahead, record conflict, unscoreable) |

Initial content: research R10 (3 `future_*` unsupported pending reconciliation; `trends`, `bands`
and `local_min_followthrough` unsupported).

## FrameworkPreset (new)

A complete, named set of framework values. It lives in `SimpleTrigger/presets.py`.

| field group | members |
|---|---|
| entry guards | `buy_params` keys: `entry_enable_guards`, `entry_{adx,atr_pct,bb_width,close_norm,guard,rvol}_threshold`; `entry_bear_filter_enable`, `entry_trend_filter_enable` |
| exits | `minimal_roi`; `sell_params`: `cexit_enable_profit_checks`, `cexit_max_days`, `cexit_take_profit`, `enable_exit_signal`, `exit_*` |
| stop | `stoploss`, `use_atr_adaptive_stoploss`, `atr_stoploss_{multiplier,floor,cap}`, `use_volume_confirmation_stoploss`, `stoploss_grace_{hours,level}`, `trailing_*` |
| run shape | `timeframe`, `startup_candle_count`, `max_open_trades`, `process_only_new_candles`, `can_short` |

Two instances:
- `ARM_D_OPEN` (the default) is the resolved `NNNC_A_Raw_Open` + `NNNC_A_Raw_Open.json` values
  (research R1).
- `LEGACY_TRIGGERONLY` is the resolved `TriggerOnlyStrategy` values.

**Rule**: a preset is total. A missing key is a load error, not an inherited default.

## Component (new)

One switchable part of the framework, listed in `SimpleTrigger/components.py`.

| field | type | rule |
|---|---|---|
| name | str | e.g. `trigger`, `guard_adx`, `guard_atr_pct`, `guard_bb_width`, `guard_close_norm`, `guard_guard`, `guard_rvol`, `take_profit`, `max_hold`, `roi`, `exit_signal`, `stop_atr_band`, `stop_grace`, `stop_volume_confirm` |
| overrides | dict | the parameter values that neutralise it (for example `entry_adx_threshold: -1.0`, `stoploss_grace_hours: 0`) |
| liveness_metric | str | which recorded measurement must move: a condition-sink key, an `exit_reason_counts` key, or `trades_hash` |
| neutral_is_identity_under | list[preset] | presets where the component is already neutral (for example guards under `ARM_D_OPEN`). An arm is not run there; it is reported as "already neutral" |

**Rule**: neutralising a component may change only the parameters in its `overrides`. A test
diffs the fingerprints.

## FrameworkFingerprint (new)

A hash of the resolved framework, computed in `evaluation/fingerprint.py` without instantiating
the strategy.

| field | rule |
|---|---|
| inputs | MRO class attributes for every FrameworkPreset member, deep-merged with `<Strategy>.json` the way freqtrade does, plus the run config's `max_open_trades`, `stake_amount`, `fee` |
| excluded | entry-signal source (model or trigger), `prediction_threshold`, the classifier, and label params. These are what a twin is allowed to differ in |
| output | `framework_sha` (sha256[:16]) and the canonical resolved dict, both written to the arm record |

## SimpleTrigger arm (new strategy class)

| field | rule |
|---|---|
| class | `SimpleTrigger_<Name>` in `SimpleTrigger/`, subclass of `SimpleTriggerStrategy` |
| TRIGGER_NAME | registry name that passes the TriggerCausality rule |
| PRESET | a FrameworkPreset (default `ARM_D_OPEN`) |
| EXIT_MODE | `decoupled` (default) \| `coupled` |
| LABEL_METHOD | required iff `coupled`; supplies the gain/loss/horizon targets |
| overrides | optional, merged `{**preset, ...}`, never a fresh literal |

## ArmRecord (existing, extended)

`regime/arms/{tag}_{label}.json`, written by `regime/run_arm.py`. Any delta derived from these
records is PROVISIONAL under quality gate 2 (no distant era). New fields:

| field | type |
|---|---|
| framework_sha, framework_resolved | str, dict |
| preset, exit_mode, trigger, label_method | str \| null |
| coupled_resolution | {take_profit_net, max_days, stop, coupled_stop: `label` \| `none-in-label`} \| null |
| exit_reason_counts | {reason: int} |
| entry_condition_counts | {condition: {pass: int, total: int}}; always present for runs launched through `run_arm` |
| no_model | bool |

## FoldPlan (existing, extended)

New key: `"no_model": true`. When it is true, `folds[*].model_dir` is null, `strategy_class` is the
same for every fold, and training and generation are skipped (contract C2).

## TwinPair (new)

`evaluation/twins/<name>.json`

| field | rule |
|---|---|
| nnnc_tag, simple_tag | arm tags on the same fold plan windows |
| framework_sha_nnnc, framework_sha_simple | **must be equal**, or the comparison is refused |
| equivalence_tag | the `NNNC_TriggerGate` arm tag; per-fold `trades_hash` must equal `simple_tag`'s |

## AttributionReport (new)

`evaluation/attribution_<name>.json` + `.md`. One row per Component:

| field | rule |
|---|---|
| component, arm_tag | |
| status | `LIVE` \| `INERT` (identical `trades_hash` in every fold) \| `ALREADY_NEUTRAL` \| `ZERO_TRADES` |
| delta_pp, ci, verdict | from `compare_pooled.compare`; underpowered verdicts are reported as such |
| halves | {early: {...}, late: {...}} from `evaluation/halves.py` |
| liveness_evidence | the moved metric, with its before and after values |
| fold_detail | per-fold pp, trades |

## State transitions

TriggerCausality: `absent → PASS|FAIL|UNSCOREABLE`. It reverts to absent (stale) on any change to
the populator, trigger or checker code. A run checks this at strategy start.

TwinPair: `defined → fingerprint-matched → equivalence-verified → compared`. Each step blocks the
next one.
