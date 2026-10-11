# Contracts: Trigger-Family Exit Presets (024)

## C1 — `BaseStrategy._exit_params(trade=None)` (Framework, additive)

- **Signature:** `_exit_params(self, trade: Trade | None = None) -> ExitParams`.
- **Default body:** unchanged. It ignores `trade`, and `_exit_params(t) == _exit_params()` for any
  `t` (unit test).
- **Call sites passing `trade`:** `custom_exit`, the `custom_stoploss` after-fill ATR branch, and
  `_resolve_normal_stop(pair, trade)`.
- **Stop-path reads routed through params** (research R2): `_atr_bar_inputs(pair, params)`,
  `_resolve_normal_stop`, the `custom_stoploss` after-fill branch.
- **No family logic, trigger name or tag parsing in `Framework/`** (FR-014).
- **Neutrality gate:** for non-opting strategies the edit must not change any recorded trades
  (research R9).

## C2 — `utils/FamilyExits.py`: `FamilyExitMixin` (strategy-agnostic, FR-015)

- `_exit_params(trade=None)`:
  - computes `base = super()._exit_params(trade)`;
  - with no trade, or no family on it, returns `base`;
  - otherwise returns `dataclasses.replace(base, **self.FAMILY_EXITS[family])`.
- `family_of(tag) -> str | None` parses the `"<family>:<trigger>"` tag.
- It keeps an in-process `untagged_exits` counter for tests and logs only. The run record's
  `untagged_trades` comes from the archive (C8, F2).
- Tag grammar: `"<family>:<trigger>[+ov]"`. `family_of` ignores the suffix.
- Class-creation validation:
  - refuses a key set both in a family set and via `SELL_OVERRIDES`/`ATTR_OVERRIDES` (F3);
  - refuses `unclog_12h_hours > max_hold_days*24`, and warns when it exceeds `unclog_1d_hours`
    (F5).
- It does not know about SimpleTrigger or the registry. It reads `FAMILY_EXITS` and the tag only.
  It is usable by a future NNNC feature.

## C3 — `StopInvariantMixin._check_stop` (utils)

Reads `use_atr_adaptive_stoploss`, `static_stoploss` and `atr_stoploss_floor` from
`self._exit_params(trade)` when available, and falls back to the attributes otherwise. Its
behaviour is unchanged for a strategy whose params mirror its attributes.

## C4 — SimpleTrigger (strategy)

- `TRIGGERS` / `DISABLED_TRIGGERS` / `FAMILY_PRIORITY` / `FAMILY_EXITS`, per the data model.
- `get_entry_conditions`:
  - runs `check_runnable` per enabled trigger, refusing as today;
  - returns the OR of the masks;
  - stores the per-row winning tag, with `+ov` appended when a trigger of another family also
    fired on that row (F4).
- `populate_entry_trend`: `super()`, then overwrite `enter_tag` on signal rows with the stored
  tags.
- MRO: `SimpleTriggerStrategy(FamilyExitMixin, StopInvariantMixin, BaseStrategy)`.
- **Family exits are chosen by PRESET, not by a default flip.**
  - The new preset `ARM_D_FAMILY` = `ARM_D_OPEN`'s attributes/buy/sell plus the family exit sets
    (data model).
  - `ARM_D_OPEN` and `LEGACY_TRIGGERONLY` carry no family sets, so every family resolves to the
    shared exits. Every 023 config is therefore untouched and keeps its recorded trades. A
    regression test re-checks `st_bbo` f00 and `stl_bo` f00.
  - Declaring a family arm = trigger(s) + `PRESET = "ARM_D_FAMILY"` (User Story 1).

## C5 — Registry (`research/outcomes/triggers.py`)

`Trigger.family`, `Trigger.family_reason`. `ensure_builtins()` assigns them. The CLI `--list`
prints the family.

## C6 — `evaluation/contention.py` (CLI)

```
python -m evaluation.contention --combined <pooled.json> \
    --single reversion=<pooled.json> --single momentum=<pooled.json> \
    --config config/config_wf.json --out <contention.json>
```

- Output: a `ContentionRecord` per family per half, plus a markdown summary.
- Exit 2 on missing inputs.

## C7 — Fingerprint (`evaluation/fingerprint.py`)

Optional members, per data model. They are included only in family mode, multi-trigger mode
or disabled-side mode (F1).

Tests:
- arm D and `SimpleTrigger_Gbb` both stay `303456c1fa322b4c`;
- each of these changes the sha: one family value, the priority order, a `DISABLED_TRIGGERS`
  entry, or a trigger's family in `TRIGGER_FAMILIES` (F6).

## C8 — `regime/run_arm.py` tag accounting (F2)

The record gains `untagged_trades` (trades whose `enter_tag` has no recognised family prefix),
`family_trade_counts`, `overlap_entries` (tags with `+ov`) and per-family `exit_reason_counts`.
All are computed from the backtest archive's trades, and are absent when no trade carries a
family tag.
