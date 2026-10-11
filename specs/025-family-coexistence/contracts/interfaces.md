# Contracts: Family Coexistence and Reversion Late-Half Fix (025)

## C1 — `utils/FamilyCoexistence.py`: `FamilyCoexistenceMixin` (rule A; strategy-agnostic)

- `FAMILY_PAIR_CAP: int | None = None`.
- `confirm_trade_entry(...)`:
  1. `ok = super().confirm_trade_entry(...)`. The existing volume check runs FIRST and unchanged.
  2. If `not ok` or the cap is None, return `ok`.
  3. `fam = family_of(entry_tag)`. If None, return `ok`.
  4. `held = #open trades whose family_of(enter_tag) == fam` (via
     `Trade.get_trades_proxy(is_open=True)`).
  5. If `held >= FAMILY_PAIR_CAP`, log one refusal to the sink (C3) and return False.
- No trigger, registry or SimpleTrigger import. It reads tags only.
- Class-creation validation per the data model.

## C2 — `SimpleTriggerStrategy` rule B (in `_tags` / `get_entry_conditions`)

- `MOMENTUM_RECENCY_BARS: int = 0`.
- When > 0:
  1. `mom_recent = rolling max over the last N rows (current included) of OR(momentum-family
     masks)`, using only past and current rows (causal).
  2. `drop = (winning family == "reversion") & mom_recent & ~(any momentum mask on this row)`.
  3. Dropped rows leave the entry mask and get tag "".
  4. **Overlap rows** (a momentum mask AND a reversion mask on this row) are tagged momentum,
     `momentum:<first momentum trigger>+ov`, regardless of `FAMILY_PRIORITY` (analyze G2). Rule B
     exists to keep momentum's pairs free, and priority alone would hand the pair to reversion.
- The per-pair sink file (C3) records `coexist_recency_refused`.
- MRO: `SimpleTriggerStrategy(FamilyCoexistenceMixin, FamilyExitMixin, StopInvariantMixin,
  BaseStrategy)`.
- **`_apply_preset` merges `FAMILY_EXIT_OVERRIDES` along the MRO** (analyze G1): reversed
  `cls.__mro__`, per family, per key, nearest class last. Each tuning config states only its own
  new parameter, and earlier adopted values are inherited, not lost.

## C3 — Sink records (no trade effect; written only when `FT_ENTRY_CONDITION_SINK` is set)

- **Rule B:** `<sink>/<pair_safe>_coexist.json` = `{"counts": {"coexist_recency_refused": {"pass": n, "total": rows}}}`.
  It is merged by the existing `aggregate_sink`.
- **Rule A:** append `{"rule": "pair_cap", "family": fam}` lines to `<sink>/coexist_events.jsonl`.
- **`regime/run_arm.py`:** `aggregate_coexistence(sink)` → record `coexistence_refusals`, omitted
  when empty.

## C4 — Fingerprint

`FAMILY_PAIR_CAP` and `MOMENTUM_RECENCY_BARS` are added to the 024 optional family members only
when non-neutral. Tests:
- arm D, `SimpleTrigger_Gbb` and `SimpleTrigger_FamCombined` keep their shas;
- each active setting gives a distinct sha.

## C5 — Stage runner and decisions (`evaluation/stage.py`)

```
python -m evaluation.stage --stage <name> --previous <tag> --candidates <tag1,tag2,...> \
    --singles reversion=st_gbb,momentum=st_bgo_mx --out specs/025-family-coexistence/decisions.md
```

- For each candidate tag (plans already run):
  - pool it;
  - the early whole-arm sum (wallet), reversion-side early contribution (from `enter_tag`),
    lost-to-other (`evaluation.contention`) and the degenerate flag (R4).
- Choose per R4/R5, run `halves` on chosen vs previous (`PYTHONHASHSEED=0`), and append a
  StageDecision block to `decisions.md`.
- **Selection key (G4):** the primary key is the whole-arm f00–f07 wallet sum. The tie-break is
  `--stage coexistence` → fewer entries lost to the other family (f00–f07), and `--stage tuneN` →
  higher reversion-side f00–f07 contribution.
- Exit codes: 0 adopted, 3 not adopted (late verdict not IMPROVED), 2 missing input.
- The selection uses ONLY f00–f07 fields. A unit test plants a candidate that is best late and
  worst early, and asserts it is not chosen.
