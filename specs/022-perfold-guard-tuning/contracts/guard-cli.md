# Contract: guard tuning and trackability commands

Both are offline research tools. Neither trains a model, neither runs inside a strategy, and
neither may write to a live params file.

## `evaluation/pick_guards.py`

Scores the enumerated grid on fold **training** windows and emits a selection map.

```
PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/evaluation/pick_guards.py \
    --plan user_data/strategies/evaluation/folds/plan_GbbGuardGrid.json \
    [--fold f00] \
    --out user_data/strategies/evaluation/folds/guards_gbbgrid.json
```

**Guarantees**

- Loads data for a fold **only** through `pick_threshold.TrainWindow.from_fold`, whose
  assertion is that the window ends at or before the fold's test window starts. There is no
  code path in this tool that accepts a bare timerange or a raw fold dict for loading, so
  `test_timerange` cannot be reached by accident. *(This guarantee is inherited, not
  reimplemented.)*
- **Measures only.** Writes a `{fold_id: GuardConfig}` map and never applies it. The value
  reaches a run through `run_folds.py`'s `guards` plan key, written into the fold's generated
  `.json` derived from the frozen params snapshot.
- Refuses to emit a configuration whose `n_taken < 25` (FR-012); if a fold has no admissible
  configuration it is written as `unpowered` rather than filled with a default.
- Records `exit_config_hash` on every score and aborts if it differs from the arms' hash.
- `--window test` exists **only** for `guard_trackability.py` and is refused unless that flag
  is passed, so a test-window score cannot leak into selection.

**Exit codes**: `0` map written · `2` exit-hash mismatch · `3` every fold unpowered.

## `evaluation/guard_trackability.py`

The Principle VII gate.

```
PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/evaluation/guard_trackability.py \
    --plan .../plan_GbbGuardGrid.json \
    --out .../trackability_gbbgrid.json
```

**Guarantees**

- Computes regret against `max_c S_i(c)` over **the enumerated candidates that actually
  existed** — never against a value outside the grid (FR-005).
- The constant is chosen in **hindsight** (maximising summed test score), deliberately
  generous: if adaptivity cannot beat a fixed value chosen with full knowledge, it is not
  trackable.
- Emits `verdict: trackable | not_trackable`. **`not_trackable` is a success exit (0)** — it is
  a complete finding under FR-006, not a failure.
- Reports `n_distinct_choices` separately from the verdict, so variation is never mistaken for
  trackability.

**Exit codes**: `0` verdict emitted (either value) · `2` proxy audit against arms A/B failed, so
no verdict is licensed.

## `evaluation/run_folds.py` — the one modification

New optional plan key, mirroring `thresholds` exactly:

```json
{ "guards": { "f00": {"entry_atr_pct": 0.013, "entry_adx_threshold": 10.0,
                      "entry_rvol_threshold": 1.9, "entry_enable_guards": true } } }
```

**Guarantees**: absent ⇒ behaviour unchanged, byte-for-byte, for every existing plan.
`entry_enable_guards` is forced `True` on write regardless of input (FR-008). Values are written
into the generated fold `.json` only.
