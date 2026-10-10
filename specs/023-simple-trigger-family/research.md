# Research: SimpleTrigger Strategy Family (023)

Phase 0 output. Every decision below comes from reading the code at HEAD `10cbd6a` (2026-10-10).
Line references are to that revision.

## Findings that change the brief

These came out of the research, and the plan is built around them.

### R1. 021/022's no-model arms did NOT use arm D's stop

`SimpleStrategies/TriggerOnly.py:77-90` claims "the Open operating point, verbatim". That is true for
the exits and false for the stop:

| setting | arm D (`NNNC_A_Raw_Open`) | `TriggerOnly` (all 021 arms, `wf_gbbr`, `wf_gbbg`) |
|---|---|---|
| `use_atr_adaptive_stoploss` | **True** (`Framework/BaseNNStrategy.py:145`) | **False** (`TriggerOnly.py:87`) |
| ATR mult / floor / cap | 4.0 / −0.06 / −0.04 (`NNNC/NNNCStrategy.py:58-60`) | 2.5 / −0.04 / −0.02 (inert) |
| volume-confirmation stop | live | **inert**: `utils/ExitPolicy.py:442-443` returns the static stop when adaptive is off |
| effective stop after grace | ATR band, rvol-tightened | static −0.05 |
| `startup_candle_count` | 64 (`BaseStrategy.py:250`) | 200 (`TriggerOnly.py:65`) |
| `max_open_trades` | 10 (JSON) | from config |

**Consequence.** "Trigger alone +2.64pp" against "arm D +39.98pp" mixes the model's effect with a
stop difference. No existing number isolates the model. US3 is therefore new evidence, not a
re-measurement.

**Decision.** Two named presets (see data-model `FrameworkPreset`):
- `ARM_D_OPEN`: the default (spec FR-009). The resolved arm D framework.
- `LEGACY_TRIGGERONLY`: TriggerOnly's resolved values verbatim. It exists only so SC-001 can
  reproduce the recorded 021/022 hashes.

The SC-001 reproduction runs under `LEGACY_TRIGGERONLY`. The US3 twin runs under `ARM_D_OPEN`.
Each run record names its preset.

### R2. Arm D's headline number is pre-fix

`+39.98pp, 15/15` comes from `evaluation/pooled_rung_aro_tp0.022.json`, which has records with
`exit_config_hash c8f80f6af046d088`. That number predates `09db623` (2026-10-08 17:02, the
post-grace sign fix, which changed 191 of 1420 trades). The current-code arm D is
`evaluation/pooled_NNNC_A_Raw_Open.json` / `regime/arms/wf_aro_fNN_FNN.json` (written 2026-10-08
17:20, after the fix): **1420 trades, +42.71pp per-fold sum, 14/15 positive** (f08 −0.45pp),
exit hash `5828946b58ec7971`.

**Decision.** The US3 reference is the post-fix `wf_aro` set. The +39.98pp figure is not quoted
as arm D in any 023 artefact. Project memory needs the same correction.

### R3. The 021/022 records were written after the fix

`wf_gbbo` (2026-10-09) and `wf_gbbr` (2026-10-10) both post-date `09db623`, so their per-fold
`trades_hash` values are reproducible targets on current code. Every per-fold record exists except
`wf_gbbg` f10, which has 0 trades and no hash, and is reported as zero-trade (spec edge case).

### R4. `exit_config_hash` proves nothing for class-attribute arms

`regime/run_arm.py:211-230` hashes only the params JSON's sell/roi/stoploss/trailing plus
`SIMULATION_LOGIC_VERSION`. The trigger arms have no JSON, so every one of them hashes
`{"sell":{},"roi":null,...}` and gets `5a18305d1fafa83e`. Equal hashes there mean "no JSON
overrides", not "same exit". No buy-side fingerprint exists anywhere.

**Decision.** Add a **framework fingerprint**: a hash of the *resolved* framework (class
attributes merged with JSON, plus the run config's `max_open_trades`, stake and fee), computed by
instantiating nothing. It reads the class `__dict__` chain and applies the same JSON merge
freqtrade does (`freqtrade/strategy/hyper.py:86-100`). `run_arm` records it on every arm, NNNC
included. The twin check (FR-013) compares fingerprints. `exit_config_hash` stays as it is, for
continuity.

*Alternatives rejected*: (a) extending `exit_config_hash` would change every historical hash and
break comparability with recorded arms; (b) instantiating the strategy would trigger NNNC's model
load/training (`BaseNNStrategy.py:1374-1380`).

### R5. `run_folds.py` has no no-model path

Read from the code, not run:
- `foldplan.validate()` (`evaluation/foldplan.py:69`) rejects duplicate classes.
- `generate_fold()` (`run_folds.py:126`) would overwrite a base class with a self-referencing
  template.
- `main()` always calls `train_fold`.

The 021 plans (`strategy_class` = the base name on all 15 folds) could therefore not have run
through `run_folds` as they are written.

**Decision.** Add a plan key `"no_model": true`. When it is set:
- every fold uses the base class (validate allows the duplicate class when `model_dir` is null);
- `generate_fold`, `model_exists` and `train_fold` are skipped;
- `run_folds` asserts the class has `needs_training == False` before evaluating, which rules out
  the renamed-strategy self-training hazard structurally.

`foldplan.build()` gains a `family="SimpleTrigger"` path that emits `model_dir: null`.

### R6. Tag glob collision

`pool.arm_record` (`evaluation/pool.py:47`) and `run_folds.arm_record` (`:211`) glob
`{tag}_*.json`, so `wf_nnnc_f00` also matches `wf_nnnc_f00_s7_F00.json`, with no defined order.

**Decision.** Resolve by exact name `{tag}_{label}.json`. This is a small hygiene fix, included
because attribution (US2) runs many sibling tags.

### R7. There are no live pass-rate diagnostics

`Framework/StrategyDiagnostics.py` has ML metrics only, and `BaseStrategy.populate_entry_trend`
(`:903-975`) records nothing. Exit reasons exist per trade in the backtest archive and in pooled
rows (`evaluation/pool.py:73-79`), but not in arm records.

**Decision.**
- **Entry liveness**: add a generic, strategy-agnostic condition recorder to
  `BaseStrategy.populate_entry_trend`. Each named mask (`signal`, `volume`, `rvol`, `adx`,
  `atr_pct`, `bb_width`, `close_norm`, `guard`) is counted per pair. The counts go to a JSON sidecar
  **only when** env `FT_ENTRY_CONDITION_SINK` is set. `run_arm` always sets it, per run, so every
  harness run records liveness (FR-011). Outside `run_arm` it is off by default. The recorder
  doesn't change behaviour, which is proven by an identical `trades_hash` with the flag on and
  off.
- **Exit liveness**: `run_arm.summarize` adds `exit_reason_counts` from the archive.

This keeps Framework/ strategy-agnostic (constitution): the recorder names conditions, not
strategies.

*Alternative rejected*: an offline re-evaluation of guards from feathers. It duplicates the guard
logic and would drift, which is the defect class 021 removed.

### R8. Label targets and exit targets are independent today

- Label side: `NNNC/LabelParamsMixin.py:42-58` (`min_gain` 0.02 via `NNNCStrategy.py:228`).
- Exit side: `cexit_take_profit` (`BaseStrategy.py:562`, NET of fees, `ExitPolicy.py:742-744`).
- The gbb **buy** label has no loss term (`TrainingSignals.py:1073`: "min_loss ignored for buy"),
  and `max_drawdown` is a labeler default that no strategy passes.

**Decision: coupled mode is a parameter mapping, not new exit logic** (FR-016):

| exit field | coupled value | when the label has no such target |
|---|---|---|
| `cexit_take_profit` (net) | `net_from_price_ratio(1 + min_gain, fee_per_side)` | n/a (every buy label has a gain target) |
| `cexit_max_days` | `horizon × timeframe` in days | unchanged |
| stop | `−loss_target` as a static stop, adaptive OFF, when the label defines a loss barrier (`forward_mae` max_drawdown, `triple_barrier` lower barrier) | **unchanged**, and the record states `coupled_stop: "none-in-label"` |

The mapping lives in the SimpleTrigger family (`SimpleTrigger/coupling.py`) and produces values
that flow through the one `BaseStrategy._exit_params` (`:1050`) → `ExitParams` path. Nothing in
`utils/ExitPolicy.py` changes.

*Alternatives rejected*:
- A coupled branch inside `ExitPolicy` duplicates exit logic per mode (FR-016).
- Gross take-profit: the unit mismatch would be exactly the class of defect this protects against.

SC-007's identity test sets label targets to the values that map back to the decoupled exit, and
asserts an identical `trades_hash`.

### R9. Stop invariants have no runtime check

- The post-grace sign fix (`BaseStrategy.py:1136-1146`) is pinned only by
  `Framework/tests/test_post_grace_stop_signfix.py`.
- The floor bound is guaranteed only by the clamp at `ExitPolicy.py:446-448`.
- The "mis-scaled floor" (`a7cb6da`) was a **trade-count floor** in `evaluation/pick_guards.py`,
  not a stop floor. `floor_for_window()` (`:57`) has **zero callers**, and line 177 still applies
  the absolute 25. That is out of 023's scope; it is noted for 022's owner.

**Decision.** A strategy-agnostic `StopInvariantMixin` in `utils/StopInvariants.py`. Its
`custom_stoploss` calls `super()` and then checks, per call:
(a) ratio ≤ 0;
(b) the implied stop price ≤ `current_rate`;
(c) after grace, the implied stop **price** is no lower than `open_rate × (1 + atr_stoploss_floor)`
    (adaptive) or `open_rate × (1 + stoploss)` (static). This is checked in price space, because
    freqtrade's ratio is relative to `current_rate`, so a correct entry-anchored stop on a winning
    trade has a ratio more negative than the floor;
(d) direction: for a trade still above its stop, the first post-grace stop price is not lower than
    the last in-grace one.

A violation raises `StopInvariantError` (FR-017: it halts the run). It is mixed into
`SimpleTriggerStrategy` and `NNNC_TriggerGate` only, so the NNNC base classes are untouched
(sibling decision, Q1). The FR-007a merge moves it into the shared base. It is built in the
foundational phase so it is active in every SimpleTrigger and twin run from the start.

### R10. Label to trigger mapping, and one stale record

Live registry: 19 triggers (`python -m research.outcomes.triggers --list`). The mapping of the 29
`LabelMethod` members comes from `research/outcomes/tests/test_labelmethod_inventory.py` and
`specs/021-label-exit-coordination/label-method-audit.md`:

| trigger | label methods |
|---|---|
| `empty` | forward_mae, triple_barrier, quantile_future, mfe_mae_ratio, risk_adj_future, multi_horizon_vote, optimal_signals, local_extrema |
| `gbb` | gbb, gbb_path, gbb_fwdmae |
| `breakout*` | breakout, breakout_tb, breakout_vol, breakout_squeeze, breakout_bb, breakout_gbb |
| `votes_*` | technical_indicators, indicators2, indicators3, indicators4, breakout_consensus |
| `geometry` | (causal substitute; no port) |
| `adx` | future_sharpe, future_sortino, future_expectancy, **CONFLICT, see below** |
| `trends`, `bands` | mapped but **unscoreable**: required columns are absent from the populator |
| **unsupported** | local_min_followthrough: centred window reads 3 bars ahead |

**Conflict.** The audit doc (`label-method-audit.md:69-71`) and the registry's `adx.source` say the
three `future_*` labels reduce to `adx`. The inventory test (`:74-76`, asserting 11 members at
`:154`) puts them in `OUTCOME_ONLY`.

**Decision.** Mark the three as `unsupported (record conflict)` until a task reads the labelers and
fixes whichever record is stale. Mark `trends` and `bands` as unsupported (unscoreable).
Unsupported entries are listed, never silently mapped to `empty` (FR-004).

### R11. Causality must be measured, not argued

The registry blocks outcome and pair columns, but cannot see forward reads.
`triggers.py:239-257` documents that pandas-swallowed warm-up NaNs evade its NaN guard. Perturbation
tests exist for only three triggers.

**Decision.** Add `research/outcomes/causality.py`, using the truncation method of
`research/labelfree/metalabel/audit.py:50-102`:
- compute the populator features and the trigger on the full history, and on prefixes cut at K
  points;
- the outputs must be bit-identical (RTOL 0) over each prefix, excluding a declared warm-up;
- run it per trigger, per pair, on 3 pairs × 3 eras;
- a planted control (`local_min_followthrough`) must FAIL, or the checker itself is broken
  (liveness, Principle IV).

The result goes to `research/outcomes/trigger_causality.json`. SimpleTrigger refuses any trigger
that is not `PASS` there.

### R12. The equivalence check needs the real NNNC path

An NNNC subclass that overrides only `get_entry_conditions` still needs a loaded model and scaler
(`BaseNNStrategy.populate_indicators` raises without the main scaler). Under a new class name it
would also self-train (`get_model_path`, `:341-346`).

**Decision.** `NNNC/NNNC_TriggerGate.py` subclasses `NNNC_A_Raw_Open`, with
`NNNC/NNNC_TriggerGate.json` a copy of `NNNC_A_Raw_Open.json` (only `strategy_name` changed). Without
that JSON, freqtrade would resolve `NNNCStrategy`'s class defaults instead of the Open values, and the
fingerprint check would fail by construction. The class:
- `get_entry_conditions` calls `super()` (which keeps `predict_*` columns) and then returns the
  registry trigger;
- `get_model_path` is redirected to the arm D fold model `saved_data/NNNC_ARO_fNN/`, which exists
  for all 15 folds;
- per-fold classes come from the existing NNNC plan path, with an `ARO` model-dir override and
  `needs_training` forced False, so a missing model aborts instead of training.

SC-003 then compares the `NNNC_TriggerGate` trade set with the SimpleTrigger `ARM_D_OPEN` twin per
fold. It doesn't need to compare with arm D itself, because arm D has no trigger at entry.

Any mismatch is a parity defect, localised by diffing:
- the fingerprints;
- the condition-sink pass rates (does `BaseNNStrategy.add_additional_indicators` change a guard
  column?);
- the first divergent trade.

### R13. The statistical tooling exists, apart from the halves split

We reuse:
- `evaluation/bootstrap.py:82` `bootstrap_delta` (paired, with `MIN_DIFF_TRADES=20`);
- `evaluation/compare_pooled.py:27` `compare`, which gives verdicts including `INCONCLUSIVE_ABSORBED`
  and `_UNDERPOWERED` and covers FR-014's "say so, don't verdict";
- `evaluation/power.py:39` `report`;
- the noise constants in `evaluation/__init__.py`.

Missing: an early/late split (022 did it by hand, `report.md:436-438`) and `guard_trackability.py`,
which `run_folds.py` refers to but which doesn't exist.

**Decision.** Add `evaluation/halves.py`: f00–f07 / f08–f14, the same cut as 022, configurable,
reusing `bootstrap_delta` per half. `guard_trackability.py` is out of scope; a note is filed.

**Decision: add `evaluation/matched_count.py`.** The twin and arm D share few trades by design
(arm D enters on the model's opinion, the twin on the trigger), which is exactly what made 022's
paired test fail (111 of about 8,500 shared, p = 0.27). 022 resolved its case with a matched-count
random-draw test (+5.2 SD), but no tool for it exists in `evaluation/`. The new module draws
`len(selected)` trades from the twin's population many times and reports where the model's
selection sits. It doesn't depend on shared trades. `twins compare` includes it.

## Settled without research

- **Family location**: new top-level dir `SimpleTrigger/`, beside `SimpleStrategies/`, so
  `run_arm.strategy_group` resolves it. It doesn't inherit from `SimpleStrategy` (spec FR-005).
- **TriggerOnly**: left in place and unchanged. Its subclasses keep running. SimpleTrigger
  supersedes it for new work (FR-022).
- **Dry-run isolation**: `Basket/MomentumRegimeBasket15mFast` inherits freqtrade `IStrategy`, not
  `BaseStrategy`, so a new family dir can't reach it. FR-020 is satisfied structurally, and SC-008
  verifies it with a resolved-params diff.
- **The shared-dict hazard**: `ft_load_hyper_params` deep-merges JSON into class dicts in place,
  so SimpleTrigger re-declares `buy_params`/`sell_params` as `{**parent, ...}` (constitution's
  merge rule). A unit test asserts no key is dropped (FR-019).
