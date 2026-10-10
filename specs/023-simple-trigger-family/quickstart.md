# Quickstart / Validation Guide: SimpleTrigger (023)

These are the scenarios that prove the feature works. Run them from `user_data/strategies/` with
`../../.venv/bin/python`. Run **one backtest chain at a time** (project rule: parallel runs caused
OOM and Metal faults). Interfaces: [contracts/interfaces.md](contracts/interfaces.md).

## 0. Unit gate (minutes)

```bash
../../.venv/bin/python -m pytest SimpleTrigger/tests research/outcomes/tests evaluation/tests -q
```

Expected: everything passes. That includes:
- presets are total;
- param dicts drop no keys;
- the label map covers all 29 `LabelMethod` members;
- coupled mode with matched targets resolves to the decoupled values exactly;
- the stop invariant raises on a planted sign flip.

## 1. Causality (US1, SC-006)

```bash
../../.venv/bin/python -m research.outcomes.causality --write
```

Expected:
- `trigger_causality.json` has a row for every registered trigger;
- the `local_min_followthrough` control is `FAIL` (exit 0; exit 1 means the checker is broken);
- `trends` and `bands` are `UNSCOREABLE`.

Then confirm that a SimpleTrigger subclass naming a non-PASS trigger enters 0 trades and prints its
`***` reason.

## 2. Reproduce 021/022 (US1, SC-001)

Seven arms under `PRESET = "LEGACY_TRIGGERONLY"`: `breakout`, `breakout_vol`, `breakout_bb`,
`breakout_gbb`, `geometry`, `gbb`, and `gbb` with the `wf_gbbr` guards.

```bash
../../.venv/bin/python -m evaluation.foldplan derive --from evaluation/folds/plan_NNNC_A_Raw_Open.json \
    --family SimpleTrigger --base SimpleTrigger_Legacy_Gbb --tag st_legacy_gbb \
    --out evaluation/folds/plan_st_legacy_gbb.json
../../.venv/bin/python -m evaluation.run_folds --plan evaluation/folds/plan_st_legacy_gbb.json
```

Expected: every per-fold `trades_hash` equals the recorded `regime/arms/wf_<tag>_fNN_FNN.json`
value, in 15 of 15 folds for every arm (`wf_gbbo`, `wf_bo`, `wf_bvo`, `wf_bbo`, `wf_bgo`,
`wf_geo`, `wf_gbbr`). `wf_gbbr` totals 458 trades. Nothing appears under `saved_data/`.

## 3. Dry-run isolation (SC-008)

Diff the resolved parameters of `Basket/MomentumRegimeBasket15mFast` before and after the change.
Expected: an empty diff.

## 4. Condition sink is neutral (C4)

Run `wf_gbbo` f00 with `FT_ENTRY_CONDITION_SINK` unset and set. Expected: identical `trades_hash`,
with the sink files present only in the second run.

## 5. Twin equivalence (US3, SC-003)

```bash
../../.venv/bin/python -m evaluation.fingerprint --strategy SimpleTrigger_Gbb
../../.venv/bin/python -m evaluation.fingerprint --strategy NNNC_A_Raw_Open
# run SimpleTrigger_Gbb (ARM_D_OPEN) and NNNC_TriggerGate over the arm D plan
../../.venv/bin/python -m evaluation.twins check --nnnc-tag wf_aro --simple-tag st_gbb --equivalence-tag wf_trgate
```

Expected:
- the two `framework_sha` values are equal;
- `check` exits 0, meaning every fold's `trades_hash` matches between `NNNC_TriggerGate` and
  `SimpleTrigger_Gbb`;
- on failure, the first divergent trade and the condition-count diff localise the parity defect.
  Fix the defect before continuing; do not proceed to step 6.

## 6. Model contribution (US3, SC-004)

```bash
../../.venv/bin/python -m evaluation.twins compare --nnnc-tag wf_aro --simple-tag st_gbb
```

Expected: a paired delta with its CI, early (f00–f07) and late (f08–f14) halves, and the
shared-trade fraction, or an explicit `INCONCLUSIVE_*` / underpowered verdict. The reference is the
post-fix arm D: 1420 trades, +42.71pp, 14/15 (research R2).

## 7. Component attribution (US2, SC-005)

```bash
../../.venv/bin/python -m evaluation.attribution --full-tag st_gbb --components all
```

Expected:
- one row per component;
- guards are `ALREADY_NEUTRAL` under `ARM_D_OPEN`;
- every other row is either `LIVE`, with its moved metric, or `INERT`, flagged as a defect.

Under `ARM_D_OPEN` the stop components (`stop_atr_band`, `stop_grace`, `stop_volume_confirm`) must
be LIVE. Under `LEGACY_TRIGGERONLY`, `stop_atr_band` and `stop_volume_confirm` are expected to be
INERT. That is the research R1 finding, now measured.

## 8. Coupled mode (US4, SC-007)

Run `SimpleTrigger_Gbb` with `EXIT_MODE="coupled"` and label targets chosen to map onto the
decoupled values. Expected:
- `trades_hash` is identical to step 5's run in 15/15 folds;
- `coupled_resolution.coupled_stop == "none-in-label"` for gbb;
- zero `StopInvariantError` across every run in steps 2–8.
