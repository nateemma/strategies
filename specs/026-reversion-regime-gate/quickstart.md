# Quickstart: Regime Gate on the Reversion Family (026)

From `user_data/strategies`. Every result is PROVISIONAL (gate 2).

## 1. Causality (SC-001)

```bash
../../.venv/bin/python -m pytest -q utils/tests/test_regime_state.py
```

**Expected:** all 4 instruments pass truncation at 16 cuts; the planted centred-mean control fails.

## 2. Feasibility, US1 (no backtests)

```bash
../../.venv/bin/python -m research.regime_gate.feasibility --primary st_fam_combo --cross st_gbb \
    --out specs/026-reversion-regime-gate/feasibility.md
```

**Expected:** each instrument has its early-chosen rule, per-state/per-half reversion pp, off-bar
shares, run counts, and PASS/STOP. **Exit 3 means STOP:** write report.md and end the feature.

## 3. US2 (only if ≥1 PASS)

1. `SimpleTrigger_RegGateOff` (`REVERSION_REGIME_GATE = None`, subclass of
   `SimpleTrigger_FamCombined`) must reproduce `st_fam_combo` 15/15.
2. One `SimpleTrigger_RegGate<Inst>` per passing instrument, with the rule from `feasibility.json`.
3. Run, then:

```bash
../../.venv/bin/python -m evaluation.stage --stage regime --previous st_fam_combo --bar st_bgo_mx \
    --candidates <tags> --singles reversion=st_gbb,momentum=st_bgo_mx \
    --out specs/026-reversion-regime-gate/decisions.md
```

**Expected:** non-zero `regime_gate_refused` on every gated arm, and adoption only if late IMPROVED
against `st_bgo_mx`.
