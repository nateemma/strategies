# Interfaces: Recovery-Aware Reversion Exits (028)

## I1 — `research/exit_coupling/confirm.py` CLI

```text
python -m research.exit_coupling.confirm feasibility --prereg specs/028-recovery-aware-exits/preregistration.json \
    --out specs/028-recovery-aware-exits/feasibility.md      # writes .md and .json
python -m research.exit_coupling.confirm judge --prereg ... --feasibility .../feasibility.json \
    --out specs/028-recovery-aware-exits/verdicts.md         # writes .md and .json
```

**Both subcommands refuse, with exit 2, if any of these hold:**
- `preregistration.json` has uncommitted changes (the `.md` is not checked);
- a class constant differs from its frozen value;
- a 2022 record is older than the pre-registration commit;
- two arms share a class.

**Exit codes:**
- `feasibility`: 0 at least one PASS; 3 none (STOP); 2 a refusal, a missing input or fidelity < 0.90.
- `judge`: 0 at least one ADOPT; 1 none ADOPTED; 2 a refusal or a missing input.
  - It lists every ADOPTED candidate whose gain is under 1 pp as needing the gate-5 re-run.
  - `judge --rerun-check` then compares the `_rr` record's `trades_hash` and finalises
    `rerun_hash_match`.

**Pure functions** (unit-tested, no I/O):
- `recovery_share(df) -> (share, n_stop, n_recovered)`
- `candidate_passes(gain_pp, share, floor, applies_floor) -> bool`
- `map_paired(verdict) -> "pass" | "fail" | "inconclusive"` and `map_risk(verdict)`, the R4 mapping table
- `decide(gain_pp, paired_verdict, risk, early_gain, late_gain, ctl_trades, folds_changed, prereg) -> (verdict, failed)`
- `ci_half_width_pp(ci_pp_yr, years) -> float`

## I2 — `ExitParams` switches (US3, only if adopted)

```python
ExitParams(..., stop_min_depth: float | None = None, take_profit_atr_mult: float | None = None)
ExitPolicy.apply_stop_min_depth(stop: float, params) -> float        # min(stop, depth) when set
ExitPolicy.atr_take_profit(k: float, atr_signal: float) -> float     # clamp(k*atr, 0.01, 0.05)
BaseStrategy._apply_atr_take_profit(params, trade) -> ExitParams     # per-trade resolution

# As built (2026-10-12): only C2 was ADOPTED, so only take_profit_atr_mult exists;
# stop_min_depth / apply_stop_min_depth were not built (C1 stopped at feasibility).
# The take-profit can be declared strategy-wide (`take_profit_atr_mult = k`) or per family
# (FAMILY_EXITS / FAMILY_EXIT_OVERRIDES {"<family>": {"take_profit_atr_mult": k}}), not both.
```

**Declared via:**
- `FAMILY_EXITS = {"reversion": {...}}`, which merges; or
- a strategy-level attribute read in `_exit_params`.

**Invariant:** each field's None leaves every code path byte-identical to today.

## I3 — 2022 fold plans

`evaluation/folds/plan_st_e22_<id>.json`: `no_model: true`. Folds e00–e02 per R1, each with
`strategy_class` set to the candidate class and `arm_tag` `st_e22_<id>_eNN`. They are validated by
`foldplan.validate_plan` and run with `python -m evaluation.run_folds --plan ...`.
