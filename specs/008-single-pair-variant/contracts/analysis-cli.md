# Contract: Pair-Restricted Comparison

One new command. Run from the repo root with `PYTHONPATH=user_data/strategies`.

## `regime/pair_restricted.py`

```
python user_data/strategies/regime/pair_restricted.py \
  --arm <variant_tag> --control <control_tag> --pair LINK/USDT --windows W1,W2,W3
```

Per window, reports:

1. **Selection (US2, FR-006)** — shared / identical / variant-only / control-only, with each
   differing trade's outcome.
2. **Expectancy (US3, FR-007)** — n, mean per-trade %, sd, bootstrap CI per arm; whether the
   intervals overlap.
3. **Delta (FR-008/009)** — total-return delta with CI and the applicable noise floor, marked
   **UNRESOLVED** when smaller than the floor.
4. **Guards** — zero-trade recorded as zero-trade (FR-011); win rate suppressed below n=20
   (FR-021); `exit_reason` stripped before comparison.

Reuses `compare.py`'s bootstrap via its trade-list entry point; it does not reimplement the
statistic (FR-018).

## Invariants

- Never modify `run_arm.py`, `compare.py`, `regimedata.py`, `regime/wf/` or `Framework/`.
- No gate, exit, label or architecture parameter differs between arms (FR-019, SC-008).
- A run whose ungated-gate warning fired is discarded, not reported (FR-004).
