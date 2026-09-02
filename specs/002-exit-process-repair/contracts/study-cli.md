# Contract: Study CLI and Artifacts

**Feature**: 002-exit-process-repair | **Date**: 2026-09-02

Scripts under `regime/exit/`, run from the freqtrade root:

```
PYTHONPATH=. .venv/bin/python user_data/strategies/regime/exit/<script>.py [options]
```

`pandas` is absent from the system interpreter, so `.venv/bin/python` is required.

## Common conventions

| Convention | Rule |
|---|---|
| Exit 0 | Ran to completion. A "nothing improves it" result also exits 0 — it is a finding. |
| Exit 1 | Could not complete: missing archive, baseline mismatch, unreadable data. |
| Exit 2 | A gate rejected the run: the screen failed to reproduce the baseline. |
| Artifacts | Written under `regime/exit/`; never console-only. |
| Determinism | Same inputs produce identical artifacts. |
| No model | Phases 1–3 MUST NOT load a model or train. |

---

## `baseline.py`

Not a CLI — the pinned baseline (`BaselineConfig`) plus archive paths, imported
by everything else.

**Contract**: values are literals in source. Reading them from
`NNNC/NNNC_MLX.json` is prohibited — that file already drifted mid-feature and
no longer holds the baseline. On import it verifies the archives resolve and
the per-window trade counts are 51 / 96 / 20; a mismatch raises rather than
silently characterising a different configuration.

---

## `characterise.py`

Phase 1. Exit-mix breakdown, stop attribution, and counterfactual excursions.

| Option | Default | Meaning |
|---|---|---|
| `--extensions` | `1,3,6,12,24` | Holding extensions in hours |
| `--out` | `regime/exit/` | Output directory |

**Produces**: `exit_mix.csv`, `stopped_trades.csv`, `counterfactual.csv`, and a
verdict line on stdout.

**Postconditions**: profitable and losing stops reported as separate populations
throughout; trades excluded from an extension for insufficient history are
counted and reported; no model is loaded.

---

## `replay.py`

Phase 3. The counterfactual screen.

| Option | Default | Meaning |
|---|---|---|
| `--candidates` | all | Candidate names to score |
| `--validate-only` | off | Run only the baseline-reproduction check |

**Precondition (FR-014)**: replaying the CURRENT rule MUST reproduce the recorded
baseline outcomes before any candidate is scored. Exit code 2 if it does not —
a screen that cannot reproduce the known answer will mis-score every candidate.

**Produces**: `screen_results.csv` with, per candidate, the change in total
outcome versus the recorded baseline and a `PASS`/`REJECTED` verdict.

**Contract**: a `REJECTED` candidate MUST NOT proceed to a backtest (FR-016).
The screen models the exit stack only — not position limits, one-open-trade-per-pair
blocking, or capital freed by an earlier exit — so its disagreement with Phase 4
is expected and is itself reported.

---

## `run_arm.py` (existing, MODIFIED)

Phase 4. Currently overrides only buy-space parameters via `--set`.

**Required extension**: sell-space overrides, plus the `roi` table and
`stoploss` block, which live outside `params.buy` in the strategy JSON.

**Contract additions for this feature**

- The exit configuration under test is written into the per-arm JSON copy; the
  user's `NNNC/NNNC_MLX.json` is never modified.
- `entry_config_hash` is emitted and MUST match the baseline — entry is frozen.
- `exit_mix` is emitted per arm, so category-shifting is visible.
- An arm whose `trades_hash` equals the baseline's is `INERT` — a defect, not a
  result about the intervention.

---

## Artifact schemas

### `exit_mix.csv`
`window, exit_reason, n, share, mean_pct, median_pct, total_pct, median_dur_h, win_rate`

### `stopped_trades.csv`
`pair, window, open_date, close_date, age_h, profit_ratio, initial_stop_pct, final_stop_pct, stop_above_exit, ended_in_profit`

**Contract**: `initial_stop_pct` is expected to be −5.00% on every row. Any other
value means the grace level became reachable and invalidates research R1 — halt
and re-derive rather than reporting.

### `counterfactual.csv`
`pair, open_date, ended_in_profit, extension_h, best_pct, worst_pct, would_reach_take_profit, would_reach_roi, sufficient_history`

### `screen_results.csv`
`candidate, kind, motivating_measurement, baseline_total_pct, candidate_total_pct, delta_pct, verdict`

**Contract**: `motivating_measurement` is non-empty for every row. A candidate
without one is not a candidate (FR-009).
