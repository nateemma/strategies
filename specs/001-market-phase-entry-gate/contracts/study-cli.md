# Contract: Study CLI

**Feature**: 001-market-phase-entry-gate | **Date**: 2026-09-02

The study exposes its interface as command-line scripts, matching the `btc_corr/` convention: `__file__`-relative, re-runnable from a clean checkout with no manual path editing (FR-006).

All invocations are from the **freqtrade root** (`/Users/philprice95/projects/freqtrade`), not from the strategies directory.

```
PYTHONPATH=. .venv/bin/python user_data/strategies/regime/<script>.py [options]
```

`pandas` is absent from the system interpreter, so `.venv/bin/python` is mandatory (R11).

---

## Common conventions

| Convention | Rule |
|---|---|
| Exit code 0 | Ran to completion. Does **not** imply a favourable result — a FAIL verdict also exits 0. |
| Exit code 1 | Could not complete: missing data, failed validation, unresolvable configuration. |
| Exit code 2 | A gate rejected the run: lookahead validation failed, or a required prior artifact is absent. |
| stdout | Human-readable table, same style as `exp4_regime.py`. |
| Artifacts | Written beside the script under `regime/`; never console-only (FR-006). |
| Determinism | Same inputs produce byte-identical artifacts. Any randomness is seeded and the seed recorded. |

---

## `phase_a_classify.py`

Emits phase labels at all three levels.

| Option | Default | Meaning |
|---|---|---|
| `--level` | `all` | `asset` \| `market` \| `btc` \| `all` |
| `--instruments` | `all` | Comma-separated instrument names |
| `--timerange` | full history | `YYYYMMDD-YYYYMMDD` |
| `--out` | `regime/labels/` | Output directory |

**Produces**: one label artifact per instrument per pair, plus `regime/instrument_catalogue.csv`.

**Postconditions**: every emitted label is causal (FR-002); market-level bars below the coverage rule carry `UNKNOWN` (R4); the `date` column is tz-aware UTC and aligns bar-for-bar with the source OHLCV.

---

## `phase_a_lookahead.py`

Truncated-versus-full-history validation. This is the gate, not a report.

| Option | Default | Meaning |
|---|---|---|
| `--cuts` | three spread across the data | Truncation points as `YYYYMMDD` |
| `--instruments` | `all` | Which instruments to validate |

**Produces**: `regime/lookahead_validation.csv` — one row per instrument per cut, with the count of differing labels.

**Contract**: exit code 2 and a non-zero differing-label count for any instrument means that instrument MUST NOT proceed to Phase B. freqtrade's own `lookahead-analysis` is not an acceptable substitute (R9).

---

## `phase_b_characterize.py`

State distribution, dwell, and transitions per instrument per era.

| Option | Default | Meaning |
|---|---|---|
| `--eras` | `W1,W2,W3` | Named windows from research R3 |
| `--dwell-source` | required | Path to the host trade export used to derive the dwell threshold |

**Precondition**: `--dwell-source` must be supplied. The threshold is derived from the **host's own** measured trade-duration distribution; there is no default, deliberately, so that a figure from another strategy family cannot be borrowed by accident (R6).

**Produces**: `regime/actionability.csv`, `regime/transitions_<instrument>.csv`, `regime/dwell.png`.

**Postconditions**: every instrument carries `SURVIVING` or `ELIMINATED` with a quoted measured reason (SC-003). Instability may only be claimed from a W1-versus-W3 difference, never W1-versus-W2 (Principle II).

---

## `phase_b_risk_normalize.py`

Re-analysis of the prior taxonomy's per-state drawdown, normalized for time-in-state.

**Produces**: `regime/risk_normalized.csv` and a one-line verdict on stdout stating whether the previously observed roughly 4× risk discrimination is **CONFIRMED** or **REFUTED** (FR-012, SC-004).

---

## `phase_c_trackability.py`

The stop gate.

| Option | Default | Meaning |
|---|---|---|
| `--horizons` | `H` grid | Forward horizons in bars |
| `--arms` | required | Trade exports for the aggressive and conservative configurations |
| `--constant` | `0.69` | The fixed configuration to beat |

**Precondition**: `--arms` must reference realized trade exports. The script MUST refuse to run against passive price series — this is the specific substitution that invalidated the prior BTC gate (FR-013).

**Produces**: `regime/trackability.csv` (one row per instrument × window) and `regime/VERDICT.md`.

**Contract**: prints exactly one of `VERDICT: PASS <instrument> <level> H=<n>` or `VERDICT: FAIL`. On FAIL, no downstream mechanism script may run (FR-017). Both outcomes exit 0 — a FAIL is a completed measurement, not an error.

---

## `run_arm.py`

Backtest runner for Phase C and D arms.

| Option | Default | Meaning |
|---|---|---|
| `--strategy` | required | Host strategy class |
| `--window` | required | `W1` \| `W2` \| `W3` — resolves to a literal pinned date range |
| `--threshold` | required | `prediction_threshold` for this arm |
| `--tag` | required | Arm label used in output filenames |

**Contract**

- Resolves `--window` to a literal `YYYYMMDD-YYYYMMDD` and passes it via `-t`. Today-relative `-n`/`-o` are never used (R8).
- Routes `--threshold` through the `<Strategy>.json` parameter file, because that file **overrides** class-level `buy_params` (R5). Writing the value only in Python would silently do nothing.
- Leaves the user's `NNNC/NNNC_MLX.json` unmodified — it writes a per-arm copy. That file is untracked working state.
- Runs with volume/liquidity constraints active (FR-028).
- Exports realized trades to a per-arm path.
- Emits the three liveness counts: model-condition pass count, full-chain pass count, trade count (R7).

**Produces**: `regime/arms/<tag>_<window>.json` conforming to the `MechanismArm` schema in [artifacts.md](./artifacts.md).

**Postcondition**: if `chain_pass_count` equals the baseline's, the arm is marked `INERT` and its trading figures are withheld from any comparison table (FR-031).
