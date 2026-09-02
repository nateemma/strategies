# Quickstart: Validating the Market-Phase Framework

**Feature**: 001-market-phase-entry-gate | **Date**: 2026-09-02

Runnable validation scenarios proving each phase works end to end. Implementation belongs in `tasks.md`; this is the run-and-check guide.

Interfaces are defined in [contracts/study-cli.md](./contracts/study-cli.md); output schemas in [contracts/artifacts.md](./contracts/artifacts.md); entity rules in [data-model.md](./data-model.md).

---

## Prerequisites

```bash
cd /Users/philprice95/projects/freqtrade
```

| Requirement | Check |
|---|---|
| Virtualenv interpreter | `.venv/bin/python -c "import pandas, numpy, talib"` — the system Python has no pandas |
| 15m OHLCV present | `ls user_data/data/binanceus/*_USDT-15m.feather \| wc -l` → 77 |
| BTC span | `BTC_USDT-15m.feather` covers 2021-01-01 → 2026-08-31 |
| Host strategy | `user_data/strategies/NNNC/NNNC_MLX.py` exists; a trained model is available |

> **Before you start**: a live dry run (`MomentumRegimeBasket15mFast`) has been executing since 2026-08-23. Phases A–C touch no strategy code and cannot disturb it. Do not modify shared `Framework/` files while it runs.

---

## Scenario 1 — The classifier emits causal labels (User Story 1)

```bash
PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_a_classify.py --level all
```

**Expect**: `regime/instrument_catalogue.csv` listing every instrument with `status = PENDING`, exactly one row with `is_baseline = True`, and label artifacts under `regime/labels/`.

**Check**:

- Every instrument's `states` includes `UNKNOWN`.
- Market-level labels are `UNKNOWN` before roughly 2022 and never `UNKNOWN` inside W1/W2/W3. An `UNKNOWN` inside an evaluation window is a defect, not a state to interpret.
- The `date` column is tz-aware UTC and aligns bar-for-bar with the source feather. A naive timestamp produces an all-NaN merge downstream and is easy to miss.

---

## Scenario 2 — Lookahead validation is a gate, not a report (FR-007)

```bash
PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_a_lookahead.py
```

**Expect**: `regime/lookahead_validation.csv` with `n_differing = 0` on every row. Exit code 0.

**Failure is informative, not fatal to the feature** — it means an instrument is wrong, so fix or drop it. `first_diff_date` localises the cause:

| Symptom | Likely cause |
|---|---|
| Divergence at the very end of the overlap | Centred rolling window |
| Divergence propagating backwards from the cut | `ffill` or `bfill` reaching across the boundary |
| Market-level instruments only | Cross-sectional operation over a panel containing future rows — the highest-risk case in this study |

**Do not** substitute freqtrade's `lookahead-analysis`. It does not inspect study code that runs outside a strategy.

---

## Scenario 3 — Characterisation eliminates unactionable instruments (User Story 1)

First measure the **host's own** trade durations — never borrow a figure from another strategy family:

```bash
PYTHONPATH=. .venv/bin/python user_data/strategies/regime/run_arm.py \
  --strategy NNNC_MLX --window W1 --threshold 0.69 --tag baseline
```

Then characterise:

```bash
PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_b_characterize.py \
  --dwell-source regime/arms/baseline_W1_trades.json
```

**Expect**: `regime/actionability.csv` covering every instrument × era × state, and every catalogue row now `SURVIVING` or `ELIMINATED` with a quoted measured reason.

**Check**:

- `share` sums to 1.0 per instrument × era; transition-matrix rows sum to 1.0.
- Any `DROP_UNSTABLE` cites a **W1-versus-W3** difference. A W1-versus-W2 claim is invalid — those windows are three months apart and share a regime.
- The dwell threshold printed at startup came from the export above, not a constant in the source.

**If zero instruments survive**, the feature stops here and that is the finding. Record it and stop.

---

## Scenario 4 — The risk-discrimination question gets a straight answer (FR-012)

```bash
PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_b_risk_normalize.py
```

**Expect**: one stdout line — `CONFIRMED` or `REFUTED` — plus `regime/risk_normalized.csv`.

**Why it matters**: the original taxonomy showed a roughly 4× drawdown spread (28.1% best state, 118.3% worst) while the worst state also held 1.55× the bars. If normalization refutes the spread, the primary motivation for treating phase as a risk-modulation signal is gone, and that must be stated plainly rather than buried in a table.

---

## Scenario 5 — ⛔ The stop gate (User Story 2)

Produce the two bracketing configurations, then run the verdict:

```bash
for T in 0.55 0.80; do
  for W in W1 W2 W3; do
    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/run_arm.py \
      --strategy NNNC_MLX --window $W --threshold $T --tag "thr$T"
  done
done

PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_c_trackability.py \
  --arms regime/arms/ --min-trades <N from T013> --status-quo 0.69
```

**Expect**: `regime/trackability.csv`, `regime/VERDICT.md`, and exactly one stdout verdict line.

**Check before believing either outcome**:

- Every `trackability.csv` row's `source` points at a **trade export**, never a price series. This is the substitution that invalidated the prior BTC gate.
- `beats_constant` compares against `constant_error` — the **better of the two fixed choices** — and *not* against `status_quo_error`. A rule that beats the status quo while losing to the better fixed choice has demonstrated threshold tuning, not trackability. Both columns exist so this cannot be lost.
- `n_blocks_excluded` is small relative to `n_blocks_qualifying`. A verdict resting on a handful of blocks is weak evidence regardless of its margin.
- No row supporting a `PASS` has `n_trades = 0`.
- A `PASS` rests on at least two windows including a temporally distant pair. W1 + W2 alone is not a `PASS`.
- One window passing and another failing is `FAIL`, not "mixed".
- A `PASS` at exactly one horizon in exactly two windows is the weakest admissible result — report it as such.

**On FAIL**: write the negative result to `regime/README.md` with the comparison table, commit, and **stop**. Do not build any mechanism. This is a completed finding of equal weight to a positive one, and the prior on it is real — the same test on a different parameter returned NO at every horizon.

**On PASS**: record the winning instrument, level, and horizon. Only that instrument proceeds.

---

## Scenario 6 — Mechanism arms, only after a PASS (User Story 3)

Order is fixed: **M3 → M1 → M2**, then M4 only if M3 or M1 showed signal.

```bash
for W in W1 W2 W3; do
  PYTHONPATH=. .venv/bin/python user_data/strategies/regime/run_arm.py \
    --strategy NNNC_MLX_PhaseThr --window $W --tag m3_phase_thr
done
```

**Check every arm, every time**:

| Check | Why |
|---|---|
| `chain_pass_count` differs from baseline | If equal, the arm is `INERT` — a defect, **not** a null result. This repository has five recorded cases of parameters confirmed as loaded that changed nothing. |
| `timerange` is a literal date range | A today-relative window makes arms run on different days non-comparable. |
| `exit_config_hash` matches baseline | Proves exit invariance. |
| `volume_filter_active` is true | Otherwise fills exceeding available liquidity inflate the result. |
| `n_trades > 0` | A zero-trade arm is inconclusive, not zero-return. |
| Sign holds in W3 | An improvement that reverses in the distant era is era-specific and rejected. |
| `seed_paired` for sub-1pp deltas | Below one percentage point, run-to-run variance dominates. |

**The single most likely false null**: `NNNC/NNNC_MLX.json` **overrides** class-level `buy_params` and is untracked working state. An arm that sets `prediction_threshold` only in Python will silently run at 0.69 and look like a clean null. `run_arm.py` routes the value through a per-arm copy of that JSON — verify via `chain_pass_count` that it took effect, and never rewrite the user's own file.

---

## Scenario 7 — M4, and its extra gate (User Story 4)

Only if M3 or M1 showed signal.

```bash
# 1. Orthogonality first (Principle IX) — before any performance measurement
# 2. Regenerate scalers — a comparison against stale normalization is void
PYTHONPATH=. .venv/bin/python -m freqtrade ... Framework CreateScalers
# 3. Retrain, then run arms as in Scenario 6
```

**Check**: phase-feature return correlation against the existing feature set is measured and recorded **before** performance is evaluated. A phase feature strongly correlated with an existing one is that feature in disguise and dilutes the set rather than adding to it.

---

## Definition of done

| # | Outcome |
|---|---|
| 1 | Every instrument passes lookahead validation with zero differing labels |
| 2 | Every instrument carries `SURVIVING` or `ELIMINATED` with a quoted measured reason |
| 3 | The risk-discrimination question is answered `CONFIRMED` or `REFUTED` with normalized figures |
| 4 | `VERDICT.md` states one `PASS` or `FAIL`, backed by per-window evidence from realized trades |
| 5 | On `FAIL` — the negative result is committed and no mechanism was built |
| 6 | On `PASS` — at least three no-retrain arms measured, each with a liveness proof |
| 7 | The whole study re-runs from a clean checkout with no manual path or parameter edits |
