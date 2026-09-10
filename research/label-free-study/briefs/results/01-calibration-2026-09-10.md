# Brief 01 — Simulator calibration report (2026-09-10)

**Feature**: [`specs/009-labelfree-direct-policy/`](../../../../specs/009-labelfree-direct-policy/)
**Gate**: protocol §5 kill condition / SC-001 · **Verdict: PASS** (attempt 3 of a 5-attempt budget)

## What was tested

Per clarification Q1=B and FR-008a, calibration replays the entry/exit series the **arbiter**
actually produced for the frozen control signal and validates the simulator's **accounting layer
only** — fee application, fill timing, the liquidity cap, stake sizing and the portfolio layer. The
simulator does not implement the deployment exit stack, so exits priced by that stack are taken as
given data.

| Item | Value |
|---|---|
| Window | W1 (`20250901`–`20260831`) |
| Frozen signal | `NNNC_MLX`, production config |
| Archive | `backtest-result-2026-09-10_10-17-08.zip` |
| Control record | 99 trades, +9.8346% on a 10,000 wallet |

## Result — all three SC-001 conditions met

| Condition | Tolerance | Measured | |
|---|---|---|---|
| Trade counts identical | exact | 99 = 99 = 99 matched | ✅ |
| Total return | ≤ 0.10 pp | **0.00000000 pp** | ✅ |
| Max per-trade profit | ≤ 1 bp | **0.00000000 bps** | ✅ |

Entry fills reconcile 99/99, exit fills 99/99. Peak concurrency 6 against a cap of 10.

## Component attribution (Principle VI — a bare total mismatch is not a diagnosis)

Attempt 1 failed, and the residual was **perfectly localised** rather than diffuse:

| exit_reason | trades | max diff (attempt 1) |
|---|---|---|
| `take_profit` | 40 | 0.0 bps |
| `model_exit` | 28 | 0.0 bps |
| `unclog_12h` | 6 | 0.0 bps |
| `trailing_stop_loss` | 21 | 408 bps |
| `roi` | 4 | 1708 bps |

Everything the strategy itself decides fills at the bar open and reconciled exactly on the first
attempt. Only exits priced by the **deployment stack** disagreed, because they fill at a
stack-computed level (the stop, or the ROI threshold) rather than at a bar open. Those are
explicitly out of scope (FR-008a), so the replay takes their realised rate as given. Adding
`trailing_stop_loss` (attempt 2) and then `roi` (attempt 3) closed the gap to zero.

Two facts confirmed exactly against the arbiter, worth recording because everything downstream
depends on them:

- **P&L formula**: `amount·close_rate·(1−fee_close) − amount·open_rate·(1+fee_open)`, max error 0.0
- **Fee**: 0.001 per side (10 bps), matching the cost model's Binance.US taker tier
- **Fill convention**: `open_rate` is the **open of the bar stamped as `open_date`** (99/99)

## Retracted finding, and what replaced it

An earlier draft of this report claimed that **72% of the control's W1 trades could not be filled
under the protocol's liquidity cap**, and recommended re-running the control under the cap before
Phase 6. **That was wrong and is retracted.**

The 72% figure came from comparing each trade's notional against 10% of the **fill** bar's quote
volume (t+1). The gate does not work that way. `BaseStrategy.custom_stake_amount` reads
`dataframe.iloc[-1]` — the last analysed candle, i.e. the **decision** bar (t) — and caps stake at
`quote_volume / QUOTE_VOLUME_HEADROOM_MULT` (10.0, a BaseStrategy default active for every family).

Re-measured against the decision bar:

| Measured against | Trades over cap | Median stake/cap |
|---|---|---|
| Fill bar (t+1) — **wrong** | 71 of 99 (72%) | 1.97× |
| **Decision bar (t) — correct** | **0 of 99 (0%)** | **0.64×** |

The source comment states the guarantee directly: *"any order that does enter is ≤10% of the candle
by definition."* The walk-forward harness records `volume_filter_active: True` for every arm.

**Consequences:**

1. **No control re-run is required.** The designated control record (`regime/wf/pooled_nnnc.json`)
   was already produced under the cap, so the Phase 6 comparison is like-for-like as specified. The
   earlier recommendation would have *weakened* the control and made SC-003/SC-005 easier to pass —
   the opposite of the study's "verbatim or stricter" rule.
2. **The simulator had the same defect and is fixed.** `sim.py` now assesses liquidity on the
   decision bar. This matters beyond diagnostics: had it stood, every trained policy would have
   been charged a cap the arbiter does not apply, systematically distorting the objective.

Recorded rather than quietly corrected, because the retracted version would have changed a baseline.

## Reproduce

```bash
cd user_data/strategies
../../.venv/bin/python -m research.labelfree.calibrate \
    --strategy NNNC_MLX --window W1 \
    --zip user_data/backtest_results/backtest-result-2026-09-10_10-17-08.zip
../../.venv/bin/python -m pytest research/labelfree/tests/ -q     # 9 passed
```

## Status

The gate is open. The simulator is now available to briefs 2, 4, 5 and 6 as the shared training
environment, per protocol §1.
