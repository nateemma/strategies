# Contract: Simulator (`research/labelfree/sim.py`)

Consumed by this feature and by briefs 2, 4, 5 and 6. Breaking changes require re-running the
calibration gate.

## Entry points

```
simulate(features, prices, volumes, policy, costs, portfolio, *, mode) -> SimulatedRun
replay(trade_series, prices, volumes, costs, portfolio)               -> SimulatedRun
```

- `mode ∈ {train, eval}` — selects soft-min vs hard liquidity cap (FR-005) and training vs
  full/pessimistic cost scale (FR-004).
- `replay` takes a **fixed** entry/exit series and applies only the accounting layer. This is the
  calibration path (FR-008, FR-008a); it never consults a policy.

## Invariants (each maps to a requirement; violation is a defect, not a tuning matter)

| Invariant | Requirement |
|---|---|
| A decision at bar `t` reads only bars `≤ t` | FR-003 |
| Fills occur at bar `t+1`; during calibration specifically at `t+1` **open** | FR-003 |
| Exposure ∈ [0, 1] per pair (long-only spot) | FR-003 |
| Executed Δw ≤ 10% of the bar's quote volume; remainder carries forward | FR-005 |
| `mode=train` uses soft-min cap; `mode=eval` uses hard cap | FR-005 |
| Fee (per side, real tier) + estimated half-spread + vol-scaled slippage are inside the returned net return | FR-004 |
| Half-spread comes from the OHLCV estimator with a floor; never a flat constant | FR-004a |
| Concurrent positions ≤ `max_open_trades`; stake and `tradable_balance_ratio` as configured | FR-006 |
| Trade list matches the backtest-archive shape | FR-007 |
| The simulator does **not** implement take-profit, adaptive stop or grace | FR-008a |

## Outputs

`SimulatedRun` per data-model §3, including `requested_delta` and `unfilled_carry` so that
cap-binding is measurable rather than inferred, and `exit_reason` so SC-018's band-vs-deployment
share can be computed.

## Failure modes the contract requires to be explicit

- Missing bars for a pair in a fold → that pair is excluded from the fold's universe and the
  exclusion is recorded (FR-015a), never silently forward-filled.
- Zero or missing volume on a bar → the cap binds to zero for that bar; the request carries.
- A fold with too few trades to bootstrap → reported inconclusive, not pooled silently.
