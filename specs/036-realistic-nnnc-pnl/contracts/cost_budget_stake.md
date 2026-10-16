# Contract: CostBudgetStakeMixin (`utils/CostBudgetStakeMixin.py`)

```python
class CostBudgetStakeMixin:
    STAKE_COST_BUDGET_BPS: float | None = None   # OFF
    STAKE_COST_MODEL: str | None = None          # path to research/live_pnl/cost_model.json
```

The mixin's two attributes are classified in `MODEL_KEY_IGNORED`, because they change exits and
stakes, not weights.

## When off (default)

- `custom_stake_amount` and `confirm_trade_entry` are exactly the parent's.
- Every existing arm keeps its trades hash, fingerprint and model key (FR-014).

## When on

| step | rule |
|---|---|
| inputs | the pair's tier, side buy, the rolling 24 h quote volume V up to and including the signal candle |
| S_max | the largest S with cost_in(S / V) ≤ budget, from the committed model (monotone, so bisection) |
| stake | min(proposed, S_max) |
| headroom check | `confirm_trade_entry` accepts if amount × rate ≤ S_max (replacing the 1/10-candle rule); the MIN_QUOTE_VOLUME and min-stake floors are kept |
| record | requested and granted stakes kept per trade, for the liveness check and the realised/requested ratio |

## Refusals

- The cost model is missing, or its sha differs from the one the plan declares: refuse to start.
- The pair has no tier: refuse. There is no per-pair default (Principle X).
