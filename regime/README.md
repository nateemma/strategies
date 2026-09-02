# Market-Phase Framework and Phase-Adaptive Entry Gate

Study for spec-kit feature `001-market-phase-entry-gate`. See
`specs/001-market-phase-entry-gate/` for the spec, plan, and task list.

Extends the `btc_corr/` regime work (Exp 4) into a three-level phase framework —
asset, market breadth, and BTC — then asks one question: does a phase label
predict which of two entry-aggressiveness settings wins the coming period,
better than always using the better fixed setting?

## Evaluation windows (pinned; research R3)

| Window | Range | Role |
|---|---|---|
| W1 | `20250901-20260831` | Deployment regime — **carries the headline verdict** |
| W2 | `20240601-20250531` | Near-era confirmation |
| W3 | `20230101-20231231` | Temporally distant era |

W1 and W2 are only three months apart and share a regime; they cannot establish
persistence between them. W3 is the window that can.

## Running

All scripts run from the freqtrade root — the system Python has no pandas:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/<script>.py

## Results

_Pending._ Phase A (classify) not yet run.
