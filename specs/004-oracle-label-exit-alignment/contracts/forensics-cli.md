# Contract — `regime/trade_forensics.py`

The one new interface in this feature. Read-only over
`user_data/backtest_results/`. It never runs a backtest and never writes to
`regime/arms/`.

## Invocation

    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/trade_forensics.py [options]

Run from the repository root.

## Options

| option | meaning |
|---|---|
| `--archive PATH` | the backtest zip to read |
| `--latest` | read the newest archive named by `.last_result.json` |
| `--match-trades N` | resolve the archive by trade count, disambiguated by `--strategy` |
| `--strategy NAME` | strategy name inside the archive |
| `--section {exits,costs,pairs,excursions,all}` | which report to emit; default `all` |
| `--json PATH` | also write the structured result |

Exactly one of `--archive`, `--latest`, `--match-trades` is required.

## Guarantees

1. **Coverage is always reported.** Every report states `n_attributed /
   n_trades`. Anything below 100% is a defect, not a rounding note (SC-002).
2. **The archive actually used is always printed** — path, strategy name and
   trade count — so a mis-resolved archive is visible rather than silent (R5).
3. **Mechanisms are mapped, not raw.** `trailing_stop_loss` is reported as the
   ATR adaptive stop, and the mapping table is printed with the report (R2).
4. **Every attribution carries its trade count** (FR-003).
5. **Read-only.** It opens nothing outside `user_data/backtest_results/` and
   writes only where `--json` says.

## Sections

**exits** — per mechanism: `n_trades`, `sum_pnl`, `mean_pnl`, `mean_duration_h`,
`share_of_total_pnl`, ordered by `sum_pnl` ascending so the largest destroyer is
first.

**costs** — `n_trades`, `total_fees`, `net_pnl`, `gross_pnl`,
`fees_as_share_of_gross`, `loss_survives_costs`.

**pairs** — per pair: `n_trades`, `sum_pnl`, share of trades; plus a comparison
line when two arms are read, so a changed pair mix is visible (FR-020).

**excursions** — from `max_rate` / `min_rate`, normalised to `open_rate`: the
distribution of favourable and adverse excursion, the share of trades whose
adverse excursion reached the operative stop before their favourable excursion
reached the take-profit (stop-before-payoff, FR-004), and the share whose
favourable excursion exceeded the take-profit level (truncation, FR-005).

## Non-goals

Does not run backtests, does not modify `run_arm.py`, does not judge a variant —
the gate in `regime/ORACLE_GATE.md` does that.
