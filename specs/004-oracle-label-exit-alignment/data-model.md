# Data Model — 004 Oracle-First Label/Exit Alignment

Five entities. Nothing is persisted beyond JSON arm records, backtest archives
and written findings; there is no database.

---

## 1. TradeRecord (read-only, from the backtest archive)

One trade, as freqtrade emits it inside
`user_data/backtest_results/<archive>.zip` → `strategy.<name>.trades[]`.

| field | use in this feature |
|---|---|
| `exit_reason` | the mechanism that closed the trade — the primary diagnostic axis |
| `profit_abs`, `profit_ratio` | P&L attribution per reason |
| `trade_duration` | minutes held; detects truncation by a hold limit |
| `fee_open`, `fee_close`, `stake_amount` | cost burden (FR-018, FR-019) |
| `pair` | pair-mix comparison (FR-020) |
| `max_rate`, `min_rate` | realised favourable / adverse excursion — answers stop-before-payoff and truncation (FR-004, FR-005) WITHOUT recomputation |
| `open_rate`, `close_rate` | normalise excursions to the entry price |

**Validation**: every trade must fall into exactly one exit reason, and the
reader must report coverage as `n_attributed / n_trades`; SC-002 requires 100%.

---

## 2. ExitReasonBreakdown (derived)

For one arm: a partition of its TradeRecords by MECHANISM.

| field | notes |
|---|---|
| `mechanism` | the mapped reason, not freqtrade's raw label |
| `n_trades` | required alongside every attribution (FR-003) |
| `sum_pnl`, `mean_pnl` | signed |
| `mean_duration_h` | |
| `share_of_total_pnl` | signed share; may exceed 100% when contributions offset |
| `stopped_before_peak_pct` | from `max_rate`/`min_rate` |

**Mechanism mapping (R2) — mandatory**:

| freqtrade `exit_reason` | mechanism |
|---|---|
| `stop_loss` | ATR adaptive stop (initial) |
| `trailing_stop_loss` | **ATR adaptive stop (moved)** — NOT the native trailing stop, which is disabled |
| `take_profit` | `cexit_take_profit` |
| `roi` | ROI table |
| `unclog_12h`, `unclog_1d`, `max_hold` | hold-window limits |
| `rsi_*`, `metric_overbought` | profit-check exits |
| `force_exit` | end-of-backtest, excluded from mechanism attribution but counted in coverage |

Reporting `trailing_stop_loss` unmapped would attribute a large share of trades
to a mechanism that is switched off.

---

## 3. CostProfile (derived)

For one arm: the trading-cost burden, so cost drag is not misread as label
quality.

| field | notes |
|---|---|
| `n_trades` | the driver — cost scales almost linearly with it |
| `total_fees` | from each trade's own fee fields and stake |
| `net_pnl`, `gross_pnl` | `gross = net + fees` |
| `fees_as_share_of_gross` | |
| `loss_survives_costs` | boolean — does the arm still lose gross of fees? |

---

## 4. OracleArm

One evaluated configuration. Extends the existing arm record; adds no new
storage format.

| field | notes |
|---|---|
| `tag`, `window` | keys into `regime/arms/<tag>_<window>.json` |
| `strategy` | the host class; oracle arms use an `NNNC_Ora*`/`NNNC_Oracle` class |
| `guards` | `on` or `neutralised` — never `off` (R7) |
| `n_trades`, `trades_hash`, `exit_config_hash` | liveness and configuration identity |
| `wallet_profit_pct`, `max_drawdown_pct`, `profit_factor`, `win_rate` | |
| `returncode` | MUST be 0; a failed or stale record is not evidence (FR-026) |
| `is_lookahead` | always true for oracle arms; forces the ceiling labelling (FR-012) |

**Validation**: an oracle arm is never promotable, whatever it returns.

---

## 5. GateVerdict (derived)

The result of applying the oracle gate to a variant.

| field | notes |
|---|---|
| `variant` | |
| `w1_return`, `w1_trades` | the acceptance inputs |
| `verdict` | `pass` / `fail` / `inconclusive` |
| `reason` | |

**Rules (R9)**:

- `pass` — W1 guards-off oracle return **> 0** AND `w1_trades >= 200`
- `inconclusive` — trades identical to the variant's control (absorbed), or a
  non-zero `returncode`
- `fail` — anything else, including an improvement that remains negative

A `fail` records progress but does not license training. Classifier metrics are
never an input (FR-008).

---

## Relationships

    OracleArm 1───* TradeRecord        (via its backtest archive)
    OracleArm 1───1 ExitReasonBreakdown
    OracleArm 1───1 CostProfile
    OracleArm 1───1 GateVerdict        (variants only; baselines have none)

`TradeRecord` is the only primary source. Everything else is derived from it or
from the existing arm records.
