# Draft input for /speckit-specify (feature 036): realistic P&L for NNNC

Realistic P&L improvement for NNNC (feature 036).

**Background:**
- **Capital sits idle.** Over the 15 walk-forward folds, the NNNC control (`wf_aro_g24`, ARO) averages
  0.37 open trades and uses about 1.7% of the 10,000 wallet. Its median stake is about 300 USDT, not
  the configured 900, because `custom_stake_amount` caps each order at 1/10 of the candle's quote
  volume on Binance.US's thin books.
- **The per-trade edge is not the bottleneck.** Label (032-034) and input (035, both32) screens found
  no model improvement.
- **The sizer did not hold forward.** The control-sizer result (+8.5 pp, PROVISIONAL) was INCONSISTENT
  on a blind 35-day forward window (research/control_sizer/report.md). The frozen June model's
  operating point had drifted: 20% agree share instead of 50%.
- **Backtest returns are not realistic P&L.** The backtest fills at the touch with no slippage model,
  so loosening the stake cap only inflates returns (memory project_liquidity_and_capacity: PF is flat
  across cap multipliers). Any larger size must be justified from fill and depth evidence, not from
  backtests.

**Goal:** measure and raise NNNC's realistic dollar P&L on Binance.US. Realistic means returns that
would survive real execution: size against available depth, slippage, and a model kept current.

**Stories (confirm in spec):**
1. **(P1) A live NNNC dry-run on a refreshed model.**
   - Retrain ARO on data up to the latest download, and pick the threshold by the standard
     walk-forward rule (validation tail).
   - Run a second dry-run bot alongside the momentum basket, isolated by its own port, database,
     log and user_data paths.
   - Use strict model mode, so a missing model stops the bot rather than training one live.
   - Record per order: signal time and price, requested stake and the stake after the cap, fill time
     and price, unfilled or timed-out orders, and an order-book depth snapshot (top N levels both
     sides) at order time.
   - Declare a retrain cadence. The forward window showed operating-point drift within about 3
     months.
   - Write a daily comparison of the bot's trades against a backtest of the same days and the same
     model (signal parity: same entries? same exits?).
2. **(P2) An execution model.**
   - From the recorded depth, estimate the price impact of a given stake: walk the book, plus fees.
   - Apply it in backtests as a size-dependent cost, and re-state the 15-fold control under it.
   - Validate it against the dry-run record. Note that freqtrade dry-run fills are simulated, so the
     depth snapshots are the real evidence, not the dry-run fill prices.
3. **(P3) A size-aware stake cap.**
   - Replace the fixed 1/10-candle-volume cap with a cap set by the execution model, for example the
     largest stake whose estimated impact stays under a declared bps budget.
   - Produce a capacity curve: realistic P&L against the cap, judged with the existing harness
     (judge_one, paired, risk). Default off; adoption is the user's decision.
4. **(P4, user decision) Wallet sharing with the momentum basket.** With about 98% of the wallet
   idle, measure the contention if NNNC and the basket share one wallet. This is deployment, not
   framework optimisation (memory feedback_framework_not_portfolio).

**Out of scope:**
- real-money trading;
- new pairs or venues;
- label, input or architecture changes;
- touching the running momentum dry-run bot (pid 2661, config_mom_dryrun_15m.json).

**Constraints:**
- default off and byte-identical when off;
- Framework/ stays strategy-agnostic;
- parameter dicts merge;
- no pair-specific code;
- one trainer or backtest chain at a time;
- secrets only via ~/.freqtrade_env;
- downloaded and recorded market data stays in its own directory, is never committed, and is parsed,
  not executed;
- promotion is the user's decision.

**Open questions for clarify:**
- Is US4 in scope?
- The retrain cadence (monthly, or per 90-day fold)?
- The minimum dry-run duration before US2/US3 are judged (for example 4 or 8 weeks)?
- The impact budget in bps for US3?
