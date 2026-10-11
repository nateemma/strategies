# Research: Trigger-Family Exit Presets (024)

Phase 0 output. Each item: Decision / Rationale / Alternatives. Code facts were read on
2026-10-11 at `fe4ecb0`.

## R1 — Where per-trade exit parameters enter the framework

**Fact.** Every exit and stop decision `BaseStrategy` makes reads its numbers from one method,
`BaseStrategy._exit_params()` (`Framework/BaseStrategy.py:1096`), which builds a
`utils.ExitPolicy.ExitParams`. `custom_exit` passes it to `custom_exit_decision`
(metric_overbought → take_profit → unclog_12h → unclog_1d → max_hold, `utils/ExitPolicy.py:620`).
`custom_stoploss` and `_resolve_normal_stop` pass it to `atr_stop_fraction`.
`ExitParams` ALREADY has `unclog_12h_hours` (12.0), `unclog_12h_min_profit` (0.005) and
`unclog_1d_hours` (24.0), and `custom_exit_decision` already reads them. `_exit_params()` just
never sets them, so they sit at their defaults.

**Decision.** Add an optional `trade=None` argument to `_exit_params`, and pass `trade` at its
three call sites. The default implementation ignores it and returns the same object as today. A
strategy that opts in overrides `_exit_params(trade)` and returns
`dataclasses.replace(base, **family_overrides)`, chosen by the trade's tag.

**Rationale.** It is one additive hook with no family logic in `Framework/` (FR-014,
constitution). The unclog times become per-family through fields the single exit model already
honours, so there is no second implementation of the exit order (the 021 single-source rule).
That retires `HoldWindowMixin` for this purpose.

**Alternatives rejected.**
- Wrapping `custom_exit` and suppressing reasons, as `HoldWindowMixin` does. It can only veto,
  never add, and a vetoed take-profit hides the later checks on that bar.
- A per-family subclass per trade. freqtrade instantiates one strategy.
- Editing the hardcoded hours inside `custom_exit_decision`. That changes every strategy.

## R2 — Stop-path reads that bypass `_exit_params`

**Fact.** Three stop-side reads use attributes, not params:
- `_atr_bar_inputs` gates `rvol` on `self.use_volume_confirmation_stoploss`;
- `_resolve_normal_stop` checks `self.use_atr_adaptive_stoploss` and falls back to
  `self.stoploss`;
- `custom_stoploss`'s after-fill branch checks `self.use_atr_adaptive_stoploss`.

`StopInvariantMixin._check_stop` reads `use_atr_adaptive_stoploss`, `stoploss` and
`atr_stoploss_floor` from `self`.

**Decision.** Route those reads through `self._exit_params(trade)`:
`params.use_volume_confirmation_stoploss`, `params.use_atr_adaptive_stoploss`,
`params.static_stoploss`, `params.atr_stoploss_floor`. Default params mirror the attributes
exactly (R1), so this is behaviour-neutral for every non-opting strategy. Byte-identity is checked
on recorded arms (R9).

**Rationale.** Without this, a family override of the ATR band or volume-confirm would be
silently INERT on the stop path (constitution IV). That is the exact failure mode 023 found in
`TriggerOnly`.

**Alternative.** Restrict family sets to custom_exit fields only. Rejected because FR-002 names
the ATR band and volume-confirm.

## R3 — How the family travels with a trade

**Fact.** `BaseStrategy.populate_entry_trend` sets `enter_tag = "model_entry "` on signal rows,
then ANDs the guard conditions into `enter_long`. freqtrade copies the signal row's `enter_tag`
onto the trade (`trade.enter_tag`), which is available in `custom_exit`, `custom_stoploss` and
`confirm_trade_exit`. **Nothing in the repo reads `enter_tag`** (grep of Framework, utils,
SimpleTrigger, NNNC, regime, evaluation). `run_arm.trades_hash` hashes pair, open_date,
close_date and profit_ratio. It does **not** include `enter_tag`, so retagging cannot disturb a
reproduction check.

**Decision.** Tag format `"<family>:<trigger>"`, e.g. `reversion:gbb` and
`momentum:breakout_gbb`. The opting strategy rewrites `enter_tag` on its signal rows after
`super().populate_entry_trend()`. The family is parsed from the prefix before `:`. A tag with no
recognised prefix gets the shared exits and is counted (FR-005).

**Rationale.** Keeping the trigger name in the tag makes per-trigger attribution free in the
results archive. Rewriting after `super()` keeps `BaseStrategy` untouched for tagging.

## R4 — Family classification

**Decision.** Add a `family` field (with a one-line `family_reason`) to the registry's `Trigger`.
- Unset (None) is allowed for construction, so existing tests still build bare triggers.
- `ensure_builtins()` registers every builtin with a family.
- A test asserts that every registered trigger has a family (SC-006).

Classification per spec FR-001. Reversion = buys weakness (low guard_metric, a confirmed valley,
oversold votes, a failed breakout). Momentum = buys strength (high band position, guard
> 0.2, a new 20-bar high). Unassigned = broad filters that fire on most bars (`adx`,
`votes_ind*`, `votes_consensus`), `empty`, and the controls.

**Alternative.** A separate mapping file. Rejected: the family belongs with the condition it
describes, and the registry is already the single source for triggers.

## R5 — Unclog equivalence with 023's `_MomExit` arms (the SC-001 risk)

`st_bgo_mx` used `HoldWindowMixin` with `HOLD_WINDOW_HOURS=24` and `cexit_take_profit=10.0`.
The 024 momentum set uses `unclog_12h_hours=24.0` and `take_profit=10.0` through params.

Case analysis, profit `p` at age `a`:
- **a < 12:** both arms reach the same decision (no time exit fires).
- **12 ≤ a < 24:**
  - The mixin's super returns `unclog_12h` when p > 0.005, and the mixin suppresses it. That
    returns None without evaluating `unclog_1d`, which cannot fire before 24h anyway.
  - The params version's `unclog_12h` condition is false, and so is `unclog_1d`.
  - The `max_hold` check is skipped on that bar under the mixin, but `max_hold` is 10 days, so
    the skip cannot matter.
  - The only call after the time checks is `strategy_custom_exit`, a no-op in this chain. The
    mixin also skips it.
  - Both arms reach the same decision.
- **a ≥ 24:** both evaluate `unclog_12h` (p > 0.005), then `unclog_1d` (p ≥ 0). Identical.

**Prediction:** byte-identical trades, verified by SC-001 against `st_bgo_mx`'s recorded
`trades_hash`.

## R6 — Combined arm entry and overlap

**Decision.** `SimpleTriggerStrategy` accepts `TRIGGERS` (a tuple of names) as well as
`TRIGGER_NAME`. Exactly one of them must be set, and `TRIGGER_NAME="x"` behaves as `TRIGGERS=("x",)`.
- `DISABLED_TRIGGERS` (a tuple) removes a side without changing anything else (FR-009).
- `FAMILY_PRIORITY` defaults to `("reversion", "momentum", "unassigned")`.
- The entry mask is the OR of the enabled triggers' masks. Each signal row is tagged with the
  highest-priority family whose trigger fired, and within a family, with the first listed trigger.
- Every enabled trigger must pass `check_runnable` (the 023 causality gate), exactly as today.

**Rationale.** It reuses 023's G-contract, and the reproduction tests (SC-002) are exact by
construction. With one side disabled, the mask and the exits equal the single-trigger arm's.

## R7 — Contention measurement

**Decision.** `evaluation/contention.py` compares the combined arm's pooled trades with each
single-family arm's pooled trades, keyed on (fold, pair, open_date).
- **Lost** = in the single-family arm, not in the combined one.
- Each lost entry gets a cause, from the combined arm's trades open at that instant:
  1. `pair_held_by_other_family`: the same pair has an open trade tagged with the other family;
  2. `pair_held_same_family`: path divergence inside the family;
  3. `slots_full`: open trades ≥ `max_open_trades` (10 in `config_wf.json`);
  4. `divergence`: none of the above, an earlier difference cascading.
- **Gained** entries (in the combined arm, not in the single-family one) are reported too.
- Everything is split early/late.

**Rationale.** It answers "what did combining cost each family" from data the harness already
writes. No new backtest instrumentation is needed.

## R8 — Fingerprint coverage

**Decision.** Extend C9's `OPTIONAL_ATTRIBUTE_MEMBERS` pattern. Set only on opting strategies:
- `TRIGGERS`;
- `DISABLED_TRIGGERS`;
- `FAMILY_PRIORITY`;
- the RESOLVED family exit sets (preset + overrides, fully expanded);
- the trigger→family map restricted to the arm's triggers.

Non-opting arms keep their shas (arm D `303456c1fa322b4c`, asserted by a test).

## R9 — "Live dry-run" safety: the premise was wrong, and the check stays cheap

**Fact.** The live dry-run (`MomentumRegimeBasket15mFast`, strategy path
`user_data/strategies/Basket`) derives from freqtrade's `IStrategy` directly and imports nothing
from `Framework/` or `utils/` (grep of `Basket/MomentumRegimeBasket15m*.py`). `BaseStrategy`
edits cannot reach it. The spec's statement that it "imports BaseStrategy" (FR-006, SC-004) came
from the feature request and is not true of the current code. The risk FR-006 guards against is
therefore lower than stated.

**Decision.** Keep SC-004 as written (`evaluation/dump_strategy.py`, identical before and after,
as in 023 T032). It costs seconds, and it also covers a future live strategy built on
`BaseStrategy`. The binding neutrality evidence moves to recorded arms. Re-run `st_gbb` f00 and
arm D's `wf_aro` f00 through the edited `BaseStrategy` and require an identical `trades_hash`.
These are the strategies that actually use the edited code.

## R10 — Comparison and reporting

**Decision.** Reuse 023's tools:
- `compare_pooled.compare_data` for paired bootstrap, run with `PYTHONHASHSEED=0` (the
  bootstrap's set-iteration nondeterminism is still an open user decision);
- `halves.py` for the early/late split;
- per-fold wallet sums for the "sum of the two single-family arms" reference. That is a number,
  not a trade set, so it is not bootstrapped;
- per-fold Pearson correlation between the families' single-family arms (constitution IX).

Every output carries the gate-2 PROVISIONAL label.
