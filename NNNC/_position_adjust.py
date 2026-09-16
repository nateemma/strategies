"""Repeat-signal position adjustment for the NNNC family.

WHY THIS EXISTS, and what it is NOT.

Measured 2026-09-15 on walk-forward fold f08 (`docs/WALKFORWARD_TODO.md` item 3): freqtrade
admits an entry only when the pair has no open position
(`freqtrade/optimize/backtesting.py:1552`), and **every repeat signal on a pair that already
holds one is silently dropped** -- not even counted, because `_collate_rejected` is called only
on the `max_open_trades` path. Re-running f08 with `--enable-position-stacking` took 102 trades
instead of 71 and the global cap then rejected a further 23, so **at least 54 of roughly 125
entry signals were being discarded: ~43%.**

`--enable-position-stacking` is backtest-only and freqtrade states its results cannot be
reproduced in dry/live, so it measures the size of the loss, not a fix. This mixin is the
deployable form of the same idea -- and it is a DIFFERENT TRADE, which is the point of
measuring it rather than assuming the `--eps` number carries over:

  position stacking   N independent trades, each with its OWN entry price and its own stop
  this mixin          ONE trade whose entry price is the weighted average of its fills

The second is strictly more dangerous when the position is underwater: adding lowers the
average entry, which moves the ATR-derived stop DOWN with it, so a losing position both grows
and gains room to lose more. The `--eps` upper bound flatters what this can capture, and it is
the reason the drawdown side of the comparison is not optional.

DESIGN, deliberately one parameter space (Principle VIII):

  * Adds fire on a REPEAT ENTRY SIGNAL only -- the same `enter_long` the strategy already
    produces. No new signal, no new threshold, nothing else to tune.
  * The add is the same fixed stake as the initial entry, matching the fixed-stake convention
    the rest of this repository is evaluated under.
  * `max_entry_position_adjustment` caps the number of adds.
  * Adds are UNCONDITIONAL on profit. Gating them on being in profit (pyramiding) or on being
    underwater (deliberate averaging down) are each a second parameter space and a separate
    experiment; the unconditional form is the faithful analogue of the signals that were
    actually masked.

`iloc[-1]` on the analyzed dataframe is this repository's audited "previous completed candle"
convention in backtesting (`project_momentum_lookahead_audit`), and is what
`BaseStrategy.custom_stoploss` already uses at line 1044.
"""
from __future__ import annotations


class RepeatSignalPositionAdjustMixin:
    """Add to an open position when the strategy re-signals entry on that pair."""

    position_adjustment_enable = True

    # Number of ADDITIONAL entries allowed beyond the first. 2 keeps worst-case exposure on one
    # pair at 3x the fixed stake, against a 10-slot book -- so a fully pyramided pair is 30% of
    # the wallet's trade capacity, not an unbounded concentration.
    max_entry_position_adjustment = 2

    # PYRAMIDING GATE (2026-09-15). False reproduces the unconditional arm, which was REJECTED:
    # return not resolved (-0.455 pp/yr) and drawdown reliably worse (12/13 folds, p=0.0017).
    #
    # That rejection came with a diagnosis rather than just a number. Grouping trades by adds
    # received gave win rates 0.791 / 0.709 / 0.286 for 0 / 1 / 2 adds -- the repeat signal is
    # ADVERSE-SELECTED, arriving disproportionately while the first entry is going against you.
    # Setting this True adds only to positions already in profit, which is the one place that
    # table says value could survive.
    position_adjust_require_profit = False

    # Minimum bars between adds. Without it a signal persisting across consecutive candles
    # spends the whole budget in 30 minutes, which is a different intervention (faster sizing)
    # than the one being tested (acting on repeat signals).
    position_adjust_min_bars = 4

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs):
        if self.dp is None or self.dp.runmode.value in ("plot", "other"):
            return None
        # Exits are owned entirely by custom_exit / custom_stoploss. This hook only ever adds;
        # returning a negative here would partially close a position behind their back.
        if getattr(self, "position_adjust_disabled", False):
            return None

        df, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        if df is None or len(df) == 0:
            return None

        last = df.iloc[-1]
        if int(last.get("enter_long", 0) or 0) != 1:
            return None

        # `current_profit` is the ratio against the position's CURRENT average entry, so after
        # an add the bar for the next one is measured from the averaged basis, not the original
        # fill. That is the correct reading: it asks whether the position as it now stands is
        # winning, which is the question the gate is meant to pose.
        if self.position_adjust_require_profit and current_profit <= 0:
            return None

        n_entries = trade.nr_of_successful_entries
        if n_entries > self.max_entry_position_adjustment:
            return None

        bars = (current_time - trade.date_last_filled_utc).total_seconds() / 60.0
        if bars < self.position_adjust_min_bars * int(self.timeframe[:-1]):
            return None

        stake = self.config.get("stake_amount")
        if not isinstance(stake, (int, float)):
            return None
        if max_stake is not None and stake > max_stake:
            stake = max_stake
        if min_stake is not None and stake < min_stake:
            return None
        return float(stake)
