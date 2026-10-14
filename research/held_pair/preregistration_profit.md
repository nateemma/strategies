# 031 follow-up — in-profit held-pair add (pyramiding): pre-registration

Machine-readable: [preregistration_profit.json](preregistration_profit.json).

**Setting (frozen).** `HELD_PAIR_ADD = {"in_profit_only": True}`:
- one full-stake add, made only when the held trade is in profit at the add candle's open;
- the last fill must be at least 1 h old;
- no add on a candle where the trade's own exit fires (the exit-due guard);
- 24 h grace.

It was chosen by [screen2.py](screen2.py) as the only variant positive in both halves on ARO, GBR
and GFM, with drawdown not raised and a risk-matched gain above zero on all four models.

**Evidence labels.**
- NNNC (ARO, GBR, GFM, MLX; f00–f14) is **EXPLORATORY**. Both halves have been seen for the
  held-pair question, and no NNNC model covers 2022.
- SimpleTrigger reversion and momentum on 2022 (e00–e02) are **CONFIRMATORY**. This variant was
  never run there.
- SimpleTrigger 2023–26 is exploratory.

**Rule.** The same as the 031 DCA pre-registration:
- NNNC: `judge_one`.
- SimpleTrigger: `judge_2022` on 2022.
- On both: zero adds means INCONCLUSIVE.
- Same verdict maps.
- Reported for every book: exposure, max drawdown, the leverage control and the crowd-out split.

**Exempt records**: `hp_aro_dca_p` f00–f07, the fidelity run.

**Fidelity.**
- First run: 86.8%, FAIL. All 7 unmatched adds fell on the take-profit candle: the add was checked
  before the exits and suppressed them.
- Fixed with the exit-due guard. The rerun matched 47 of 47 adds; arm +3.19 pp vs screen
  +3.24 pp.

**Regime check.** [regime_check.json](regime_check.json): the volatility regime does not separate
add P&L, so there is no regime gate.

The switch stays off by default. Enabling it is the user's decision.
