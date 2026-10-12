# 028 US2 — verdicts on the judging window

*CONFIRMATORY (pre-registered) -- PROVISIONAL under quality gate 2 (2022 adjoins f00; not a distant era)*

Folds e00, e01, e02 (0.822 years). Control `st_e22_rev2`: -5.42 pp, 969 trades.

| candidate | 2022 gain pp (a) | paired (b) | risk Sharpe / maxDD (c) | 2023–26 early / late (d) | folds changed | CI half-width pp | verdict |
|---|---|---|---|---|---|---|---|
| C4 | — | — | — | +8.44 / +6.27 | — | — | STOPPED_AT_FEASIBILITY |
| C2 | +4.95 | INCONCLUSIVE_ARM_LOSES (p 0.5499) | UNCHANGED / UNCHANGED | +4.46 / +4.17 | 3/3 | +16.25 | **ADOPT** |
| C1 | — | — | — | +3.29 / +1.88 | — | — | STOPPED_AT_FEASIBILITY |
| C42 | — | — | — | +4.23 / +4.08 | — | — | STOPPED_AT_FEASIBILITY |

## C2 (`SimpleTrigger_RevOnlyVolreg2_XC2`)

- arm -0.47 pp vs control -5.42 pp (919 vs 969 trades); per fold e00 +2.02, e01 +1.15, e02 +1.78
- replay predicted +0.97 pp; arm realised +4.95 pp
- paired: INCONCLUSIVE_ARM_LOSES, delta_pp +4.95 (reported only), CI [-13.756685908954656, 25.78852532556938] pp/yr, p 0.5499; flagged INCONCLUSIVE_ARM_LOSES
- risk (272 days, realised daily pp): Sharpe **UNCHANGED** -0.03 vs -0.32 (delta +0.30, CI [-0.16, +0.69], p 0.166); max drawdown **UNCHANGED** 15.66 vs 17.59 (delta -1.93, CI [-7.96, +1.47], p 0.570)
- exit reasons arm {'force_exit': 7, 'roi': 66, 'stop_loss': 4, 'take_profit': 419, 'trailing_stop_loss': 350, 'unclog_12h': 70, 'unclog_1d': 3} vs control {'force_exit': 3, 'roi': 11, 'stop_loss': 4, 'take_profit': 534, 'trailing_stop_loss': 339, 'unclog_12h': 74, 'metric_overbought': 1, 'unclog_1d': 3}
- gate-5 re-run: not needed

## Reading (T015)

- **The stop-side mechanism did not carry over to 2022.**
  - Only 46.9% of 2022 stop exits recovered to +2% within the max hold, against 63.4% / 57.5% in
    2023–26. That is below the pre-registered 0.50 floor, and close to momentum's 45%, where C4
    already failed.
  - The replay agrees on the sign: C4 −0.04, C1 −1.29, C42 −2.93 pp.
  - In a falling market, giving a stopped trade more room loses more often than it recovers.
  - C4, C1 and C42 stop at feasibility, as pre-registered. They were not run on 2022.
- **C2 (ATR-scaled take-profit) is ADOPTED.**
  - It gains in every 2022 fold, and the per-fold gains are similar in size (+2.02, +1.15, +1.78).
  - Its 2023–26 gains were +4.46 / +4.17, so it is positive in all three windows and in two regimes.
  - Its mechanism doesn't depend on recovery. The take-profit scales with volatility, so on calm
    bars (k × ATR below 2.2%) it banks gains earlier, and on volatile bars it lets them run.
    - Take-profit exits fall from 534 to 419; ROI exits rise from 11 to 66.
    - Stop exits are about level (350 vs 339). Trades fall from 969 to 919: exits change, so later
      entries change.
- **Flags and power.**
  - The paired verdict is INCONCLUSIVE_ARM_LOSES: the arm still loses money in 2022 (−0.47 pp),
    only less than the control (−5.42 pp). The pre-registered mapping passes this, because the
    comparison arm is the same strategy, not a broken one.
  - The window's power is ±16.25 pp: 3 folds can't make a 4.95 pp gain significant
    (p 0.55). This is a persistence verdict, not a significance test, as pre-registered.
  - Risk is UNCHANGED on both measures, with point estimates in C2's favour (Sharpe −0.03 vs
    −0.32; max drawdown 15.66 vs 17.59).
- **Replay vs arm.**
  - 2022 realised +4.95 pp against a replay prediction of +0.97 (5.1×), while 2023–26 realised
    about half the replay (027).
  - The replay ignores re-sequencing, and the sign of that effect is not stable. Don't use the
    replay to size gains, only to choose what to test.
- **Recommendation (R4).** C2 is the only ADOPTED candidate, so the reversion exit NNNC should
  inherit is the current exit with **take-profit = 2.897 × signal-bar ATR, clamped [1%, 5%]**.
  C42 cannot bring C4 back.
