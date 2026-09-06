# NNNC_MLX_Open — adopted as the forward configuration, NOT yet live-validated

Decision taken 2026-09-06. `NNNC/NNNC_MLX_Open.py` + `.json` pin the
configuration the 2026-09-05/06 study arrived at: gbb label at gain threshold
**0.02**, the six quality guards **open**, `prediction_threshold` **0.80**.

Verified: the pinned strategy reproduces the study arm `r02t08`
BYTE-IDENTICALLY on all three windows with no `--set` flags.

## Measured

    configuration                          W1      W2      W3   summed  trades  worst DD
    PRODUCTION  0.007, guards on, 0.69  +10.45  +10.31   +3.37   +24.13     332     5.25%
    NNNC_MLX_Open  0.02, open, 0.80      +6.94  +16.91  +10.25   +34.10    1489     8.54%

    per-window detail (NNNC_MLX_Open)
    window   trades    ret%     DD%    PF   win%   Calmar   Sortino
    W1          431   +6.94    7.90  1.16   65.4     4.61      0.87
    W2          763  +16.91    8.54  1.31   65.0    10.39      3.74
    W3          295  +10.25    2.34  1.49   68.1    22.98      3.11

## The case for it

- **+9.97pp summed**, winning W2 and W3; positive in every window including the
  temporally distant 2023.
- **Risk is comparable, not worse in the way a single window suggested.** Worst
  -window drawdown 8.54% against 5.25%; return per worst-window drawdown 3.99
  against 4.60. Better Sortino on two windows of three. (An earlier "5x
  drawdown" framing of mine compared W1 only, which is production's best DD
  window — it was misleading.)
- **Headroom is the deciding argument.** At the production gate a PERFECT model
  is worth only 0.68pp more than the real one, so no model work there can ever
  pay. Here the volume-matched gap is ~25pp and the model captures 47% of it.
- **4.5x the trade volume.** At 431 W1 trades a 1pp effect is measurable; at
  production's 95 it is not. Every null in this study was measured at the count
  where nothing is resolvable.

## The case against, stated plainly

Production wins **W1 by 3.51pp** — and W1 is the repo's stated acceptance
window. That is a real cost and the decision accepts it knowingly, on the
grounds that the W1 gap sits inside this study's own noise band while the
headroom difference is a factor of 37.

## What remains before this goes live

1. **Seed robustness.** Every arm here is single-seed. Repeat at 2–3 `MLX_SEED`
   values; sub-pp differences are not trustworthy without it.
2. **Volume/liquidity check.** 1489 trades against 332 is a 4.5x change in
   fill demand. Constitution V: confirm the extra fills are realistic, not
   phantom. `regime/trade_forensics.py --section pairs` and the volume filter.
3. **Cost re-check at the new volume.** Fees scale with trade count; the study
   measured cost on oracle arms, not on this one.
4. **Its own trained model.** It currently ALIASES `NNNC_MLX_Lbl02`'s weights
   via `get_model_path`. That is correct for proving configuration equivalence,
   and must stay that way until a deliberate retrain under its own name is done
   — otherwise a fresh backtest silently trains on the evaluation window (this
   happened once already and produced 290 trades instead of 431).
5. **Dry run** before any capital.

## Hazard recorded

`BaseNNStrategy.get_model_path` keys the model directory off
`self.__class__.__name__`. **A renamed configuration with no model of its own
will TRAIN ONE INSIDE THE BACKTEST**, fitting on the evaluation window and
silently producing different numbers — 290 trades vs 431 here, with no error.
Any strategy that is a pure re-parameterisation of another MUST override
`get_model_path`. Precedents: `NNNC_Brk_H1`, `NNNC_NoModel`, `NNNC_Oracle`,
and now this.
