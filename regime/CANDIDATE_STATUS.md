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

---

# Seed robustness — RUN, and it changes the numbers

Three seeds (MLX_SEED 42 / 1 / 7), each a fresh model, same pinned config.

    seed       W1 tr  W1 ret%   W2 tr  W2 ret%   W3 tr  W3 ret%   summed
    42           431     6.94     763    16.91     295    10.25    34.10
    1            321     5.72     540     8.54     196     7.25    21.51
    7            413     2.97     698    11.15     262    11.88    26.00
    PROD          95    10.45     181    10.31      56     3.37    24.13

    spread    min      max     mean      sd
    W1      +2.97    +6.94    +5.21    2.03
    W2      +8.54   +16.91   +12.20    4.28
    W3      +7.25   +11.88    +9.79    2.34
    summed +21.51   +34.10   +27.20    6.38

    beating production summed (+24.13): 2/3
    positive in all three windows:      3/3
    beating production on W1 (+10.45):  0/3

    val_mcc:  42 -> 0.6078   1 -> 0.6010   7 -> 0.6053

## What this corrects

**The +9.97pp summed advantage was seed luck.** Seed 42 is the MAXIMUM of the
three. The honest comparison is **mean +27.20 against production's +24.13 —
+3.07pp with a standard deviation of 6.38pp**, i.e. indistinguishable. The
figure quoted when this configuration was adopted overstated it by 3x.

**The W1 disadvantage is NOT noise — it is consistent.** 0 of 3 seeds beat
production on the acceptance window, and the W1 mean is +5.21 against +10.45.
This is the most robust single finding in the comparison.

**val_mcc is stable across seeds (0.6010-0.6078)**, so the P&L spread is not a
training-quality artefact. It is the configuration's own variance, which at
~300-430 trades per window is large.

## What survives

- **Positive in all three windows, 3 of 3 seeds** — including the temporally
  distant 2023. That property is genuinely seed-robust.
- **The headroom argument is untouched.** ~25pp of volume-matched headroom
  against production's 0.68pp is a CEILING measurement, independent of which
  seed trained the model.
- **The volume argument is untouched.** 300-430 W1 trades against 95 is what
  makes a 1pp effect measurable at all.

## Revised basis for the decision

The case for `NNNC_MLX_Open` no longer rests on measured return — on return it
is indistinguishable from production on summed and consistently worse on W1. It
rests on **headroom and measurability**: it is the only configuration in this
repo where model work can move the result, and where an effect of the size this
study keeps producing could actually be detected.

That is a legitimate basis for a research platform. It is NOT a basis for
claiming better performance, and the earlier framing in this document has been
corrected accordingly.
