# Reversal selector screen -- results (2026-10-21)

Pre-registration `preregistration.json` (728887e4), code 20b0a068, raw numbers `results.json`. Instrument: the
ARO width book, 412 early trades (matches 034) + 519 late, every trade ranked (0% unranked).

## Verdict: FAIL

agree = the trade's pair is in the half of the 10 pairs that fell most over the prior 24 h.

| half | agree n / pp per trade | disagree n / pp per trade | gap | AUC (-rank vs pp > 0) |
|---|---|---|---|---|
| early | 216 / 0.101 | 196 / 0.071 | **+0.030** | 0.583 |
| late | 265 / 0.025 | 254 / 0.035 | **-0.010** | 0.426 |
| pooled | | | +0.009 | 0.479, permutation p = 0.37 |

The early split goes the expected way, about a third of the control-at-share gap (0.100). The late split
reverses, so rules 1-3 all fail.

## Descriptive
- The rank is close to orthogonal to ARO's p_buy (Spearman +0.07 both halves), so it would have been new
  information, had it held.
- Within ARO's own agree trades the gap is +0.009 early and -0.024 late, so it adds nothing on top of the model.
- 3-day variant: gap -0.008 / +0.022, AUC 0.49 / 0.48. Nothing there either.
- The early AUC (0.58295) matches the 034 control's (0.58305) to 3 decimals by coincidence. They are
  different scores (rho 0.07).

## Reading
The cross-sectional 1-day reversal is real across all bars (xsec screen, t -4.9), but it carries no
information inside the gbb width book. The gbb trigger already selects bars after a fall, so the trades are
pre-conditioned on the same move. Per the rule, no sizer arm follows.
