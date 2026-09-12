# Pre-registration: turnover-matched P&L control (brief 02 follow-up)

**Written 2026-09-12, BEFORE any run.** Recording this after seeing results voids it.

## Why

Brief 02 measured every arm against a P&L control and found the return ordering is an
**exact** turnover ordering (Spearman = −1.000 over 5 configurations):

| config | log_ret | mean_w | turnover |
|---|---|---|---|
| `pnl_control` | −1.0894 | 0.447 | 647.9 |
| `armA_warmstart` | −0.5251 | 0.301 | 312.1 |
| brief 01 `utility-band-g1.0` | −0.3187 | 0.270 | 194.3 |
| `armB_a0.25_thr0.02` | −0.1331 | 0.125 | 105.3 |
| `armC_auxvol` | −0.1083 | 0.426 | 15.7 |

Arm C holds the **same exposure** as the control (0.426 vs 0.447) and trades it **41× less**.
So the arms' apparent advantage may be entirely a cost effect, with the label prior and the
auxiliary target contributing nothing beyond damping turnover. That confound cannot be
separated by any run performed so far.

## Hypothesis under test

**H0 (confound explains everything):** a P&L control whose *realised* turnover is matched to
an arm's will match that arm's return. Nothing is left for the prior to explain.

**H1 (residual prior value):** at matched turnover the arm still beats the control, and the
residual is the contribution of the label prior (Arm B) or the non-directional auxiliary
target (Arm C).

## Design

- Fold **f08**, 10 pairs, identical panel, purge 48 / embargo 48 (post-C1 fix).
- Lever: **`gamma`**, which scales perceived cost inside training (`sim.net_returns(w, o,
  c * cfg.gamma, ...)`). It is the only live turnover lever that does not change the
  objective — `turnover_control` is a policy-side learned band, identical across all arms,
  and is deliberately left alone so the matched control differs from `pnl_control` in
  turnover **only**.
- **Stage 1 (bracket)**: `pnl_control` at gamma ∈ {2, 5, 10, 20, 50}, seed 1 only.
- **Stage 2 (match)**: the gamma bracketing Arm C's turnover (15.7) and the gamma bracketing
  Arm B's (105.3), each at seeds 1–5.

## Decision rule — FROZEN

1. **Match criterion.** A control is "turnover-matched" to an arm if its IQM realised
   turnover lies within **2×** of the arm's (Arm C: 7.9–31.4; Arm B: 52.7–210.6). If no
   swept gamma lands in the window, the match FAILS and the comparison is not made — an
   unmatched control is the same trap this experiment exists to escape.
2. **Verdict.** Compare the arm's IQM return against the matched control's, paired by seed,
   with a bootstrap 95% CI on the difference.
   - CI contains zero → **H0: the prior adds nothing measurable.** Arm C's advantage is a
     cost result and the "labels helped" direction stays withdrawn.
   - CI excludes zero, arm higher → **H1: residual prior value**, quoted with its interval.
   - CI excludes zero, control higher → the arm is *worse* than turnover-matching alone.
3. **Degeneracy.** Any control whose mean exposure falls below the 0.05 participation floor
   is INCONCLUSIVE and its return is withheld (SC-015) — raising gamma far enough will
   eventually buy low turnover by refusing to trade, which is not a match.
4. **Exposure sanity.** A match is only admissible if the control's mean exposure is also
   within 2× of the arm's. Matching turnover while holding a tenth of the position is not
   matching.
5. This settles **direction only**, on one fold. It is not a walk-forward result and cannot
   clear SC-001.
