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

---

## Addendum 2026-09-12 — the lever changed; the decision rule did NOT

The original design named **`gamma`** as the matching lever. Two levers have now failed, for
different reasons. Recording this because changing a lever after seeing results is a
deviation, and an undocumented one would make the eventual match unauditable.

**The frozen decision rule is unchanged** — match windows (turnover AND exposure within 2×),
the participation-floor exclusion, and the seed-paired bootstrap CI verdict all stand as
written. Only the mechanism for *reaching* the match changes.

### Lever 1 — `gamma` (scales perceived cost in training): FAILED, does not move turnover

| gamma | log_ret | mean_w | turnover |
|---|---|---|---|
| 1 (baseline) | −1.0894 | 0.447 | 647.9 |
| 2 | −0.8574 | 0.306 | 554.3 |
| 5 | −0.8992 | 0.510 | 555.7 |
| 10 | −0.7688 | 0.306 | 526.1 |
| 20 | −0.8500 | 0.379 | 557.5 |
| 50 | −1.4326 | 0.826 | 646.7 |

A 50× range moves turnover 19% at best, non-monotonically, and γ=50 returns to baseline with
exposure blown up to 0.826. Gamma changes *what* the policy holds, not how often it re-trades.
Neither window reached.

### Lever 2 — `lam_turnover` (explicit λ|Δw| in the loss): FAILED twice over

First, it saturates. λ ∈ {0.5, 2, 10, 50, 200} — a 400× range — all give turnover 0.1,
exposure 0.167, return −0.0367. The fine bracket below 0.5 does not help: λ=0.001 already
gives turnover **2.2**, under Arm C's window of 7.9–31.4.

| lam | log_ret | mean_w | turnover |
|---|---|---|---|
| 0.001 | −0.0169 | 0.074 | 2.2 |
| 0.005 | −0.0236 | 0.107 | 0.2 |
| 0.02 | −0.0297 | 0.135 | 0.1 |
| 0.08 | −0.0343 | 0.156 | 0.1 |
| 0.3 | −0.0360 | 0.164 | 0.1 |

Second, and disqualifying regardless of the numbers: **it changes the policy architecture.**
`turnover_control="penalty"` builds a *different* model — no `nn.Linear(hidden, 2)` band head,
and `__call__` returns the raw exposure instead of running `band_hold`, the sequential scan
that holds the previous position. The tell was a 12× speedup: penalty runs take 14.5–18.7s
against 168–205s for band runs, because the band's per-timestep Python scan is gone.

A control that differs from `pnl_control` in architecture is not a turnover-matched control.
The design says "differs in turnover **only**", so these runs are inadmissible on their face.

### Lever 3 — `band_width`: the correct one

`ClassifierEncoderPolicy` already accepts `band_width` (default 0.10) and applies it at
`policy.py:145` as `softplus(band(z)) * band_width`. It scales the learned no-trade zone
with the **architecture unchanged** — same band head, same `band_hold` scan, same parameter
count — which is exactly what the design requires. It was simply never exposed through
`ArmConfig`.

Bracket: `band_width` ∈ {0.1 (baseline), 0.3, 1.0, 3.0, 10.0}, seed 1.

### If lever 3 also fails

Then the honest result is that **this trainer cannot produce a turnover-matched control**,
and the confound is *unresolvable* with the current policy rather than resolved in either
direction. That is reported as the finding. It is not licence to compare against an
unmatched control.
