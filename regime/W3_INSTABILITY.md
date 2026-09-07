# Why W3 disagrees — it is a near-zero net, not a different regime

Measured 2026-09-07, after W3 contradicted W1 on three consecutive exit-side
changes. The pattern looked like it might mean something. It does not.

## The three disagreements

| change | W1 | W3 |
|---|---|---|
| production take-profit interaction | +1.281pp | **−0.748pp** |
| label 0.02 vs 0.007, sell head live | +0.086pp | **−0.453pp** |
| `gbb` → `gbb_path` | −0.091pp | **+0.486pp** |

Three in a row on the same side of the strategy is the kind of thing that
usually indicates a structural difference. Here it indicates arithmetic.

## The measurement

| window | trades | gross win | gross loss | NET | **net/gross** | take-profit exits | worst 4 losses |
|---|---|---|---|---|---|---|---|
| W1 | 176 | 2,492 | −1,348 | 1,145 | **29.8%** | 32 (18%) | −262 |
| W2 | 347 | 3,337 | −2,285 | 1,052 | 18.7% | 40 (12%) | −407 |
| **W3** | 127 | 1,092 | −940 | **152** | **7.5%** | 7 (6%) | **−426** |

**W3 retains 7.5% of its gross flow as profit, four times less than W1.** Its
net (+152) is a small residual of two large opposing numbers, and its four worst
trades alone total −426 — **2.8x the entire window's profit**.

Any perturbation that reshuffles a handful of trades therefore moves W3's
percentage return disproportionately, and the SIGN of that move is close to
arbitrary. That is the whole explanation for the three "disagreements".

## The secondary cause: the take-profit barely fires in W3

Take-profit exits are 18% of trades in W1, 12% in W2, and **6% in W3**. Prices
in that window rarely reach +2.2% inside the hold, so the exit mix shifts toward
stops and the ROI table. That is also why the take-profit INTERACTION showed up
most strongly on W3 — there is least take-profit activity there to begin with,
so changing it perturbs a thin, fragile balance.

## How to apply this

- **Do not read a W3 sign-flip as evidence about a change.** Its net/gross of
  7.5% makes it the least informative of the three windows on this family, which
  is a sharper statement of research R7's "W3 is near-flat".
- **Report W3 with its net/gross ratio**, not only its trade count. A window can
  have a respectable trade count and still be uninformative if its net is a thin
  residual — 127 trades sounds adequate and is not.
- **This is not a reason to drop W3.** It remains the only temporally distant
  era available (Constitution II), and a change that HURTS it badly is still a
  warning. What it cannot do is adjudicate sub-1pp effects.
- The same test is worth running on any window before treating its verdict as
  informative: `net / (gross_win − gross_loss)`.

Closes the residual item recorded in `PATH_LABEL_GATE.md`.
