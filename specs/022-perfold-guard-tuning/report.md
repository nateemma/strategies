# Feature 022 — Per-fold guard tuning: result

**Status**: terminated at the gate, as designed. 23 of 40 tasks; Phases 6–7 are unreachable.
**Total compute**: one grid scoring of **7 seconds**, plus no backtests at all.

---

## Result 1 (US1) — the premise was an artefact

Arm B's **+18.76pp**, the number this feature was built to explain, is **97.8% in-sample**. Its
guard values were hyperopted on `20240511-20260501`
(`strategy_NNNC_CGP_MLX_2026-05-31_15-03-45.fthypt`, 6-of-6 exact match), a window covering
**9 of 15 folds and 53.5% of all test-window days**.

| | folds | wallet |
|---|---|---|
| inside the tuning window | 8 | **+18.34pp** |
| outside it | 6 | **+0.42pp** |

The three largest contributors — f07, f11, f08, together 70% of the total — are all three
inside it. Out of sample, tuned guards earn **+0.07pp/fold against arm A's +0.18pp**: no better
than open guards, and both far below the 3.350 pp/yr floor. **There was never a +18.76pp to
explain.**

## Result 2 (Phase 4) — the proxy is trustworthy, and measured so

| arm | proxy_score | measured live EV |
|---|---|---|
| A | +0.000461 | +0.000155 |
| B | **+0.013955** | **+0.014671** |

Within 5% on arm B, right sign on A, and it independently reproduced arm B's **zero-trade f10**
as `nan`. Audited before use because feature 021's experience with this instrument was a 3.6e-5
match on one population and the **wrong sign** on another.

## Result 3 (US2) — the gate: INDETERMINATE

Under FR-017 a fold is powered only if its **selected** configuration clears 25 trades on its
own test window. **3 of 15 do** (f03, f04, f11). Spec **Branch 0 fires**: fewer than 10 powered
⇒ `indeterminate`, and no other branch is evaluated.

**A dynamic scheme is not refuted here — it is not assessable.** The per-fold optimum chosen on
a training window trades too little on the matching test window to be measured in 12 of 15
folds. That is a weaker claim than "adaptivity loses", and it is the honest one.

### A false positive this nearly produced, and the correction

The first implementation reported **`trackable`** with a paired interval excluding zero. It was
wrong: the comparison ran on **n=3 folds**, and in 2 of those 3 the adaptive regret was
*structurally* **0.00000** — the adaptive pick was the only candidate surviving the test-side
floor, so it was compared against itself. The power test had been applied to candidate
*availability on train* instead of to the *selected configuration on test*.

Branch 0 exists precisely to stop this, and it was added during `/speckit-analyze` as finding
A3 — four hours before it caught a live false positive that would have licensed the expensive
phase. The gate then refused arm C mechanically (T010), exiting 2 and quoting the verdict.

## Result 4 — the answer to "dynamic, or a safe middle range?"

Three strategies over the same 64-point grid. **`proxy_score`, not a return** (FR-018).

| strategy | config | folds measurable | mean | worst fold |
|---|---|---|---|---|
| **ADAPTIVE** per-fold train optimum | varies, 4 distinct | **3 / 15** | — | — |
| **BEST CONSTANT** (hindsight, max mean) | atr 0.02, adx open, rvol open | 10 / 15 | **+0.01091** | −0.00650 |
| **ROBUST / MAXIMIN** (max worst fold) | **atr 0.006, adx open, rvol 1.0** | **15 / 15** | +0.00196 | **−0.00121** |

**A safe middle range exists, is identifiable, and is the only one of the three measurable in
every fold.** The robust configuration has a **5× milder worst fold** than the best constant
(−0.00121 vs −0.00650) at a fifth of the mean — the classic mean-versus-robustness trade, and
the robust side is the only one that never stops trading.

**Where the middle actually is, which is not the middle of the grid**: the four configurations
measurable in all 15 folds **all hold `atr_pct` open** and differ only in `rvol`. Tightening
`atr_pct` is what collapses trade counts below measurability — consistent with it being the
heaviest lifter in the ablation (−7.33pp) and therefore the most selective. So the safe region
is *`atr_pct` open or near-open, `adx` open, `rvol` mildly positive (~1.0)* — and `rvol` is the
one axis that can be tightened without destroying power, which matters because `rvol` is a
mandatory tradeability constraint rather than a quality guard.

## Corrections made during this feature

- **`entry_enable_guards` does NOT gate `volume > 0`.** Commit `e7cf8cf` (2026-09-08,
  "volume+rvol mandatory") moved `volume > 0` and `rvol` outside the flag; only the five quality
  thresholds sit behind it. BaseStrategy's `buy_params` comment still claims otherwise and is
  **stale**. I cited that comment as current fact in FR-008, in `TriggerOnly.py` and in feature
  021's report, and the correction is now recorded in all three. Nothing measured changes —
  feature 021's arms never lost the volume check, which is the better explanation of their
  identical `trades_hash` than the backstop I originally proposed.
- **`rvol` is a mandatory constraint, not a quality guard**, despite being one of the three
  tuned axes.
- **`atr_pct` and `adx` are the heaviest lifters, not `rvol`.** The feature was proposed
  assuming rvol dominant; that holds for its share of gate *rejections*, not for return.

## What remains open

Arm D's **+39.98pp at 15/15 folds** is still explained by nothing measured. It is *not*
in-sample contaminated — its per-fold thresholds were selected on each fold's own training
window, verified as 15 distinct values. That is now the only live thread from features 021 and
022 combined, and it is a model-side question, not a guard-side one.

## Outstanding, not done

T024–T035 (arm C, arm E) are unreachable by FR-006. T037–T040 (polish) remain: the Principle
VIII deviation note, the full-suite run, and propagating this outcome into feature 021's
follow-up section.
