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

---

# CORRECTION (2026-10-10) — the floor was mis-scaled, and it was my defect not the adaptation's

The user asked what was stopping the adaptive scheme from firing, and whether that meant the
adaptation was wrong. It did not. **Two defects, both in my implementation.**

1. **The power test checked the wrong thing** — candidate *availability on train* rather than
   the *selected configuration on test*. This is what produced the first run's false
   `trackable` on n=3.
2. **FR-012's floor was applied as an absolute count to both windows.** This harness's train
   windows are **anchored and growing** — 730 days at f00 to 1990 at f14 — against 90-day test
   windows: **8.1× to 23.4× longer, 15.2× on average**. A flat 25 on a train window therefore
   admits configurations trading **~1.7 trades per 90 days**, which can never clear 25 on test.
   **The adaptive scheme was selecting unmeasurable configurations because nothing stopped it.**

Rate-matching the floor to 25 trades per 90-day-equivalent (203 at f00, 378 at f07, 553 at f14)
took powered folds from **3 to 9**. FR-012 is amended and `floor_for_window()` now implements
the rate in code.

## The verdict stands, and deliberately

**9 of 15 powered, against a pre-registered threshold of 10 ⇒ `indeterminate`.** FR-019 forbids
relaxing the threshold after seeing the result, and 9 is exactly the case the rule exists to
govern. I have not moved it.

## But the corrected run answers the real question

**The per-fold optimum is strongly concentrated**, which is the trackability question answered
descriptively even though the gate is formally indeterminate:

| configuration | folds selecting it |
|---|---|
| `atr 0.013 / adx 20 / rvol 1.9` | **8 of 15** |
| `atr 0.02 / adx open / rvol 1.0` | 3 |
| `atr 0.013 / adx 10 / rvol 3.0` | 2 |
| `atr 0.013 / adx 20 / rvol 1.0` | 1 |

Eight of fifteen folds pick the **identical** configuration and all fifteen pick one of **four**.
A per-fold optimum that concentrated is close to a constant — **there is very little to adapt
to**, which supports a fixed safe setting over a dynamic scheme and is consistent with the
robust/maximin result above.

**There is a real early/late shift** (Principle II): f00–f09 concentrate on
`atr 0.013 / adx 20 / rvol 1.9`, while f12–f14 move to `atr 0.02 / adx open / rvol 1.0`. So the
optimum has roughly **two regimes across 15 folds**, not fifteen — which is the shape under
which a two-state scheme might work and a per-fold scheme is overkill.

Note the selected `atr 0.013` is the **historical hyperopt value**, recovered independently here
from training windows that in six folds do not overlap the 2024-05-11…2026-05-01 tuning window.
That is mild out-of-sample support for the *level* of `atr_pct`, which is a different and much
weaker claim than arm B's +18.76pp return.

---

# Two-state scheme — it degenerates to a constant, and that constant loses

Run as Principle VII's trackability test applied to a **coarser** scheme, not as a bypass of the
per-fold gate. Pre-registered before looking at any result: **one** indicator (mean
`atr_pct_roll` over the fold's own TRAIN window, pooled across pairs — computable at decision
time, already a column the guards read), split at its **median across folds**, each regime's
config chosen **leave-one-out** so the fold under test never contributes to choosing its own
config, rate-matched floor throughout.

**The hindsight split was deliberately NOT used.** The corrected gate run showed f00–f09
favouring one config and f12–f14 another; splitting at f10 *because of that* and scoring on the
same folds fits the boundary on the outcome, which is what Principle VII forbids on a handful of
windows.

## Finding 1 — there is no second state

| regime | config selected (leave-one-out) |
|---|---|
| HI volatility | `atr 0.013 / adx 20 / rvol 1.9` |
| LO volatility | **`atr 0.013 / adx 20 / rvol 1.9`** |

**Both regimes pick the identical configuration.** The two-state scheme *is* a constant. The
volatility indicator carries no information about which guard setting to use, so there is no
second state to switch to — which is a cleaner negative than a close-run comparison would have
been.

Why: the pre-registered indicator does not reproduce the drift I saw in the hindsight optimum.
The HI/LO labels **interleave** (HI HI HI LO LO HI LO LO HI HI HI LO HI LO LO) rather than
forming the early/late blocks the hindsight optimum suggested. An indicator chosen *because* it
reproduced that pattern would be the fitted mapping the constitution prohibits, so this limit is
accepted rather than searched around.

## Finding 2 — as a constant, it loses

Over the 9 powered folds (n = 8 usable in the paired comparison):

| strategy | mean regret |
|---|---|
| **best constant** (hindsight) `atr 0.02 / adx open / rvol open` | **+0.006765** |
| two-state (= `atr 0.013 / adx 20 / rvol 1.9`) | +0.017752 |
| robust / maximin `atr 0.006 / adx open / rvol 1.0` | +0.017952 |

**Paired two-state − constant = +0.010987, 95% CI [+0.000199, +0.021735] — excludes zero.** The
two-state scheme is significantly *worse* than a single fixed value.

Note the best constant is chosen in **hindsight** and is therefore not implementable; the honest
deployment comparison is two-state against robust, and those are **indistinguishable on mean
regret (+0.0178 vs +0.0180)** while robust has the better worst fold. So nothing is gained by
switching.

## What this closes

Adaptivity has now failed at **both** granularities on the same grid: per-fold is
`indeterminate` (9/15 powered, below the pre-registered 10), and two-state degenerates to a
constant that is beaten. The constitution records four adaptive schemes having lost to static
values in this project; **this is the fifth**, and the only honest remaining question about guard
values is *which constant*, not *whether to switch*.

On that question the evidence points one way: the robust/maximin setting —
**`atr_pct` open-to-near-open, `adx` open, `rvol` mildly positive (~1.0)** — is the only
configuration measurable in all 15 folds, with a worst fold 5× milder than the
mean-maximising constant.

---

# WHICH CONSTANT? — and a correction to the "safe middle range" above

Asked whether `BaseStrategy`'s guard defaults should change. **On this evidence, no — and the
analysis argues for keeping them.**

## The robust config I reported was a HINDSIGHT selection

The maximin ("safe middle range") above was computed over **TEST**-window scores. That is a
selection made with knowledge of the outcome — the same flaw that makes arm B's +18.76pp 97.8%
in-sample. Re-selected over **TRAIN** windows only, maximin and max-mean both pick a *different*
configuration, and it is materially worse out of sample:

| selection | config | folds measurable | mean | **worst fold** |
|---|---|---|---|---|
| maximin on **TEST** *(hindsight — what was first reported)* | atr 0.006 / adx open / rvol 1.0 | 15/15 | +0.00196 | −0.00121 |
| maximin on **TRAIN** *(leakage-free)* | **atr 0.013 / adx 20 / rvol 1.9** | **10/15** | +0.00467 | **−0.02457** |
| max-mean on TRAIN *(leakage-free)* | same | 10/15 | +0.00467 | −0.02457 |
| **CURRENT DEFAULT** (all open) | atr −1 / adx −1 / rvol −1 | **15/15** | +0.00046 | **−0.00225** |

Honestly selected, the "robust" constant has a worst fold **~10× worse** than the current
default and trades in a third fewer folds. On the properties that matter for a *default* — never
being disastrous, and continuing to trade — **the existing all-open setting is the better of the
two legitimate candidates.** That independently supports the deliberate 2026-09-06 choice to
open the guards "for headroom and measurability".

## Why no default is changing on this evidence regardless

- **These are `proxy_score` figures, not returns** (FR-018). The arm below is what converts one
  into a return.
- **The grid held 3 of 5 quality thresholds fixed**, so "which constant" is answered only within
  that slice.
- **`BaseStrategy.buy_params` is inherited by the live dry-run.** Changing it is a behaviour
  change to a running 28-day dry-run.

## Arm F — `GbbRobustOpen`, the static TRAIN-selected constant

Static, therefore **not** gated by Principle VII: the gate blocks adaptivity, and a fixed
configuration is an ordinary arm. The values sit in the strategy class rather than in
`run_folds.py`'s `guards` plan key, which is why `_require_trackable_verdict` is not involved —
a useful demonstration that the gate discriminates correctly rather than blocking everything
guard-shaped.

**Pre-registered expectation**: higher mean than the default, ~10× worse worst fold, trading in
10 of 15 folds. **Grounds to change a default** would be beating arm A (+2.64pp, 7/15) by more
than the 3–4pp/yr floor with a paired interval excluding zero — and a higher mean bought with a
10× worse worst fold is not grounds.

## Arm F result — promising, NOT resolved, and my prediction was wrong in both directions

| arm | wallet | pp/yr | folds + | trades | worst fold |
|---|---|---|---|---|---|
| **A** guards open, no model | +2.64 | +0.72 | 7/15 | 8,150 | **−10.61** |
| **F** static clean-selected constant | **+25.34** | **+6.89** | **13/15** | 458 | **−2.00** |

### The selection is clean, and that was checked rather than assumed

A constant selected from the train scores of **all** folds would be contaminated: **14 of 15
test windows lie inside f14's train window** (anchored design), so such a selection has seen
almost every fold's test data — arm B's error one level up. The only fully leakage-free static
selection uses **f00's train window alone** (`20210101-20230101`), which precedes every test
window. **It picks the identical configuration, ranked 1 of 52.** So arm F's config is
recoverable from data ending 2023-01-01 and evaluated from 2023-01-01 onward.

Residual caveat: the `atr 0.013` *level* originated in the contaminated 2024–2026 hyperopt, so
the grid's candidate set was informed by it even though the choice among candidates was clean.
The pre-2023 data independently ranking it first is reassuring but not the same as having
proposed it from clean data.

### My pre-registered prediction was wrong, and in the opposite direction

I recorded, before the run: *"higher mean, ~10× worse worst fold, 10/15 folds powered."*

- worst fold: **−2.00pp against arm A's −10.61pp — 5× BETTER, not 10× worse** (ratio 0.2×)
- powered folds: **6/15**, not 10
- wallet: +25.34pp, far above what the proxy's mean implied

The proxy got the *direction* of the mean right and the *worst-fold* prediction badly wrong.
That is a second measured limit on the instrument, alongside feature 021's sign error.

### Why this is not yet grounds to change a default

**Paired per-fold delta F−A = +6.17 pp/yr, 95% CI [−8.19, +20.65] — the interval SPANS ZERO.**
Arm F beats arm A in 10 of 15 folds and clears the floor on the point estimate, but the
fold-level bootstrap does not resolve it, largely because arm A carries a −10.61pp fold.

**The instrument used here is not the one FR-015 specifies.** FR-015 requires
`bootstrap_delta` on the **symmetric difference of trades**, which is far tighter than a
bootstrap over 15 fold-level observations — "shared-trade arms are PAIRED; a flat 1pp floor
overstated an oracle arm's resolution 4.7×". Running it needs the trade lists from the backtest
artefacts and has not been done.

**Pooled power is adequate** — 458 trades against the 378 the 3.350 pp/yr floor was derived
from — even though only 6 of 15 folds individually clear the 25-trade per-fold floor.

### Revised answer on the defaults question

Earlier in this report I answered **no** on the basis of a proxy prediction of a 10× worse worst
fold. **That prediction was wrong**, so the basis for the "no" is gone. The honest position now
is **not resolved, and favourable in direction**: +6.89pp/yr with a *better* worst fold than the
current default and 13/15 folds positive, against a paired interval that spans zero.

The next step is the FR-015 trade-level paired test, not a defaults change. And even if that
resolves, `BaseStrategy.buy_params` is inherited by the live dry-run, so the change belongs to a
deliberate promotion decision rather than to this feature.

### FR-015 trade-level paired test — NOT RESOLVED, p = 0.27

Run with `evaluation/bootstrap.py::bootstrap_delta` on the symmetric difference of trades, the
instrument FR-015 specifies, from the repo root (its `BT_RESULTS` path is relative — my first
invocation reported every archive as "pruned" when the real issue was my working directory).

```
observed  +22.70pp      95% CI [-16.34, +62.45]      p = 0.2706      resolved: False
shared 111   arm_F_only 347   arm_A_only 8039   identical 108
annualised +6.17 pp/yr  CI [-4.44, +16.97]
```

**The paired instrument cannot help here, and the reason is structural.** `bootstrap_delta` is
tight when two arms *share* their entries — its own docstring: "for an exit-side change sharing
every entry, only the handful of genuinely differing trades carry variance". Arm F and arm A
**share 111 trades out of ~8,500**. Arm A contributes 8,039 unique trades, including the fold
that lost 10.61pp. So the comparison is effectively unpaired and the interval is dominated by
arm A's own variance.

This is also the *"judge at a powered operating point"* trap in its original form: arm A trades
**8,150** and arm F **458**, an 18× difference in selectivity. Two arms that far apart in trade
count are not resolvable against each other by this harness, whatever their point estimates.

## FINAL ANSWER on changing the base-class defaults: NO, on the evidence available

Arm F is better on **every point estimate** — wallet +25.34pp vs +2.64pp, 13/15 folds positive
vs 7/15, worst fold −2.00pp vs −10.61pp — and the configuration was selected leakage-free from
data ending 2023-01-01 and ranked 1 of 52 there. That is genuinely encouraging.

But **p = 0.2706 and the interval spans zero by a wide margin.** This repository's standing rule
is that a P&L claim comes from the pooled harness and that sub-floor differences are not
resolvable; here the difference is *above* the floor on the point estimate and still
unresolvable, because the two arms barely share trades.

So: **no defaults change.** `BaseStrategy.buy_params` is inherited by a live dry-run, and
changing it on an unresolved comparison would be exactly the kind of promotion this project has
repeatedly recorded as premature.

**What would resolve it** — and this is a separate study, not an extension of 022: compare at a
*matched operating point*. Arm A at open guards trades 18× more than arm F; restricting arm A to
a matched selectivity (or running arm F with loosened thresholds to a matched trade count) would
make the paired instrument work as designed, because the entry sets would then overlap. Until
then the honest statement is that the tightened guard constant **looks better on every measure
and is not demonstrably better on any of them.**

### MATCHED OPERATING POINT — the guards' selection carries real information (p < 1e-5)

The guards *are* the selector, so selectivity cannot be held fixed while the guards change. The
matched test is therefore the `random_matched` control pattern feature 021 already used: hold
trade COUNT fixed and vary only WHICH trades.

```
arm A          8,150 trades   mean profit_ratio +0.000155
arm F            458 trades   mean profit_ratio +0.008961

458 drawn at random from arm A, 20,000 times:
  mean +0.000164   95% range [-0.003266, +0.003480]
  arm F: +5.2 SD above that distribution   one-sided p = 0.00000
```

**Decomposition**: F's 458 trades at A's per-trade quality would be worth **+0.15pp**; they are
actually worth **+25.34pp**. The advantage is **quality, not trade count.**

Caveat: random subsampling does not model concurrency — a genuinely 458-trade version of arm A
would free slots and could admit different bars. The null tested is "a random 458 of the trades
arm A actually took", which is the right null for *is F's subset better than an arbitrary
subset*, and is not a simulation of a sparser arm A.

## REVISED FINAL ANSWER on the defaults — and I have moved twice, so here is the ledger

| claim | status |
|---|---|
| Tightened guards select better bars than random at matched count | **ESTABLISHED** — +5.2 SD, p < 1e-5 |
| The config was chosen leakage-free | **ESTABLISHED** — f00 train only (ends 2023-01-01), ranked 1 of 52 |
| Arm F's wallet, folds-positive and worst fold all beat arm A | **ESTABLISHED as point estimates** — +25.34 vs +2.64pp, 13/15 vs 7/15, −2.00 vs −10.61pp |
| The wallet DIFFERENCE is statistically resolved | **NOT ESTABLISHED** — p = 0.2706, the arms share 111 of ~8,500 trades |

**My position changed twice and both changes were forced by evidence, not preference.** First
"no", on a proxy prediction of a 10× worse worst fold — that prediction was wrong by a factor of
50 (−2.00pp actual against a predicted ~−22pp). Then "not resolved, favourable". Now: **the
mechanism is demonstrated and the magnitude is not.**

So the honest recommendation is no longer "no". It is: **this is a legitimate promotion
candidate, and the decision is a deliberate one rather than an inference from this feature.**
What argues for it — a selection mechanism proven at p < 1e-5, a leakage-free selection, better
performance on every point estimate including drawdown. What argues for caution — the wallet
delta is formally unresolved, `atr 0.013` originated in a contaminated hyperopt even though
clean data independently ranks it first, three of five quality thresholds were never varied, and
`BaseStrategy.buy_params` is inherited by a live 28-day dry-run.

**What would close it**: arm F re-run on a second, temporally distant universe or era — the
persistence requirement (Principle II) that no single 15-fold result satisfies on its own.

### PRINCIPLE II PERSISTENCE — the selection holds in both halves; the WALLET edge does not

There is **no temporally distant unused era** in this dataset: `binanceus` history is either
selection (pre-2023) or evaluation (2023–2026), `data/binance/` is empty, and new downloads are
out of scope. But the config was selected on data ending **2023-01-01**, so **all 15 folds are
already out-of-sample** — splitting them gives two temporally distinct clean windows, which is
the persistence test.

| | arm F | arm A |
|---|---|---|
| **EARLY** f00–f07 (2023-01 → 2025-03) | **+16.82pp**, 7/8 folds, 255 tr, EV +0.010162 | **+20.43pp**, 5/8, 4,513 tr, EV +0.001079 |
| matched-count (255 drawn from A) | **+4.2 SD, p = 0.00000 — HOLDS** | |
| **LATE** f08–f14 (2025-03 → 2026-09) | **+8.53pp**, 6/7 folds, 203 tr, EV +0.007452 | **−17.78pp**, 2/7, 3,637 tr, EV −0.000992 |
| matched-count (203 drawn from A) | **+3.1 SD, p = 0.00040 — HOLDS** | |

**Finding 1 — the selection effect IS persistent.** Arm F's trades beat a matched-count random
draw from arm A in **both** halves, at p < 1e-5 and p = 0.0004. Principle II is satisfied for the
claim *"tightening these guards selects better bars"*. That is now established on two distinct
out-of-sample periods, not one.

**Finding 2 — the WALLET advantage is NOT persistent, and this matters.** Arm A's wallet
**beats** arm F in the early half (+20.43 vs +16.82) despite far worse per-trade quality, because
it trades 18× more. Arm F only wins the full period because **arm A collapses in the late half
(−17.78pp)**. So the +25.34 vs +2.64 headline is driven by one period, and the per-trade quality
finding is the durable part — not the wallet gap.

**Finding 3 — but arm F is the only one positive in both halves.** F: +16.82 and +8.53. A: +20.43
then −17.78. Combined with the worst-fold figures (−2.00pp vs −10.61pp), the case for the
tightened constant is **robustness**, not expected return: it gives up the better early-period
wallet in exchange for not having a losing regime.

## THE ANSWER, final

**For a default, the evidence now supports the tightened constant — on robustness grounds, with
the expected-return claim explicitly withdrawn.**

- *Established, twice, out of sample*: the selection carries information (p < 1e-5, p = 0.0004).
- *Established*: positive in both halves, worst fold 5× milder than the open default.
- *Withdrawn*: that it earns more. Arm A wins the early half, and the full-period gap is one
  regime's collapse. The paired full-period delta remains p = 0.2706.

Remaining caveats, unchanged: `atr 0.013` originated in a contaminated hyperopt (clean data
independently ranks it first); three of five quality thresholds were never varied; and
`BaseStrategy.buy_params` is inherited by a live dry-run, so promotion is a deliberate decision.
