# Path-aware gbb label — oracle gate screen

> **⚠️ FIGURES CORRECTED 2026-09-07 — see `regime/ORACLE_CORRECTION_2026-09-07.md`.**
> Every oracle figure below was measured on an instrument that entered up to TWO
> BARS EARLY, because `augment_training_signals` dilates buy labels backward and
> the oracle read the labels after augmentation. All ceilings were biased LOW.
> **Realised-return results are unaffected** (real models never read the
> augmented labels as an entry signal) and stand as written.

**Status**: gate PASSED 2026-09-07. Training is licensed but NOT yet spent.
Labeler: `LabelMethod.gbb_path` (27), added additively — `gbb` (17) is
byte-for-byte unchanged and every prior result stays reproducible.

## What the label does

One condition added to each side of `labels_gbb`, mirrored:

- **BUY**: the worst drawdown BEFORE the target is first reached must not breach
  `stop_frac` (default −0.05). Removes labels that certify a gain the ATR stop
  would have pre-empted.
- **SELL**: the best rise BEFORE the fall is first reached must stay below
  `take_profit` (default 0.022). Removes labels the take-profit would have
  pre-empted, which the sell head could never have acted on.

Verified as a **strict subset** of the incumbent on both sides, and monotonic in
both barriers (looser stop keeps more; looser take-profit keeps more). On SOL it
removes 3.9% of buy labels and 14.4% of sell labels, matching the standalone
measurements in `path_label.py` (4.2%) and `path_label_sell.py` (11.4%).

## The gate (`ORACLE_GATE.md`): PASS = W1 guards-off return > 0 AND ≥200 trades

| label | trades | return | max DD | PF | win | verdict |
|---|---|---|---|---|---|---|
| `gbb` 0.007 | 2126 | −28.113% | 31.85% | 0.796 | — | **FAIL** |
| `gbb` 0.02 | 1561 | +17.791% | 7.63% | 1.200 | 60.5% | PASS |
| **`gbb_path` 0.02** | 1560 | **+23.991%** | 8.75% | **1.282** | **61.2%** | **PASS** |

**+6.200pp over the incumbent, on ONE fewer trade.** That is the important
shape: the constraint does not buy its improvement by trading less. Guards-off
the oracle is slot- and capital-constrained, so removing bad labels lets better
ones through — count holds, quality rises. Drawdown is slightly worse
(7.63% → 8.75%), which is the one figure that moved the wrong way.

## Guards-on ceiling (at NNMT's own gate)

| window | `gbb` 0.02 | `gbb_path` 0.02 | Δ |
|---|---|---|---|
| **W1** | 145tr +8.733% | 143tr **+10.704%** | **+1.971pp** |
| W2 | 292tr +9.053% | 286tr **+14.913%** | +5.860pp |
| W3 | 98tr +2.747% | 98tr +2.747% | +0.000pp |

W3's returns coincide to three decimals while the trade hashes differ, so the
sets are not identical — treat W3 as uninformative here rather than as evidence
of inertness.

Note the guards-on W1 ceiling now **exceeds the real model's +10.708%** for the
first time in this feature. Under `gbb` 0.02 the ceiling was +8.733%, i.e.
BELOW the real model — the headroom was negative. Under `gbb_path` it is
+10.704%, level with it. Headroom is no longer negative; it is approximately
zero.

## What this licenses, and what it does not

**Licensed**: two retrains (seeds 42 and 7) at `gbb_path`, compared against the
existing `NNMT_Lbl02_s42/s7`. That is the gate's entire purpose — it refuses
retrains on labels that cannot pay, and this one can.

**NOT licensed — and this feature has earned the caution the hard way**: any
expectation that the model will realise it. Every ceiling movement measured here
has failed to translate:

| change | ceiling movement | realised on W1 |
|---|---|---|
| label 0.007 → 0.02 | **+45.9pp** guards-off (FAIL → PASS) | **−0.157pp** |
| label 0.007 → 0.02, sell head live | same | **+0.086pp** |
| `gbb` → `gbb_path` | **+6.2pp** guards-off | **UNMEASURED** |

A 46pp ceiling gain bought −0.16pp of realised return on this family. There is
no basis for assuming +6.2pp buys more. The gate says the label COULD pay; only
a retrain says whether the model learns it, and the honest prior from this
family's own record is that it will not.

The one structural reason to be marginally less pessimistic: the earlier null
was measured when the model's entry opinion did not reach the trades at all
(byte-identical trade hashes). The sell head is now live, so a label change has
a live channel to act through — which is what made the 0.02 re-test move from
INCONCLUSIVE to a measured null.

---

## THE RETRAIN RESULT (2026-09-07) — NULL on the acceptance window

Two models trained (`MLX_SEED` 42/7, 49 and 53 min), compared against
`NNMT_Lbl02_s42/s7` — identical in every respect except `TRAINING_TYPE`.
Evaluated in both exit regimes. 12 arms, all `returncode` 0.

### Exit signal ON (the promoted production regime)

| window | gbb s42/s7 | path s42/s7 | **path effect** | seed spread | paired 2sd |
|---|---|---|---|---|---|
| **W1** (acceptance) | +11.56 / +11.58 | +11.45 / +11.51 | **−0.091pp** | 0.067 | 0.16 |
| W2 (in-sample) | +10.78 / +10.47 | +10.92 / +10.60 | +0.138pp | 0.319 | 0.22 |
| W3 (in-sample, near-flat) | +1.01 / +1.19 | +1.45 / +1.73 | +0.486pp | 0.280 | 1.51 |

### Exit signal OFF (the control regime)

| window | path effect | note |
|---|---|---|
| **W1** | **+0.000pp** | **BYTE-IDENTICAL trades, both seeds** |
| W2 | +0.048pp | identical on seed 42 |
| W3 | +0.527pp | differs |

**Verdict: the path constraint is a NULL on W1.**

With the exit signal OFF the label does not reach the trades at all — W1 trade
hashes are byte-identical for both seeds, which is INCONCLUSIVE under the
standing protocol, not refuted. With the signal ON it does reach the trades in
every window, and the W1 effect is **−0.091pp against a 0.16pp band**: within
noise, and pointing the wrong way.

## The scoreboard this completes

**A pre-registered prediction, and it held.** Before spending the retrains, the
recorded prior was "another null", on the grounds that this family has never
converted a ceiling gain into realised return:

| change | ceiling movement (W1 guards-off) | realised W1 |
|---|---|---|
| label 0.007 → 0.02 | **+45.9pp** (gate FAIL → PASS) | −0.157pp |
| label 0.007 → 0.02, sell head live | same | +0.086pp |
| **`gbb` → `gbb_path`** | **+6.2pp** (PASS → PASS) | **−0.091pp** |

Three ceiling movements totalling ~52pp have produced −0.16, +0.09 and −0.09pp
of realised return. And `val_trading_mcc` pointed at the LOSING label both
times it was consulted (0.76 vs 0.61 for the old threshold; 0.61 vs 0.59 for
the incumbent labeler), so it was right about learnability and useless about
return, exactly as FR-019 assumes.

## Was the gate wrong to license this?

No, and the distinction matters. `ORACLE_GATE.md` states its own scope: a PASS
says a label **could** pay, not that a model **will** learn it. Its job is to
refuse retrains on labels that cannot pay under any model — which it did, on
`gbb` 0.007 (−28.1%, FAIL). It correctly declined to refuse this one.

The gate is a NECESSARY-not-sufficient screen and behaved as designed. The cost
of the licence was 100 minutes of training, against a screen that has already
saved far more by killing 0.007 before anyone trained on it.

## Where this leaves the path label

`LabelMethod.gbb_path` **stays in the codebase, unused by any production
strategy.** It is additive, `gbb` (17) is untouched, and the labeler is
correct — verified as a strict subset, monotonic in both barriers, and matching
the standalone excursion measurements. It is a better TARGET on the evidence of
the ceiling. It is not a better STRATEGY.

Do not re-open it on this family without new information. The one thing that
would justify revisiting: a configuration in which the model's opinion actually
drives entries — which this feature has established NNMT does not have, and
cannot reach by opening its guards (−34.15pp).

**Residual open item**: W3 is the only window where the path label helps
(+0.49pp exit-on, +0.53pp exit-off, consistent across both seeds and both
regimes). W3 is in-sample, near-flat, and inside its own 1.5pp band, so it
carries no weight — but it is now the THIRD consecutive exit-side change where
W3 has disagreed with W1. That pattern is worth a look on its own terms rather
than being filed as noise a third time.
