# NNNC: path-aware label and exit signal — both REJECTED, no retrain spent

> **⚠️ FIGURES CORRECTED 2026-09-07 — see `regime/ORACLE_CORRECTION_2026-09-07.md`.**
> Every oracle figure below was measured on an instrument that entered up to TWO
> BARS EARLY, because `augment_training_signals` dilates buy labels backward and
> the oracle read the labels after augmentation. All ceilings were biased LOW.
> **Realised-return results are unaffected** (real models never read the
> augmented labels as an entry signal) and stand as written.

Measured 2026-09-07 at the request to try `gbb_path` on NNNC with exit signals
enabled. Both questions answered with backtests only. **No training was spent,
and the screens are why.**

## 1. The path-aware label: PASSES the gate, but the gain does not survive the guards

| W1 guards-off (the standing gate) | trades | return | PF | verdict |
|---|---|---|---|---|
| `gbb` (incumbent) | 1333 | +26.544% | 1.368 | PASS |
| `gbb_path` | 1342 | **+29.656%** | 1.424 | PASS |

**+3.112pp, and on MORE trades** — a better shape than NNMT, where the same
constraint gained +6.2pp on one fewer trade.

But the gate is measured guards-OFF, and NNNC does not run guards-off. At its
ACTUAL gate:

| W1 guards-on | trades | ceiling | headroom vs real |
|---|---|---|---|
| perfect entry, `gbb` | 85 | +13.934% | **+3.488pp** |
| perfect entry, `gbb_path` | 84 | +13.596% | +3.150pp |
| real model | 95 | +10.446% | — |

**`gbb_path`'s ceiling is 0.338pp LOWER at the gate NNNC actually runs.** The
entire guards-off improvement is absorbed by the guard chain, which was already
rejecting the marginal bars the path constraint removes.

**Verdict: do NOT retrain NNNC on `gbb_path`.** There is nothing to capture —
the label is not better where this family operates. Two retrains saved (~100
minutes), by running the guards-ON ceiling alongside the guards-OFF gate.

**Process note worth keeping.** `ORACLE_GATE.md`'s screen is guards-OFF by
design, because it asks whether a label can pay under ANY gate. That is the
right question before a retrain on a family whose gate is open or movable. For a
family with a TIGHT, fixed gate it is necessary but insufficient: the guards-on
ceiling is the more decisive screen and costs one extra arm. Run both.

## 2. The exit signal: W1 unresolved, W2 RESOLVED HARM

Deltas vs exit signal OFF, on the real model. Trade counts are identical across
every arm (95 / 184 / 57), so this is pure exit timing.

| arm | W1 (acceptance) | W2 | W3 |
|---|---|---|---|
| exit ON rvol 3.0 | +0.54 | −0.26 | −0.08 |
| exit ON rvol 2.0 (= entry) | +0.38 | −0.52 | +0.41 |
| **exit ON rvol 1.0** | **+1.02** | **−1.71** | +0.28 |
| exit ON rvol 0.5 | +0.53 | −2.44 | +0.37 |
| exit ON fully open | −0.41 | −3.18 | −0.08 |

Paired-by-trade resolution at rvol 1.0, with each window's informativeness:

| window | net/gross | effect | paired 2sd | verdict |
|---|---|---|---|---|
| W1 | 40.1% | +1.02pp | 2.05pp | within noise |
| **W2** | 23.4% | **−1.71pp** | 1.53pp | **RESOLVED — harmful** |
| W3 | 32.7% | +0.28pp | 1.04pp | within noise |

**Verdict: do NOT enable the exit signal on NNNC.** The only RESOLVED effect is
harm, and W2's degradation is monotonic across all five gate values
(−0.26 → −0.52 → −1.71 → −2.44 → −3.18), which is a pattern rather than noise.
W1's +1.02pp does not clear its own band.

Note NNNC's windows are all far more informative than NNMT's W3 (net/gross 40.1%
/ 23.4% / 32.7% against 7.5%), so a resolved W2 result here carries real weight
— unlike the W3 sign-flips in `W3_INSTABILITY.md`.

## 3. Two cross-family findings

**The interior optimum at rvol 1.0 replicates.** On NNMT (entry rvol 0.2) and on
NNNC (entry rvol 2.0) the best exit gate is 1.0 in both cases, and fully open is
worst in both. Two families with a ten-fold difference in entry tuning agree on
the exit value, which suggests 1.0 is a property of the volume-confirmation
mechanism rather than of either family's gate. Research R5's intuition — that
the exit gate should mirror the entry — is now refuted on BOTH families.

**But the exit SIGNAL is family-specific.** Same mechanism, opposite verdicts:
NMMT +0.78pp on W1 and positive 3/3 windows; NNNC +1.02pp unresolved on W1 and
resolved harm on W2. NNMT has a genuine multi-task sell head; NNNC's sell side
is a single-task by-product. That is the plausible reason, and it is consistent
with the standing rule that families tune independently.

**The prior does not reproduce, though.** NNNC's exit signal was recorded at
−5.15pp. At no gate value here is it worse than −0.41pp on W1. The old figure
was measured at a single unswept gate on a different exit stack and should be
treated as superseded.

## 4. NNNC has headroom; NNMT does not

The most useful number here for future work:

| family | headroom at its own gate (W1) |
|---|---|
| NNMT | **−4.558pp** (a perfect model is WORSE) |
| **NNNC** | **+3.488pp** |

NNNC's previously recorded headroom was 0.68pp, measured 2026-09-05 before the
exit repair, the label move to 0.02 and the threshold retune. It is not a
contradiction — the configuration changed and the ceiling rose with it.

**So NNNC, not NNMT, is where model-side work can still pay on this repository.**
`gbb_path` is simply not the lever: its ceiling at NNNC's gate is no better.
Anything proposed for NNNC should be screened against that +3.488pp, which is
what a perfect ENTRY signal is worth there.

---

# ADDENDUM — the open gate, and a correction to section 1

Prompted by the observation that if `gbb_path` does better with the guards off,
that is evidence about the GUARDS, not just about the label. It is — and section
1's "do not retrain" was scoped to the closed gate and stated too broadly. The
open-gate arms were then built and trained (`NNNC_Open_Path_s1/s7`, seeds 1 and
7, 6-7 minutes each).

## What the guards cost, and whether it is reachable

| W1 | trades | return |
|---|---|---|
| CLOSED gate, real model (production) | 95 | **+10.446%** |
| CLOSED gate, ceiling | 85 | +13.934% → headroom **+3.488pp** |
| OPEN gate, real model (3 seeds: 42/1/7) | ~320-440 | **+8.72 / +5.72 / +2.97%** |
| OPEN gate, ceiling | 1333 | +26.544% → headroom **+17.824pp** |

**The guards discard roughly half the achievable CEILING and five times the
headroom — and none of it is reachable.** At the open gate the real model
returns +4.3% on two paired seeds (+5.8% averaged over three), against
production's +10.4%. Opening the gate exposes 17.8pp of theoretical room and
LOSES 5-6pp of realised return.

**So the guards are not wrong. They are doing the selection the model cannot
do.** That is the same conclusion feature 005 reached for NNMT by a different
route, and it is now established on both families: the guard chain substitutes
for a model that cannot discriminate, and removing it is only an improvement for
a perfect model.

## The path label at the open gate: also a NULL

At the adopted threshold 0.80, identical for every configuration:

| window | gbb s1/s7 | path s1/s7 | path effect | seed spread | paired 2sd |
|---|---|---|---|---|---|
| **W1** | +5.72 / +2.97 | +4.17 / +6.01 | **+0.747pp** | 2.747 | 8.98 |
| W2 | +8.54 / +11.15 | +15.45 / +9.71 | +2.738pp | 5.733 | 9.95 |
| W3 | +7.25 / +11.88 | +7.91 / +11.67 | +0.226pp | 4.623 | 4.56 |

**+0.747pp on W1, inside a 2.747pp seed spread and an 8.98pp band.** The label's
+3.112pp ceiling advantage at the open gate does not convert, exactly as it did
not on NNMT. Section 1's verdict was reached for the wrong reason and is correct
for the right one: `gbb_path` is not a lever on NNNC at either gate.

## A METHOD ERROR worth recording

The first pass at this comparison chose each configuration's threshold on the
in-sample windows (max mean of W2/W3), to keep W1 clean. That rule picked **0.75
for three of four configurations**, and reported a path effect of **+4.961pp**.

Both were artefacts. At 0.75 **W1 is negative for all four configurations**
(−9.13, −4.61, −9.15, −7.99) while W3 reaches +23.7%. The in-sample windows
prefer a threshold at which the acceptance window loses money, and the two path
seeds landed on DIFFERENT thresholds (0.80 and 0.75), so the "+4.961pp" compared
configurations that were not alike.

    threshold   W1 (acceptance)      W2/W3 (in-sample)
      0.70      -11.7 to -16.8%      strongly negative W2, +19 to +22% W3
      0.75       -4.6 to  -9.2%      positive W2, +20 to +26% W3
      0.80       +3.0 to  +6.0%      +8 to +15% W2, +7 to +12% W3
      0.85       zero trades on 7 of 12 arms — INCONCLUSIVE

**In-sample threshold selection is not safe on this family.** The per-window
optima genuinely disagree — previously recorded as 0.82/0.80/0.75 — and here the
disagreement is severe enough to flip the sign of the acceptance window. Use the
adopted value, or select on a rule that includes W1's behaviour explicitly and
accept the resulting selection bias, but do not assume in-sample transfers.

The 0.85 zero-trade arms were correctly recorded as INCONCLUSIVE rather than as
zero-return results (constitution experiment gate 6).

## Net

- `gbb_path`: rejected on NNNC at BOTH gates. Null at the open gate (+0.747pp,
  inside noise), slightly negative at the closed one.
- The open gate: **not a candidate on W1** — it loses 5-6pp to production across
  three seeds, despite five times the headroom.
- The headroom at the open gate (+17.824pp) is real and remains **unreachable by
  any label change tested**. It is reachable only by a model that can actually
  discriminate, which is the thing this repository has repeatedly failed to
  build. That is the honest statement of where NNNC stands.
