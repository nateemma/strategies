# US2 result — the hold-window package (T022-T029)

**Verdict: UNCHANGED (indistinguishable), and negative-leaning on both
components.** Neither the hold window nor the tail label given that window
moves W1 by a distinguishable amount, and both point the wrong way.

## Arms

| arm | label | hold window | model |
|---|---|---|---|
| `ctl003` | h48 / 0.005 | 12h | host |
| `h1` | h48 / 0.005 (control's) | **24h** | host, reused byte-identically |
| `l2` | **h96 / 0.03** (C3 tail) | **24h** | retrained |
| `l1` (US1) | h48 / 0.02 | 12h | retrained |

`h1` reuses the host's weights via a `get_model_path` override, so the ONLY
difference from the control is the exit rule — no seed noise contaminates the
hold-window measurement.

## W1 — OUT-OF-SAMPLE, the only acceptance window

    arm                          tr     ret%    DD%    PF   win%
    control (h48/0.005, 12h)     97    -0.74   3.74  0.94   58.8
    H1  hold window ALONE        97    -1.09   3.98  0.91   56.7
    L2  tail label + 24h hold    98    -1.34   4.10  0.90   55.1
    L1  h48/0.02, 12h (US1)      98    -0.20   3.80  0.98   60.2

## Separated contributions (FR-011, SC-005)

    hold window alone             H1 - control = -0.35pp
    tail label GIVEN that window  L2 - H1      = -0.25pp
    ------------------------------------------------------
    package total                 L2 - control = -0.60pp

All three are below the 1.0pp power threshold on 97 control trades, so each is
individually indistinguishable from noise. What is notable is that they are
consistent in SIGN and they COMPOUND: every step away from the control costs a
little return, a little profit factor and a little win rate, while adding
drawdown (3.74% -> 3.98% -> 4.10%).

The decomposition is valid because H1 and L2 share a hold window and differ
only in label, and H1 and the control share a label and differ only in the
hold window. US1's L1 is NOT subtracted here — it carries a third, different
label, so a three-way decomposition against it would be arithmetically invalid
(T028).

## In-sample (excluded from the verdict, FR-013)

    W2:  H1 +0.35pp   L2 +0.59pp
    W3:  H1 +0.57pp   L2 +0.59pp

Both arms improve on BOTH in-sample windows while degrading out-of-sample.
That is the classic signature of fitting the training period, and it is the
reason FR-013 excludes W2/W3 from acceptance. Judged on summed return across
all three windows — the metric this study used before the windows were pinned —
L2 would look like a WIN (+0.58pp summed). On W1 alone it is a loss.

## The central finding: label economics do not predict return

    arm        r:r   clears stop   W1 delta
    control   0.57      26.0%       0.00pp
    L1        0.86      40.6%      +0.54pp
    L2        1.28      66.8%      -0.60pp

Reward:risk more than doubled and the fraction of labelled moves that clear the
stop went from a quarter to two thirds — and out-of-sample return did not
follow. It rose slightly at the intermediate cell and fell at the best-economics
cell. The relationship is not merely weak; it is NON-MONOTONIC.

This is the fifth metric in this study to fail as a proxy for realised return,
after EV-per-signal, precision, val_mcc and MCC-at-learnability. `val_mcc` again
pointed the wrong way: 0.5271 for L2 against 0.5165 for L1, while L2 returned
1.14pp less.

## What this does and does not establish

It DOES establish that the label-specification hypothesis, as posed, is
answered: the labels were mis-specified relative to the risk taken, that was
fixed, the fix reached the trades on every window, and out-of-sample return did
not improve. The mechanism was never in doubt and is not what failed.

It does NOT establish that the tail label alone would have failed — R4 declined
to build that arm because its payoff peaks at 17.0h against a 12h forced exit,
so it would have been harvested early by construction. That consequence was
accepted in advance and is restated here.

It also does not rule out that the effect is real but smaller than this design
can see. 97 W1 trades cannot resolve half a point. Distinguishing a 0.5pp effect
would need roughly an order of magnitude more trades than the guard chain
currently admits, which is a statement about the arm's trade volume, not about
the label.

## Note on the liveness check

`exit_config_hash` MATCHED between H1 and the control even though H1's exit
behaviour genuinely differs. The hash fingerprints exit PARAMETERS, and the hold
window is a method override rather than a parameter, so the hash cannot see it.
The trade-hash comparison did catch it (same 97 trades, different hash — exit
timing changed, entries did not). Anyone reusing this harness for a code-level
exit change should rely on the trade hash, not the exit-config hash.
