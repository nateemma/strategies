# T007 — Instrument validation. **CAUSE FOUND, AND IT WAS MATERIAL.**

**Verdict: a NAMED CAUSE was found and corrected (FR-002 first branch). A 9.6%
residual remains and is carried as the bound on every figure this feature
produces.**

## The cause: training augmentation leaked into the oracle's entry signal

`Framework/BaseNNStrategy.augment_training_signals` (line 678), invoked inside
`get_trading_classes`, begins:

    # Trick 1: artificially extend positive signals two entries earlier
    for i in range(1, 3):
        buys[valid_indices - i] = 1.0

**Every buy label is copied onto the two PRECEDING bars.** Those bars never
satisfied the label's own condition — they are synthetic, and deliberately so:
the trick exists to give the classifier more positive examples and a softer
decision boundary. That is legitimate for TRAINING.

The oracle read `get_training_labels()`, which runs AFTER augmentation. So the
"perfect entry" arm was entering **up to two bars early**, on bars whose forward
move had not yet been established.

### The evidence that identified it

Compliance with the raw label, by reference bar — the signature is that
compliance peaks FORWARD of the entry, which is what a BACKWARD dilation
produces:

| reference bar | compliance |
|---|---|
| entry−2 | 64.1% |
| entry−1 | 81.4% |
| entry | 84.1% |
| **entry+1** | **88.3%** ← peak |
| entry+2 | 82.1% |
| entry+3 | 77.9% |
| any of entry, +1, +2 | **96.6%** |

96.6% of entries had a genuine label inside the two-bar dilation window. A simple
off-by-one was already excluded (entry−1 and entry−2 are both WORSE).

## The correction

`NNMT_Oracle` and `NNNC_Oracle` now AND the augmented trading class with the
**unaugmented** `get_train_buy_signals()` — the pre-augmentation source, since
augmentation is applied by its caller. This removes only the dilated bars and
preserves the rest of the label logic (sells overriding buys, the
buy-followed-by-sell override).

**`Framework/` is untouched.** Augmentation remains live for training, where it
is doing its intended job. This is an oracle-arm change only, and it is inherited
automatically by `NNMT_Oracle_Lbl02`, `NNMT_Ora_Path` and `NNNC_Open_Path`.

**Compliance: 84.1% → 90.4%** (123 of 136 entries).

## The effect on the baselines — LARGE, and it changes conclusions

| arm | W1 before | W1 corrected | Δ |
|---|---|---|---|
| NNMT oracle @0.007 | 155tr +6.150% | 150tr +6.544% | +0.394 |
| **NNMT oracle @0.02** | 145tr +8.733% | **136tr +11.809%** | **+3.076** |
| NNNC oracle @0.02 (guards on) | 85tr +13.934% | 78tr +13.914% | −0.020 |
| **NNMT guards-off gate** | 1561tr +17.791% | **1317tr +62.006%** | **+44.215** |
| **NNNC guards-off gate** | 1333tr +26.544% | **1163tr +71.552%** | **+45.008** |

W2/W3 move the same way: NNMT @0.02 W2 +9.053 → +13.458, W3 +2.747 → +7.517.

The guards-off arms move most because they carry ~1300 trades, each degraded by
the early entry; guards-on NNNC barely moves because its tight gate rejects most
dilated bars anyway.

## **THE FOUNDING PARADOX IS RESOLVED — and it was the instrument, not the exit**

This feature exists because a perfect-entry arm earned LESS than the real model.
At the label NNMT actually runs, that is **no longer true**:

    NNMT W1, label 0.02:   real model  176tr  +10.708%
                           ORACLE      136tr  +11.809%   headroom +1.101pp

The oracle now beats the real model, as it always should have. **The paradox was
an artefact of feeding the oracle a deliberately-degraded label.**

At the pre-inheritance label 0.007 the oracle still loses (+6.544% against
+10.708%, −4.164pp) — that label remains genuinely bad, which is a real finding
and unchanged.

## Blast radius — what this invalidates elsewhere

**Every oracle figure in this repository was measured on the degraded
instrument.** The bias is always in the same direction (early entry against a
−2%..−4% stop is strictly worse), so every ceiling was understated.

| affected result | status |
|---|---|
| Feature 005's headroom **−4.558pp** | control was label 0.007; corrected it is **−4.164pp** — still negative, conclusion SURVIVES |
| Feature 005's "model work on NNMT is foreclosed" | **NEEDS QUALIFICATION.** True at the 0.007 control, but at the CURRENT 0.02 config headroom is **+1.101pp**, not negative |
| `gbb_path` gate numbers on both families | measured against a degraded incumbent (+17.791 / +26.544%); the incumbents are now +62.006 / +71.552%. **Gate verdicts need re-running** |
| Feature 004's whole diagnosis | measured on the degraded instrument AND at label 0.007. Already known stale (R1); now doubly so |
| `WHY_ORACLES_UNDERPERFORM.md` | its central claim — that a perfect model underperforms a real one — is **substantially wrong for NNMT at 0.02**. Needs correction |

**What does NOT change**: the entry-side guard findings (guard chain selects,
`entry_guard_threshold` inert, super-additivity), the exit-signal promotion
(+0.78pp, measured on real models throughout), the W3 net/gross instability, and
the exchange survey. None of those depend on an oracle.

## The residual bound (FR-002, second branch)

**9.6% of corrected oracle entries still do not satisfy the raw label at the
entry bar.** Remaining candidates, untested: window-edge horizon truncation
(T004), limit-fill timing (T005), and the difference between the strategy's
trimmed dataframe and a full-history feather read.

**Every figure this feature produces carries a ±9.6%-of-entries uncertainty on
the composition of the oracle's entry set.** No conclusion in this feature may
rest on a margin narrower than that unless T004/T005 close it further.
