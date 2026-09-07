# Path-aware gbb label — oracle gate screen

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
