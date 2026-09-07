# NNNC: path-aware label and exit signal — both REJECTED, no retrain spent

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
