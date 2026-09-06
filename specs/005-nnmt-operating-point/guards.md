# T012 — NNMT guard attribution (US3)

Leave-one-out over NNMT's six quality guards at the control configuration, plus
the joint arm. All 21 arms `returncode` 0. Guards were neutralised by the six
values, never by `--guards off`, so the entry `volume > 0` check and the
exit-side guard block stayed live throughout.

Resolution is the paired 2sd bootstrap over each arm's symmetric difference with
the control (T008's rule), NOT a flat 1pp floor.

## W1 — the acceptance window

| neutralised | trades | return | Δ vs control | differing trades | paired 2sd | verdict |
|---|---|---|---|---|---|---|
| control | 176 | +10.708% | — | — | — | — |
| `entry_rvol_threshold` | 174 | +8.965% | −1.743pp | 20 | 3.20pp | within noise |
| `entry_atr_pct` | 392 | +4.995% | −5.713pp | 302 | 9.66pp | within noise |
| `entry_adx_threshold` | 542 | +9.015% | −1.693pp | 490 | 12.05pp | within noise |
| `entry_bb_width_threshold` | 234 | +10.971% | +0.263pp | 82 | 3.89pp | within noise |
| **`entry_guard_threshold`** | **176** | **+10.708%** | **+0.000pp** | **0** | — | **INERT** |
| `entry_close_norm_threshold` | 205 | +11.471% | +0.763pp | 33 | 3.23pp | within noise |
| **all six (joint)** | **2798** | **−23.442%** | **−34.150pp** | 2870 | 18.95pp | **resolved** |

W2 and W3 are in-sample and reported only; the same shape holds. **W3 carries
126 control trades and is near-flat** (R7), so nothing below rests on it.

| neutralised | W2 Δ | W3 Δ |
|---|---|---|
| `entry_rvol_threshold` | −2.711pp | −0.454pp |
| `entry_atr_pct` | −3.636pp | +1.289pp |
| `entry_adx_threshold` | −6.010pp | +2.834pp |
| `entry_bb_width_threshold` | +0.152pp | +1.348pp |
| **`entry_guard_threshold`** | **+0.000pp** | **+0.000pp** |
| `entry_close_norm_threshold` | −0.751pp | +0.561pp |
| all six (joint) | −29.726pp | +4.610pp |

## FR-010 — `entry_guard_threshold` is INERT, not beneficial

Neutralising it from 0.3 to 5.0 leaves the trade set **byte-identical in all
three windows** — zero differing trades, delta exactly 0.000pp. It is not a
weak guard, a saturated guard or a confirmed-correct guard: on NNMT it does
nothing at all, and the swept value 0.3 is meaningless.

This is the eighth recorded inert parameter in this repository, and it echoes
NNNC, where `guard_metric` moved the trade count only 49 → 51.

## FR-011 — the individual effects do NOT sum to the joint effect

| window | Σ individual | joint | ratio |
|---|---|---|---|
| W1 | −8.123pp | **−34.150pp** | **4.20×** |
| W2 | −12.956pp | −29.726pp | 2.29× |
| W3 | +5.578pp | +4.610pp | 0.83× |

On the acceptance window the joint effect is **more than four times** the sum of
the individual effects. **The guard chain is strongly interacting, not additive.**
No single guard is load-bearing; the chain is load-bearing as a conjunction.
Removing any one guard leaves the other five to catch most of what it caught,
which is exactly why every leave-one-out lands within noise while removing all
six costs 34pp.

That has a direct methodological consequence: **a leave-one-out cannot rank these
guards, and must not be used to justify opening any of them individually.** The
one thing it does establish unambiguously is which guard is inert, because that
conclusion rests on identical trade hashes rather than on a return difference.

## FR-009 — attribution, stated at the resolution the data supports

**On realised return, no individual guard is resolved on W1.** Every
leave-one-out delta is smaller than its own paired 2sd band. Reporting any of
them as "worth −5.7pp" would be exactly the error T008 was written to prevent.

**On trade count the picture is unambiguous**, and trade counts need no
significance test:

- `entry_adx_threshold` is the dominant filter — removing it takes 176 → 542
  trades on W1 (3.1×) and 347 → 914 on W2 (2.6×).
- `entry_atr_pct` is second — 176 → 392 on W1 (2.2×).
- `entry_bb_width_threshold` and `entry_close_norm_threshold` are mild
  (176 → 234 and 176 → 205).
- `entry_rvol_threshold` is nearly inert at NNMT's setting — 176 → 174 trades.
  It is set to 0.2 (see the baseline note on `decimals=1`), which is so loose
  that it rejects almost nothing. This is the sharpest contrast with NNNC, where
  rvol was the DOMINANT guard at its setting of 2.0.
- `entry_guard_threshold` is inert.

## What this adds to the Phase 2 gate

Phase 2 showed the model's entry opinion contributes nothing and a perfect one
contributes less. Phase 3 explains it: **NNMT's entries are selected by an
interacting conjunction of adx and atr_pct, with rvol and guard_metric
contributing nothing at their current settings.** The model is downstream of a
filter that has already made the decision.

It also explains why opening the gate — the intervention that worked on NNNC —
cannot simply be transplanted. Guards-off on NNMT costs **−34.15pp on W1** and
produces 2798 trades. On NNNC the equivalent move was the doorway to ~25pp of
headroom because there the model discriminated once the guards stopped binding.
Here there is no headroom behind the door: the Phase 2 oracle says a perfect
model is worth nothing even at the current gate.

**This is a coherent, and negative, picture of the family**: the guards are the
strategy, the model is not, and the label the model is fitted to is worse than
the model. Phase 4 tests whether the inherited label change moves that.
