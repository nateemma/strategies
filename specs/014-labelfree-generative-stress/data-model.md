# Data Model: labelfree-generative-stress

Phase 1. Entities, fields, and the invariants that make the gates checkable.

---

## 1. `Path` — one synthetic panel

**Unit**: one path is a **complete resampled panel across every pair in the fold set**, not a
per-pair series. A bank of 1,000 paths therefore holds 1,000 whole-universe histories.

A path has the **same shape as a real fold panel**, which is what lets the existing surrogate
score it without modification:

```
open, high, low, close   float[n]    prices
volume                   float[n]
rvol_stop                float[n]    volume / rolling(20).mean() — the STOP's rvol,
                                     not the `rvol` column (different quantity, same name)
cost                     float[n]    per-bar, per-side, from brief 1's cost model
p_sell, p_hold, p_buy    float[n]    the model's own class probabilities
rvol, atr_pct_roll, guard_metric, close_norm, adx, bb_width   float[n]   guard inputs
```

**Everything is resampled together, in blocks.** Inside a block every column is real and mutually
consistent — including the model's reaction to the market state, which is why no re-inference is
needed (research.md R1). Only the joins between blocks are synthetic.

---

## 2. `Bootstrap` — stationary, geometric block lengths

Politis & Romano (1994). A block starts at a uniformly random index; each subsequent bar
continues the block with probability `1 − 1/L` and starts a new one otherwise, wrapping at the
series end.

| parameter | value | why |
|---|---|---|
| `L` (mean block length, bars) | **swept over {96, 144, 192, 240, 288}** | the brief's 1–3 days at 15-minute bars. Swept, not chosen — reporting one value as *the* answer would hide the sensitivity (Principle IV applied to the one parameter this feature has). Required by FR-003a and SC-006a. |
| index sharing | **one index across all pairs** | keeps the cross-section intact, so pairs that crashed together still do |
| selection among passing `L` | **the SHORTEST that passes calibration** | see below |

**Selecting the block length: shortest that passes, NOT best calibrated.** The calibration
statistic improves monotonically with block length, because longer blocks preserve more real
structure and splice less often — measured across the sweep, KS ran 0.0252 → 0.0119 from L=96 to
L=288. At the limit, where a block spans the whole series, every path *is* the real series:
calibration would be perfect and the harness would resample nothing.

So **selecting by best calibration drives toward the degenerate case**. The rule is the most
resampling consistent with a faithful generator — the shortest passing `L`. This was discovered
during implementation, when the sweep script picked L=288 on precisely the wrong rule; it is
recorded here so a re-run does not have to rediscover it.
| `n_paths` | ≥ 1,000 (FR-010) | |
| path length | the source fold's own bar count | keeps synthetic and real comparable without a length correction |
| wrap | yes | the stationary variant's requirement; without it, bars near the end are under-sampled |

**Why geometric and not fixed blocks**: with fixed blocks a bar's probability of starting a block
depends on its position modulo `L`, which makes the resampled series non-stationary. The geometric
variant removes that artefact for one extra line.

**Pairs share one block index** (revised by the pre-implementation analyze pass). The fold panels
are date-aligned, so applying the same index sequence to every pair keeps the cross-section
intact: when the index selects a moment, every pair contributes its own bar from that moment.

The original design resampled pairs **independently**, which would have destroyed cross-pair
co-movement and **understated joint drawdown** — and `dd_ratio` is one of the three calibration
statistics, so the method's known bias pointed straight at a gate it had to pass. Fixing the cause
beats widening the band to accommodate it.

The cost is less path diversity, since the cross-section is tied together. That is the correct
trade: a harness that understates joint drawdown is worse than one with fewer distinct paths,
because joint drawdown is what a portfolio actually experiences.

---

## 3. `Bank` — a persisted set of paths with provenance

```
generator        str      "stationary_block_bootstrap"
block_length     int      L
seed             int
fold_ids         list     the source panels
n_paths          int
source_range     str      earliest..latest source bar
feature_columns  list     every column carried, in order
source_sha256    str      hash of the source panels
created          iso8601
```

**The provenance exists to make a stale bank detectable** (FR-002). This is not hypothetical: a
bank generated before a panel changes would silently stress the wrong thing, and this study has
the precedent — feature 012 recorded that a cache key must cover everything that changes the
panel, and feature 013's panel cache keys on the side-car's content hash for exactly that reason.

`load()` recomputes `source_sha256` and **refuses a bank whose source has moved**, rather than
warning.

---

## 4. `Calibration` — the Tail-GAN criterion at PRE-REGISTERED tolerances

The incumbent is run on the bank; its per-trade P&L is compared with **the same policy scored by
the same evaluator on the real panels**.

**The evaluator is held constant, deliberately.** Comparing surrogate-on-synthetic against the
378 archived *freqtrade* trades would conflate two differences — synthetic vs real paths, and the
surrogate vs freqtrade. That second gap is known and measured: level offset **−5.30 ± 23.46 pp**
over 21 candidates. A failed test could not be attributed, and with SC-001 the only gate and a
failure a kill, the feature could have died of a confound. The archived trades are reported as a
**secondary sanity reference**, never gating.

| statistic | tolerance | direction it guards |
|---|---|---|
| `ks_distance` — two-sample KS on per-trade P&L | **≤ 0.10** | the body of the distribution |
| `es_ratio` — 5% expected shortfall, synthetic ÷ real | **0.7 – 1.4** | the loss tail |
| `dd_ratio` — median max-drawdown, synthetic ÷ real-panel | **0.7 – 1.6** | sustained runs |

**Fixed in research.md R3 before any calibration run** (SC-002). The KS threshold is set from what
the comparison can resolve: at n = 378 the 95% critical value against an identical distribution is
1.36/√378 = **0.0700**, so 0.10 admits sampling noise without admitting a different shape.

The tail bands are **deliberately asymmetric**. A synthetic tail *fatter* than real makes the
harness conservative and is tolerable; a synthetic tail *thinner* than real is the dangerous
direction, because it would let a fragile policy through.

```
passed = (ks_distance <= 0.10) and (0.7 <= es_ratio <= 1.4) and (0.7 <= dd_ratio <= 1.6)
```

The drawdown lower bound was tightened from 0.6 to 0.7 by the analyze pass, to match ES: both are
tail statistics and "thinner than real" is the dangerous direction for both.

**A failure is a KILL, not a fallback.** Clarification Q3 left no second generator.

---

## 5. `StressReport` — a distribution, never a point estimate

```
policy            str
bank              str
n_paths           int
pnl_quantiles     {5, 25, 50, 75, 95}   pp/yr
drawdown_quantiles{50, 90, 95, 99}
loss_path_fraction float      fraction of paths with negative net return
n_inadmissible    int         paths on which the policy took no trades
incumbent         {...}       the same block, for reference (FR-012)
verdict           str         "VETO" | "NO VETO" | "INADMISSIBLE BANK"
```

**`verdict` is never an endorsement.** SC-009 and FR-014 make stress results a veto only: "NO
VETO" means the harness found no reason to object, not that the policy is good. The field name is
deliberately not `passed`, so it cannot be read as a gate.

**`n_inadmissible`**: a path on which the policy never trades is not a flat result, it is an
unusable path (FR-013). The check routes through the study's shared `admissibility.compare` with
the **incumbent-relative exposure floor** feature 013 added — the absolute `MIN_EXPOSURE` marks
production itself degenerate on a 90-day window (measured 0.0196 on f00). Reimplementing the
check locally is the exact habit that module was created to end.

---

## 6. `GeneratorCensus` — what the repo's generators can actually do

```
gan_type          str
returns           str     the observed shape or type
emits_sequences   bool
evidence          str     file:line in the running code
in_scope          bool
```

Determined **from running code, not documentation** (FR-004, SC-003). Nothing is built on it —
clarification Q3 removed every learned generator — but it ships because it is a durable fact about
the repo's tooling and is the documented reason the design pivoted (research.md R7).
