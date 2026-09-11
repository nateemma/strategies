# Phase 0 Research: Label Warm-Start with P&L Fine-Tune

**Feature**: 010-labelfree-warmstart-finetune · **Date**: 2026-09-11

Every decision resolves a Technical Context unknown or records a measurement that constrains
the design. Measurements are marked **[measured 2026-09-11]**.

---

## R1 — Encoder architecture and how the warm-start actually attaches

**Decision**: All three arms use the classifier's encoder. Arm A inherits its weights; B and C
initialise the same architecture randomly.

**The architecture** [measured 2026-09-11, from both the source and the saved tensors]:

```
resize     Linear(num_features -> 32)   -> relu
conv1      Conv1d(32, 32, kernel=2, padding=1), trimmed to seq length
ln_conv    LayerNorm(32) -> dropout 0.25
residual   x_r + x_c
lstm       LSTM(32, 32)
ln_lstm    LayerNorm(32) -> dropout 0.25   -> latent (B, 32)
```

**The attachment point already exists.** `NNNClassifierMLX_LSTM` is split into `encode()` and
`decode()` — a split added so NoisyCoconut could hook the latent. Arm A therefore inherits
`encode()` verbatim and replaces `decode()` (which ends in a 3-class softmax) with the exposure
head. No surgery on the classifier is required, and nothing in brief 01 needs forking.

**Consequence that must not be missed**: `encode()` returns a per-**window** latent taken from
the LSTM's last timestep, not a per-bar sequence. A per-bar exposure therefore requires one
`seq_len`-window per bar — `(n_bars, 16, n_features)` — rather than one streaming pass. That is
a different computation from brief 01's policy, not a re-parameterisation of it.

**Alternatives considered**: projecting brief 01's LSTM(64) weights into the classifier's
LSTM(32) (rejected at clarification — a lossy re-encoding makes a null result uninterpretable);
keeping brief 01's policy and skipping Arm A (rejected — it is the arm that resolves brief 01's
central ambiguity).

---

## R2 — The windowed encoder is FASTER than brief 01's streaming pass

**Measurement [measured 2026-09-11]**, 8,641 bars, 24 features, seq_len 16:

| | time |
|---|---|
| brief 01 streaming LSTM over the full series | 0.64 s |
| windowed encode (8,626 windows × 16 steps) | **0.37 s** — 0.6× |

**Decision**: adopt the windowed form without compute-budget concessions.

**Rationale**: this inverts the obvious expectation and is worth recording, because the naive
reading ("16× the FLOPs, therefore ~16× slower") would have justified cutting seeds or arms.
MLX parallelises across the batch dimension: the windowed form runs 16 sequential steps over a
batch of 8,626, while the streaming form runs 8,641 sequential steps over a batch of 1. Depth
dominates, not FLOPs.

**Memory is the real constraint** [measured]: the window tensor scales as
`bars × seq_len × features × pairs`.

| panel | window tensor |
|---|---|
| full train panel (139,105 bars × 10 pairs) | **2.14 GB — exceeds Metal's limit** |
| one chunk (2,880 bars × 10 pairs) | 0.04 GB |
| test panel (8,641 bars × 10 pairs) | 0.13 GB |

The existing truncated-BPTT chunking already keeps every step inside the limit. Windows MUST be
built per chunk and never for the whole panel.

---

## R3 — Arm A needs a classifier per fold, and that is the feature's dominant cost

**Decision**: train one classifier per fold on that fold's training range only. The production
model is never reused.

**Rationale**: the production model was trained on data that later folds treat as held out;
warm-starting from it would leak the test range into the initialisation. This is not a
formality — it is the same class of error as ranking on the training range, which cost feature
009 a full run.

**Cost**: one classifier for the screening fold; **fifteen** for a pooled run. Phase sequencing
must therefore put the screen first and treat the pooled classifier training as a separate
budgeted step, not an incidental prerequisite.

---

## R4 — Blend weights and the two labels

**Decision**: sweep the blend weight over {0.75, 0.5, 0.25} and test both
`MIN_BUY_GAIN_THRESHOLD` values, 0.007 (production) and 0.02.

**The brief's premise about these two labels is contradicted by the repo's own record.** Brief
02 says they "behave very differently". The recorded result (commit `7ddcfb2`) says:

```
thr 0.02 TESTED on NNNC — no out-of-sample effect
  thr 0.007 (production)  332 trades  10.45%
  thr 0.02                332 trades  10.45%
Not adopted. There is no out-of-sample evidence for it.
```

They differ substantially **as labels** — a 38% reduction in signal count, 10,406 → 9,529 — and
not at all in **P&L** on the classifier. Both are still tested, because this feature uses them
as a *prior* rather than a target and the prior's density may matter where its P&L did not. But
the expectation is recorded now: if the two labels produce the same result here, that replicates
a known finding rather than discovering one, and must be reported as such.

**Alternatives considered**: testing only the production label (rejected — the brief asks for
both, and the cost is one extra sweep); sweeping more blend weights (rejected — every value is a
trial and inflates the multiple-testing correction for no design benefit).

---

## R5 — Auxiliary target for Arm C

**Decision**: forward realised volatility over the label horizon, computed causally from the
same OHLCV the features come from, predicted by a second linear head from the shared latent.

**Rationale**: the arm exists to separate "labels help because they carry direction" from "any
auxiliary supervision regularises the objective". The target must therefore carry no directional
information, and realised volatility is the natural choice — it is forward-looking, so it takes
the same purge and embargo as the directional labels.

---

## R6 — Purge and embargo

**Decision**: purge the label horizon at each fold boundary and embargo one horizon after it,
for every arm whose objective contains a forward-looking term — which is A (its warm-start
classifier is label-trained), B (the cross-entropy term) and C (forward volatility).

**Rationale**: protocol §2, and López de Prado ch. 7. The applied purge and embargo are recorded
per fold so that a run missing either is void rather than caveated; a forward-looking term
without a purge is leakage, not a weaker result.

---

## R7 — Controls, and the architecture confound

**Decision**: two controls — the label control (production classifier) and the P&L control
(brief 01's `utility-penalty-g0.5`, mean exposure 0.130, −7.06% held-out). The P&L control is
**re-run under this feature's architecture**.

**Rationale**: brief 01's recorded numbers came from a plain LSTM(64). Comparing this feature's
arms against them would confound the label prior with the architecture change — and isolating
the prior is the entire purpose. Both figures are reported so the architecture effect is visible
separately.

**Why not brief 01's top-ranked config**: its four best-ranked arms are degenerate, flat on 100%
of bars at mean exposure 0.002–0.041. Any arm that trades beats them trivially, which would make
"did labels help" vacuous.

---

## R8 — Screening fold

**Decision**: **f08**, not the f07 the brief names.

**Rationale** [measured in feature 009]: f08 carries 71 control trades to f07's 58, and f07 is
the most *profitable* fold (+8.08% vs −0.14%). Screening on the profitable fold selects for
configurations that only work in benign conditions — the bias feature 009 identified and
corrected. f08 also keeps this screen directly comparable to brief 01's, which matters because
brief 01 is a control here.

---

## R9 — Inherited without re-derivation

From feature 009, all verified there and not re-litigated here: the calibrated simulator (zero
difference per trade against the arbiter across 99 trades); the participation floor with
mask normalisation; date-aligned pooled panels with a validity mask; ranking on a held-out
range; coverage measured as days of data inside the range rather than by start date.

**One inherited hazard is worth restating.** Four of feature 009's defects were wrong-axis
slices on pooled panels, which MLX broadcasts silently into plausible shapes and wrong numbers.
Every test added here asserts on **values**, not shapes.
