# Quickstart: labelfree-ssl-features

> ### ⚠️ CORRECTED 2026-09-15 — the numbers below are LEAK ARTEFACTS
>
> Every ρ figure in this document was measured with a forward-return target starting at
> `open[t]`, which **included bar *t*'s own move** — information the inputs reveal directly.
> The causally aligned target is `open[t+1] → open[t+1+H]`, and under it the hand-built bar is:
>
> ```
> H        3        6       12       24       48       96
> median  0.0352   0.0461   0.0396   0.0328   0.0238   0.0401
> ```
>
> **Two claims stated here as findings do not survive.** "ρ ≈ 0.15 is a horizon-3-to-5 number"
> — under causal alignment nothing reaches 0.15. And "predictability decays with horizon" — it
> is roughly **flat** at 0.024–0.046 from H=3 to H=96; the decay was the leak's own signature,
> since bar *t*'s move is proportionally the largest part of a short-horizon target.
>
> The text is left standing rather than rewritten, because a corrected document should show what
> it corrected. Current numbers: `research/labelfree/ssl/baseline.json`. Full history of the
> three generations: [`CEILING_CORRECTIONS.md`](../../research/labelfree/CEILING_CORRECTIONS.md).

All commands from `user_data/strategies` under `freqtrade/.venv/bin/python`, sequentially.

## Prerequisites

- MLX 0.32.2 on GPU (`mx.default_device()` → `Device(gpu, 0)`)
- `research/labelfree/` importable; fold panels available
- **Nothing to install and nothing to download** — `transformers` and every pretrained model are
  out of scope (SC-016)

## Step 1 — the bar, measured before anything else

```
python -m research.labelfree.ssl.screen --baseline-only --folds f05..f14
```

**Expect**, reproducing research.md R1:

**Measured 2026-09-15** (f05..f09, ~4 min — this is the guide's most expensive step, not the
quick sanity check its position suggests):

```
H         median       min       max
3         0.0344    0.0184    0.0374
6         0.0419    0.0324    0.0465
12        0.0354    0.0266    0.0513
24        0.0396   -0.0123    0.0718
48        0.0378   -0.0109    0.0656
96        0.0505    0.0142    0.0688
```

If these do not reproduce, **stop and check `forward_return`'s alignment before touching
anything else.** The comparison bar is the gate, and it has been measured wrongly twice —
once by splitting across pairs instead of time, once by starting the target at `open[t]` so it
contained bar *t*'s own move. Both looked plausible and both inverted conclusions.

**ρ≈0.15 does not reproduce at any horizon**, and predictability is roughly **flat** at
0.034–0.051 rather than decaying. The familiar decay was the leak's signature.

## Step 2 — fit an encoder

```
python -m research.labelfree.ssl.ts2vec --fold f07 --input engineered --seed 7
```

**Expect**: an embedding under `embeddings/`, fitted on the fold's training range only.

## Step 3 — causality, before believing any number

```
python -m research.labelfree.ssl.causality --fold f07 --input engineered
```

**Expect**: the deletion test bit-equal, and the **planted leak detected** (ρ jumps). If the
planted leak is *not* detected the screen is void, not negative — the test cannot see a leak it
was handed.

## Step 4 — the deployability gate

```
python -m research.labelfree.ssl.pairagnostic --embeddings embeddings/f07_engineered_s7.npz
```

**Expect**: max between-pair ratio reported against the pre-registered **0.5**. A rejection is a
finding — an embedding whose scale depends on the pair cannot be deployed here.

## Step 5 — THE GATE

```
python -m research.labelfree.ssl.screen --folds f05..f14 --out screen.json
```

**Expect** four combinations, each with its **own** bar:

```
input        H    rho_embedding   rho_handbuilt   rho_control
engineered   48   ?               ~0.024          ?            <- PRIMARY
engineered    6   ?               ~0.046          ?
ohlcv        48   ?               ~0.024          ?
ohlcv         6   ?               ~0.046          ?
```

**Measured 2026-09-15** — the primary FAILED, and the decisive column is what the encoder adds
over its **own** raw input, not the bar:

```
combination      embedding   raw input   adds      folds adding
engineered@6        0.0358      0.0461  -0.0089        0 / 10
engineered@48       0.0326      0.0238  -0.0012        5 / 10
ohlcv@6             0.0194      0.0209  -0.0007        4 / 10
ohlcv@48            0.0180      0.0121  +0.0089        7 / 10
```

**The primary is `engineered` × 48 and was fixed before any of this ran.** A secondary that passes
is a hypothesis for a follow-up, never an acceptance — four screens against one threshold is four
chances to clear a bar meant to be cleared once.

**If the primary does not beat its own bar, the feature is KILLED here and no backtest runs.**
That is the expected path.

## Step 6 — only on a pass

```
# NNNC_MLX subclass, scalers regenerated, pooled walk-forward under protocol §5
```

## Step 7 — the report

```
python -m research.labelfree.ssl.report --out research/label-free-study/briefs/results/07-2026-09-15.md
```

Written whatever the outcome.

## What a pass would and would not mean

The screen measures **learnability**, and this repo has recorded that learnability is not edge — a
+0.155 MCC once produced identical trades. Passing earns a backtest. It is never an acceptance.
