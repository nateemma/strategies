# Quickstart: labelfree-ssl-features

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

```
H       3        6       12       24       48       96
median  0.1717   0.1127   0.0944   0.0720   0.0826   0.0583
```

If these do not reproduce, stop: the comparison bar is wrong and every screen below is
meaningless. **ρ≈0.15 is a horizon-3-to-5 number, not a constant** — predictability decays with
forward horizon, so a bar quoted at one horizon is the wrong bar at another.

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
engineered   48   ?               ~0.083          ?            <- PRIMARY
engineered    6   ?               ~0.113          ?
ohlcv        48   ?               ~0.083          ?
ohlcv         6   ?               ~0.113          ?
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
