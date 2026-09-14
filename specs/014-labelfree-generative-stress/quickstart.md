# Quickstart: labelfree-generative-stress

End to end, and what each step must produce before the next may start. All commands run from
`user_data/strategies` under `freqtrade/.venv/bin/python`, sequentially.

## Prerequisites

- feature 013's per-fold side-cars present (`/tmp/lf/es_sink/f*_test`). If absent:
  `python -m research.labelfree.es.signals --folds f00..f14 --which test` (~2 min)
- `research/labelfree/es/` importable and its tests green

## Step 1 — what can the repo's generators emit?

```
python -m research.labelfree.stress.gencensus --out census.json
```

**Expect**: TabDDPM `(n, 1, F)`, WGAN `(n, 1, F)`, CTAB-GAN+ `DataFrame` — all
`emits_sequences: false`; CGAN `(n, seq_len, F)` true but out of scope and unmaintained.

This is a finding, not a gate. Nothing downstream consumes it.

## Step 2 — the bootstrap round-trips

```
pytest research/labelfree/tests/test_stress_bootstrap.py research/labelfree/tests/test_stress_bank.py -q
```

**Expect**: green. The decisive test asserts a **bank and a real fold panel score identically on
the same bars** — if they don't, the bank's shape has drifted from what the surrogate expects and
every downstream number is meaningless.

## Step 3 — build a bank (offline, once)

```
python -m research.labelfree.stress.bank --build --folds f05..f14 --block 192 --paths 1000 --seed 7
```

**Expect**: an `.npz` under `stress/banks/` with full provenance. Re-running with the same
arguments reuses it; re-running after the source panels change **refuses** rather than warning.

## Step 4 — THE BLOCKING GATE

```
python -m research.labelfree.stress.calibrate --bank stress/banks/f05-f14_L192_s7.npz
```

**Expect** the three pre-registered numbers:

```
ks_distance  <= 0.10        (n=378 -> 95% critical value 0.0700)
es_ratio     0.7 .. 1.4
dd_ratio     0.6 .. 1.6
```

**If this fails the feature is killed** — clarification Q3 left no second generator. Write the
result report saying so; do not proceed to Step 5.

The body of the distribution should match by construction, since the bootstrap resamples real
blocks. The parts that can genuinely fail are the tails: spurious gaps at block joins firing
stops, and a drawdown distribution wrong because resampling destroys long-range dependence.

## Step 5 — sweep the block length

```
for L in 96 144 192 240 288; do
  python -m research.labelfree.stress.bank --build --folds f05..f14 --block $L --paths 1000 --seed 7
  python -m research.labelfree.stress.calibrate --bank stress/banks/f05-f14_L${L}_s7.npz
done
```

**Expect**: calibration reported **per block length**. `L` is swept, not chosen — reporting one
value as *the* answer would hide the sensitivity.

## Step 6 — stress the incumbent

```
python -m research.labelfree.stress.run --bank <best-calibrated bank> --policy incumbent
```

**Expect** a distribution, never a point estimate: P&L quantiles, drawdown quantiles, and the
**loss-path fraction**.

Brief 05 measured the incumbent at ≈ **+0.17 pp/yr** on the pooled test folds under this same
surrogate, so the prior is that a large share of paths lose money. **That is a finding, not a
failure** — it would say the incumbent's positive record is thin against resampling noise, which
is consistent with a 3.35 pp/yr floor it does not clear.

## Step 7 — the report

```
python -m research.labelfree.stress.report --out research/label-free-study/briefs/results/06-2026-09-14.md
```

Written whatever the outcome, including a calibration kill.

## What this harness is not

A **veto, never a gate**. "NO VETO" means the harness found no reason to object — not that the
policy is good. No policy may be promoted on stress results alone (SC-009, FR-014).
