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

**Expect**: an `.npz` under `stress/banks/` with full provenance. **Measured**: 88.3 MB in 46 s
for 10 folds x 1,000 paths. The bank stores index vectors rather than resampled arrays, so it
stays small and is exactly reproducible from the source panels. `load()` **refuses** a bank whose
source has moved rather than warning.

## Step 4 — THE BLOCKING GATE

```
python -m research.labelfree.stress.calibrate --bank stress/banks/f05-f14_L192_s7.npz
```

Calibration compares **surrogate-on-synthetic against surrogate-on-real-panels** — the evaluator
is held constant so only the paths vary. The archived freqtrade trades appear as a secondary
sanity reference, not a gate.

**Expect** the three pre-registered numbers (measured at L=96: ks 0.0252, es 0.9317,
dd 1.0995 — all inside the bands, with no tolerance moved):

```
ks_distance  <= 0.10        (n=378 -> 95% critical value 0.0700)
es_ratio     0.7 .. 1.4
dd_ratio     0.7 .. 1.6
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

**Expect**: calibration reported **per block length**, ~4 min each. **Measured** — all five pass:

```
L     ks       es_ratio  dd_ratio
96    0.0252   0.9317    1.0995
144   0.0183   0.9488    1.0843
192   0.0149   0.9606    1.0965
240   0.0126   0.9686    1.0832
288   0.0119   0.9700    1.0721
```

**Select the SHORTEST that passes, not the best calibrated.** KS falls monotonically with block
length because longer blocks preserve more real structure — so at the limit, where a block spans
the series, calibration would be perfect and the harness would resample nothing. Selecting by
best calibration drives toward that degenerate case.

## Step 6 — stress the incumbent

```
python -m research.labelfree.stress.run --bank <best-calibrated bank> --policy incumbent
```

**Expect** a distribution, never a point estimate: P&L quantiles, drawdown quantiles, and the
**loss-path fraction**.

**Measured** (incumbent, L=96, 10,000 paths, ~190 s):

```
P&L pp/yr   p5 -2.71   p25 -0.33   p50 +0.19   p75 +0.89   p95 +2.75
loss-path fraction   0.386
9,481 scored / 519 inadmissible
```

The median **+0.193** reproduces feature 013's **+0.171** measured on the real folds to within
0.02 pp — an independent consistency check. The 90% interval straddles zero and its 5.46 pp width
sits around a 3.35 pp/yr floor the incumbent does not clear: **the edge is thin against
re-ordering**, which is a finding rather than a failure.

## Step 7 — the report

```
python -m research.labelfree.stress.report --out research/label-free-study/briefs/results/06-2026-09-14.md
```

Written whatever the outcome, including a calibration kill.

## What this harness is not

A **veto, never a gate**. "NO VETO" means the harness found no reason to object — not that the
policy is good. No policy may be promoted on stress results alone (SC-009, FR-014).
