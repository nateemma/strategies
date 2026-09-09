# Pooled walk-forward power — measured 2026-09-09

Feature 007. The harness is a **measurement instrument**; its deliverable is statistical power,
not P&L. No gate, exit, label or model configuration was changed to produce these numbers.

## Result

| | value |
|---|---|
| pooled out-of-sample trades | **378** (baseline 99) |
| pooled span | 3.68 yr-equivalent, 15 folds |
| bootstrap 95% CI (whole span) | [+13.854%, +38.545%] |
| half-width on the whole span | 12.345pp |
| **ACHIEVED FLOOR (annualised)** | **3.350 pp/yr** |
| baseline floor (single window) | 7.633 pp/yr |
| **improvement** | **2.28x** |
| 1/sqrt(n) prediction | 3.906 pp/yr |
| shortfall | **−0.556pp — pooling OVER-delivered** |
| fold-return lag-1 correlation | −0.171 |
| per-trade dispersion | 0.635 (baseline 0.767) |

SC-001 (measure and report) **SATISFIED**. SC-001a (≤4.0pp) **MET**. SC-002 (≥350 trades) **MET**.

**Pooling worked better than predicted.** R9 flagged the risk that fold returns correlate through a
shared regime, which would make the floor fall more slowly than 1/sqrt(n). It did not happen: the
lag-1 correlation is −0.171 (slightly negative, i.e. effectively independent), and per-trade
dispersion across the walk-forward is *lower* than in the single baseline window.

## What this does and does not make measurable

**The instrument now resolves ~3.4pp/yr instead of ~7.6pp.** Against the effects measured this
week:

| change | delta | before | after |
|---|---|---|---|
| exit-defect repair | +4.5 / +12.0pp | invisible | **resolvable** |
| label conflict resolution off (DDPM) | −24pp | resolvable | resolvable |
| rvol 2.5 | +1.86pp | invisible | **still invisible** |
| grace −0.10 | +1.04pp | invisible | **still invisible** |

**Parameter-scale effects (1–2pp) remain unresolvable, and always will be on this data.** That is a
limit of 5.68 years of history at ~100 trades/yr, not a defect of the harness. Nothing that
returned "not resolved" in the 2026-09-08 sweeps would resolve now. The win is that the **3–7pp
band opens** — the band containing every change that has actually mattered here.

Do not read the harness as licence to resume parameter tuning.

## Fold detail

```
fold   trades  return%  days     fold   trades  return%  days
f00        11   +1.822    90     f08        71   -0.138    90
f01        11   -1.531    90     f09        12   -0.150    90
f02         9   +1.147    90     f10         7   -0.155    90
f03        19   +2.371    90     f11        48   +3.150    90
f04        41   +3.243    90     f12        24   +2.009    90
f05        25   +0.947    90     f13        16   +4.099    90
f06        23   +1.082    90     f14         3   +0.160    85
f07        58   +8.080    90
```

Trade counts vary 3–71. The early folds are thin partly because several pairs had not yet listed.

## Geometry, and two constraints that shaped it

ANCHORED (expanding) training, 730-day minimum, 90-day test folds, first test 2023-01-01.
Anchored because production trains on all data to a cutoff; a short fixed rolling window would
change the training regime and confound the instrument with what it measures.

**SUI/USDT is excluded** (`config/config_wf.json`). Training only fires once every whitelist pair
has been seen (`TrainingEngine.maybe_train`: `pair_count == len(whitelist)`), and SUI lists
2023-10-25 — so every fold training before that date saw 10 of 11 pairs and **never trained**.
Keeping it would force the first fold to 2023-10-25: 11 folds / 2.87 yr / ~284 trades. Dropping it
gives 15 folds / 3.68 yr / 378 trades. Cost on the W1 control: SUI is 8 of 99 trades (8.1%) and
4.0% of P&L. A documented deviation from the production 11-pair universe.

**The final fold is shorter** (85 days). Dropping it to keep folds equal discarded genuine
out-of-sample data and cost ~23 trades. Folds are pooled by TRADE, not averaged by fold, so unequal
lengths are fine for the pooled statistic.

## Era sensitivity (FR-021)

The recent third alone: 98 trades, 1.22 yr, floor **6.218 pp/yr** — a 1.23x improvement. The power
comes from pooling MANY folds, not from any single era. Principle II stays checkable:
`power.py --folds-subset recent-third`.

## Train-length confound (the instrument measuring itself)

Anchored training means early folds train on ~2 years against later folds' ~5.4. Checked:

```
corr(train_days, val_mcc)       = +0.478   (n=15; critical ~0.514 at p=0.05)
corr(train_days, expectancy/tr) = +0.136
```

Training length has a marginal, **non-significant** relationship with classification quality and
essentially none with per-trade expectancy. The confound is not established. One more instance of a
classification metric failing to reach P&L.

## Two errors made producing this, both worth remembering

**The first version of this report said 0.62x and "POOLING UNDERDELIVERED".** It compared a
bootstrap CI on the 3.68-YEAR total against a ONE-YEAR baseline. Annualisation is now explicit in
`power.py`, with the whole-span figure printed separately so the two cannot be conflated. Reporting
that number would have declared a working instrument a failure — the same class of error as the
1.0pp threshold it was built to replace: arithmetic dressed as a finding.

**research.md R6 asserted `run_arm.py` accepts `--timerange`. It does not.** The string appears in
an internal helper; `--window` is required. Caught when fold f00 trained successfully then failed
evaluation in one second. Each fold now runs in its own subprocess that injects its window into the
in-memory `WINDOWS` dict — `regimedata.py` on disk stays untouched, which matters because
`phase_c_trackability.py` does `for w in WINDOWS`.

## Cost

112.6 minutes for 15 folds, sequential — under the 3.1 h budget, because anchored training makes
early folds cheap (3.9 min) and only later ones expensive.

## Reproducing

```
PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/regime/wf/foldplan.py \
  --family NNNC --base-strategy NNNC_MLX --out regime/wf/folds/plan_nnnc.json
... regime/wf/run_folds.py  --plan regime/wf/folds/plan_nnnc.json --seed 1     # ~113 min
... regime/wf/pool.py       --plan regime/wf/folds/plan_nnnc.json --out regime/wf/pooled_nnnc.json
... regime/wf/power.py      --pooled regime/wf/pooled_nnnc.json
... regime/wf/compare_pooled.py --arm <pooled_arm.json> --control regime/wf/pooled_nnnc.json
```

Seeded (`BOOTSTRAP_SEED = 0`); reruns reproduce the floor exactly.
