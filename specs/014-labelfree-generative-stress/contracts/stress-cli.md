# Contract: `research/labelfree/stress/` command surface

All commands run sequentially from `user_data/strategies` under
`freqtrade/.venv/bin/python`.

---

## `gencensus.py` — what can each generator actually emit?

```
python -m research.labelfree.stress.gencensus --out census.json
```

Probes each registered `GANType`'s declared and actual output shape **from the running code**
and records sequence capability with a `file:line` citation.

Nothing downstream consumes it — clarification Q3 removed every learned generator from scope —
but it ships as a durable finding about the repo's tooling, and it is the reason this feature
has exactly one generator.

**Does not import or instantiate any GAN model.** It reads shapes and contracts; it does not
load weights.

---

## `bootstrap.py` + `bank.py` — sample paths offline

```
python -m research.labelfree.stress.bank --build \
    --folds f05..f14 --block 192 --paths 1000 --seed 7
```

Stationary block bootstrap (Politis & Romano) over the fold panels, written to
`stress/banks/<fold-set>_L<block>_s<seed>.npz` with the provenance of data-model.md §3.

**Sampling is offline and once** (FR-001/SC-005). `load()` recomputes the source hash and
**refuses** a bank whose source panels have moved, rather than warning.

`--block` is swept, not chosen: `{96, 144, 192, 240, 288}` bars = 1–3 days.

---

## `calibrate.py` — THE BLOCKING GATE

```
python -m research.labelfree.stress.calibrate --bank <path> --out calibration.json
```

Runs the **incumbent** on the bank and compares its per-trade P&L against the real pooled
walk-forward trades, at the tolerances **pre-registered** in research.md R3:

```
ks_distance <= 0.10        (n=378 -> 95% critical value 0.0700)
0.7 <= es_ratio  <= 1.4    5% expected shortfall, synthetic / real
0.6 <= dd_ratio  <= 1.6    median max drawdown, synthetic / real
```

**Exit code is non-zero on failure**, and `run.py` refuses to proceed without a passing
`calibration.json`.

**A failure is a KILL, not a fallback** (SC-007). Q3 left no second generator, which is why the
tolerances were fixed before the first run rather than after seeing a distance.

Calibration is honestly a **plumbing-and-tails check**: the bootstrap resamples real blocks, so
the body of the distribution should match by construction. What it can genuinely catch is
spurious gaps at block joins firing stops, and a drawdown distribution wrong because resampling
destroys long-range dependence (research.md R2).

---

## `run.py` — stress a policy

```
python -m research.labelfree.stress.run --bank <path> --policy incumbent --out stress_report.json
```

Scores the policy on every path with feature 013's exit surrogate (31 ms per evaluation,
rank fidelity Spearman +0.9766 against real backtests) and reports a **distribution**:
P&L quantiles, drawdown quantiles, loss-path fraction, and the incumbent's figures beside any
candidate's (FR-012).

A path on which the policy never trades is counted **inadmissible**, not flat (FR-013). If those
dominate, the bank is reported inadmissible rather than averaging zeros into a distribution.

Output carries `verdict ∈ {VETO, NO VETO, INADMISSIBLE BANK}`. **"NO VETO" is not an
endorsement** — SC-009 forbids promoting any policy on stress results alone.

---

## `report.py` — the §6 result document

```
python -m research.labelfree.stress.report \
    --out research/label-free-study/briefs/results/06-<date>.md
```

Protocol §6 template. Leads with the generator census and the pivot it forced, then the
calibration, then the stress distribution.

Written **whatever the outcome**, including a calibration kill (FR-021).
