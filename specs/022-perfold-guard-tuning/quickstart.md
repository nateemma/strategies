# Quickstart: running feature 022

Run from the repo root (`/Users/philprice95/projects/freqtrade`) with the harness venv.
`PY=.venv/bin/python`, `PYTHONPATH=user_data/strategies`.

**Order matters.** Each step can terminate the feature with a complete result, and the cheap
steps come first by design (Principle VII).

---

## Step 1 — Provenance (US1). No compute.

Trace the pre-2026-09-06 tuned guard values and compare their tuning window to the 15 fold test
ranges.

```
git log -S'entry_atr_pct' -- user_data/strategies/Framework/BaseStrategy.py
ls user_data/strategies/**/*.fthypt
$PY user_data/strategies/evaluation/foldplan.py --show   # the 15 test ranges
```

**Expected outcome**: a `ProvenanceVerdict` with `arm_b_status` set to one of `trustworthy`,
`in_sample` or `untrustworthy`. **`untrustworthy` is a legitimate result** — it means arm B's
+18.76pp cannot be read as out-of-sample, C vs B loses its interpretation, and C vs A carries
the feature alone.

**Stop condition**: none. This step always completes and always records a verdict.

---

## Step 2 — Audit the proxy BEFORE trusting it (research.md R2).

Score arms A and B with the table-based proxy and compare to their known harness results.

```
$PY user_data/strategies/evaluation/pick_guards.py --audit-proxy \
    --against wf_gbbo,wf_gbbg
```

**Expected outcome**: the proxy separates A (+2.64pp) from B (+18.76pp) in the right direction
and roughly the right magnitude.

**Stop condition**: if it does not, the gate is **not licensed at 64 configurations**. Fall back
to two axes (`atr_pct` × `adx`, 16 points) scored by real backtests, per research.md's recorded
fallback. Do not proceed on an unaudited proxy — feature 021 had this instrument match live to
3.6e-5 on one population and get the **sign wrong** on another.

---

## Step 3 — The trackability gate (US2). One working session. **BLOCKING.**

```
$PY user_data/strategies/evaluation/pick_guards.py \
    --plan user_data/strategies/evaluation/folds/plan_GbbGuardGrid.json \
    --out  user_data/strategies/evaluation/folds/guards_gbbgrid.json

$PY user_data/strategies/evaluation/guard_trackability.py \
    --plan user_data/strategies/evaluation/folds/plan_GbbGuardGrid.json \
    --out  user_data/strategies/evaluation/folds/trackability_gbbgrid.json
```

**Expected outcome**: a `verdict`, plus `n_distinct_choices` reported separately so variation is
not mistaken for trackability.

**Stop condition — the important one**: `verdict: not_trackable` means **arm C MUST NOT run**
(FR-006) and the feature is **complete**. Write the finding up and stop. Given that
`GUARD_ABLATION.md` shows every live guard's effect flipping sign across all three eras, and
four adaptive schemes have already lost to static values in this project, this is the expected
outcome.

---

## Step 4 — Prove the axes bit (FR-013), with the negative control.

Only if Step 3 passed.

```
# one fold, one axis at a time -> trades_hash must CHANGE
for v in <4 atr_pct levels>; do
  $PY -m regime.run_arm --strategy GbbGuardGridOpen --timerange <f00 test> \
      --label F00 --tag probe_atr_$v --set entry_atr_pct=$v \
      --config user_data/strategies/config/config_wf.json
done

# NEGATIVE CONTROL: vary only the known-inert guard -> trades_hash must NOT change
$PY -m regime.run_arm ... --set entry_guard_threshold=0.1  ...
$PY -m regime.run_arm ... --set entry_guard_threshold=0.9  ...
```

**Expected outcome**: distinct `trades_hash` per level on each of the three tuned axes, and an
**identical** hash across the two `entry_guard_threshold` runs.

**Stop condition**: an axis with an unchanged hash is INERT and must be reported as a defect
(FR-013) — the grid was searching a dimension that does not exist. If the negative control
*changes*, the check itself is broken and nothing downstream can be trusted.

---

## Step 5 — Arm C (US3). 15 backtests, one at a time, ~30 min.

```
$PY user_data/strategies/evaluation/run_folds.py \
    --plan user_data/strategies/evaluation/folds/plan_GbbGuardGrid.json
```

**Expected outcome**: 15 arm records under `regime/arms/wf_gbbgrid_f*.json`, each with an
`exit_config_hash` equal to `5a18305d1fafa83e`, a `trades_hash`, and
`entry_enable_guards_written: true`.

**Verify before interpreting anything**:

```
$PY - <<'EOF'
import json, glob
h={json.load(open(f))['exit_config_hash'] for t in ('wf_gbbo','wf_gbbg','wf_aro','wf_gbbgrid')
   for f in glob.glob(f'user_data/strategies/regime/arms/{t}_f*.json')}
print('exit hashes across all arms:', h, '-> OK' if len(h)==1 else '-> MISMATCH, comparisons void')
EOF
```

---

## Step 6 — Compare, paired (FR-015), and report halves (FR-016).

```
$PY user_data/strategies/evaluation/pool.py  --plan .../plan_GbbGuardGrid.json
$PY user_data/strategies/evaluation/compare_pooled.py \
    --arm  .../pooled_gbbgrid.json --ctl .../pooled_gbbo.json
$PY user_data/strategies/evaluation/power.py --pooled .../pooled_gbbgrid.json
```

**Expected outcome**: arm C's pooled wallet return, folds-positive out of the powered folds, and
a paired interval against arm A via `bootstrap_delta` — which already handles the
symmetric-difference case correctly because the entry set moves. Plus early-half and late-half
values for every reported effect.

**Then apply the pre-registered decision rule in spec.md.** It was fixed before Step 5 ran and
FR-019 forbids altering it now.

---

## Step 7 — Arm E (US4), only if C cleared its rule.

Reuses the incumbent's 15 existing `saved_data/NNNC_ARO_f*/**.safetensors`. **No training.**

---

## Invariants to re-check at every step

- `exit_config_hash` identical across A, B, C, D, E (SC-004).
- `entry_enable_guards` true in every arm (FR-008) — it also gates `volume > 0`, and 34.26% of
  bars here have `volume <= 0`.
- No fold with fewer than 25 trades contributes a return observation (FR-017).
- The live dry-run `MomentumRegimeBasket15mFast` is untouched: nothing in this feature modifies
  `Framework/`.
