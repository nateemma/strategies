# Pooled Walk-Forward Harness

**What it is:** a measurement instrument. It answers one question — *is this arm better than that
control, and by how much, with a confidence interval you can defend?* It does not tune anything,
and nothing in `regime/wf/` may alter an entry gate, an exit parameter, a label, or a model
configuration (feature 007, FR-015).

**Why it exists:** a single-window backtest on this data has a 95% CI half-width of **7.633pp**.
A +1.86pp "improvement" measured that way has a CI of `[-4.29, +7.93]pp`, p=0.54 — it is noise.
Pooling 15 anchored folds into 378 trades drops the floor to **3.350 pp/yr**, a 2.28x improvement.
Full derivation: [`../WALKFORWARD_POWER.md`](../WALKFORWARD_POWER.md).

---

## The decision rule

| Effect size | Verdict |
|---|---|
| **≥ 3pp/yr** | Resolvable. Use this harness. |
| **1–2pp/yr** | Unresolvable, and always will be on 5.68 yr at ~100 trades/yr. **The harness is not licence to resume parameter tuning.** |

Two corollaries that have already caused mistakes:

- **Annualise before comparing floors.** A pooled CI spans ~3.7 years; the single-window baseline
  spans one. The first power report compared them directly and declared a working instrument a
  0.62x regression.
- A single-window comparison is still valid for a *large* effect — but quote the 7.633pp floor
  alongside it.

---

## The pipeline

Five stages. Run from the **repo root** (the directory containing `user_data/`), **sequentially** —
two concurrent jobs put load 60 on 14 cores and finished slower than one.

```bash
export PYTHONPATH=user_data/strategies
PY=.venv/bin/python
WF=user_data/strategies/regime/wf
```

### 1. Plan the folds

```bash
$PY $WF/foldplan.py --family NNNC --base-strategy NNNC_MLX \
                    --out regime/wf/folds/plan_nnnc.json
```

Anchored expanding windows: training always starts at `--data-start`, the test fold slides forward.
Defaults — `--min-train-days 730`, `--test-days 90`, `--data-start 2021-01-01`,
`--data-end 2026-09-07`, `--min-final-days 45`, `--config config/config_wf.json`. The plan is
**validated**: any fold whose test range is not genuinely out-of-sample is refused, not warned about.

### 2. Run the folds

```bash
$PY $WF/run_folds.py --plan regime/wf/folds/plan_nnnc.json --seed 1
```

**~113 minutes, sequential.** Generates one `NNNC_WF_f{NN}` strategy class per fold, trains it, and
evaluates it on its own out-of-sample range. Use `--dry-run` to see what it would do, and
`--folds f07,f08` to run a subset. A fold counts as `trained` only if its artefacts actually exist —
a silently-untrained fold is caught here, not three hours later.

### 3. Pool

```bash
$PY $WF/pool.py --plan regime/wf/folds/plan_nnnc.json --out regime/wf/pooled_nnnc.json
```

Concatenates every fold's trades into one set **before any statistic is computed**. That ordering is
the whole point: pooling trades then bootstrapping is far more powerful than bootstrapping each fold
then averaging.

### 4. Report the power

```bash
$PY $WF/power.py --pooled regime/wf/pooled_nnnc.json
```

Prints the minimum resolvable effect, the 1/√n prediction and the shortfall against it, the
fold-return lag-1 correlation, and a per-fold table. `--folds-subset recent-third` restricts to the
most recent third of folds.

If the shortfall exceeds 0.25pp the report says **POOLING UNDERDELIVERED** — fold returns are not
independent, so the power ceiling is below what trade count implies. That is a finding to report,
never a reason to retune the fold geometry.

### 5. Compare an arm to a control

```bash
$PY $WF/compare_pooled.py --arm regime/wf/pooled_arm.json \
                          --control regime/wf/pooled_nnnc.json
```

Pairs trades on `(fold_id, pair, open_date)` — the same pair and timestamp can legitimately recur
across folds, so the two-field key used for single windows would collide. Delegates to
`regime.tail.compare.bootstrap_delta`, the *same* statistic as single-window arms, not a second
implementation.

Output is a delta in pp/yr, a 95% CI, a p-value, and one of:

| Verdict | Meaning |
|---|---|
| `IMPROVED` / `WORSENED` | CI excludes zero |
| `UNCHANGED — CI spans zero, NOT RESOLVED` | the honest majority outcome |
| `INCONCLUSIVE_ABSORBED` | every trade is identical; the intervention changed nothing |
| `INCONCLUSIVE_UNDERPOWERED` | too few trades actually differ |
| `*** PARTIAL fold set` | prepended when either side is not a complete plan |

---

## Artefacts

```
regime/wf/
  __init__.py             pinned baseline constants — measured, never recomputed
  foldplan.py             stage 1
  run_folds.py            stage 2
  pool.py                 stage 3
  power.py                stage 4
  compare_pooled.py       stage 5
  folds/
    plan_<name>.json      the fold plan
    params_<run_id>.json  the frozen strategy params, SHA-recorded in the run manifest
    results_<run_id>_seed<N>.json
  pooled_nnnc.json        the designated NNNC_MLX control record (15 folds, 378 trades)
```

The constants in `__init__.py` were **measured** on 2026-09-09 by bootstrapping NNNC_MLX's own W1
per-trade P&L (99 trades, +9.831%) and are **pinned, not recomputed**. `TARGET_HALF_WIDTH_PP = 4.0`
is a target, and missing it is a reportable result rather than a feature failure.

---

## Traps

These are load-bearing. Both cost real debugging time.

- **A pair that lists mid-history silently prevents training.** `TrainingEngine.maybe_train` fires
  only when `pair_count == len(whitelist)`, so a window predating any whitelist listing trains
  *nothing* and then fails at predict. SUI/USDT lists 2023-10-25, so `config/config_wf.json` drops
  it — a **documented deviation** from the production 11-pair universe. Keeping SUI would force the
  first test fold to 2023-10-25 (11 folds / 2.87 yr / ~284 trades); dropping it gives 15 folds /
  3.68 yr. Measured cost on the W1 control: SUI is 8 of 99 trades and 4.0% of P&L.
- **`run_arm.py` has no `--timerange`.** `--window` is required and its choices come from
  `regime.regimedata.WINDOWS`. Do **not** add fold keys to that dict on disk — `phase_c_trackability.py`
  iterates it. `run_folds.py` injects the window into the in-memory dict inside a per-fold
  subprocess; `regimedata.py` and `run_arm.py` on disk are never modified.
- **Params are frozen at plan time.** `NNNC/NNNC_MLX.json` is untracked working state the user edits
  live; re-reading it per fold would make folds incomparable.

---

## Scope and coupling

The harness is general-purpose in intent but currently lives inside the `regime/` study directory and
is coupled to it in two places:

- `compare_pooled.py` and `power.py` import `regime.tail.compare` (deliberate reuse — one statistic,
  not two).
- `run_folds.py` drives `regime/run_arm.py` and monkeypatches `regime.regimedata.WINDOWS`.

`--family` / `--base-strategy` are already parameters, so nothing is hardcoded to NNNC. Promoting
`wf/` to a top-level shared location would mean lifting `tail/compare.py`'s statistics out too, and
giving `run_arm.py` a real `--timerange` so the monkeypatch can go.

**That promotion is a planned task**, blocked until the label-free study closes — see
[`../../docs/WALKFORWARD_TODO.md`](../../docs/WALKFORWARD_TODO.md) item 1 for the ordered work and
the reason for the gate.

## See also

- [`../WALKFORWARD_POWER.md`](../WALKFORWARD_POWER.md) — why 15 folds and 378 trades; the power measurement
- [`../../specs/007-pooled-walkforward-harness/`](../../specs/007-pooled-walkforward-harness/) — spec, plan, tasks, and `quickstart.md` (end-to-end validation scenarios)
- [`../../AGENT_GUIDE.md`](../../AGENT_GUIDE.md) — the resolvability rules this harness exists to serve
