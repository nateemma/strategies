# Quickstart: Validating Label-free Direct Policy Optimisation

**Feature**: 009-labelfree-direct-policy · **Date**: 2026-09-10

Runnable validation scenarios, in the order the gates must pass. Each scenario states its
prerequisite, the command shape, and the outcome that constitutes a pass. Implementation belongs in
`tasks.md`, not here.

## Prerequisites

```bash
cd /Users/philprice95/projects/freqtrade
source .venv/bin/activate          # Python 3.12.13, MLX with Metal GPU
cd user_data/strategies
```

- 15-minute feather data present under `user_data/data/binanceus/`.
- Feature scalers present in `saved_data/` (regenerate only if the include list changes).
- Walk-forward config `config/config_wf.json` (10-pair universe).
- Control record `regime/wf/pooled_nnnc.json` (15 folds, 378 pooled trades).
- **Run everything sequentially.** Concurrent jobs are prohibited (AGENT_GUIDE operational rules).

---

## Scenario 1 — Simulator calibration (BLOCKING)

Nothing downstream is evidence until this passes. Budget: 5 attempts (SC-012).

```bash
python -m research.labelfree.calibrate \
    --window W1 --signal-from <arbiter-backtest-archive> \
    --report research/label-free-study/briefs/results/01-calibration-2026-09-10.md
```

**Pass when all three hold** (SC-001): trade counts identical; total return within 0.10pp;
maximum per-trade difference ≤ 1 bp.

The report must also attribute any residual to fee application, fill timing or stake sizing — a
bare "totals differ" is not a diagnosis (Principle VI).

**On failure**: fix the simulator and retry. After the 5th failed attempt, stop the feature and
record the discrepancy (SC-011, SC-012). Do not write policy code.

---

## Scenario 2 — Lookahead sanity ceiling

Prerequisite: Scenario 1 passed.

```bash
python -m research.labelfree.train --fold f07 --seed 1 --leaky   # features shifted +1 bar
python -m research.labelfree.train --fold f07 --seed 1           # correct timing
```

**Pass when** the leaky run's cumulative net **log** return exceeds the clean run's by ≥ 2.0 on the
same range and seed, and the clean run does not beat its **exposure-matched** benchmark
(buy-and-hold scaled to the clean policy's own mean exposure) by more than the resolvable floor
(SC-002). Report log-returns: a leaky run's linear return overflows.

**On failure**: the gradient path is broken. This is a defect, not a null result — fix it before
running the grid (Principle IV).

---

## Scenario 3 — Determinism

Prerequisite: Scenario 2 passed. Cheap, and it de-risks SC-008 before the expensive grid.

```bash
python -m research.labelfree.train --fold f07 --seed 1 --save /tmp/a
python -m research.labelfree.train --fold f07 --seed 1 --save /tmp/b   # separate process
diff <(jq -S . /tmp/a/trades.json) <(jq -S . /tmp/b/trades.json)
```

**Pass when** the trade lists are byte-identical. If not, escalate — do **not** relax SC-008 to a
tolerance.

---

## Scenario 4 — Variant grid on one fold

Prerequisite: Scenarios 1–3 passed.

12 configurations (2 objectives × 2 turnover controls × 3 cost scales) × ≥5 seeds, on the screening
fold only.

```bash
python -m research.labelfree.sweep --fold f07 --seeds 1,2,3,4,5 --grid full
```

**Pass when**: every configuration has a trial-log record written **before** its result (SC-015);
reported figures are interquartile means with bootstrap CIs, never best-seed (SC-016); seeds
producing identical trades are counted once. Select the best two configurations for the pooled run.

---

## Scenario 5 — Pooled walk-forward verdict

Prerequisite: Scenario 4 selected two configurations. This is the long job — the control's own
15-fold run took ~113 minutes.

```bash
python -m research.labelfree.export_policy --trial <trial_id>     # -> NNDP/NNDP_MLX.py model path
python regime/wf/run_folds.py --plan regime/wf/folds/plan_nnnc.json --seed 1
python regime/wf/pool.py
python regime/wf/compare_pooled.py --arm <arm.json> --control regime/wf/pooled_nnnc.json
```

Also run the four mandatory baselines (FR-021), including the **pinned** 20/60-bar EMA trend rule
with band ±0.1 — frozen before this scenario and contributing no trial-log records (FR-021a).

**Pass when** SC-003 through SC-006 hold. A delta below 3.35pp/yr reads "not resolved", never "no
effect" (SC-009).

**Also required before any acceptance** (SC-018): report simulator P&L, arbiter P&L, their
difference, and the share of trades exited by the deployment stack rather than the band.

**Report per window** (FR-015a): the pair-universe size. Expect the distant window to run on
9 of 10 pairs — only XRP is excluded, for a 911-day delisting gap — with six of the nine training
on partial history (345-533 days rather than 729). State that data-quantity asymmetry explicitly.

---

## Scenario 6 — Cost stress, statistics, reproduction, report

```bash
python -m research.labelfree.stress --trial <trial_id> --fees 1.5,2.0 --cap 0.05
python -m research.labelfree.stats --trials research/labelfree/trials.jsonl   # DSR + PBO (CSCV)
python -m research.labelfree.reproduce --config <saved-config> --seed <n>     # fresh process
python -m research.labelfree.audit_trials --trials research/labelfree/trials.jsonl
```

**Pass when**: the delta survives 1.5× fees (SC-007); PBO ≤ 0.2 and DSR > 0 over the **full** trial
log (SC-006); a fresh process reproduces identical trades (SC-008); the audit confirms every
configuration is logged and each was written before its result (SC-015).

Finally write `research/label-free-study/briefs/results/01-<date>.md` per protocol §6, linking the
feature directory and both `/speckit-analyze` reports (FR-033, SC-017). **The report is written
even when the verdict is kill or not-resolved** — the negative result is the deliverable.

---

## Staging

Stage only files this feature creates (FR-031). Never `git add -A`. The outer freqtrade checkout
receives no commits.
