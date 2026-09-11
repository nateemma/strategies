# Quickstart: Validating Label Warm-Start with P&L Fine-Tune

**Feature**: 010-labelfree-warmstart-finetune · **Date**: 2026-09-11

Ordered validation scenarios. Implementation belongs in `tasks.md`.

## Prerequisites

```bash
cd /Users/philprice95/projects/freqtrade && source .venv/bin/activate
cd user_data/strategies
python -m research.labelfree.preflight      # inherited from brief 01
python -m pytest research/labelfree/tests/ -q    # 16 existing tests must pass first
```

**Run everything sequentially.** Window tensors reach 2.14 GB on the full panel — build them per
chunk (0.04 GB), never for the whole panel.

---

## Scenario 1 — Leakage control (BLOCKING)

Nothing that uses a label is evidence until this passes.

**Pass when**: for every fold, the label horizon is purged at the boundary and one horizon is
embargoed after it, and the applied values are recorded per fold. An arm missing either is void
(SC-012), not caveated.

---

## Scenario 2 — Warm-start actually transfers (BLOCKING for Arm A)

**Pass when**: inherited tensor names and shapes match the policy's encoder exactly; post-load
encoder weights differ from a randomly-initialised policy of the same architecture; the
classification head is absent from the inherited set.

**On failure**: the arm silently did nothing and any result from it is void — a warm-start that
loads nothing is indistinguishable from Arm B, and would be reported as one.

---

## Scenario 3 — Arm A on the screening fold

```bash
python -m research.labelfree.arms --arm warm_start --fold f08 --seeds 1,2,3,4,5
```

**Pass when**: held-out return and mean exposure are reported against the P&L control on the same
fold and seeds; the exposure/label correlation is reported as a diagnostic.

**If Arm A underperforms the P&L control in-fold**: record it and proceed to Arm B. Do not tune
Arm A — a classifier tuned to the wrong label can legitimately hurt, and tuning here would be
fitting the screen.

---

## Scenario 4 — Arm B blend sweep

```bash
python -m research.labelfree.arms --arm blended --fold f08 \
    --alphas 0.75,0.5,0.25 --labels 0.007,0.02 --seeds 1,2,3,4,5
```

**Pass when**: every configuration is logged before running; `alpha_realised` is reported beside
`alpha_nominal`; the two labels are reported separately, never pooled; no single α is presented
as best (SC-010).

Expect the two labels to agree: the repo measured 0.007 vs 0.02 as identical in P&L (332 trades,
10.45% both) though 38% apart in label count. If they agree here, that replicates a known finding
rather than discovering one, and must be reported as such.

---

## Scenario 5 — Arm C auxiliary head

```bash
python -m research.labelfree.arms --arm auxiliary --fold f08 --seeds 1,2,3,4,5
```

**Pass when**: no directional label enters the objective at any point, and the report states
whether directional content was necessary by comparing against the best blended arm.

---

## Scenario 6 — Verdict against both controls

The P&L control must be **re-run under this architecture** (FR-004b) — citing brief 01's numbers
would confound the label prior with the architecture change.

```bash
python -m research.labelfree.arms --arm pnl_control --fold f08 --seeds 1,2,3,4,5
python regime/wf/run_folds.py --plan regime/wf/folds/plan_nnnc.json --seed 1
python regime/wf/compare_pooled.py --arm <arm.json> --control regime/wf/pooled_nnnc.json
```

**Pass when**: SC-001…SC-006 hold against **both** controls. A delta below 3.35pp/yr reads "not
resolved", never "no effect".

**Always required, even on a kill**: the report records whether labels **helped or hurt**
relative to the P&L control (SC-009), and states both P&L-control figures — re-run and brief 01's
original — so the architecture effect is separable (SC-010a).

---

## Scenario 7 — Report and audit

```bash
python -m research.labelfree.trials --audit
```

Write `research/label-free-study/briefs/results/02-<date>.md` per protocol §6, linking the
feature directory and both analyze reports. **Written even on a kill** — the directional finding
is the deliverable.

## Staging

Stage only files this feature creates or modifies. Never `git add -A`.
