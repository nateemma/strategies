# Contract: Walk-Forward Harness CLI

Four commands under `regime/wf/`. All run from the **repo root** with
`PYTHONPATH=user_data/strategies`. All are sequential (R12).

## `foldplan.py` — build and validate a plan (FR-001, FR-002)

```
python user_data/strategies/regime/wf/foldplan.py \
  --family NNNC --anchored --min-train-days 730 --test-days 90 \
  --data-start 2021-01-01 --data-end 2026-09-07 \
  --out regime/wf/folds/plan_<run_id>.json
```

Emits the plan and a validation table. **Exit non-zero if any fold fails validation**, naming the
fold — a plan that cannot be validated must not reach execution.

Output: one line per fold (`fold_id`, train range, test range, `strategy_class`), then
`VALIDATED: n folds` or `REFUSED: <fold_id> — test range intersects its own training range`.

## `run_folds.py` — train and evaluate, fold by fold (FR-003..FR-005, FR-016)

```
python user_data/strategies/regime/wf/run_folds.py \
  --plan regime/wf/folds/plan_<run_id>.json --seed 1 [--folds f00,f01] [--dry-run]
```

Per fold, in order: generate the fold class + its `.json` (R4) → train → **verify the artefact
exists** (R5; never trust the exit code) → invoke `run_arm.py --timerange <test range>` (R6) →
record the fold.

`--dry-run` prints the fold sequence and the generated class/`.json` paths without training —
the cheap way to confirm identity uniqueness before spending hours.

Resumable: a fold already `evaluated` is skipped, so an interrupted run continues rather than
restarting.

## `pool.py` — pool per-trade results (FR-006, FR-012, FR-020)

```
python user_data/strategies/regime/wf/pool.py --plan <plan.json> --out regime/wf/pooled_<run_id>.json
```

Reads each fold's arm record, extracts trades from the archive named in `result_file` **using
`compare.py`'s existing accessor** (R7, FR-013), tags each trade with its `fold_id`, and writes the
pooled set. Zero-trade folds are recorded as zero-trade (FR-012). If any planned fold is missing,
the output is marked `status: partial` and every downstream report inherits that marking (FR-020).

## `power.py` — the headline report (FR-007..FR-011, FR-021)

```
python user_data/strategies/regime/wf/power.py --pooled <pooled.json> [--folds-subset recent-third]
```

Reports: pooled trade count; bootstrap 95% CI and **half-width (the achieved floor)**; the 7.633pp
baseline and the improvement ratio; the **minimum resolvable effect**; the 1/sqrt(n) prediction and
any shortfall with the inter-fold correlation explaining it; per-fold counts and returns; folds
where seeds gave identical trades; and pair-group expectancy when the universe is expanded.

`--folds-subset` recomputes over an era subset (FR-021), so era sensitivity stays inspectable and
Principle II remains checkable.

## Arm-vs-control on pooled folds (FR-013)

No new command. Pooled comparison routes through the existing
`regime/tail/compare.py`, which owns the paired-bootstrap statistic. `power.py` reports the
**floor of one set**; `compare.py` reports the **delta between two**. They must not both implement
a bootstrap of the same quantity.

## Invariants across all commands

- Never modify `run_arm.py`'s record schema (FR-018) or `Framework/` (FR-019).
- Never alter a gate, exit, label or model parameter (FR-015, SC-008).
- Identical `trades_hash` ⇒ inconclusive, never refuted. Zero trades ⇒ zero-trade, never
  zero-return.
- Same inputs ⇒ same floor and same verdicts (FR-022, SC-005).
