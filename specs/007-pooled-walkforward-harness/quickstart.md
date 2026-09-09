# Quickstart: Pooled Walk-Forward Evaluation Harness

Validation scenarios proving the harness works end-to-end. Run everything from the **repo root**
(the directory containing `user_data/`), sequentially.

**Prerequisites**: repo `.venv`; feathers covering 2021-01-01..2026-09-07; a trained production
`NNNC_MLX` for reference; nothing else running (`pgrep -f 'freqtrade backtesting'` empty).

---

## S1 — The plan validates, and refuses a bad fold (FR-001, FR-002, SC-003)

```
PYTHONPATH=user_data/strategies .venv/bin/python \
  user_data/strategies/regime/wf/foldplan.py --family NNNC --anchored \
  --min-train-days 730 --test-days 90 --data-start 2021-01-01 --data-end 2026-09-07 \
  --out regime/wf/folds/plan_test.json
```

**Expect**: ~15 folds, first test starting 2023-01-01, `VALIDATED: 15 folds`, exit 0.

**Negative control** — hand-edit one fold's `test_start` to fall inside its training range and
re-validate. **Expect**: exit non-zero, the offending `fold_id` named, and no execution possible.
A plan that silently accepted this would invalidate every number downstream.

## S2 — Fold identities are unique before any compute is spent (FR-003, R4)

```
... run_folds.py --plan regime/wf/folds/plan_test.json --seed 1 --dry-run
```

**Expect**: one generated class per fold (`NNNC_WF_f00`..), each with a distinct `saved_data/`
path and its own `.json` whose `strategy_name` equals its class. **Two folds sharing a class or a
model path is a hard failure** — that is the silent-weight-overwrite mode, and this dry run is the
cheap place to catch it.

Confirm each generated `.json` carries the production entry gate, not `NNNCStrategy`'s
all-guards-open class defaults.

## S3 — One fold end-to-end (FR-004, FR-005, R5, R6)

```
... run_folds.py --plan regime/wf/folds/plan_test.json --seed 1 --folds f00
```

**Expect**: training runs; the artefact check passes on `saved_data/NNNC_WF_f00/*.safetensors`;
`run_arm.py` is invoked with the fold's `--timerange`; an arm record appears carrying `result_file`.

**Critical check**: the fold's test range must be strictly later than its training range. If the
model directory was empty and the backtest "succeeded" anyway, the model was trained **inside the
backtest** — treat any fold whose artefact check was skipped as void.

## S4 — Full run and the pooled floor (FR-006..FR-009, SC-001, SC-002)

```
... run_folds.py --plan regime/wf/folds/plan_test.json --seed 1     # ~3.1 h, sequential
... pool.py  --plan regime/wf/folds/plan_test.json --out regime/wf/pooled_test.json
... power.py --pooled regime/wf/pooled_test.json
```

**Expect**:

| check | target |
|---|---|
| pooled trades | ≥ 350 (SC-002), against 99 today |
| achieved floor (CI half-width) | ≤ 4.0pp (SC-001), against 7.633pp |
| improvement ratio | ≈ 1.9x or better |
| min resolvable effect | stated explicitly (SC-004) |
| per-fold breakdown | present (FR-010) |

**If the floor does not fall as 1/sqrt(n) predicts**, the report must state the shortfall and the
inter-fold correlation (FR-009, SC-007). That is a legitimate result, not a failed run — it would
mean the power ceiling is lower than data volume implies, which is worth more than a passing number.

## S5 — Reproducibility (FR-022, SC-005)

Re-run `power.py` on the same pooled file. **Expect**: byte-identical floor and CI
(`BOOTSTRAP_SEED = 0`). Any drift means the statistic is not seeded and no verdict from it is
trustworthy.

## S6 — Seed robustness on a subset (FR-011, R11)

```
... run_folds.py --plan regime/wf/folds/plan_test.json --seed 7 --folds f00,f04,f08,f12
```

**Expect**: for each of the 4 folds, either differing trades (two real measurements) or a report
that the seeds produced **identical** trades — which is ONE measurement, not two. Silence here
would let a doubled sample size be claimed from duplicated data.

## S7 — Era sensitivity and Principle II (FR-021)

```
... power.py --pooled regime/wf/pooled_test.json --folds-subset recent-third
```

**Expect**: floor and pooled return recomputed on recent folds only. Used to show an effect
survives in the deployment regime rather than being carried by distant folds.

## S8 — The instrument changed nothing (FR-015, SC-008)

```
git -C user_data/strategies status --short Framework/ regime/run_arm.py regime/tail/compare.py
git -C user_data/strategies diff --stat NNNC/NNNC_MLX.json
```

**Expect**: empty. The harness measures; it must not have moved a gate, an exit, a label, or the
comparison statistic.

---

## Success criteria coverage

| SC | scenario |
|---|---|
| SC-001 floor ≤ 4.0pp | S4 |
| SC-002 ≥ 350 pooled trades | S4 |
| SC-003 all folds out-of-sample | S1, S3 |
| SC-004 floor + min resolvable effect stated | S4 |
| SC-005 reproducible | S5 |
| SC-006 pair-group expectancy | phase E (not in the first run) |
| SC-007 sqrt(n) shortfall reported | S4 |
| SC-008 no configuration changed | S8 |
