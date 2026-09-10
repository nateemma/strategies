# Quickstart: Single-Pair (LINK) Variant

From the repo root, sequentially.

## S1 — Isolation is real (US1, FR-002/003/004)

```
grep -c '"LINK/USDT"' user_data/strategies/config/config_link.json      # whitelist is exactly LINK
python -c "import json;d=json.load(open('user_data/strategies/NNNC/NNNC_MLX_LINK.json'));\
print(d['strategy_name'], sorted(d['params']))"                         # buy AND sell present
```

**Expect**: `strategy_name` = `NNNC_MLX_LINK`; params carry `buy`, `sell`, `roi`, `stoploss`,
`trailing`. A buy-only file is a FAILURE — it reverts the exit stack.

## S2 — Train, and verify it happened (FR-005)

```
PATH="$PWD/.venv/bin:$PATH" zsh user_data/strategies/scripts/test_strat.sh \
  -c user_data/strategies/config/config_link.json -f -t 20220101-20250831 NNNC NNNC_MLX_LINK
ls user_data/strategies/saved_data/NNNC_MLX_LINK/*.safetensors
grep -c 'COMPLETELY UNGATED' /tmp/link_train.log
```

**Expect**: exit 0, artefact exists, **warning count 0**. A non-zero warning count VOIDS the run.

## S3 — Evaluate on all three windows

```
... run_arm.py --strategy NNNC_MLX_LINK --window W1 --tag link_only \
      --config user_data/strategies/config/config_link.json
```
**Expect**: `rc=0`. Trade counts to expect, from the control's LINK trades:
**W1 ~8, W2 ~26, W3 ~5**. A very low count is the expected result, not a failure.

Hold these in mind when reading S4: W1's noise floor is **7.633pp measured on 99 trades**, and
LINK contributes 8. Total return cannot resolve anything here. Per-trade expectancy can — at n=8
with sd 1.47 the standard error is ~0.52pp.

## S4 — The feature's actual answer (US2, US3)

```
... regime/pair_restricted.py --arm link_only --control nnnc_v2 --pair LINK/USDT --windows W1,W2,W3
```

**Expect**, per window: the three-way selection split; per-trade expectancy with intervals for
both arms; and any total-return delta marked UNRESOLVED against the floor.

**Reading it**: identical trade sets ⇒ cross-pair training contributes nothing to LINK selection
(a positive finding). Different and worse ⇒ cross-pair training carries real signal. Different and
better ⇒ dilution — but see Principle X in plan.md before concluding anything shippable.

## S5 — Seed robustness (FR-016)

Retrain with `MLX_SEED=7`; identical trades across seeds is ONE measurement, not two.

## S6 — Nothing else changed (SC-008)

```
git -C user_data/strategies status --short Framework/ regime/run_arm.py regime/tail/compare.py
```
**Expect**: empty.

## SC coverage

| SC | scenario |
|---|---|
| SC-001 | S1, S2 |
| SC-002 | S4 |
| SC-003 | S4 |
| SC-004 | S4 |
| SC-005 | conclusion doc |
| SC-006 | phase F only (P3) |
| SC-007 | conclusion doc |
| SC-008 | S6 |
