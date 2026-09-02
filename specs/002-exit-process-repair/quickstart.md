# Quickstart: Validating the Exit-Process Study

**Feature**: 002-exit-process-repair | **Date**: 2026-09-02

Interfaces are in [contracts/study-cli.md](./contracts/study-cli.md); entity
rules in [data-model.md](./data-model.md); the resolved mechanism in
[research.md](./research.md).

---

## Prerequisites

```bash
cd /Users/philprice95/projects/freqtrade
```

| Requirement | Check |
|---|---|
| Virtualenv | `.venv/bin/python -c "import pandas, freqtrade"` |
| Baseline archives | The three results named in `regime/arms/applied_W{1,2,3}.json` exist |
| Baseline trade counts | 51 / 96 / 20 for W1 / W2 / W3 |
| Clean tree | `git status --short Framework/` is empty — Phase 4 edits and reverts this file |

> **Read research.md first.** It resolves what fires the stop exits, and it
> records one refuted hypothesis that costs 9.24pp if acted on.

---

## Scenario 1 — The baseline is pinned, not read (FR-028b)

```bash
PYTHONPATH=user_data/strategies .venv/bin/python -c \
  "from regime.exit.baseline import BASELINE; print(BASELINE)"
```

**Expect**: grace 5.6h, take-profit 0.024, ROI 3%, stoploss −0.05, exit signals off.

**Check**: these values are literals in `baseline.py`. `NNNC/NNNC_MLX.json`
currently holds *different* values — it was overwritten mid-feature with a
hyperopt output. If the printed baseline matches the JSON rather than the values
above, the pinning has been lost and every downstream measurement is against the
wrong configuration.

---

## Scenario 2 — Characterisation (Phase 1)

```bash
PYTHONPATH=user_data/strategies .venv/bin/python \
  user_data/strategies/regime/exit/characterise.py
```

**Expect**: `exit_mix.csv`, `stopped_trades.csv`, `counterfactual.csv`, and a
verdict on whether the stop cuts noise or cuts losers.

**Check**:

- `initial_stop_pct` is **−5.00% on every row**. Anything else means the grace
  level became reachable, contradicting research R1 — stop and re-derive.
- Profitable and losing stops appear as **separate populations** in every table.
  They are different phenomena: 50.6% of stopped trades had their stop end up
  *above* entry.
- Trades excluded from an extension for insufficient history are **counted**,
  not silently treated as "did not recover".

---

## Scenario 3 — ⛔ The screen must reproduce the baseline first (FR-014)

```bash
PYTHONPATH=user_data/strategies .venv/bin/python \
  user_data/strategies/regime/exit/replay.py --validate-only
```

**Expect**: replaying the *current* rule reproduces the recorded per-window
totals. Exit 0 on success, exit 2 on failure.

**This gate is not optional.** A screen that cannot reproduce the known answer
will mis-score every candidate, and the error would be invisible — candidates
would simply rank wrongly. Do not score anything until this passes.

---

## Scenario 4 — Screening candidates (Phase 3)

```bash
PYTHONPATH=user_data/strategies .venv/bin/python \
  user_data/strategies/regime/exit/replay.py
```

**Expect**: `screen_results.csv` ranking candidates by change against the
baseline, each carrying its motivating measurement.

**Check**:

- The four candidates research already eliminated appear with their reasons and
  are **not re-tested**: grace *level* (inert), the `return 1.0` correction
  (refuted, −9.24pp), `trailing_*` (inert), the hyperopt exit config (rejected
  on a W3 sign reversal).
- At least one candidate is rejected here without consuming a backtest — that is
  what the screen exists for (SC-006).

---

## Scenario 5 — Confirmation (Phase 4, survivors only)

```bash
for W in W1 W2 W3; do
  PYTHONPATH=user_data/strategies .venv/bin/python \
    user_data/strategies/regime/run_arm.py --strategy NNNC_MLX --window $W \
    --tag <candidate> --set-sell <param>=<value>
done
```

**Check every arm**:

| Check | Why |
|---|---|
| `trades_hash` differs from baseline | Equal ⇒ **INERT**, a defect, not a null. Four of nine exit settings already do nothing. |
| `entry_config_hash` matches baseline | Entry is frozen by feature 001. |
| `exit_mix` reported | Converting stops into equally-bad time-based exits improves the mix without improving the outcome. |
| Sign holds in W3 | The test that rejected both the hyperopt config and `rvol 2.5`. |
| `n_trades` reported | A few-trade arm is inconclusive. |

**If you edit `Framework/BaseStrategy.py`**: revert immediately after the arm
(`git checkout Framework/BaseStrategy.py`). A live dry run imports that file.

---

## Definition of done

| # | Outcome |
|---|---|
| 1 | The stop's behaviour is explained with a stated verdict and figures |
| 2 | Profitable and losing stops analysed separately, and the report says whether one change can serve both |
| 3 | The screen reproduced the baseline before scoring anything |
| 4 | Every candidate cites its motivating measurement; excluded ones carry reasons |
| 5 | At least one candidate rejected at the screen, saving a backtest |
| 6 | No model loaded for training; no new weights produced |
| 7 | An accepted candidate improves risk-adjusted return without reversing sign in W3 — or the report states that nothing does, which is a successful outcome |
