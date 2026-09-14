# Walk-forward harness — TODO / open threads

Living list for the pooled walk-forward harness (`regime/wf/`, feature 007, delivered
2026-09-09). Usage guide: `regime/wf/README.md`. Power write-up:
`regime/WALKFORWARD_POWER.md`. Design record: `specs/007-pooled-walkforward-harness/`.

---

## 1. Promote the harness out of `regime/` — **DONE 2026-09-14**

Moved to top-level **`evaluation/`**. The gate (feature 010 closed) cleared when briefs 01-05
all closed; 011, 012 and 013 had by then leaned on the harness heavily, which is what made the
promotion worth doing rather than merely tidy.

All five steps completed in order:

1. **`run_arm.py --timerange`** — `--window` and `--timerange` are now alternatives with
   `--label` carrying the naming slot. This removed `run_folds.py`'s in-memory `rd.WINDOWS`
   monkeypatch, which was the single largest thing welding `wf/` to `regime/`. `WINDOWS` on
   disk was left alone, so `phase_c_trackability.py` is unaffected.
2. **Statistics lifted, not duplicated** — the paired bootstrap moved to
   `evaluation/bootstrap.py`, and `regime/tail/compare.py` imports it back and re-exports the
   names its own callers use. Verified: `regime.tail.compare.bootstrap_delta is
   evaluation.bootstrap.bootstrap_delta` → `True`. The regime-specific single-window helpers
   (`load`, `verdict`, `report`, `ARMS`, `OOS_WINDOW`) stayed behind, which is where they
   belong.
3. **`wf/` moved** via `git mv` so history follows. `BOOTSTRAP_SEED = 0` and every pinned
   baseline constant carried across unchanged.
4. **Importers updated** — `research/labelfree/folds.py`, `AGENT_GUIDE.md`, the evaluation
   protocol, the brief set, `regime/WALKFORWARD_POWER.md`, `docs/HYPERPARAMETER_STUDY.md`, and
   `specs/008`'s do-not-modify list (which would otherwise have guarded a path that no longer
   exists). **Closed feature specs `007`-`012` were deliberately left naming the old path** —
   they are records of what was built when, and `evaluation/README.md` carries a note so a
   stale path resolves to an explanation.
5. **Reproduced exactly**: `evaluation/pooled_nnnc.json` still gives **15/15 folds, 378 pooled
   trades, floor 3.350 pp/yr**. 165 labelfree tests pass; `regime/`'s own single-window callers
   (`screen_candidates.py`, `pair_restricted.py`) import clean.

One thing the reproducibility check caught that a grep would not: `folds/plan_nnnc.json`
records `params_snapshot` as an absolute repo path and `run_folds.py:143` **reads it back**, so
the move had silently broken re-running that plan until the recorded path was updated too.

## 2. Upstreaming to freqtrade — parked, low priority

freqtrade core backtesting has no walk-forward support at all (zero matches for
walk-forward / anchored-fold / purged-CV / combinatorial in `freqtrade/`). freqai has the
*mechanism* — `data_kitchen.py` splits on `train_period_days` / `backtest_period_days` —
but it is one rolling window, per-pair, with no anchoring, no purge/embargo at the label
horizon, no pooling of trades across pairs, and no PBO/DSR. It also only works for models
behind the freqai interface, which the MLX strategies are not.

So the novel parts here (pooling for power, purge/embargo, CSCV/DSR on the pooled set)
have no upstream equivalent — but pooling assumes a shared decision process across pairs,
which freqai's per-pair model registry does not have.

The repository boundary means this can only ever be an upstream PR against
`freqtrade/freqtrade`, maintained separately from this fork. If it is ever worth doing,
the shippable slice is **purged/embargoed splits for freqai** — small, self-contained,
obviously correct, and it does not require them to adopt pooling.

---

## 3. Known limits — recorded, not defects

- **Pooling underdelivers against 1/sqrt(n).** Fold returns are not independent, so the
  power ceiling is below what trade count implies. `power.py` reports the shortfall and
  the lag-1 correlation. This is a finding (research R9), never grounds to retune the fold
  geometry.
- **1-2pp effects remain unresolvable** on 5.68 yr at ~100 trades/yr, and always will be.
  Even the honest ceiling (every fold the data allows, at double the universe) is ~2.5-2.8pp.
- **SUI/USDT is excluded** from `config/config_wf.json` — a documented deviation from the
  production 11-pair universe, costing 8 of 99 W1 trades and 4.0% of P&L, in exchange for
  15 folds / 3.68 yr instead of 11 folds / 2.87 yr.
