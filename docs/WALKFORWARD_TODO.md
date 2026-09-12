# Walk-forward harness — TODO / open threads

Living list for the pooled walk-forward harness (`regime/wf/`, feature 007, delivered
2026-09-09). Usage guide: `regime/wf/README.md`. Power write-up:
`regime/WALKFORWARD_POWER.md`. Design record: `specs/007-pooled-walkforward-harness/`.

---

## 1. Promote the harness out of `regime/` — BLOCKED until the label-free study closes

**Why.** `regime/wf/` is a general-purpose measurement instrument parked inside a finished
study's scratch space. `regime/` holds ~35 one-off analysis scripts and ~30 markdown
write-ups from the regime-switching study (`breakout_dist.py`, `GUARD_ABLATION.md`,
`WHY_ORACLES_UNDERPERFORM.md`, …). The harness is now the standard way every arm gets
judged — it should not live there.

**Not `Framework/`.** `Framework/` stays strategy-agnostic *runtime* code that strategies
inherit from. This is offline evaluation tooling and would be the only non-runtime thing
in it. Target is a sibling top-level directory — `evaluation/` (or top-level `wf/`).

**Gate: do not start until feature 010 (`labelfree-warmstart-finetune`) is closed.**
`research/labelfree/folds.py` imports `regime.wf.foldplan`, and `research/labelfree/`
paths are recorded in `trials.jsonl`. A mid-study path change invalidates the trials log
and breaks reproducibility of runs already recorded.

**The work, in order:**

1. **Prerequisite — give `run_arm.py` a real `--timerange`.** Worth doing on its own
   merits. `run_arm.py` currently requires `--window`, with choices from
   `regime.regimedata.WINDOWS`. `run_folds.py` therefore monkeypatches that dict in an
   in-memory per-fold subprocess. That hack is the single largest thing welding `wf/` to
   `regime/`. Do NOT solve it by adding fold keys to `WINDOWS` on disk —
   `phase_c_trackability.py` iterates it.
2. **Lift the statistics.** `compare_pooled.py` and `power.py` import
   `regime.tail.compare` (`bootstrap_delta`, `_trades`, `_scale`, `MIN_DIFF_TRADES`).
   This reuse is *deliberate* — one statistic for both single-window and pooled arms, not
   two implementations — so it must move with the harness, not be duplicated. Single-window
   callers in `regime/` then import it from the new home.
3. **Move `wf/` itself**, keeping `BOOTSTRAP_SEED = 0` and the pinned baseline constants in
   `__init__.py` byte-identical. They were MEASURED on 2026-09-09 and are pinned, not
   recomputed; changing them silently rebases every past verdict.
4. **Update importers** — `research/labelfree/folds.py`, `AGENT_GUIDE.md:152-180`,
   `research/label-free-study/00-evaluation-protocol.md`, the brief set, and
   `specs/008-single-pair-variant/contracts/analysis-cli.md` (which names `regime/wf/` in a
   do-not-modify list).
5. **Re-run one known fold plan end-to-end** and confirm `pooled_nnnc.json` reproduces 15
   folds / 378 trades / floor 3.350 pp/yr exactly. The move is only complete when the
   control record is bit-reproducible.

`--family` / `--base-strategy` are already parameters with NNNC as a *default*, not a
hardcode, so nothing else is family-specific.

---

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
