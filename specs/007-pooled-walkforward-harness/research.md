# Phase 0 Research: Pooled Walk-Forward Evaluation Harness

All items below are RESOLVED. Tasks must act on them, not re-derive them.

## R1 — The noise floor is measured, not assumed. DONE.

Bootstrap on NNNC_MLX W1's own per-trade P&L (99 trades, +9.831%, archive
`backtest-result-2026-09-08_19-55-22.zip`): 95% CI **[2.383%, 17.649%]**, half-width **7.633pp**,
sampling SD 3.893pp. Paired over an arm's symmetric difference the half-width is **~6.1pp**.
Method validated: reconstructed 983.1/10000 = 9.831% matches `run_arm`'s reported figure exactly.
This is the baseline every power claim is measured against. Do not recompute it; reuse it.

## R2 — The achievable ceiling is ~2.7x, not 10x. DONE.

Data on disk: **2021-01-01 .. 2026-09-07, 2,075 days (5.68 yr)**. At current rates (NNNC 99/yr,
NNMT 183/yr, 11 pairs):

| train | test span | NNNC trades 11 / 22 pairs | floor 11 / 22 pairs |
|---|---|---|---|
| 24 mo | 3.68 yr | ~365 / ~730 | 3.98 / 2.81 pp |
| 18 mo | 4.19 yr | ~414 / ~829 | 3.73 / 2.64 pp |
| 12 mo | 4.68 yr | ~464 / ~928 | 3.53 / 2.49 pp |

Consequence, and it must be stated in the final report: **1-2pp parameter effects remain
unresolvable after this feature ships.** Everything that returned "not resolved" this session would
still return "not resolved". The feature's value is making the **3-7pp band** measurable — the band
containing the changes that have actually mattered (exit-defect repair +4.5pp / +12.0pp).

## R3 — Anchored training, 24-month minimum, 90-day test folds. DONE (decision).

Production trains on all data to a cutoff (~44 months), so an **anchored expanding** window is the
faithful analogue; a short fixed rolling window changes the training regime and confounds the
instrument with what it measures. ≥24 months before the first test places fold 1's test at
2023-01-01 → 1,345 test days → ~15 folds at 90 days.

Alternatives rejected: rolling 12mo (best floor, largest quality risk); rolling 18mo (middle on
both, still a departure from production).

## R4 — Fold identity must be a real class, generated. DONE.

`BaseNNStrategy.get_model_path` builds the path from `self.__class__.__name__`. Two consequences,
and together they are the most dangerous failure mode in this feature:

- Folds sharing a class **overwrite each other's weights**.
- A subclass with no model of its own **trains inside its own backtest**, fitting the evaluation
  window, silently, with returncode 0.

So each fold gets a generated class `NNNC_WF_f{NN}` with its own `saved_data/` directory. Generated,
not hand-written, because each also needs its own `.json`: file params beat class params
(`deep_merge_dicts(file_params, class_params)`, file wins — freqtrade/strategy/hyper.py:88-97), so a
fold class subclassing `NNNC_MLX` **without** a `.json` silently falls back to `NNNCStrategy`'s
class params, which have **every entry guard open**. The `.json`'s `strategy_name` must equal the
class name or freqtrade aborts with `Invalid parameter file provided`.

Note for a future NNMT phase: `NNMT_MLX` has **no** `.json`, so its fold classes must also have
none — that absence *is* the matching config.

## R5 — Verify the model artefact, never the exit code. DONE.

`scripts/test_strat.sh` returns 0 even when training crashed. The only reliable check is
`ls saved_data/<CLS>/*.safetensors`. FR-004 exists for this; a fold whose artefact is missing is
INCOMPLETE, and its absence must abort pooling for that fold rather than silently yielding a fold
with no trades.

## R6 — Reuse run_arm.py per fold via --timerange. DONE.

`run_arm.py` accepts `--timerange` in addition to `--window`, so a fold's arbitrary test range is
expressible without touching `regimedata.py`'s WINDOWS or `run_arm.py`'s schema (FR-018). Each fold
run produces a normal arm record carrying `result_file`, which is what makes pooling possible at all.

## R7 — Pool from archives; there is no other per-trade source. DONE.

freqtrade silently ignores `--export-filename`, so `user_data/backtest_results/*.zip` is the only
per-trade source. `compare.py._trades()` already implements archive extraction keyed off a record's
`result_file`; `pool.py` must reuse that accessor rather than re-implementing it (FR-013).

## R8 — Feed compare.py; do not fork the statistic. DONE.

`compare.py` (31183cb) pairs on `(pair, open_date)`, excludes shared-and-identical trades (zero
variance), signs unique trades, uses 20,000 resamples with `BOOTSTRAP_SEED=0`, and guards
`MIN_DIFF_TRADES=20` against a degenerate CI. Pooled comparison must route through it. A second,
divergent bootstrap in `power.py` would produce two answers to the same question — `power.py`
computes the *single-set floor*, `compare.py` computes *arm-vs-control deltas*; they must not
overlap.

## R9 — The sqrt(n) assumption is a hypothesis to TEST, not to rely on. DONE.

All floor projections in R2 assume independent folds. Fold returns may be correlated through a
shared market regime, in which case pooling reduces the floor more slowly. FR-009/SC-007 require
reporting the shortfall and the inter-fold return correlation. **A floor that fails to fall as
predicted is the feature's most valuable possible finding** — it would mean the power ceiling is
lower than the data volume suggests, and must be reported, never hidden.

## R10 — Pair expansion is an open question, not an assumption. DONE.

A prior study concluded the gbb edge is structurally the high-volatility alt and that expanding the
universe has no lever for **edge**. This feature wants **samples**, which is a different question.
It must be tested, not assumed: added pairs must be reported with their own trade count and
per-trade expectancy (FR-014), and an expansion whose added pairs materially underperform the
incumbents is reported as an invalid source of power (SC-006). Deferred to phase E — the floor must
be demonstrated at the current universe first.

## R11 — Seeds are a later phase, but FR-011 still applies. DONE.

Two seeds over 15 folds is 30 trainings (6.3 h) and is not justified before the floor is shown.
Phase D runs seed 7 on a **4-fold subset**. The entry gate is often tight enough that two seeds
produce identical trades (observed on W1 this session) — that is ONE measurement, not two, and
FR-011 requires detecting and reporting it wherever seeds are run.

## R12 — Sequential execution. DONE.

Two concurrent jobs put load 60 on 14 cores and ran **slower** than serialising. All fold work is
strictly sequential. To kill a run use `pkill -9 -f 'freqtrade backtesting'` — never bare
`freqtrade`, which matches a live `freqtrade trade` dry run.
