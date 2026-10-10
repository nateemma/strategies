# Phase 0 Research: Per-Fold Guard Tuning

Every NEEDS CLARIFICATION from Technical Context is resolved below. Decisions are stated with
the repository evidence that settled them.

---

## R1 — Where the code lives

**Decision**: `evaluation/pick_guards.py` and `evaluation/guard_trackability.py`, plus one
optional `guards` plan key in `evaluation/run_folds.py`. Nothing in `Framework/`.

**Rationale**: `evaluation/` already owns the fold plan, pooling, power **and per-fold
parameter selection**. `pick_threshold.py` is the exact precedent: it selects a per-fold
`prediction_threshold` on the fold's own training window and routes the choice into the run
through a plan key. `pick_guards.py` is its direct sibling and reuses its leakage-safety type.
`regime/` owns running one arm, which `run_arm.py` already does without modification.

**Alternatives considered**: `Framework/` — rejected outright, it is runtime code a live
dry-run inherits and this is offline research. `regime/` — rejected because the artefacts
produced (a fold-keyed selection map, a pooled comparison) are `evaluation/`'s data model, and
splitting selection across two packages would duplicate the fold-plan reader. A new top-level
package — rejected; it would need its own copy of the fold plan, pooling and bootstrap.

## R2 — How the grid is scored, and why it is NOT 960 backtests

**Decision**: score the 64 configurations on each fold's **training window** using the existing
per-bar **outcome table**, with the concurrency model from feature 021 — not a backtest.
960 scorings run in minutes. **Arm C itself remains a real harness backtest.**

**Rationale**: scoring a guard configuration needs "what would a trade opened at each admitted
bar have returned under the fixed exit", which is precisely what `research/outcomes/table.py`
stores, keyed by `exit_config_hash` so a changed exit invalidates it. A guard configuration is a
per-bar boolean mask over the candidate entries, so the pipeline is: mask → concurrency
selection (`leaderboard.py::_select_concurrent`, `max_open` 10) → pooled mean. 960 real
backtests would cost ~30 hours and make the gate more expensive than the arm it gates, which
defeats Principle VII's "kill or confirm in an afternoon".

**THE PROXY MUST BE AUDITED BEFORE IT IS TRUSTED — this is a task, not a caveat.** Arms A and B
are already measured on the harness (+2.64pp and +18.76pp). Score both with the proxy and check
it ranks and approximates them. Feature 021's own lesson is that this instrument reproduced
live mean `profit_ratio` to 3.6e-5 on one population and had the **wrong sign** on another, so
its fidelity is a measured property and not an assumption. **If the proxy cannot separate A from
B, it cannot be trusted to rank 64 configurations, and the gate must fall back to a reduced
grid scored by real backtests.**

**Alternatives considered**: a freqtrade hyperopt per fold — rejected; it is a search where an
enumerated grid is wanted (FR-005 needs a finite candidate set), it adds the documented
"silently inert parameter" hazards, and `-j>1` needs the pickle-by-value guard. Scoring by
entry *rate* like `pick_threshold.py` — rejected; a rate is not a return, and which guard
setting *pays* is the whole question.

## R3 — Proving each grid axis actually bit (FR-013), with a negative control

**Decision**: for one representative fold, run the harness at configurations varying **one axis
at a time** and compare `regime/run_arm.py`'s per-fold `trades_hash`. Three axes × two
additional levels ≈ 6 runs. Then run **one more pair varying only `entry_guard_threshold`** and
assert the hash is **UNCHANGED**.

**Rationale**: `trades_hash` is already recorded per fold, so no new instrumentation is needed.
The negative control is the load-bearing part: `GUARD_ABLATION.md` records `guard_metric` as
**completely inert** — "byte-identical trades and returns on all three windows, zero extra
trades… the eighth recorded parameter that is accepted, applied and does nothing". So it is a
parameter *known* to do nothing, which makes it a free test that the inertness check can
actually *detect* inertness. A check that cannot fail is worthless, and feature 021 established
that pattern by planting a duplicate to prove its duplication guard fired.

**Alternatives considered**: reading the console or the `.fthypt` file — the spec forbids the
console, and there is no hyperopt here to produce an `.fthypt`. Comparing returns instead of
hashes — rejected; two different trade sets can coincidentally return the same figure, whereas
a hash collision is not a plausible failure mode.

## R4 — The trackability gate: regret, and trailing-best versus a constant

**Decision**: with `S_i(c)` the proxy score of configuration `c` on fold `i`'s **test** window:

```
adaptive choice      c_i   = argmax_c  (score of c on fold i's TRAIN window)
best constant        c*    = argmax_c  sum_i S_i(c)          <- chosen in HINDSIGHT
regret_i(adaptive)         = max_c S_i(c)  -  S_i(c_i)
regret_i(constant)         = max_c S_i(c)  -  S_i(c*)
```

**The gate passes only if mean `regret(adaptive)` < mean `regret(constant)`** with a paired
interval excluding zero, over all 15 folds.

**Rationale**: regret against `max_c S_i(c)` measures the choice against **the best of the 64
candidates that actually existed**, which is what FR-005 demands — this repository has
previously measured a trackability baseline against a third value that was never an option, and
recorded that as an error. The constant is chosen **in hindsight**, which is deliberately
generous to it: if adaptivity cannot beat a single fixed value selected with full knowledge of
the outcome, it is definitively not trackable. That is the same bar that rejected the dynamic
lookback ("trailing-best loses to a constant at every horizon").

**Prior expectation, recorded before running**: `GUARD_ABLATION.md` shows **every** live
guard's effect flipping sign across the three eras (`-adx` −5.95/−0.49/+4.64; `-atr_pct`
−7.33/−1.30/+0.18; all "consistent? NO"), and all-six-off swings from −24.91pp to **+15.92pp**.
Four adaptive schemes have already lost to static values here. **The gate is expected to fail**,
and FR-006 makes that a complete finding.

**Alternatives considered**: a constant chosen on folds `1..i-1` — rejected as a weaker bar
than hindsight and harder to interpret at `i=1`. Accuracy of the chosen configuration rather
than regret — rejected; choosing among 64 options is a decision, and accuracy against a
single "right" label misrepresents it.

## R5 — Paired statistics (FR-015) with no new implementation

**Decision**: reuse `evaluation/bootstrap.py::bootstrap_delta` unchanged, passing pooled trade
lists.

**Rationale**: its docstring already describes the exact case: trades identical in both arms
"contribute exactly zero and are excluded", while "trades unique to one side enter with their
own sign (the membership/symmetric-difference case, **correct when the entry set itself
moves**)". Arm C vs arm A is precisely a moving entry set. Writing a second paired bootstrap
would risk the two drifting, which is the duplication failure feature 021 spent its foundational
phase removing.

**Alternatives considered**: `research/outcomes/bootstrap.py`'s `paired_block_bootstrap_delta`
— that operates on per-bar series for the outcome table, not on arm records with trade lists;
wrong level. A fresh implementation — rejected on duplication grounds.

## R6 — How a per-fold guard choice reaches the fold run

**Decision**: add one optional `guards` plan key to `evaluation/run_folds.py`, keyed by
`fold_id`, exactly mirroring the existing `thresholds` key (`run_folds.py:194` reads it,
`:237` passes it to `generate_fold`). The chosen values are written into the fold's **generated**
`.json`, still derived from the frozen params snapshot. A new `SimpleStrategies/GbbGuardGridOpen.py`
is a thin `TriggerOnly` subclass that reads them.

**Rationale**: the mechanism exists and is already leakage-audited for `prediction_threshold`;
reusing its shape means the guard choice travels the same proven path rather than a parallel
one. `pick_threshold.py`'s own docstring records why this matters: the value is written into the
generated file, "never applied by this script directly — this script only measures", and never
into the live working params file.

**Alternatives considered**: `run_arm.py --set key=value` overrides — workable for a handful of
folds but the override would live in the shell driver rather than in a reviewable plan artefact,
and FR-019's pre-registration is better served by a committed map. Editing the strategy class
per fold — rejected; 15 near-identical classes is the duplication this repository keeps paying
for.

---

## Open risk carried into tasks

**The proxy may not be good enough** (R2). If the A-vs-B audit fails, the gate cannot be run at
64 configurations in one session, and the fallback is a coarser grid (e.g. 3×3×3 = 27) scored by
real backtests on training windows — 405 runs, which is still too many. The honest fallback is
then to reduce to **two** axes (`atr_pct` × `adx`, 4×4 = 16) and accept that `rvol` is held at
its open value. This is recorded now so the decision is not improvised under time pressure.
