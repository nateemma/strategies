# Phase 0 Research: labelfree-jump-model-gate

**Date**: 2026-09-15 · Literature anchors in `research/label-free-study/01-literature-review.md`

---

## R1 — The sparse jump model objective

**Decision.** Fit by three-block coordinate descent over centroids `μ`, the state sequence `s`, and
feature weights `w`:

```
minimise   Σ_t Σ_j w_j (x_tj − μ_{s_t, j})²  +  λ · Σ_t 1[s_t ≠ s_{t−1}]
subject to w on the simplex with an L1 budget (the "sparse" part)
```

Blocks alternate: centroid update (a weighted k-means mean), state update (dynamic programming,
R2), weight update (sparse k-means, Witten & Tibshirani 2010).

**Rationale.** This is the Bemporad–Breschi–Piga–Boyd (2018) jump model with the Nystrup–Kolm–
Lindström (2020) financial framing and the Shu–Yu–Mulvey (2024) sparse feature weighting the brief
names. The switching penalty λ is the *whole* mechanical difference from the eight instruments
`regime/VERDICT.md` already rejected — it is why the brief exists.

**Alternatives considered.** HMM (`hmmlearn`): no explicit switching cost, ~3× the turnover in the
source study, and Phase C's instruments already cover threshold-on-a-statistic. Changepoint
detection (`ruptures`): offline by construction, so its states are smoothed by definition — it is
the leak, not the method.

---

## R2 — Filtered vs smoothed inference: the exact line this feature is built on

**Decision.** The state update solves the recursion

```
V_t(k) = w-weighted‖x_t − μ_k‖²  +  min_j [ V_{t−1}(j) + λ·1[j ≠ k] ]
```

- **FILTERED (online, the honest path):** `s_t = argmin_k V_t(k)`, forward pass only.
  `V_t` depends on `x_1..x_t` and nothing later. This is the deployed path.
- **SMOOTHED (Viterbi, the leak reference):** run the forward pass to `T`, take
  `argmin_k V_T(k)`, and **back-track**. The traceback is what makes `s_t` depend on bars after
  `t`. That single backward pass *is* the leak being measured.

**Rationale.** Stating it as one line of difference is deliberate. The brief mandates that the
smoothed variant be run first and shown to be materially better — fixed at **≥ 7.633 pp**, the
pinned noise floor `evaluation.bootstrap.BASELINE_HALF_WIDTH_PP` (FR-009). If the two paths are not
separated by exactly this and nothing else, the measured gap is not a leak size, it is a
confound.

**Consequence for fitting.** Fitting itself uses the smoothed DP — that is legitimate, because a
refit at time `τ` sees only data before `τ` (R3). What must never happen is the *deployed* state
series being produced by a traceback.

---

## R3 — Refit schedule, and what "expanding window" means at a fold boundary

**Decision.** Refit every **90 days** on an expanding window. Per clarification Q2=A, **refits
continue through a fold's test range**, each using only bars strictly before its own refit point.
Between refits, `μ` and `w` are frozen and states are produced by the filtered recursion.

**Rationale.** This is what the deployed system does — it keeps re-learning — and each fit remains
causal. Freezing at the fold boundary would leave the gate running on a model up to a year stale
by fold 15, which measures staleness rather than the method.

**Standardisation is part of the fit, not a preprocessing step.** Feature means and scales are
computed on the expanding window **up to the refit point only** and then applied forward.
Recorded failure (brief 07): standardisation statistics fitted over the full series including the
scoring split was a leak that survived two other controls.

**At each refit, states are re-mapped by mean volatility** so label identity does not swap: the
state with the higher fitted realised-volatility centroid is always the "high-vol" state. Without
this, "block the high-vol state" silently inverts at a refit boundary.

---

## R4 — Features

**Decision.** Three causal features per pair, plus the same three computed on BTC as a market-wide
block: trailing realised volatility, trailing downside (semi-) volatility, and return-sign
persistence. All computed on closed bars only, by the same code path the deletion test (R7)
verifies.

**Rationale.** Named by the brief. They are deliberately few — the sparse weighting is there to
decide which matter, and a wide feature battery is the thing this repository has repeatedly found
adds nothing (recorded: an 85-indicator battery added exactly zero to a triple-barrier target).

---

## R5 — Hyperparameters, pre-registered and frozen

**Decision.**

- **K = 2 states** (bull/bear), matching the source study. A `K = 3` run may be reported **only**
  as a labelled sensitivity and **cannot produce the verdict**.
- **λ** is calibrated **once**, on the **first fold's training window only**, by targeting the
  switching rate the source study reports, then **frozen** for every fold and every refit.
- The incumbent's own gate parameters stay frozen at their production values (rvol 2.0, guard 0.3,
  grace 6.5h/−0.27, TP 0.024, ATR mult 2.5).

**Rationale.** Principle VIII — one parameter space at a time, and the sample here is thin. A
per-fold λ search would be a joint search over the jump model and the incumbent's gate, which is
how brief 05 reached PBO 0.7128 over 48,015 evaluations. One pre-registered calibration on one
training window is not a search.

**Alternative rejected.** Selecting λ by fit quality per fold. Beyond the overfitting, it repeats
brief 06's "selection-by-best-calibration" error, where the criterion fell monotonically in the
tuned parameter and drove the choice to a degenerate limit.

---

## R6 — The gate rule

**Decision.** A hard **entry** gate: while the filtered state is the high-volatility state, no new
entries. Open positions and all exits are untouched — the exit stack (ATR-adaptive stop,
stoploss-grace, ROI) is the incumbent's and stays the incumbent's.

**Rationale.** The brief specifies use (a) hard gate. Leaving exits alone keeps the comparison to
one changed thing.

---

## R7 — Causality proven end-to-end, on the artefact the backtest actually reads

**Decision.** The state series is precomputed to a per-(pair, timestamp) artefact, and the gated
strategy consults **that file**. The deletion test truncates the input history at many cut points
and requires the states over the overlapping region to be **bit-identical** to those computed from
full history. The test runs against **the artefact the backtest consumes**, not a separate
in-memory computation.

**Rationale.** Principle V: framework lookahead analysis is not sufficient evidence; lookahead is
measured end to end by comparing truncated-history against full-history output. Pointing the test
at a different object than the backtest reads is precisely how a leak survives a passing test —
brief 07's encoder was trained on 70% of the *test* range while the design said otherwise, and the
split code looked right.

---

## R8 — Why real gated backtests, not a filter over the recorded trades

**Measured 2026-09-15** from `evaluation/pooled_nnnc.json` (378 trades, 15 folds):

| | |
|---|---|
| `max_open_trades` cap | 10 |
| peak concurrency in any fold | **9** |
| folds reaching the cap | **0 of 15** |
| folds reaching ≥8 | 3 of 15 |

**Decision.** Primary evaluation is **real gated backtests**. The cheap filter is used only for
exploration and its agreement with the real runs is reported.

**Rationale.** The cap never binds, so the concurrency channel cannot manufacture a trade. But
freqtrade also refuses a second position on a pair that already holds one — and **that** channel
*is* reachable: blocking a pair's first trade makes a later signal on the same pair takeable, a
trade the ungated run never recorded and a filter can never see.

Feature 013 settles the cost-benefit: its exit surrogate passed review and a **single** real
backtest exposed four independent fidelity bugs. Here the honest path is cheap — a hard entry gate
does not change the model, so **no retraining is required**.

---

## R9 — Reusing fold weights without retraining inside the backtest

**Decision.** Each gated fold class overrides `get_model_path()` to return the **ungated** fold
class's directory.

**Rationale.** `Framework/BaseNNStrategy.py:333` keys the model path off
`self.__class__.__name__`. A new class name means a new `saved_data/` directory, and
`evaluation/run_folds.py` documents the consequence in its own module docstring: a class with no
model of its own **trains inside its own backtest**, fitting the evaluation window silently, with
returncode 0. Overriding `get_model_path` is the established pattern in this repository — ten
`NNNC/*.py` classes already do it.

**Verification.** Before any gated arm is evaluated, assert the gated class resolves to an
existing `*.safetensors` under the ungated fold's directory, and that no new `saved_data/`
directory is created by the gated run.

---

## R10 — Drawdown and its intervals

**Decision.**

- **Equity curve**: additive fixed-stake in percentage points, `pp = profit_abs ×
  scale_pp_per_abs` (the factor already carried on each pooled trade row), ordered by close time
  within a fold.
- **Max drawdown**: peak-to-trough of that cumulative curve. **Time underwater**: the fraction of
  the fold's span spent below the running maximum.
- **Per-fold CI**: stationary block bootstrap over that fold's **trade sequence**, reusing
  `research/labelfree/stress/bootstrap.py::block_index`.

**Rationale for additive, not compounded.** Recorded failure (brief 013): `prod(1+pnl)` over
pooled trades treats concurrent positions as sequential full-wallet bets and read a real
+23.46 pp/yr as −19.06.

**Rationale for a block bootstrap.** Drawdown is a **path** statistic — an iid resample of trades
destroys the ordering the statistic is made of. Brief 06 built and **accepted** exactly this
instrument, measuring that 38.6% of re-orderings lose. The CI it produces will be wide; that is
the honest width, not a defect.

**Adaptation flagged.** `BLOCK_LENGTHS = (96, 144, 192, 240, 288)` in that module is calibrated
for **15-minute bars**. A trade sequence is not a bar sequence, so the block length MUST be
re-chosen in trade units and the choice recorded — inheriting the bar-panel constant would be a
silent unit error.

**Why per-fold.** SC-011 asks for a drawdown fall whose CI excludes zero in ≥10 of 15 folds, so
the CI is required per fold. A pooled CI cannot answer it, and 15 paired fold deltas also give the
sign count Principle II requires.

---

## R11 — The cross-check, and auditing the package's inference mode

**Decision.** `jumpmodels 0.1.1` is installed into `freqtrade/.venv` and run as an independent
cross-check; the in-repo model is primary and carries the verdict. Agreement between the two state
sequences is reported as a number.

**WHAT THE CROSS-CHECK CAN AND CANNOT TEST, settled by measurement on 2026-09-15.** An
end-to-end agreement threshold turned out to be the wrong instrument, and the investigation
that showed why is worth more than the threshold was:

| configuration | agreement |
|---|---|
| independent fits, raw features handed to the package | 0.4863 |
| …after standardising for it (`jump.py` does no scaling of its own) | 0.6243 |
| …against `SparseJumpModel`, `max_feats = kappa²`, `jump_penalty = lam·√d` | 0.8935 |
| our inference, **their weights** | 0.9360 |
| our inference, **their weights and centroids** | **0.9708** |

The last row is the answer: with matched parameters the two agree at 0.97, so **the inference
recursion is right**. The residual gap is not a correctness defect in either implementation —
it is which local optimum the coordinate descent lands in. Both solutions sit at the *identical*
L1 budget (1.725 = kappa) and both zero `sign_persistence`; they differ in that ours loads the
pair's own volatility (0.752 / 0.606) while theirs loads BTC's (0.747 / 0.609) — near mirror
images.

So FR-016 splits: a **hard** gate on matched-parameter agreement, which tests something an
implementation can be wrong about, and an **advisory** report on independent agreement, which
tests a tie-break between equally valid optima. A threshold on the latter is a threshold on a
coin flip.

**The bimodality is itself a finding.** "Regime driven by this pair's own volatility" and
"regime driven by BTC's volatility" are different economic gates — idiosyncratic versus
market-wide. The objective is close to indifferent between them here, so the report must state
which optimum produced the verdict rather than leaving it implicit. It also means the seed
ensemble (FR-022) is doing more work than guarding arithmetic: it samples which of the two
economic stories the fit tells.

**Dry-run verified 2026-09-15**: pure-Python wheel; numpy, pandas, scipy, scikit-learn and
matplotlib all already satisfied; **no new transitive dependencies**.

**The audit that matters.** The package's inference path MUST be read and confirmed to be
**filtered, not smoothed**. Its default is not assumed. A library whose `predict` runs a Viterbi
traceback would silently reintroduce the exact leak this feature exists to measure, and the
cross-check would then "agree" with our smoothed reference while disagreeing with our honest one —
a failure mode that looks like a bug in *our* code.

**Interpreter.** Both implementations run under `freqtrade/.venv/bin/python`, asserted via
`sys.executable`. Recorded failure: `cma` was installed into the pyenv base by mistake, a
*different* Python 3.12.13 carrying no pandas or scipy.

---

## R12 — The redundancy check runs before performance (Principle IX)

**Decision.** Before any return or drawdown number is read, measure the overlap between bars the
jump gate blocks and bars the incumbent's **rvol ≥ 2.0** guard already rejects. Report the
contingency table and the share of jump-gate blocks that are *novel*.

**Rationale.** The incumbent's guard chain rejects ~97% of candidates with rvol dominant. A model
fitted on realised and downside volatility is at real risk of being that same axis in disguise.
Principle IX requires correlation against what is already there to be checked **first**.

**The threshold is pre-registered** (FR-024, SC-019): `novel_block_rate < 0.20` → verdict
`redundant_with_rvol`.

- **Overlap near-total** (`novel_block_rate < 0.20`) → the gate is inert *because rvol got there
  first*. Reported as a defect under Principle IV, explicitly **not** as "regime gating does not
  help".
- **Overlap partial** (`≥ 0.20`) → the blocked set is distinct and the comparison is meaningful.

**Measured context for how likely that is.** Parsing all 32 rows of `regime/VERDICT.md`'s evidence
table (2026-09-15): **18 rows — 56% — had `n_taken == n_marginal`**, the gate admitting every
marginal trade, all at `edge_pct` exactly 0, with a single `beats_bar True` in the whole table.
Over half of Phase C's evidence is gates that never fired. (Every earlier attempt to parse that
table returned 3 rows, because the `rule` column itself contains `|` characters and over-splits
those lines; rebuilding fields from the row ends fixes it.)

---

## R13 — Seeds: the fit is stochastic even though the backtest is not

**Decision.** Run the jump-model fit across **≥5 seeds** per `00-evaluation-protocol.md` §4.
Report the interquartile mean with a stratified-bootstrap interval, never a single seed or the
best one. Require cross-seed state agreement **≥ 0.95** (FR-022, FR-023, SC-017).

**Rationale.** `fit()`'s coordinate descent starts from a k-means initialisation, which is
seed-dependent. Everything downstream — which bars are blocked, the block rate, the drawdown
delta — inherits that randomness, even though the backtests themselves are perfectly deterministic
because the gated arms reuse the ungated weights and train nothing. Determinism in the expensive
half of the pipeline is easy to mistake for determinism in the whole of it.

The failure this guards against is recorded here in a sharper form: multi-task head weights moved
entries **less than a random seed did**, which is what closed that line. A regime gate is
precisely the kind of object that can look decisive while being initialisation noise, and a
single-seed run cannot tell the two apart.

**Consequence for the kill gate.** SC-017 makes the comparison explicit: if the per-fold block
rate varies more across seeds than it does between the gated and ungated arms, the gate is an
initialisation artifact and the result is a KILL — not a finding about regimes.

---

## Open risks carried into Phase 1

1. **The gate is inert.** Most likely single outcome, via R12. Pre-registered as a KILL that is
   reported as a gate failure, not a regime-gating failure.
2. **The smoothed variant does not clear 7.633 pp.** Pre-registered as a stop: it would mean
   the online path is itself leaking (FR-009 / SC-014).
3. **The two implementations disagree.** Pre-registered as a stop (FR-016).
4. **Return falls while drawdown falls.** Not success — a trade-off, reported as one (SC-013).
