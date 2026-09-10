# Phase 0 Research: Label-free Direct Policy Optimisation

**Feature**: 009-labelfree-direct-policy · **Date**: 2026-09-10
**Sources**: `research/label-free-study/01-literature-review.md` (§A direct policy optimisation),
`00-evaluation-protocol.md`, `briefs/01-direct-policy-optimisation.md`, `AGENT_GUIDE.md`,
`.specify/memory/constitution.md`, and measurements taken in this session (recorded below).

Every decision below either resolves a Technical Context unknown or records a measurement that
constrains the design. Measurements are marked **[measured 2026-09-10]**.

---

## R1 — Compute substrate

**Decision**: MLX, GPU device, single machine, sequential runs.

**Rationale**: Protocol §1 says "MLX first; a PyTorch mirror is acceptable if MLX blocks
something". The repo's production NN path is MLX and the models here are tiny. `mlx.nn.LSTM`,
`GRU`, `RNN`, `Linear` and `Sigmoid` all exist **[measured 2026-09-10]**, so nothing in the
planned architecture requires a mirror.

**Alternatives considered**: PyTorch mirror (deferred — only if a specific MLX gap appears;
would be recorded as a deviation, not adopted silently). TensorFlow (rejected: the repo's Keras
path is the one prior work found overfits, and the MLX path is production).

---

## R2 — Half-spread estimator (resolves the Q4 clarification into one estimator)

**Decision**: Abdi–Ranaldo (2017) close-high-low estimator as primary, floored at 1 basis point
per side; Corwin–Schultz (2012) computed alongside as a sensitivity check only.

**Rationale**: The data is OHLCV only — columns are exactly
`['date','open','high','low','close','volume']` **[measured 2026-09-10]** — so the spread must be
estimated. Corwin–Schultz produces negative estimates frequently enough that truncation becomes a
material part of the estimator's behaviour; Abdi–Ranaldo is the later, more stable construction on
the same inputs. Running CS alongside costs nothing and gives a cheap robustness read on whether
any result depends on the spread model. The floor exists because FR-004a forbids the estimate
collapsing to zero on quiet bars, which is exactly where phantom fills previously appeared.

**Alternatives considered**: Flat constant (rejected by FR-004a — under-costs illiquid pairs by
the widest margin, precisely where this repo twice found phantom edges). Live order-book
collection (rejected: adds a data-acquisition dependency outside this feature's scope, and the
historical windows cannot be back-filled that way).

---

## R3 — Differentiable liquidity cap

**Decision**: Cap executed exposure change at 10% of the bar's quote volume. In training use a
soft-min (`-softplus(-x/τ)·τ` form, τ a fixed small constant); in evaluation apply the hard
`min`. Unfilled remainder carries to the next bar in both.

**Rationale**: Protocol §1 requires "soft-min in training so the gradient does not vanish where
the cap binds; the hard cap applies in evaluation". A hard `min` zeroes the gradient on the capped
branch, so the policy would never learn that requesting more than the book can fill is futile —
which is the exact failure the cap exists to prevent (`us_spot_market_study.md` wall 2). τ must be
logged with each trial because it changes the training signal.

**Alternatives considered**: Hard cap in training (rejected: dead gradient). Penalty term instead
of a cap (rejected: lets the policy buy the penalty and keep the phantom fill).

---

## R4 — Policy backbone

**Decision**: Single-layer `mlx.nn.LSTM` with hidden size 64, `seq_len = 16` matching
`BaseNNStrategy.seq_len` **[measured 2026-09-10]**. The existing MLX Mamba block
(`MLX/mamba_mlx.py`) is a registered variant only if the LSTM clears the sanity ceiling.

**Rationale**: The brief is explicit — "start with the smallest thing that can work … Do **not**
start with the transformer". Matching `seq_len` keeps the policy's input identical to what the
production classifier sees, which is what makes the comparison a test of the *objective* rather
than of the architecture.

**Alternatives considered**: Mamba first (deferred, not rejected — it is the brief's other named
option but is larger). Transformer (explicitly excluded by the brief).

---

## R5 — Objectives

**Decision**: Implement both, as separate logged trial families.
1. **Utility**: `−(mean(r_net) − λ/2 · var(r_net))` over pooled minibatches (Kubo & Nakagawa
   2025 — unbiased under minibatching, which naive Sharpe is not).
2. **Robust Sharpe**: pooled Sharpe plus a SoftMin worst-window Sharpe over 90-day windows
   (DeePM).

**Rationale**: Literature review §A: the *naive* direct-P&L version overfits (val 1.61 → test
0.48 Sharpe) and only the structured version holds. Running both is how this feature distinguishes
"the objective works" from "one particular objective works". Minibatch-unbiasedness matters
because the training loop is minibatched by construction.

**Alternatives considered**: Plain mean return (rejected: no risk term, and the literature's
failure mode). Sortino (deferred: the repo already measured Sortino and Sharpe selecting the same
configuration, so it adds a trial without adding information).

---

## R6 — Turnover control

**Decision**: Both variants, as the brief requires. (A) learned no-trade band — head emits
`lo_t`/`hi_t`, exposure held via `clamp(w_{t-1}, lo_t, hi_t)` (Imaki 2021). (B) plain sigmoid
exposure plus explicit `λ·|Δw|` penalty.

**Rationale**: These encode the same economic idea (do not trade small edges) with different
gradients: the band makes inaction the default and is differentiable through the clamp's active
branch; the penalty prices action continuously. The repo's own operating-point finding — a
threshold sweep moved W1 from −12.7% to +6.9% without retraining — is what makes the band
plausible, and the brief predicts the learned policy will be flat most of the time.

**Alternatives considered**: Threshold applied post-hoc (rejected: that *is* the current
classifier's operating point, i.e. the fourth baseline, not a new method).

---

## R7 — Sequence handling

**Decision**: Truncated BPTT with a 96-bar (one-day) burn-in preceding each loss window; the
burn-in contributes state but no gradient.

**Rationale**: Brief design section. At 15m bars, 96 bars is one day — enough to warm recurrent
state so the first scored bar is not decided by a zero hidden state.

---

## R8 — Trial accounting, PBO and DSR

**Decision**: Append-only JSONL at `research/labelfree/trials.jsonl`, one object per configuration,
written **before** the run starts and updated in place with the result afterwards (the pre-write
carries `result: null`). Deflated Sharpe Ratio per Bailey & López de Prado (2014); PBO via CSCV
(Bailey et al. 2015). PBO > 0.2 fails the feature regardless of point estimate.

**Rationale**: Protocol §4. Writing before the result is what makes the trial count honest — a log
written afterwards silently omits abandoned runs, which is the mechanism the deflation is meant to
correct for. Append-only JSONL survives interrupted runs, which matters because the protocol
requires sequential execution over long jobs.

**Alternatives considered**: SQLite (rejected: heavier, and the audit requirement is "every
configuration appears", which a text log satisfies and makes reviewable in a diff).

---

## R9 — Reproducibility / determinism

**Decision**: Seed via `MLX_SEED` (AGENT_GUIDE: `train_seed` is inert). Verify byte-identical
trades across two *separate processes* as an early task, before the variant sweep.

**Rationale**: Protocol §4 requires byte-identical reproduction. A seeded MLX computation on GPU
reproduced to the last printed digit across two separate processes **[measured 2026-09-10]**, and
AGENT_GUIDE documents seeded GAN sampling as byte-reproducible across processes — so the
requirement is plausible. But a matmul is a much weaker test than a full training loop with
reductions and optimiser state. If a real training run proves non-deterministic, that is a finding
to escalate under SC-008, **not** a reason to weaken the gate to a tolerance.

**Alternatives considered**: Accept a numerical tolerance (rejected — protocol says byte-identical;
weakening a gate is forbidden by the study's "verbatim or stricter" rule).

---

## R10 — The W3 universe problem (material risk, newly measured)

**Measurement [measured 2026-09-10, corrected during implementation]**: the walk-forward universe
is 10 pairs (`config/config_wf.json`). Coverage is measured as **days of actual data inside the
training range**, not by start date — a start-date test is wrong here and dangerously so:

| W3 training coverage (2021-01-01 .. 2022-12-31) | Pairs |
|---|---|
| Full (~729 days) | SOL, LTC, BCH |
| Partial but usable (345-533 days) | AAVE 533, AVAX 407, DOT 393, **ZEC 352**, LINK 351, NEAR 345 |
| Effectively absent (12 days) | **XRP** |

**W3 trains on 9 of 10 pairs; only XRP is excluded.** XRP begins 2021-01-01 but carries a
**911-day gap** (the Binance.US delisting, Dec 2020 -> Jul 2023), leaving 12 days inside the range.
A `first_bar <= train_start` test reports XRP as *covered* and would train on almost nothing while
claiming a full universe — the exact trap this rule exists to avoid. The universe rule is therefore
">= 180 days of actual data inside the training range" (`research/labelfree/folds.py`).

**Decision**: proceed per the Q5 clarification (730-day minimum retained, universe size reported per
window, control run on the identical universe within each window).

**Residual risk, now much smaller than first assessed**: six of the nine W3 pairs train on partial
history (345-533 days rather than 729), so the distant window's policies see less data than W1's.
**ZEC is present**, so the earlier concern that the distant window could not see the pair prior
studies found carries the edge does not apply. What remains is a data-quantity asymmetry to state in
the result report, not a missing-pair confound.

**Alternatives considered**: Shorter W3 training to keep the universe (rejected in Q5 — swaps one
confound for another). Restricting all windows to the 4 deep-history pairs (rejected in Q5 —
breaks comparability with the control's existing pooled record).

---

## R11 — Calibration target and the accounting layer

**Decision**: Calibration replays the arbiter's own realised entry/exit timestamps (Q1 = B) and
compares per-trade profit within 1 bp, total within 0.10pp, trade counts equal. The simulator
reproduces: taker fee per side, fill at the following bar's open, the 10%-of-volume cap, stake
sizing, `max_open_trades`, and `tradable_balance_ratio`.

**Portfolio parameters [measured 2026-09-10]** from `config/config_wf.json`:
`max_open_trades = 10`, `stake_amount = 900`, `tradable_balance_ratio = 0.99`,
`dry_run_wallet = 10000`, `timeframe = 15m`, whitelist = 10 pairs.

**Rationale**: With `max_open_trades = 10` and a 10-pair whitelist the position cap is rarely the
binding constraint, so a calibration mismatch will almost certainly be fee application, fill
timing, or stake sizing — which is what the per-trade comparison localises. Diagnosing before
tuning (Principle VI) means the calibration must report *which* of those three disagrees, not just
that the totals differ.

---

## R12 — Where the code lives

**Decision**: `research/labelfree/` for the simulator, policy, training loop and trial log (new
directory, created by this feature). The deployable wrapper goes in a new `NNDP/` strategy
directory following the repo's family convention. Nothing is added to `Framework/`.

**Rationale**: AGENT_GUIDE's directory map puts strategy families at the top level and reserves
`Framework/` for strategy-agnostic base classes; the constitution makes that a hard rule
("Framework must stay strategy-agnostic"). The policy is research code consumed by one family, so
it does not belong in `Framework/`. `Predictors/` holds the classifier base hierarchy and is the
intended home for *predictor* implementations, but this feature's policy is not a predictor in
that hierarchy — it emits exposure, not class probabilities — so it stays in `research/labelfree/`
until the family is proven.

**Alternatives considered**: Putting the simulator in `Framework/` (rejected: constitution).
Putting it in `utils/` (rejected: `utils/Classifier*` is documented as legacy).
