# Brief 01 — Direct policy optimisation: train on net P&L, no labels

**Family:** differentiable trading (Moody–Saffell 2001 → Deep Momentum Networks 2019 → DeePM 2026)
**Type:** unsupervised (raw normalised features → exposure → realised net return)
**Priority:** 1 · **Budget:** 3–4 Claude Code sessions · **Depends on:** `00-evaluation-protocol.md`

## Hypothesis

A small sequence model that outputs a long-only exposure per pair, trained by gradient descent
on the **net-of-cost realised return** of that exposure (with a turnover control), will convert
more of the available ~ρ 0.15 signal into P&L than the current classifier, because it:

1. has no learnability/tradeability conflict (`regime/LABEL_CEILING_VS_LEARNABILITY.md`) — the
   objective *is* the P&L;
2. learns the operating point (when to act, how much, when to stay flat) that the threshold
   sweep in `regime/CLOSING_THE_LEARNABILITY_GAP.md` §2 found by hand;
3. sees fees, spread and the fill cap during training, so it cannot learn trades that only
   pay under phantom fills (`us_spot_market_study.md` wall 2).

**Expected effect if true:** ≥ +3.35pp/yr over `NNNC_MLX` pooled walk-forward. The oracle
ceiling at label 0.02 was +26.5% vs the model's +6.9% at the 0.80 operating point (W1, guards
off), so there is headroom; the ρ≈0.15 ceiling says most of it will not be captured.

## Prior evidence (see `01-literature-review.md` §A)

Works on daily liquid futures with costs in the loss (Lim 2019; Wood 2021; DeePM 2026);
independent re-implementation shows the *naive* version overfits (val 1.61 → test 0.48 Sharpe)
and only the structured version holds. Crypto evidence is confounded (funding income, zero
slippage). **Untested at 15-min long-only spot with fill caps** — that is what this brief
establishes.

## Design

### Simulator (deliverable 1, blocks everything)
`research/labelfree/sim.py` per protocol §1. Calibration test: feed the frozen `NNNC_MLX`
signal (1 when the classifier says buy, held until its own exit rule fires) and match the
freqtrade backtest P&L for W1 to within fee rounding. Until this passes nothing else is
evidence.

### Policy
- Input: the same normalised feature sequence NNNC uses (`seq_len` window from
  `FeatureNormalizer`), pooled across the whitelist (`aggregate_pairs = True` semantics).
- Backbone: start with the smallest thing that can work — a single-layer LSTM (64) or the
  existing MLX Mamba block (`MLX/mamba_mlx.py`). Do **not** start with the transformer.
- Head: `w_t = sigmoid(·) ∈ [0,1]` exposure per pair, then a **no-trade band**
  `w_t = clamp(w_{t-1}, lo_t, hi_t)` with `lo/hi` also emitted by the head (Imaki 2021).
  Variant B: plain sigmoid + explicit `λ·|Δw|` turnover penalty. Run both.
- Portfolio layer: `max_open_trades` and stake sizing from config; positions above the cap
  are pro-rated by exposure.

### Loss (run both, log as separate trials)
- **Utility:** `−(mean(r_net) − λ/2·var(r_net))` over pooled batches (Kubo & Nakagawa 2025 —
  unbiased under minibatching).
- **Robust Sharpe:** pooled Sharpe + SoftMin worst-window Sharpe (DeePM), windows = 90-day
  folds.
- `r_net` = position × next-bar log return − γ·(fee + half-spread + slippage(vol))·|Δw_exec|,
  with `Δw_exec` liquidity-capped (soft-min in training). γ ∈ {0.5, 1.0, 1.5}; evaluate at 1.0
  and 1.5.
- Truncated BPTT with a 96-bar (1-day) burn-in before the loss window.

### Training
- Anchored folds exactly as `regime/wf/foldplan.py`; one policy per fold, trained on the
  fold's training range only.
- ≥5 seeds per config; report IQM. Early stopping on the *training* objective only (no
  validation split from the test fold).
- Apple Silicon: MLX. Model is tiny; expect minutes per fold.

### Freqtrade wrapper (deliverable 2)
`NNDP/NNDP_MLX.py` — a `BaseNNStrategy` subclass whose `get_predictions` loads the policy and
returns exposure; `populate_entry_trend` enters when `w_t` crosses the band's upper edge,
`custom_exit` exits when it crosses the lower edge. Keep the ATR stop and grace logic
**unchanged** so the comparison with `NNNC_MLX` isolates the entry/exit signal. Pin
`get_model_path` per AGENT_GUIDE's silent-failure note.

## Experiments (in order; each is a logged trial family)

1. **Calibration** — simulator ≡ freqtrade on the frozen NNNC signal.
2. **Sanity ceiling** — train the policy with lookahead deliberately enabled (features shifted
   by +1). It must produce absurd returns; if it does not, the gradient path is broken.
3. **Utility vs robust-Sharpe × band vs penalty × γ** — 12 configs × 5 seeds, single fold
   first (fold f07, the richest), then full 15-fold pooled run for the best two.
4. **Cost stress** on the best config (protocol §4).
5. **Ablation:** same policy trained with MSE on forward return (the supervised twin) — to show
   whether the objective or the architecture is doing the work.

## Acceptance / kill

Protocol §5 verbatim. Specific kills for this brief:
- Calibration fails after one session → stop; fix the simulator before any modelling.
- Best pooled config vs `NNNC_MLX` < +3.35pp/yr with CI including zero → "not resolved";
  proceed to brief 2 anyway (it is cheap and shares this code), then stop the family.
- Turnover > 3× control at the accepted point → flag; require survival at 2× fees.

## What to record for the next brief
Trained fold policies, per-fold trial log, the simulator calibration report, and the
learned exposure distribution (how often the policy is flat — the operating-point finding
predicts it will be flat most of the time).

## Starter prompt for Claude Code

```
Read AGENT_GUIDE.md, then research/label-free-study/README.md, 00-evaluation-protocol.md
and briefs/01-direct-policy-optimisation.md. Implement research/labelfree/sim.py per
protocol §1 in MLX, and its calibration test against the frozen NNNC_MLX signal on W1
(regime/regimedata.py). Do not write any policy code until the calibration test passes and
its report is in briefs/results/01-calibration-<date>.md. Then implement the policy/loss
variants in the brief, logging every configuration to research/labelfree/trials.jsonl
before running it. Run everything sequentially (AGENT_GUIDE operational rules). Stage only
the files you create.
```
