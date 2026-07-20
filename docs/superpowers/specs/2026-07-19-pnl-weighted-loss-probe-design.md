# P&L-magnitude-weighted loss probe — design spec

**Date:** 2026-07-19
**Status:** draft, awaiting review
**Family:** NNNC (single-task 3-class Buy/Hold/Sell classifier)

## Motivation

The classifier trains on a classification loss (focal, val_mcc-monitored) that
treats every sample equally, but what we care about is **P&L**, and MCC is not a
P&L predictor. This probe tests the cheapest possible version of "optimize for
P&L, not accuracy": **weight each training sample's loss by the magnitude of its
realised forward move**, so the model prioritises getting the high-P&L trades
right rather than all trades equally.

It is the cheap gate for a larger idea (differentiable-Sharpe / P&L objective).
Design discipline (established this session): **gate on the learnable gbb signal
first**, not on triple-barrier. If up-weighting P&L can't beat equal-weighting on
a signal we *know* carries edge (gbb, MCC 0.60), it won't rescue TB (MCC~0,
R²<0). Only if it wins here does the objective-vs-information question stay open
for TB.

## Non-goals

- Not on triple-barrier (that's the *next* step if this shows life).
- Not changing the val_mcc monitor (isolate the loss change; see Caveats).
- Not a differentiable trading simulator — this is a loss-*weighting* tweak only.
- Not combined with the ponder head yet (isolate the loss effect on the N0 base).

## The magnitude is already computed

`Framework/TrainingSignals.py::labels_forward_return_mae_cap` (the gbb labeler)
already computes, per row:
- `mfe = (max_future - close) / close` — max favorable excursion (buy P&L proxy)
- `sell_mfe = (close - min_future) / close` — sell P&L proxy

It returns only the boolean label and discards these. The probe exposes them as a
per-row weight.

## Mechanism

Per training row `i`, a P&L weight:

```
mag_i = mfe_i        if label_i == Buy
        sell_mfe_i   if label_i == Sell
        baseline     if label_i == Hold        # Holds carry no trade P&L
mag_i = clip(mag_i, 0, quantile(mag, 0.95))     # tame heavy tails (ZEC)
w_i   = (1 - alpha) + alpha * (mag_i / mean(mag_i))   # blend; normalised mean≈1
```

- **`alpha` blend knob** (swept, per "hyperopt the method choice"):
  `alpha=0` → identical to the current unweighted loss (built-in control /
  identity check). `alpha=1` → pure magnitude weighting. Sweep `{0, 0.5, 1.0}`.
- **Holds baseline**: Hold rows get the mean magnitude (weight ≈ 1 after
  normalisation), so the probe up-weights high-excursion Buy/Sell rows without
  distorting the Hold class.
- **Normalisation** keeps the total loss scale ≈ unchanged so the LR / clip
  behave the same across alpha.
- The weight multiplies the existing focal loss per sample; class weights and
  focal gamma are unchanged.

## Plumbing

1. **Labeler** (`labels_forward_return_mae_cap`): also return the per-row
   magnitude array (or a sibling function that does), aligned with the label
   Series. Buy→mfe, Sell→sell_mfe, Hold→NaN/baseline sentinel.
2. **TrainingEngine.prepare_training_data**: carry the magnitude array through the
   same train/test split, `df_to_tensor` windowing, and `offset` slicing that
   labels get, so `w` stays aligned with training rows. **Non-GAN only**
   (`gan_type=NONE`): `enhance_training_data` passes through, so there is no
   augmentation reshuffling to realign — weights map 1:1 to rows.
3. **MLXClassifierNary.train**: accept an optional per-sample weight array; batch
   it alongside `(X, y)` in `_batch_iter`; pass `w_batch` into the loss.
4. **Loss** (`multi_class_focal_loss_mlx`): add an optional per-sample weight
   multiplier (default all-ones → current behaviour).

All additions default to the current behaviour when the weight/alpha is absent,
so production and other strategies are unaffected.

## Strategy

`NNNC_MLX_PnlLoss` — inherits the **plain non-GAN `NNNC_MLX`** base (gan_type=NONE,
**post_gan_scaling=False**; NOT `NNNC_DDPM_MLX`, whose `NNNC_DDPM_` prefix means GAN
and which carries `post_gan_scaling=True`). Plain LSTM (no ponder) isolates the
loss effect. Sets `pnl_loss_alpha` (stamped onto the classifier via
`PnlLossStrategyMixin`, alongside `train_seed`). Distinct-name subclasses per alpha
for the sweep, each retrains its own model.

Note: the `NNNC_MLX` path uses the name-aware df `main_scaler`. If it is stale
(pre `di_diff_scaled`/`spread_ma`, like `gan_scaler_a` was), `CreateScalers` is
required first (a shared-artifact regen — flag before running).

## Validation gate

Same discipline that separated the ponder signal from the mirages:

- **Paired seed-robust A/B**: for alpha ∈ {0, 0.5, 1.0}, retrain at seeds
  {1, 7, 13} (and 42), non-GAN, **pinned window** `20240629-20260619`. Compare
  alpha>0 vs the alpha=0 control *paired by seed*.
- **`alpha=0` must reproduce the current N0 non-GAN baseline** (~13% band) —
  identity check.
- **Reject criterion (upfront)**: if no alpha beats alpha=0 across seeds (sign
  consistent, gap outside the paired-seed noise floor ~±0.5pp) → the objective was
  not the bottleneck; the information ceiling is loss-independent. Documented
  reject, don't proceed to TB.
- **Escalate only on a robust win**: if alpha>0 beats alpha=0 paired across seeds
  → (a) try P&L-based model selection (monitor), (b) combine with N=2 ponder,
  (c) *then* test on triple-barrier targets.

## Caveats

- **Monitor mismatch**: training loss is P&L-weighted but best-model selection
  stays on val_mcc. If weighting helps but MCC-selection masks it, the probe
  under-reads. Accepted for the cheap first pass (changing the monitor risks the
  documented F1/precision-collapse failure modes); P&L-based selection is a
  follow-up only if the probe shows life.
- **Heavy tails**: without the 95th-pct clip, a few ZEC mega-moves dominate the
  gradient. Clip is load-bearing; log the clip fraction.
- Same structural caveats as the strategy: ZEC-concentrated, ~2yr, in-sample.

## Success criteria

- `alpha=0` reproduces the non-GAN N0 baseline (identity).
- Weights verified aligned with rows (a unit check on a small slice: Buy rows
  carry mfe, Sell rows carry sell_mfe, Holds carry baseline).
- Experimental verdict: a robust paired win for some alpha → escalate; otherwise
  documented reject (objective was not the bottleneck).
