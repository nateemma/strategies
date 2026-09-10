# Brief 07 — Self-supervised representations as extra features (null test)

**Family:** self-supervised time-series representation learning (TS2Vec / TS-TCC / masked
PatchTST) and pretrained candlestick foundation models (Kronos-small, Chronos-Bolt) as feature
extractors
**Type:** unsupervised · **Priority:** 7 · **Budget:** 1 session · **Expected result: null**

## Hypothesis (stated so it can be killed cleanly)

An embedding learned without labels from the pooled OHLCV/feature stream carries information
about forward returns that the hand-built `DataframePopulator` indicators do not. If true,
adding the embedding to the input of *either* the classifier (`NNNC_MLX`) or the brief-1
policy improves pooled-WF return by the floor.

## Why the prior is null

- `us_spot_market_study.md`: an 85-indicator battery and a learned 1D-CNN on raw OHLCV
  both hit the same ρ≈0.15 ceiling; the CNN was *worse* than hand-crafted indicators.
- Review §E: SSL embeddings learn second-moment structure (volatility, similarity), not
  conditional mean; generic foundation models have negative OOS R² on returns; Kronos
  "captures volatility clustering, not alpha"; an independent 5-min BTC test found it
  indistinguishable from Brownian motion.
- Zero-shot inference is cheap, so the test is worth one session precisely because a positive
  result would be surprising and a negative one closes the door.

## Design

- **Encoders:** (a) TS2Vec trained on the pooled normalised feature sequences of the fold's
  training range (contrastive, ~hours on MLX/MPS); (b) Kronos-small zero-shot embeddings of
  the raw OHLCV window (inference only, no fine-tuning — fine-tuning hurt in two studies);
  (c) Chronos-Bolt forecast-as-feature (predicted next-bar return quantiles).
- **Leakage guard:** encoders are fit per fold on the training range only; embeddings for a
  bar use only bars ≤ t.
- **Consumers:** `NNNC_MLX` with the embedding concatenated to `include_list` (a subclass —
  never modify the production class; pin nothing, this one *should* retrain), and brief 1's
  policy with the same concatenation.
- **Control:** the same consumer with an equally-sized block of random Gaussian features (to
  detect capacity effects rather than information).
- **Quick screen before any backtest:** Spearman ρ of each embedding dimension (and of a
  ridge projection of all dimensions) against forward return, walk-forward, versus the
  hand-built features' ρ≈0.15. If the ridge projection does not exceed 0.15 on nested OOS,
  stop — no backtest needed.

## Acceptance / kill

Protocol §5. Kill after the quick screen if no encoder's projection beats the hand-built
feature ceiling; kill after the backtest if the random-feature control matches.

## Starter prompt for Claude Code

```
Read AGENT_GUIDE.md ("Feature Engineering Rules"), research/label-free-study/README.md,
00-evaluation-protocol.md and briefs/07-self-supervised-features.md. Implement
research/labelfree/ssl/ with per-fold TS2Vec training on the pooled normalised feature
stream, Kronos-small and Chronos-Bolt zero-shot feature extraction (inference only), a
random-Gaussian control block, and the walk-forward Spearman screen against forward return.
Only if the screen beats ρ=0.15 on nested OOS, add the features to an NNNC_MLX subclass and
run the pooled walk-forward. Write briefs/results/07-<date>.md either way.
```
