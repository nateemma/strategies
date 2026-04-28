# Anomaly — Anomaly Detection Strategies

A different angle on signal generation: instead of training on labelled
buy/sell points (which are scarce and imbalanced), train an
autoencoder-style model on "normal" hold data, then flag bars where the
model has unusually high reconstruction error as buy/sell candidates.

Inherits from `BaseNNStrategy` (`Framework/BaseNNStrategy.py`) via
`NNAnomalyStrategy`.

## Main files

| File | What it does |
|---|---|
| `NNAnomalyStrategy.py` | Family base.  Wires the autoencoder pipeline into `BaseNNStrategy` — train on "normal" rows, predict reconstruction error, threshold to entry/exit. |
| `NNAnomalyClassifier.py` | Autoencoder classifier (encoder + decoder).  High reconstruction error on a bar means "doesn't look like the training distribution" → candidate trade signal. |
| `NNGANomalyStrategy.py` | GANomaly variant of the strategy. |
| `NNGANomalyClassifier.py` | GANomaly classifier — trains a generator + discriminator + encoder triplet so anomaly score combines reconstruction error and discriminator output. |
| `GANomaly_README.md` | Background notes on the GANomaly architecture (kept inline since it predates this README). |

## When to prefer over NNNC/NNMT

Anomaly-detection strategies don't need labelled buy/sell training data
— they only need a clean "normal" period.  Useful when:

* The market regime has shifted and labelled signals from the past
  no longer represent the current behaviour.
* You want to flag "unusual" conditions for further inspection rather
  than commit to a directional trade.

The flip side: anomaly score doesn't tell you whether to buy or sell,
so the strategy adds heuristic rules on top.
