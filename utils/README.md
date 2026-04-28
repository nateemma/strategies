# utils — Shared Utility Code

Cross-cutting code used by every strategy family: classifier base
classes, dataframe helpers, indicator factories, signal builders,
wavelet transforms, and forecaster glue.

Nothing in this directory is a strategy — it's all infrastructure.

## Classifier bases

| File | What it does |
|---|---|
| `ClassifierKeras.py` | Root TF/Keras classifier base class.  Owns the model lifecycle (load/save/path/checkpointing) and a generic `train()` loop using EarlyStopping + ReduceLROnPlateau + ModelCheckpoint. |
| `ClassifierKerasNary.py` | N-ary (multi-class softmax) Keras classifier.  Used by `NNNClassifier` variants. |
| `ClassifierKerasBinary.py`, `ClassifierKerasTrinary.py` | Binary and trinary Keras classifiers (specialised loss + metrics). |
| `ClassifierKerasMultiTask.py` | Multi-task Keras classifier — six task heads, focal losses, task weights, custom metrics. |
| `ClassifierKerasAnomaly.py`, `ClassifierKerasEncoder.py` | Autoencoder-style classifiers for the Anomaly family. |
| `ClassifierKerasLinear.py`, `ClassifierKerasTFT.py` | Linear and Temporal-Fusion-Transformer variants. |
| `ClassifierMLX.py` | Root MLX classifier base.  Same API as `ClassifierKeras.py` but uses `mlx.nn.Module` and `safetensors` for weights. |
| `ClassifierMLXNary.py` | MLX N-ary base.  Mirrors `ClassifierKerasNary` — manual training loop with EarlyStopping / ReduceLROnPlateau / ModelCheckpoint. |
| `ClassifierMLXMultiTask.py` | MLX multi-task base.  Per-task focal losses with per-task alpha/gamma; gradient clipping; non-finite loss/grad rejection; pre-training data filtering for NaN/Inf rows. |
| `ClassifierSklearn.py` | sklearn classifier base.  Same train/save/load contract but works with 2-D DataFrames. |
| `ClassifierDarts.py`, `ClassifierPyTorch.py` | Adapters for Darts and PyTorch classifiers (less commonly used). |

## Dataframe + indicators

| File | What it does |
|---|---|
| `DataframePopulator.py` | Adds the standard set of pre-approved technical indicators to a freqtrade dataframe.  This is the single source of truth for what the NN models see — adding indicators here is what changes feature engineering. |
| `DataframeUtils.py` | Helpers: 2-D ↔ 3-D tensor conversion, scaler fitting/persistence, rolling-window normalization, one-hot encoding. |
| `custom_indicators.py` | Project-specific indicators that aren't in TA-Lib or finta. |

## Signal generation

| File | What it does |
|---|---|
| `TradingSignals.py` | Future-aware label generators (peak detection variants).  Selected by `TRAINING_TYPE`. |
| `MarketRegimes.py`, `Risk.py`, `Flow.py`, `Momentum.py`, `Profit.py` | Per-task label computation for the multi-task classifiers (NNMT family). |

## Time-series / wavelets

| File | What it does |
|---|---|
| `Wavelets.py` | Wavelet transforms (DWT, CWT, etc.) for the TSPredict family. |
| `Forecasters.py` | Pluggable forecaster registry — wraps Prophet, ARIMA, custom DWT/FFT forecasters under one API. |

## Loss / metric helpers

| File | What it does |
|---|---|
| `CustomLoss.py`, `CustomLossMLX.py` | Multi-class focal loss (Keras and MLX). |
| `CustomMetric.py`, `CustomMetricMLX.py` | Per-class precision / F1 / MCC / accuracy metrics for the manual MLX training loops. |
| `Environment.py` | Prints the runtime environment (TF / MLX / package versions) at strategy startup. |
