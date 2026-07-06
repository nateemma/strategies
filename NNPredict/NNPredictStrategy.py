# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNPredictStrategy — base class for Neural Network Regression strategies.

Parallel to NNNCStrategy (n-ary classification) but predicts a continuous
future-gain target directly rather than a class label. Entry/exit signals
are derived from a rolling-quantile threshold on the predicted gains, so
the threshold adapts per-pair / per-regime.

The three classification-specific paths in BaseNNStrategy are bypassed here:

  * get_training_labels       — overridden to return continuous future_gain
                                 (1D float array) instead of TradingAction
                                 class indices.
  * prepare_training_data     — overridden so the per-pair label slicing
                                 does NOT one-hot-encode the float targets.
  * get_training_class_weights — overridden to return None; bincount on
                                 negative continuous targets would crash.
  * get_predictions           — overridden to consume continuous regressor
                                 output and convert it to TradingAction
                                 integers via a per-pair rolling-quantile
                                 threshold.

Markov smoothing is disabled (only meaningful for discrete-state output).
"""

import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)
sys.path.append(str(Path(__file__).parent.parent))

from Framework.BaseNNStrategy import BaseNNStrategy, StrategyConfig
from Framework.BaseStrategy import (
    ModelType,
    NormalizationType,
    TradingAction,
)

import NNPredictRegressor


class NNPredictStrategy(BaseNNStrategy):

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                # "%train_lebels": {"color": "lightgreen"},
                "predict_buy": {"color": "green"},
                # "%train_sell": {"color": "orange"},
                "predict_sell": {"color": "red"},
                "%train_gain": { "color": "blue"},
                "%predict_gain": { "color": "purple"},
                "fisher_ss": { "color": "lightsteelblue"}
            },
        },
    }

    buy_params = {
        "entry_adx_threshold": 0.02,
        "entry_atr_pct": 0.001,
        "entry_bb_width_threshold": 0.0,
        "entry_close_norm_threshold": 0.0,
        "entry_enable_guards": False,
        "entry_guard_threshold": -0.2,
        "entry_rvol_threshold": 1.0,
        "prediction_threshold": 0.3,
    }

    sell_params = {
        "cexit_enable_profit_checks": True,
        "cexit_max_days": 3,
        "cexit_take_profit": 0.02,
        "enable_exit_signal": True,
        "exit_close_norm_threshold": 0.0,
        "exit_guard_threshold": 0.0,
    }

    strategy_config = StrategyConfig(
        normalization=NormalizationType.ROLLING_ROBUST,
        model_type=ModelType.KERAS,
        needs_training=True,
        seq_len=16,
    )

    augment_training_data = False  # signal augmentation is binary-only — not meaningful for continuous targets
    use_markov_smoothing = False   # Markov transition matrix is for discrete states

    # Regression target caps. After ATR scaling, gain is in ATR-units
    # (multiples of per-bar volatility). The signed peak/trough over the
    # forward horizon naturally distributes around ±3-5 ATR for typical
    # crypto data, so ±8 clips only the genuine tail and keeps the bulk
    # of the target distribution continuous (avoids a bimodal saturated
    # target that traps MSE at predict-the-mean). Kept symmetric so MSE
    # doesn't bias the regressor toward one side.
    target_max_gain: float = 8.0
    target_max_loss: float = 8.0

    # Floor on atr_pct to avoid divide-by-near-zero on dead pairs.
    atr_floor: float = 1e-3

    # Signal logic — z-score with magnitude floor (Option A).
    # Replaces rolling-quantile thresholding, which was variance-inverted:
    # smoother predictions produced TIGHTER q90/q10 bands → MORE noise-driven
    # signals (see feedback_nnpredict_label_smoothing_backfires in memory).
    # Z-score keeps the per-pair adaptive normalization but uses a fixed σ
    # threshold (independent of prediction width) and adds an absolute
    # magnitude floor so tight-distribution scenarios can't over-fire.
    rolling_window: int = 200
    entry_z: float = 1.0         # require pred >= entry_z σ above rolling mean
    min_magnitude: float = 0.10  # absolute floor on |pred|; raised by adaptive

    # Adaptive magnitude floor — multiplied against rolling std of predictions.
    # Effective floor = max(min_magnitude, mag_std_mult * rolling_std).
    # KEPT AT 0 — empirically (2026-05-28) prediction std does NOT correlate
    # with prediction quality. Winning pairs have HIGHER std (bold confident
    # predictions); losing pairs have lower std (timid noise). At mult>=2.5
    # the adaptive floor filters out winning trades while leaving losing
    # trades alone. See feedback_nnpredict_adaptive_magnitude_inverse.
    mag_std_mult: float = 0.0

    # Magnitude-floor mode:
    #   "absolute"   — floor = max(min_magnitude, mag_std_mult * rolling_std)
    #                  (the existing behaviour; unchanged default).
    #   "percentile" — floor = rolling `magnitude_pctile` quantile of |pred|.
    # The absolute floor (0.10) is calibrated on the TARGET scale (~0.23) but is
    # applied to the raw predictions, which regress-to-mean and compress to a much
    # smaller, predictor-dependent scale (Ridge vs LSTM differ). A percentile
    # floor is scale-invariant: it always passes the top (1-pctile) fraction of
    # |pred|, so it self-calibrates to whatever amplitude the model outputs.
    magnitude_floor_mode: str = "absolute"
    magnitude_pctile: float = 0.60

    # entry_quantile / exit_quantile kept for backward compat but UNUSED
    # under the new logic.
    entry_quantile: float = 0.90
    exit_quantile: float = 0.10

    # for a regressor, we need a smaller prediction window. Picked off
    # half-period of fisher_ss to break the "predict current state" shortcut
    # the model finds when H ≈ fisher_ss period / 2 (where -fisher_ss[i+H]
    # ≈ +fisher_ss[i]).
    HORIZON = 4

    # Label smoothing window (bars on each side of i+H). The training target
    # becomes the centered rolling mean of forward returns over a (2*W+1)-bar
    # window centered at i+H. W=0 disables smoothing (original behaviour).
    # KEEP AT 0 by default — smoother predictions trigger MORE rolling-quantile
    # signals (the q90/q10 bands tighten with prediction variance), producing
    # catastrophic stop-bleed. See feedback_nnpredict_label_smoothing_backfires
    # in memory. Re-enable only if signal logic is also revised.
    LABEL_SMOOTH_WINDOW: int = 0

    # Target formulation:
    #   "point"     — H-bar-forward point return (close[i+H] vs close[i]). Default.
    #   "excursion" — the dominant signed move over the next H bars (largest
    #                 favorable rise vs adverse fall), i.e. the move a trade could
    #                 actually capture rather than the arbitrary endpoint. Causal
    #                 forward mirror of recent_gain, same sign/ATR/cap convention.
    # Validated on the wavelet-coeff family (NNPredict_Coeff_Exc): "excursion" was
    # the one change where ρ AND OOS P&L moved together. Feature-agnostic, so any
    # NNPredict strategy can opt in. Judge on walk-forward P&L, not ρ/R².
    target_mode: str = "point"

    # =========================================================================
    # Classifier (regressor) selection
    # =========================================================================

    def get_classifier_type(self):
        return NNPredictRegressor.RegressorType.LSTM

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> Any:
        reg, _ = NNPredictRegressor.create_regressor(
            classifier_type, pair, num_features, seq_len
        )
        return reg

    # =========================================================================
    # Strategy-specific features
    # =========================================================================

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Adds recent_gain — the backward-window mirror of the future_gain
        target — as an autoregressive feature for the regressor:

            up   = (close[i] - min(close[i-H : i])) / close[i]   (rise from recent low)
            down = (close[i] - max(close[i-H : i])) / close[i]   (fall from recent high; <= 0)
            raw  = up if up > -down else down
            gain = raw / max(atr_pct[i], atr_floor)

        Sign convention matches the future_gain target: positive recent_gain
        means we recently rose (largest backward excursion was a rise), so the
        feature lines up with target_positive = will rise. Strictly causal
        (window is close[i-H:i], excluding current).

        Pre-normalized to [-1, +1] by dividing the ATR-unit gain by
        max(target_max_gain, target_max_loss) (the same cap used on the target),
        so the feature can sit in pre_normalized_columns and the shared scaler
        does not need to refit.
        """

        dataframe = super().add_additional_indicators(dataframe)

        close = dataframe["close"].astype(float)
        horizon = int(self.HORIZON)

        atr_pct = (
            pd.Series(dataframe.get("atr_pct", pd.Series(np.zeros(len(close)))))
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        atr_pct = np.maximum(atr_pct, self.atr_floor)

        past = close.shift(1)
        past_max = past.rolling(horizon, min_periods=1).max().to_numpy(dtype=np.float32)
        past_min = past.rolling(horizon, min_periods=1).min().to_numpy(dtype=np.float32)
        close_arr = close.to_numpy(dtype=np.float32)

        up = (close_arr - past_min) / close_arr
        down = (close_arr - past_max) / close_arr
        raw = np.where(up > -down, up, down)

        recent_gain = raw / atr_pct
        recent_gain = np.nan_to_num(recent_gain, nan=0.0, posinf=0.0, neginf=0.0)
        cap = float(max(self.target_max_gain, self.target_max_loss))
        recent_gain = np.clip(recent_gain, -cap, cap) / cap

        dataframe["recent_gain"] = recent_gain.astype(np.float32)

        # current_gain: backward H-bar close-to-close return in ATR-units,
        # pre-normalized to [-1, +1]. Backward mirror of the forward training
        # target — get_training_labels shifts this column forward by H to
        # produce
        #     labels[i] = current_gain[i+H]
        #               = (close[i+H] - close[i]) / close[i] / atr_pct[i+H]
        # i.e., the model is asked to predict the 4-bar forward return in
        # ATR-units (price-direction target). Replaces the prior -fisher_ss
        # target which had high training metrics but was not actually a good
        # buy/sell indicator — accurate fisher prediction does not translate
        # to tradeable price-direction signals.
        past_close = close.shift(horizon).to_numpy(dtype=np.float64)
        close_full = close.to_numpy(dtype=np.float64)
        gain_atr = (close_full - past_close) / past_close / atr_pct
        gain_atr = np.nan_to_num(gain_atr, nan=0.0, posinf=0.0, neginf=0.0)
        current_gain = np.clip(gain_atr, -cap, cap) / cap
        dataframe["current_gain"] = current_gain.astype(np.float32)

        # Strategy-local registration so recent_gain isn't dropped by
        # rolling_dataframe_normalise (which drops anything not in include_list)
        # and isn't fit by the shared scaler (it's already pre-normalized to
        # [-1, +1]). Idempotent across calls — same pattern as
        # BaseNNStrategy.process_one_hot_columns uses for derived columns.

        add_columns = ["recent_gain", "current_gain", "atr_norm", "ema_fast_norm"]
        for col in add_columns:
            if col not in self.include_list:
                self.include_list.append(col)
            if col not in self.pre_normalized_columns:
                self.pre_normalized_columns.append(col)

        return dataframe

    # =========================================================================
    # Training labels — continuous future_gain target
    # =========================================================================

    def get_training_labels(self, dataframe: DataFrame):
        """Return training labels as the H-bar-forward shift of `current_gain`.

        `current_gain` is the backward close-to-close return in ATR-units,
        pre-normalized to [-1, +1], computed once in add_additional_indicators.
        Shifting it forward by H produces

            labels[i] = current_gain[i+H]
                      = (close[i+H] - close[i]) / close[i] / atr_pct[i+H]

        — i.e., the same close-to-close formulation, looking forward instead
        of backward. Output is in [-1, +1] (since current_gain is already
        pre-normalized). val_loss values will be ~50× smaller than the prior
        ATR-unit targets, but R² and ρ are scale-invariant.

        Tail rows without a full forward horizon get label=0 from fillna; the
        training tensor drops them via the seq_len offset, so this is benign.

        Note: atr_pct is sampled at i+H here (consistent with current_gain's
        backward definition), not at i as the prior implementation used.
        """
        if self.target_mode == "excursion":
            return self._excursion_labels(dataframe)

        self.dbg_curr_df = dataframe

        if "current_gain" not in dataframe.columns:
            raise ValueError(
                "current_gain column not found — add_additional_indicators "
                "must be called before get_training_labels."
            )

        horizon = int(self.HORIZON)
        smooth_w = int(getattr(self, "LABEL_SMOOTH_WINDOW", 0))
        forward_gain = dataframe["current_gain"].shift(-horizon)
        if smooth_w > 0:
            forward_gain = forward_gain.rolling(
                window=2 * smooth_w + 1, center=True, min_periods=1
            ).mean()
        labels = forward_gain.fillna(0.0).to_numpy(dtype=np.float32)

        # Expose for debugging — write through self.dbg_curr_df, which is
        # the dataframe the plot pipeline reads. Writing to the local
        # `dataframe` parameter alone wasn't surviving downstream — that
        # produced a stale %train_gain in the plot that didn't match the
        # current_gain column. Mirroring %train_buy / %train_sell convention.
        self.dbg_curr_df["%train_gain"] = labels

        if getattr(self, "dbg_verbose", False):
            self.debug_print(
                f"        future_gain target: mean={labels.mean():.4f} "
                f"std={labels.std():.4f} min={labels.min():.4f} max={labels.max():.4f}"
            )

        return labels

    def _excursion_labels(self, dataframe: DataFrame):
        """Dominant signed excursion over the forward window [i+1 .. i+H]:
            up   = (max(high[i+1:i+H]) - close[i]) / close[i]   (rise potential)
            down = (close[i] - min(low[i+1:i+H])) / close[i]    (fall potential)
            raw  = up if up >= down else -down                  (dominant move)
            gain = raw / atr_pct  -> clip to ±cap -> /cap in [-1, 1]
        Same sign convention + ATR normalization + cap as current_gain, so the
        z-score / magnitude signal logic is unchanged. Feature-agnostic."""
        self.dbg_curr_df = dataframe
        close = dataframe["close"].astype(float)
        high = dataframe["high"].astype(float)
        low = dataframe["low"].astype(float)
        atr_pct = pd.Series(
            dataframe.get("atr_pct", pd.Series(np.zeros(len(close)), index=close.index))
        ).fillna(0.0)
        atr_pct = np.maximum(atr_pct.to_numpy(dtype=float), self.atr_floor)
        h = int(self.HORIZON)

        c = close.to_numpy(dtype=float)
        fwd_high = high.rolling(h).max().shift(-h).to_numpy(dtype=float)
        fwd_low = low.rolling(h).min().shift(-h).to_numpy(dtype=float)
        up = (fwd_high - c) / c
        down = (c - fwd_low) / c
        raw = np.where(up >= down, up, -down)
        gain_atr = np.nan_to_num(raw / atr_pct, nan=0.0, posinf=0.0, neginf=0.0)
        cap = float(max(self.target_max_gain, self.target_max_loss))
        labels = np.nan_to_num(np.clip(gain_atr, -cap, cap) / cap).astype(np.float32)

        self.dbg_curr_df["%train_gain"] = labels
        return labels

    # =========================================================================
    # Training data prep — skip one-hot encoding (targets are continuous)
    # =========================================================================

    def prepare_training_data(
        self,
        dataframe: List[DataFrame],
        labels: List[Any],
        norm: bool = True,
        pair_names: Optional[List[str]] = None,
    ):
        """Mirror of BaseNNStrategy.prepare_training_data minus the
        one_hot_encode(labels, 3) step. Continuous targets pass through as
        1D float32 arrays."""

        if len(dataframe) < 1:
            raise ValueError("No dataframes passed in")

        num_pairs = len(dataframe)
        num_rows = np.shape(dataframe[0])[0]
        max_index = num_pairs * num_rows

        if max_index <= 1:
            raise ValueError(
                f"Insufficient data for training: max_index={max_index} "
                f"(num_pairs={num_pairs}, num_rows={num_rows}). "
                f"Need at least 2 total rows across all pairs."
            )

        aggr_tsr_train = None
        aggr_tsr_test = None
        aggr_train_labels = None
        aggr_test_labels = None

        for i in range(num_pairs):

            pair_labels = np.asarray(labels[i], dtype=np.float32)
            df_norm = self.scale_dataframe(dataframe[i]) if norm else dataframe[i]

            split_idx = int(self.TRAIN_DATA_SPLIT * len(df_norm))
            buffer_size = self.seq_len - 1

            train_end = split_idx - buffer_size
            train_df = df_norm[:train_end]

            test_start = train_end
            test_df = df_norm[test_start:]

            train_labels = pair_labels[:train_end]
            test_labels = pair_labels[test_start:]

            pair_name = (
                pair_names[i]
                if pair_names is not None and i < len(pair_names)
                else None
            )
            train_df, train_labels = self.enhance_training_data(
                train_df, train_labels, pair_name=pair_name
            )

            tsr_train = self.dataframeUtils.df_to_tensor(
                train_df, self.seq_len, method=self.tensor_method
            )
            tsr_test = self.dataframeUtils.df_to_tensor(
                test_df, self.seq_len, method=self.tensor_method
            )

            if self.tensor_method == 0:
                offset = self.seq_len - 1
            else:
                offset = 0

            train_labels = train_labels[offset:]
            test_labels = test_labels[offset:]

            if aggr_tsr_train is None:
                aggr_tsr_train = tsr_train
                aggr_tsr_test = tsr_test
                aggr_train_labels = train_labels
                aggr_test_labels = test_labels
            else:
                aggr_tsr_train = np.concatenate([aggr_tsr_train, tsr_train], axis=0)
                aggr_tsr_test = np.concatenate([aggr_tsr_test, tsr_test], axis=0)
                aggr_train_labels = np.concatenate(
                    [aggr_train_labels, train_labels], axis=0
                )
                aggr_test_labels = np.concatenate(
                    [aggr_test_labels, test_labels], axis=0
                )

        # df_to_tensor with method=3 (the default on Apple Silicon) returns
        # mlx.core.array, which sklearn.utils.shuffle in BaseNNStrategy can't
        # index. Coerce to numpy here so the shuffle step works regardless of
        # backend; both regressor backends accept numpy in train().
        return (
            np.asarray(aggr_tsr_train),
            np.asarray(aggr_tsr_test),
            np.asarray(aggr_train_labels),
            np.asarray(aggr_test_labels),
        )

    # =========================================================================
    # Class weights — N/A for regression
    # =========================================================================

    def get_training_class_weights(self, train_labels=None, validation_labels=None):
        """Regression has no classes. Returning None (rather than running
        BaseNNStrategy's bincount-based code, which crashes on negative
        continuous values)."""
        return None

    # =========================================================================
    # Predictions — continuous gain → rolling-quantile → TradingAction
    # =========================================================================

    def get_predictions(self, dataframe: DataFrame, classifier):
        """Run the regressor, then threshold the continuous predictions via
        a rolling per-pair quantile to produce HOLD/BUY/SELL integers."""

        df_norm = self.scale_dataframe(dataframe)
        df_tensor = self.dataframeUtils.df_to_tensor(
            df_norm, self.seq_len, method=self.tensor_method
        )
        # df_to_tensor with method=3 (default on Apple Silicon) yields an
        # mlx.core.array; the Keras predict path can't consume that. Coerce
        # to numpy here — both regressor backends accept numpy in predict().
        df_tensor = np.asarray(df_tensor)

        pred_gains = classifier.predict(df_tensor)
        pred_gains = np.asarray(pred_gains, dtype=np.float32).reshape(-1)

        original_length = len(dataframe)
        if len(pred_gains) < original_length:
            pad = np.zeros(original_length - len(pred_gains), dtype=np.float32)
            pred_gains_full = np.concatenate([pad, pred_gains])
        else:
            pred_gains_full = pred_gains[:original_length]

        gains_series = pd.Series(pred_gains_full)
        window = max(int(self.rolling_window), self.seq_len + 1)
        mp = max(self.seq_len, 2)

        # Z-score relative to rolling stats: how unusual is the current
        # prediction vs the last `window` bars on this pair? Z is dimensionless
        # so the threshold (entry_z) doesn't shift when the prediction
        # distribution widens or tightens — the variance-inversion failure
        # mode of the prior quantile logic is eliminated.
        pred_mean = gains_series.rolling(window, min_periods=mp).mean()
        pred_std = gains_series.rolling(window, min_periods=mp).std()
        z_score = (gains_series - pred_mean) / (pred_std + 1e-6)

        # Adaptive magnitude floor: max of static base and N * rolling std.
        # On pairs where the model is uncertain (high pred_std), the floor
        # rises proportionally so only outlier-magnitude signals fire. On
        # confident pairs (low pred_std), the static base applies.
        if self.magnitude_floor_mode == "percentile":
            # Scale-invariant floor: rolling percentile of |pred|. Self-calibrates
            # to the predictor's output amplitude, so compression / predictor
            # choice doesn't shift the effective threshold.
            adaptive_floor = (
                gains_series.abs()
                .rolling(window, min_periods=mp)
                .quantile(self.magnitude_pctile)
            )
        else:
            adaptive_floor = np.maximum(
                self.min_magnitude, self.mag_std_mult * pred_std
            )

        actions = np.full(original_length, TradingAction.HOLD, dtype=int)
        # BOTH conditions required: unusual ranking (z) AND meaningful
        # magnitude relative to the per-pair adaptive floor.
        buy_mask = (z_score > self.entry_z) & (gains_series > adaptive_floor)
        sell_mask = (z_score < -self.entry_z) & (gains_series < -adaptive_floor)
        actions[buy_mask.fillna(False).to_numpy()] = TradingAction.BUY
        actions[sell_mask.fillna(False).to_numpy()] = TradingAction.SELL

        if getattr(self, "dbg_verbose", False):
            n_buy = int((actions == TradingAction.BUY).sum())
            n_sell = int((actions == TradingAction.SELL).sum())
            self.debug_print(
                f"        predicted gains: mean={pred_gains_full.mean():.4f} "
                f"std={pred_gains_full.std():.4f} "
                f"min={pred_gains_full.min():.4f} max={pred_gains_full.max():.4f}; "
                f"actions buy={n_buy} sell={n_sell}"
            )

        # Stash continuous gain for downstream debug / stats — the classifier
        # path stores per-class probabilities; the equivalent here is the raw
        # predicted gain.
        dataframe["%predict_gain"] = pred_gains_full

        return actions
