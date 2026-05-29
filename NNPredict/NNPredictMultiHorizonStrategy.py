# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# type: ignore
# pylint: disable=import-error

"""
NNPredictMultiHorizonStrategy — multi-output variant of NNPredictStrategy.

Trains a single regressor with H output heads, jointly predicting forward
returns at multiple horizons (e.g., H=[2, 4, 8]). At signal-time the H
predictions are combined via a configurable rule (default: "all heads
agree on direction AND any head exceeds magnitude floor").

Each horizon's diagnostics show meaningful per-pair differences (Stage 1
sweep in 2026-05-28 session), motivating the joint training rather than
single-best-horizon selection.

Overrides relative to NNPredictStrategy:
  - HORIZONS list replaces the single HORIZON integer
  - get_training_labels returns (N, len(HORIZONS)) instead of (N,)
  - get_predictions takes the multi-output predict() result and applies
    a combination rule per the strategy's signal logic
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

from Framework.BaseStrategy import TradingAction
from NNPredictStrategy import NNPredictStrategy
import NNPredictRegressor


class NNPredictMultiHorizonStrategy(NNPredictStrategy):

    # Multiple horizons predicted jointly. Override in subclass to set.
    HORIZONS: list = [2, 4, 8]

    # Combination rule for the H output heads:
    #   "all_agree" — fire only when all horizons have same direction sign
    #                 AND max magnitude exceeds min_magnitude
    #   "any_strong" — fire when any horizon's prediction passes z + magnitude
    SIGNAL_COMBINATION: str = "all_agree"

    # =========================================================================
    # Classifier (regressor) selection
    # =========================================================================

    def get_classifier_type(self):
        return NNPredictRegressor.RegressorType.LSTM

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> Any:
        # Note: subclasses (the concrete MLX strategy) override this to use
        # the multi-horizon regressor type.
        return super().get_classifier(classifier_type, pair, seq_len, num_features)

    # =========================================================================
    # Training labels — multi-horizon
    # =========================================================================

    def get_training_labels(self, dataframe: DataFrame):
        """Compute multi-horizon labels (shape N, len(HORIZONS)), cache the
        2D matrix on self per-pair, and return just the middle-horizon 1D
        column so BaseNNStrategy.populate_indicators can store it as a
        single dataframe column (`%train_labels` assignment requires 1D).

        prepare_training_data() retrieves the full 2D matrix from the cache
        keyed by curr_pair, restoring the multi-horizon labels before they
        reach the regressor's train() call.
        """
        self.dbg_curr_df = dataframe

        if "current_gain" not in dataframe.columns:
            raise ValueError(
                "current_gain column not found — add_additional_indicators "
                "must be called before get_training_labels."
            )

        horizons = list(self.HORIZONS)
        labels_2d = np.zeros((len(dataframe), len(horizons)), dtype=np.float32)
        for i, h in enumerate(horizons):
            labels_2d[:, i] = (
                dataframe["current_gain"]
                .shift(-int(h))
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )

        # Cache the full 2D labels keyed by pair so prepare_training_data
        # can retrieve them. Always overwrite (idempotent on re-call).
        if not hasattr(self, "_mh_labels_by_pair"):
            self._mh_labels_by_pair = {}
        self._mh_labels_by_pair[self.curr_pair] = labels_2d

        # Expose the middle horizon (e.g. H=4) for the plot pipeline.
        mid_idx = len(horizons) // 2
        self.dbg_curr_df["%train_gain"] = labels_2d[:, mid_idx]

        if getattr(self, "dbg_verbose", False):
            for i, h in enumerate(horizons):
                col = labels_2d[:, i]
                self.debug_print(
                    f"        H={h} target: mean={col.mean():.4f} "
                    f"std={col.std():.4f} min={col.min():.4f} max={col.max():.4f}"
                )

        # Return 1D so BaseNN's `dataframe["%train_labels"] = labels` works.
        return labels_2d[:, mid_idx]

    def prepare_training_data(
        self,
        dataframe,
        labels,
        norm: bool = True,
        pair_names=None,
    ):
        """Substitute the cached 2D labels for the 1D placeholders returned
        by get_training_labels, then defer to the parent's loop."""
        cache = getattr(self, "_mh_labels_by_pair", None)
        if cache and pair_names is not None:
            labels_2d_list = []
            for i, p in enumerate(pair_names):
                if p in cache:
                    labels_2d_list.append(cache[p])
                else:
                    # Fallback to the 1D placeholder if cache miss.
                    labels_2d_list.append(np.asarray(labels[i]).reshape(-1, 1))
            labels = labels_2d_list
        return super().prepare_training_data(
            dataframe, labels, norm=norm, pair_names=pair_names
        )

    # =========================================================================
    # Predictions — multi-horizon → combined TradingAction
    # =========================================================================

    def get_predictions(self, dataframe: DataFrame, classifier):
        """Run the multi-output regressor and combine the H heads into
        BUY/HOLD/SELL via the strategy's SIGNAL_COMBINATION rule."""

        df_norm = self.scale_dataframe(dataframe)
        df_tensor = self.dataframeUtils.df_to_tensor(
            df_norm, self.seq_len, method=self.tensor_method
        )
        df_tensor = np.asarray(df_tensor)

        pred_gains = classifier.predict(df_tensor)  # (N, H)
        pred_gains = np.asarray(pred_gains, dtype=np.float32)
        if pred_gains.ndim == 1:
            pred_gains = pred_gains.reshape(-1, 1)

        n_horizons = pred_gains.shape[1]
        original_length = len(dataframe)
        if pred_gains.shape[0] < original_length:
            pad = np.zeros(
                (original_length - pred_gains.shape[0], n_horizons), dtype=np.float32
            )
            pred_gains_full = np.concatenate([pad, pred_gains], axis=0)
        else:
            pred_gains_full = pred_gains[:original_length]

        # Per-horizon z-score against per-horizon rolling stats.
        window = max(int(self.rolling_window), self.seq_len + 1)
        mp = max(self.seq_len, 2)

        z_scores = np.zeros_like(pred_gains_full)
        for j in range(n_horizons):
            col = pd.Series(pred_gains_full[:, j])
            mean = col.rolling(window, min_periods=mp).mean()
            std = col.rolling(window, min_periods=mp).std()
            z = (col - mean) / (std + 1e-6)
            z_scores[:, j] = z.fillna(0.0).to_numpy()

        actions = np.full(original_length, TradingAction.HOLD, dtype=int)

        if self.SIGNAL_COMBINATION == "all_agree":
            # Buy: ALL horizons have z > entry_z AND any horizon exceeds
            # magnitude floor (the strongest head sets the case).
            all_buy_z = (z_scores > self.entry_z).all(axis=1)
            all_sell_z = (z_scores < -self.entry_z).all(axis=1)
            any_buy_mag = (pred_gains_full > self.min_magnitude).any(axis=1)
            any_sell_mag = (pred_gains_full < -self.min_magnitude).any(axis=1)
            buy_mask = all_buy_z & any_buy_mag
            sell_mask = all_sell_z & any_sell_mag
        elif self.SIGNAL_COMBINATION == "any_strong":
            # Fire whenever any single horizon clears both gates.
            buy_each = (z_scores > self.entry_z) & (
                pred_gains_full > self.min_magnitude
            )
            sell_each = (z_scores < -self.entry_z) & (
                pred_gains_full < -self.min_magnitude
            )
            buy_mask = buy_each.any(axis=1)
            sell_mask = sell_each.any(axis=1)
        else:
            raise ValueError(
                f"Unknown SIGNAL_COMBINATION: {self.SIGNAL_COMBINATION}"
            )

        actions[buy_mask] = TradingAction.BUY
        actions[sell_mask] = TradingAction.SELL

        if getattr(self, "dbg_verbose", False):
            n_buy = int((actions == TradingAction.BUY).sum())
            n_sell = int((actions == TradingAction.SELL).sum())
            self.debug_print(
                f"        multi-horizon preds: shape={pred_gains_full.shape} "
                f"per-H mean={pred_gains_full.mean(axis=0).round(4).tolist()} "
                f"std={pred_gains_full.std(axis=0).round(4).tolist()}; "
                f"combination={self.SIGNAL_COMBINATION}; "
                f"actions buy={n_buy} sell={n_sell}"
            )

        # Expose the middle horizon's prediction for plotting — keeps the
        # %predict_gain trace meaningful.
        dataframe["%predict_gain"] = pred_gains_full[:, n_horizons // 2]

        return actions
