# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W0613
# type: ignore
# pylint: disable=import-error

"""
NNPredict_Coeff — TS_Coeff reimagined on the NNPredict (BaseNNStrategy) framework.

TS_Coeff's idea: decompose the (normalised) ``gain`` series into wavelet
coefficients over a rolling window and use that rich, multi-scale representation
as features to predict future gain. This port keeps exactly that feature
construction but drops it into NNPredictStrategy, so it inherits the modern
framework instead of TSPredict's hand-rolled machinery:

  * one persisted RobustScaler (fit lazily over include_list) vs ad-hoc per-chunk
    scalers,
  * train-once + save + load-and-infer (live-safe on a ~950-candle buffer) vs the
    walk-forward per-chunk retrain that can't run live,
  * the pluggable regressor set (Ridge / LSTM / MLX) selected by enum,
  * GAN hook, and parallel hyperopt that already works for the NN family.

The coefficient vector is a ~50-300-dim representation (transform-dependent), so
unlike TS_Gain's 16 smoothed lags this is a setting where a nonlinear predictor
(LSTM/MLX) can plausibly beat the linear baseline. Predictor defaults to Ridge
(the linear floor any nonlinear model must beat); swap get_classifier_type to an
LSTM/MLX regressor to test that.

Note: DWT is not batchable in PyWavelets, so the per-window transform is a Python
loop (same cost profile as TS_Coeff). Use WaveletType.SWT for a batched path.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)
sys.path.append(str(Path(__file__).parent.parent))

from NNPredictStrategy import NNPredictStrategy
import NNPredictRegressorRidge
import utils.Wavelets as Wavelets
from utils.Scalers import scaler_exists, load_scaler, save_scaler


class NNPredict_Coeff(NNPredictStrategy):

    # --- wavelet coefficient feature config ---
    wavelet_type = Wavelets.WaveletType.DWT
    wavelet_size = 64  # rolling window fed to the transform (power of 2)

    # cover the wavelet window + sequence + horizon warmup
    startup_candle_count = 128

    # DWT coefficients of the (small) gain series are naturally bounded well
    # inside [-1, 1] (measured: |c| < 0.33, 99.9% within ±0.07) but low-amplitude
    # (per-column std ~0.015) and per-column scales differ. So we give them their
    # OWN persisted RobustScaler and add them as pre_normalized columns: the base
    # features keep using the shared main_scaler, while the coefficients are
    # standardised by this dedicated scaler and pass through untouched. Delete
    # saved_data/<coeff_scaler_name>.* after changing wavelet_type/size.
    coeff_scaler_name = "coeff_scaler"

    # Ridge regularization strength. Higher tames the OOS outliers that the
    # high-dimensional coeff input produces at the default alpha=1.
    ridge_alpha = 1.0

    # Optional coeff dimensionality reduction: if set to K, the coefficient block
    # is reduced to K whitened PCA components before use (cuts the redundant
    # seq_len × n_coeff blowup). None = keep all coefficients (RobustScaler only).
    coeff_pca_components = None

    # Optional low-lag smoothing of the gain series BEFORE the wavelet transform
    # (denoises the coeff features without adding lag). None | "hma" | "zlema".
    gain_smoother = None
    gain_smooth_period = 8

    # target_mode ("point" | "excursion") is inherited from NNPredictStrategy.

    _wavelet = None  # lazily created transform instance

    # =====================================================================
    # Predictor — Ridge linear baseline (the floor a nonlinear model must beat)
    # =====================================================================
    def get_classifier_type(self):
        return NNPredictRegressorRidge.RegressorTypeRidge.RIDGE

    def get_classifier(self, classifier_type, pair, seq_len, num_features) -> Any:
        reg, _ = NNPredictRegressorRidge.create_regressor_ridge(
            classifier_type, pair, num_features, seq_len
        )
        reg.alpha = self.ridge_alpha
        return reg

    # =====================================================================
    # Lifecycle
    # =====================================================================
    def bot_start(self, **kwargs) -> None:
        super().bot_start(**kwargs)
        # Own the feature lists so the coefficient columns we append below don't
        # leak into the class-shared include_list of sibling NN strategies.
        self.include_list = list(self.include_list)
        self.pre_normalized_columns = list(self.pre_normalized_columns)

    # =====================================================================
    # Strategy-specific features — wavelet coefficients of the gain series
    # =====================================================================
    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        # NNPredictStrategy adds recent_gain / current_gain (and registers them).
        dataframe = super().add_additional_indicators(dataframe)

        if "gain" not in dataframe.columns:
            raise ValueError(
                "NNPredict_Coeff: 'gain' column not found — expected from "
                "DataframePopulator before add_additional_indicators."
            )

        gain = dataframe["gain"].astype(float)
        if self.gain_smoother:
            gain = self._smooth_gain(gain, self.gain_smoother, self.gain_smooth_period)
        coeff_table = self._build_coefficient_table(gain.to_numpy(dtype=float))
        coeff_table = self._scale_coeffs(coeff_table)

        # Add each (scaled) coefficient as a named feature column and register it
        # in BOTH include_list (so it's kept) and pre_normalized_columns (so the
        # shared main_scaler skips it — we've already scaled it ourselves).
        for i in range(coeff_table.shape[1]):
            col = f"coeff_{i}"
            dataframe[col] = coeff_table[:, i]
            if col not in self.include_list:
                self.include_list.append(col)
            if col not in self.pre_normalized_columns:
                self.pre_normalized_columns.append(col)

        return dataframe

    def _scale_coeffs(self, coeff_table: np.ndarray) -> np.ndarray:
        """Transform the coefficient columns with a dedicated, persisted
        transformer (fit-if-missing, mirroring the framework's main_scaler): a
        RobustScaler by default, or a whitened PCA(K) when coeff_pca_components is
        set. Base features keep using the shared main_scaler; these outputs are
        added as pre_normalized columns."""
        loc = self.get_storage_location()
        offset = self.wavelet_size - 1  # zero warmup rows — exclude from the fit
        k = self.coeff_pca_components
        base = f"coeff_pca_{k}" if k else self.coeff_scaler_name
        # smoothing changes the coefficients, so give it its own persisted transform
        tag = f"_{self.gain_smoother}{self.gain_smooth_period}" if self.gain_smoother else ""
        name = base + tag

        if scaler_exists(loc, name):
            transformer = load_scaler(loc, name)
        else:
            fit_rows = np.nan_to_num(
                coeff_table[offset:] if len(coeff_table) > offset else coeff_table
            )
            if k:
                transformer = PCA(n_components=k, whiten=True).fit(fit_rows)
            else:
                transformer = RobustScaler().fit(fit_rows)
            save_scaler(transformer, loc, name)

        out = transformer.transform(np.nan_to_num(coeff_table))
        return np.clip(out, -10.0, 10.0)

    @staticmethod
    def _wma(s: pd.Series, period: int) -> pd.Series:
        w = np.arange(1, period + 1, dtype=float)
        return s.rolling(period).apply(
            lambda x: np.dot(x, w) / w.sum(), raw=True
        )

    def _smooth_gain(self, s: pd.Series, method: str, period: int) -> pd.Series:
        """Low-lag causal smoothing of the gain series. HMA (Hull) or ZLEMA
        (zero-lag EMA) — both denoise with minimal phase lag, so the coeff
        features stay noise-reduced without shifting the signal in time."""
        period = int(period)
        if method == "zlema":
            lag = (period - 1) // 2
            delagged = s + (s - s.shift(lag))
            out = delagged.ewm(span=period, adjust=False).mean()
        elif method == "hma":
            half = max(2, period // 2)
            sqrtp = max(2, int(round(period ** 0.5)))
            out = self._wma(2 * self._wma(s, half) - self._wma(s, period), sqrtp)
        else:
            raise ValueError(f"unknown gain_smoother: {method}")
        return out.fillna(s)  # keep warmup rows finite

    def _build_coefficient_table(self, gain: np.ndarray) -> np.ndarray:
        """Rolling wavelet decomposition of the gain series.

        Returns a (nrows, n_coeffs) array where row i holds the flattened
        transform coefficients of the window ending at bar i. The first
        (wavelet_size - 1) rows are zero (warmup), matching TS_Coeff's offset.
        """
        if self._wavelet is None:
            self._wavelet = Wavelets.make_wavelet(self.wavelet_type)
            self._wavelet.set_lookahead(int(self.HORIZON))

        nrows = len(gain)
        if nrows < self.wavelet_size:
            return np.zeros((nrows, 1), dtype=float)

        windows = sliding_window_view(gain, self.wavelet_size)
        rows = [
            self._wavelet.coeff_to_array(self._wavelet.get_coeffs(w.copy()))
            for w in windows
        ]
        result = np.asarray(rows, dtype=float)

        table = np.zeros((nrows, result.shape[1]), dtype=float)
        offset = self.wavelet_size - 1  # align window end to its bar
        table[offset : offset + result.shape[0]] = result
        return table
