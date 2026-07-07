# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W0613
# type: ignore
# pylint: disable=import-error

"""
NNWavelet — TS_Wavelet reimagined on the NNPredict (BaseNNStrategy) framework.

TS_Wavelet's idea: decompose the gain series into wavelet coefficients, train a
model to predict the *future* coefficient vector, then INVERSE-transform the
predicted coefficients back into a gain and trade on that. This differs from
NNPredict_Coeff, which uses the coefficients only as features to predict a scalar
gain directly.

This port keeps that idea but drops it into the modern framework:
  * one persisted scaler set + train-once/save/load-and-infer (live-safe),
  * the pluggable regressor set (MLX MLP / Ridge) via get_classifier_type,
  * NNPredict's z-score buy/sell logic reused UNCHANGED.

How it fits NNPredict's single-scalar pipeline:
  * Features (model INPUT): the SCALED current coefficient vector (+ the base
    NNPredict indicators), inherited from NNPredict_Coeff's feature construction.
  * Target (model OUTPUT): the RAW future coefficient vector — a 2-D label matrix
    returned by get_training_labels(). prepare_training_data slices labels along
    axis 0, so a 2-D target passes through cleanly.
  * The regressor is MULTI-OUTPUT and reconstructs the gain INSIDE its predict()
    (inverse wavelet transform -> last value), so it hands get_predictions() a
    1-D gain array exactly like every other NNPredict regressor.

Because the reconstructed prediction is in raw gain units (not the ATR-normalised
target scale the base uses), the static magnitude floor would be mis-calibrated;
we use the scale-invariant percentile floor instead.

Reconstruction requires the full, invertible coefficient set, so PCA reduction
and gain smoothing (both optional in NNPredict_Coeff) are disabled here.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "NNPredict"))

from NNPredict_Coeff import NNPredict_Coeff
from WaveletForecaster import WaveletRegressorType


class NNWaveletStrategy(NNPredict_Coeff):

    # --- reconstruction requires full, invertible coefficients ---
    coeff_pca_components = None  # no dimensionality reduction on this path
    gain_smoother = None         # coeffs must match the gain we reconstruct

    # own scaler namespace so we don't collide with NNPredict_Coeff's
    coeff_scaler_name = "nnwavelet_coeff_scaler"

    # reconstructed gain is in raw gain units, not the normalised target scale,
    # so the static min_magnitude floor would be wrong — use the self-calibrating
    # percentile floor.
    magnitude_floor_mode = "percentile"

    # High-conviction gate. The default NNPredict gate (z=1.0/p=0.60) floods this
    # predictor with ~2000 thin-conviction trades whose friction sinks OOS P&L.
    # A walk-forward turnover sweep found a clean inverted-U peaking here
    # (z3.0/p0.97 → +1.11% mean, 3/4 windows, 341 trades); looser bleeds, tighter
    # discards good trades faster than it saves on hard regimes.
    entry_z = 3.0
    magnitude_pctile = 0.97

    # raw future-coeff target columns (kept OUT of include_list => not features)
    RAW_PREFIX = "wcoeff_raw_"

    _n_coeffs = 0  # output dimension, set when the coeff table is built
    _coeff_target = None  # stashed 2-D training target (per-pair, set below)

    # =====================================================================
    # Predictor — MLX multi-output MLP (the neural forecaster)
    # =====================================================================
    def get_classifier_type(self):
        return WaveletRegressorType.MLX

    def get_classifier(self, classifier_type, pair, seq_len, num_features) -> Any:
        reg = classifier_type.value(pair, seq_len, num_features)
        # hand the regressor the seeded wavelet + output dim for reconstruction
        reg.wavelet = self._seeded_wavelet()
        reg.n_coeffs = int(self._n_coeffs)
        return reg

    def _seeded_wavelet(self):
        """The wavelet instance used to build the coeff table — its coeff_slices
        are already seeded (populated during _build_coefficient_table), which
        array_to_coeff needs to rebuild the coefficient structure."""
        if self._wavelet is None:
            # defensive: seed on a dummy window if the table wasn't built yet
            import utils.Wavelets as Wavelets
            self._wavelet = Wavelets.make_wavelet(self.wavelet_type)
            self._wavelet.set_lookahead(int(self.HORIZON))
            dummy = np.zeros(self.wavelet_size, dtype=float)
            self._wavelet.coeff_to_array(self._wavelet.get_coeffs(dummy))
        return self._wavelet

    # =====================================================================
    # Features — scaled current coeffs (input) + raw coeffs (target source)
    # =====================================================================
    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        # Grandparent adds recent_gain/current_gain and registers them. We skip
        # NNPredict_Coeff's own add_additional_indicators because we need BOTH the
        # raw coefficient table (for the reconstruction target) and the scaled
        # features, built from a single transform pass.
        dataframe = super(NNPredict_Coeff, self).add_additional_indicators(dataframe)

        if "gain" not in dataframe.columns:
            raise ValueError(
                "NNWavelet: 'gain' column not found — expected from "
                "DataframePopulator before add_additional_indicators."
            )

        gain = dataframe["gain"].astype(float).to_numpy(dtype=float)
        raw = self._build_coefficient_table(gain)  # (N, C) raw coefficients
        self._n_coeffs = raw.shape[1]

        # RAW future-coeff target source — added to the dataframe but NOT to
        # include_list, so scale_dataframe drops it before tensorising (it never
        # becomes a model feature).
        for i in range(raw.shape[1]):
            dataframe[f"{self.RAW_PREFIX}{i}"] = raw[:, i]

        # SCALED current coeffs = model input features (own persisted scaler,
        # registered as pre_normalized so the shared main_scaler skips them).
        scaled = self._scale_coeffs(raw)
        for i in range(scaled.shape[1]):
            col = f"coeff_{i}"
            dataframe[col] = scaled[:, i]
            if col not in self.include_list:
                self.include_list.append(col)
            if col not in self.pre_normalized_columns:
                self.pre_normalized_columns.append(col)

        return dataframe

    # =====================================================================
    # Target — the RAW future coefficient vector (multi-output)
    # =====================================================================
    def get_training_labels(self, dataframe: DataFrame):
        """The real training target is the raw coefficient vector at t+HORIZON
        (the future window's coefficients) — a (N, C) matrix stashed for
        maybe_train. The framework stores this return in the 1-D "%train_labels"
        debug column, so we return the scalar future gain (which a perfect
        reconstruction of the coeff target would itself produce)."""
        self.dbg_curr_df = dataframe
        h = int(self.HORIZON)
        raw_cols = [f"{self.RAW_PREFIX}{i}" for i in range(self._n_coeffs)]
        raw = dataframe[raw_cols].to_numpy(dtype=np.float32)

        future = np.zeros_like(raw)
        if len(raw) > h:
            future[:-h] = raw[h:]  # target[t] = coeffs at t+h
        self._coeff_target = np.nan_to_num(future).astype(np.float32)

        # 1-D debug/plot label: the actual future gain (val[-1] of the perfectly
        # reconstructed future window == gain[t+h]).
        fgain = dataframe["gain"].astype(float).shift(-h).fillna(0.0)
        fgain = fgain.to_numpy(dtype=np.float32)
        self.dbg_curr_df["%train_gain"] = fgain
        return fgain

    def maybe_train(self, dataframe: DataFrame, labels, curr_pair: str) -> DataFrame:
        """Swap the 1-D debug labels for the real 2-D coefficient target before
        training. get_training_labels stashed it moments ago for this same pair."""
        return super().maybe_train(dataframe, self._coeff_target, curr_pair)
