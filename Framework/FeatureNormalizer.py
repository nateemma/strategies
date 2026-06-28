# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
FeatureNormalizer - feature-selection + normalization mixin.

A generic, strategy-agnostic engine extracted verbatim from BaseNNStrategy. It
owns the feature-list selection (include_list / pre_normalized_columns /
one_hot_columns), the scaler + PCA state and persistence, and every method that
consumes those (rolling_dataframe_normalise, normalise_for_gan, clean_for_tensor,
apply_pca, get_normalized_size, etc.). The list *contents* remain overridable
class attributes (subclasses/families inject their own); the methods operate on
whatever those attributes contain. Mixed into BaseNNStrategy (listed first), so
self.* lookups (debug_print, dataframeUtils, get_storage_location) still resolve
via BaseStrategy through the MRO and every call site is unchanged.
"""

from typing import Optional, List, Any, Dict
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

from utils.Scalers import scaler_exists, save_scaler, load_scaler
from Framework.BaseStrategy import (
    MarketRegime,
    RiskLevel,
    FlowDirection,
    MomentumDirection,
)


class FeatureNormalizer:
    """Feature-list selection + scaling/PCA engine. Stateful only in the scaler/
    PCA/feature-list class attributes below; everything else reads via self."""

    # =========================================================================
    # Scaler / PCA / feature-list state
    # =========================================================================

    main_scaler = None
    main_scaler_name = "main_scaler"
    gan_scaler_a = None
    gan_scaler_a_name = "gan_scaler_a"
    pca_components = None
    pca_mean = None
    pca_n_components = None
    pca_feature_columns = None
    pca_explained_variance = None  # per-component variance (for whitening transform)
    pca_col_min = None  # per-component min from training data (for scaling to [-1, 1])
    pca_col_max = None  # per-component max from training data (for scaling to [-1, 1])
    pca_name = "pca_data"
    pca_passthrough_columns = [
        "rsi_scaled",
        "mfi_scaled",
        "ema_fast_norm",
        "fastk_scaled",
        "di_diff_scaled",
    ]
    use_pca_reduction = False
    include_list = [
        # "ad_scaled",
        "adx_scaled",
        "aroonosc_scaled",
        "atr_norm",
        "bb_position",
        "bb_width",
        "cci_scaled",
        # "close_norm",
        "di_diff_scaled",
        # Calendar features dropped — empirical correlations with buy/sell
        # signals are all < 0.06 on type-17 labels, mutual information barely
        # above zero. Net cost (GAN modeling overhead + passthrough complexity)
        # exceeded the negligible signal benefit. Reinstate (uncomment + drop
        # from pre_normalized_columns) if a stronger time-of-day signal appears
        # under a different label scheme.
        # "doy_cos",
        # "doy_sin",
        # "dow_cos",
        # "dow_sin",
        "ema_fast_norm",
        "fast_diff",
        # "fastd_scaled",
        "fastk_scaled",
        "fisher_ss",
        "cg_ss",
        # "dymi_scaled",
        # "fisher_wr",
        # "flow",
        "gain_norm",
        # guard_metric (scaled RMI) is the variable type-17 labels are
        # derived from (alongside bb_width); including it gives the
        # classifier direct access to one of the two deciding inputs.
        # Split into unimodal guard_metric_pos / guard_metric_neg
        # (DebugAnalyseIndicators BC 0.56-0.60 on SOL/ZEC/ICP) — same
        # treatment as macd_pos/neg. Original guard_metric is still
        # computed in DataframePopulator for non-NN consumers.
        # "guard_metric",
        "guard_metric_pos",
        "guard_metric_neg",
        # "hour_cos",
        # "hour_sin",
        # log_volume_norm dropped — composite rank 14, individual buy/sell
        # correlation ~0.08. Its only meaningful structure was the joint
        # with rsi_scaled (real |ρ|=0.27-0.35), which the GAN persistently
        # collapses to ~0 across every augmentation block. Passthrough only
        # fixes the marginal, not joints with non-passthrough features.
        # "log_volume_norm",
        # macd_norm split into unimodal pos/neg components — the original
        # is bimodal around 0 (trend-up vs trend-down clusters), which
        # MLP-regression-based GANs collapse to the conditional mean.
        # Splitting gives both the GAN and classifier unimodal features
        # with explicit sign. Original macd_norm is still computed in
        # DataframePopulator for non-NN consumers (TSPredict, etc.).
        "macd_pos",
        "macd_neg",
        "macdhist_norm",
        # "macdsignal_norm",
        "mfi_scaled",
        # "minus_di_scaled",
        # mod_sin/cos dropped together with the other calendar features above.
        # "mod_cos",
        # "mod_sin",
        # "momentum",
        # "obv_scaled",
        # "plus_di_scaled",
        # "pv_trend",
        # "regime",
        # "risk",
        "rsi_scaled",
        # "rsi_sma",  # ABLATION C
        "sar_ratio",
        "spread_ma",
        # "volume_sma_norm",
        # vwap_ratio also bimodal — same split treatment as macd_norm.
        "vwap_pos",
        "vwap_neg",
        # "willr_scaled",
    ]
    pre_normalized_columns = [
        "ad_scaled",
        "adx_scaled",
        "aroonosc_scaled",
        "atr_norm",
        "bb_position",
        # bb_width removed — DataframePopulator no longer fixed-normalises
        # it (was /0.028 + clip to [-1, 1]). main_scaler now z-scores it
        # like other unbounded ratio features.
        "cci_scaled",
        "close_norm",
        "doy_cos",
        "doy_sin",
        "dow_cos",
        "dow_sin",
        "ema_fast_norm",
        "fastd_scaled",
        "fastk_scaled",
        "fast_diff",
        "fisher_ss",
        "cg_ss",
        "dymi_scaled",
        "gain_norm",
        "guard_metric",
        # guard_metric_pos / guard_metric_neg follow the macd_pos/neg
        # convention: derived via clip(lower=0) from a [-1, 1] feature,
        # so they're already in [0, 1] — pre-normalized.
        "guard_metric_pos",
        "guard_metric_neg",
        # "hour_cos",
        # "hour_sin",
        "mod_sin",
        "mod_cos",
        "log_volume_norm",
        "macd_norm",
        # macd_pos / macd_neg follow macd_norm's "pre-normalized to
        # [-1, 1]" convention since they're derived via clip(lower=0).
        "macd_pos",
        "macd_neg",
        "macdhist_norm",
        "mfi_scaled",
        "minus_di_scaled",
        "obv_scaled",
        "plus_di_scaled",
        "pv_trend",
        "rsi_scaled",
        "rsi_sma",
        "willr_scaled",
    ]
    one_hot_columns = []

    @staticmethod
    def get_pca_path(storage_location: str, name: str) -> Path:
        """Path for PCA artifact (same directory as scalers)."""
        return Path(storage_location) / f"{name}.pkl"
    @staticmethod
    def pca_data_exists(storage_location: str, name: str) -> bool:
        """Check if PCA data exists at the given location."""
        return FeatureNormalizer.get_pca_path(storage_location, name).exists()
    @staticmethod
    def save_pca_data(
        storage_location: str,
        name: str,
        components: np.ndarray,
        mean: np.ndarray,
        n_components: int,
        feature_columns: Optional[List[str]] = None,
        explained_variance_ratio: Optional[np.ndarray] = None,
        explained_variance: Optional[np.ndarray] = None,
        col_min: Optional[np.ndarray] = None,
        col_max: Optional[np.ndarray] = None,
    ) -> None:
        """Save PCA artifacts for later load."""
        path = FeatureNormalizer.get_pca_path(storage_location, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"    Saving PCA data to {path}")
        payload = {
            "components": components,
            "mean": mean,
            "n_components": n_components,
            "feature_columns": feature_columns,
            "explained_variance_ratio": explained_variance_ratio,
            "explained_variance": explained_variance,
            "col_min": col_min,
            "col_max": col_max,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
    @staticmethod
    def load_pca_data(storage_location: str, name: str) -> Dict[str, Any]:
        """Load PCA artifacts."""
        path = FeatureNormalizer.get_pca_path(storage_location, name)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload
    def process_one_hot_columns(self, dataframe: DataFrame) -> DataFrame:
        """Process one-hot encoded columns"""
        fixed_class_counts = {
            "regime": len(MarketRegime),
            "risk": len(RiskLevel),
            "flow": len(FlowDirection),
            "momentum": len(MomentumDirection),
        }
        for col in self.one_hot_columns:
            if col in dataframe.columns:
                num_classes = fixed_class_counts.get(col, len(dataframe[col].unique()))
                ohe = self.dataframeUtils.one_hot_encode(dataframe[col], num_classes)
                for i in sorted(set([0, num_classes - 1])):
                    new_col = f"{col}_{i}"
                    dataframe[new_col] = ohe[:, i]
                    if not new_col in self.pre_normalized_columns:
                        self.pre_normalized_columns.append(new_col)
                    if not new_col in self.include_list:
                        self.include_list.append(new_col)

                if col in dataframe.columns:
                    dataframe = dataframe.drop(columns=[col])
                if col in self.include_list:
                    self.include_list.remove(col)
            else:
                print(f"    Warning: Column {col} not found in dataframe")
        return dataframe
    def get_normalized_size(self, dataframe: DataFrame):
        """Get the size of the normalized dataframe"""

        if self.use_pca_reduction and self.pca_n_components is not None:
            return len(self.pca_passthrough_columns) + self.pca_n_components

        norm_size = len(self.include_list)
        for col in self.one_hot_columns:
            one_hot_exists = f"{col}_0" in dataframe.columns
            original_exists = col in dataframe.columns

            if one_hot_exists:
                pass
            elif original_exists:
                norm_size += 2
            else:
                print(
                    f"WARNING: {col} not found in dataframe when calculating normalized size. "
                    f"This column should be added by add_additional_indicators() before get_normalized_size() is called."
                )
                norm_size += 2

        for col in self.include_list:
            if col not in dataframe.columns:
                print(f"WARNING: column {col} not found in dataframe")
                norm_size -= 1

        return norm_size
    def check_columns_included(self, required_columns: list, function_name: str):
        """Check that all required columns are in the include_list."""
        missing_columns = []
        for col in required_columns:
            if col not in self.include_list:
                missing_columns.append(col)

        if missing_columns:
            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: Missing columns in include_list for {function_name}()\n"
                f"{'='*80}\n"
                f"Missing columns: {missing_columns}\n"
                f"These columns will not be available in the normalized dataframe.\n"
                f"Please add them to self.include_list in BaseNNStrategy.\n"
                f"\nCurrent include_list: {self.include_list}\n"
                f"{'='*80}\n"
            )
            raise ValueError(error_msg)
    def rolling_dataframe_normalise(self, df: DataFrame):
        """Normalise the dataframe using the rolling robust scaler"""

        df_to_scale = df.copy()

        # process one-hot encoded columns (modifies lists so do first)
        df_to_scale = self.process_one_hot_columns(df_to_scale)

        # drop any columns that are not in the include list
        drop_list = []
        for col in df_to_scale.columns:
            if col not in self.include_list:
                drop_list.append(col)
        if len(drop_list) > 0:
            df_to_scale = df_to_scale.drop(columns=drop_list)

        # should be redundant, but just in case drop debug columns
        df_to_scale = self.dataframeUtils.remove_debug_columns(df_to_scale)

        # get list of columns not in pre-norm list:
        needs_norm_columns = [
            col for col in df_to_scale.columns if col not in self.pre_normalized_columns
        ]

        # Clean data BEFORE fitting scaler (replace NaNs and infs with 0)
        df_to_scale = df_to_scale.replace([np.inf, -np.inf], 0)
        df_to_scale = df_to_scale.fillna(0)

        # Use saved scaler or create new one
        if self.main_scaler is None:
            scaler_dir = self.get_storage_location()
            if scaler_exists(scaler_dir, self.main_scaler_name):
                self.main_scaler = load_scaler(scaler_dir, self.main_scaler_name)
                self.debug_print(f"    Main scaler loaded from {scaler_dir}")
                if self.pca_data_exists(scaler_dir, self.pca_name):
                    pca_payload = self.load_pca_data(scaler_dir, self.pca_name)
                    self.pca_components = pca_payload["components"]
                    self.pca_mean = pca_payload["mean"]
                    self.pca_n_components = pca_payload["n_components"]
                    self.pca_feature_columns = pca_payload.get("feature_columns")
                    self.pca_explained_variance = pca_payload.get("explained_variance")
                    self.pca_col_min = pca_payload.get("col_min")
                    self.pca_col_max = pca_payload.get("col_max")
                    self.debug_print(
                        f"    PCA data loaded: n_components={self.pca_n_components}"
                    )
            else:
                self.main_scaler = RobustScaler()

                self.debug_print(f"    Fitting columns:  {needs_norm_columns}")
                self.debug_print(f"    Skipping columns: {self.pre_normalized_columns}")

                if needs_norm_columns:
                    self.main_scaler.fit(df_to_scale[needs_norm_columns])
                    save_scaler(self.main_scaler, scaler_dir, self.main_scaler_name)
                    self.debug_print(
                        f"    Created and fitted main scaler with {len(needs_norm_columns)} features (skipping {len(self.pre_normalized_columns)} pre-normalized columns)"
                    )
                else:
                    self.debug_print(
                        "    No columns require fitting (all pre-normalized or one-hot)."
                    )

        # Transform using the fitted scaler. df_to_scale is already a local
        # copy of the caller's df (line above), so transform in place rather
        # than taking a second full-frame copy.
        df_scaled = df_to_scale
        df_scaled[needs_norm_columns] = self.main_scaler.transform(
            df_to_scale[needs_norm_columns]
        )

        # Clip extreme values to prevent NaN
        extreme_cols = []
        for col in df_scaled.columns:
            col_min = df_scaled[col].min()
            col_max = df_scaled[col].max()
            if col_min < -10 or col_max > 10:
                extreme_cols.append((col, float(col_min), float(col_max)))
        if extreme_cols:
            print(f"    Columns exceeding ±10 before clipping: {extreme_cols}")

        df_scaled = np.clip(df_scaled, -10, 10)

        if "date" in df_scaled.columns:
            raise ValueError(
                "ERROR: Date column found in scaled data. This indicates a bug in scaling."
            )

        df = df_scaled
        df = df.fillna(0)

        return df
    def normalise_for_gan(self, dataframe: np.ndarray | pd.DataFrame):
        """Performs selective minmax scaling suitable for use with a GAN"""

        if self.use_pca_reduction:
            return (
                dataframe.copy() if isinstance(dataframe, pd.DataFrame) else dataframe
            )

        if self.gan_scaler_a is None:
            scaler_dir = self.get_storage_location()
            if scaler_exists(scaler_dir, self.gan_scaler_a_name):
                self.gan_scaler_a = load_scaler(scaler_dir, self.gan_scaler_a_name)
                self.debug_print(f"    GAN scaler A loaded from {scaler_dir}")
                self.debug_print(f"      columns: {self.gan_scaler_a.feature_columns}")
                if not isinstance(dataframe, pd.DataFrame):
                    gan_df = pd.DataFrame(
                        dataframe, columns=self.gan_scaler_a.feature_columns
                    )
                else:
                    gan_df = dataframe.copy()
                non_norm_cols = [
                    col
                    for col in gan_df.columns
                    if col not in self.pre_normalized_columns
                ]

            else:
                if not isinstance(dataframe, pd.DataFrame):
                    raise ValueError("Dataframe must be a pandas DataFrame")

                gan_df = dataframe.copy()
                non_norm_cols = [
                    col
                    for col in gan_df.columns
                    if col not in self.pre_normalized_columns
                ]

                self.gan_scaler_a = MinMaxScaler(feature_range=(-1, 1))
                self.debug_print(f"    GAN Scaler A, fitting columns:  {non_norm_cols}")
                self.gan_scaler_a.fit(gan_df[non_norm_cols])
                self.gan_scaler_a.feature_columns = list(dataframe.columns)
                self.debug_print(f"      columns: {self.gan_scaler_a.feature_columns}")
                save_scaler(self.gan_scaler_a, scaler_dir, self.gan_scaler_a_name)
        else:
            if not isinstance(dataframe, pd.DataFrame):
                gan_df = pd.DataFrame(
                    dataframe, columns=self.gan_scaler_a.feature_columns
                )
            else:
                gan_df = dataframe.copy()

        non_norm_cols = [
            col for col in gan_df.columns if col not in self.pre_normalized_columns
        ]

        gan_df[non_norm_cols] = self.gan_scaler_a.transform(gan_df[non_norm_cols])

        return gan_df
    def denormalise_from_gan(self, gan_df: np.ndarray | pd.DataFrame):
        """Denormalise the dataframe from the GAN scaler"""

        if self.use_pca_reduction:
            return gan_df.copy() if isinstance(gan_df, pd.DataFrame) else gan_df

        if self.gan_scaler_a is None:
            raise ValueError("GAN scaler A is not loaded")

        if isinstance(gan_df, pd.DataFrame):
            dataframe = gan_df.copy()
        else:
            columns = self.gan_scaler_a.feature_columns
            dataframe = pd.DataFrame(gan_df, columns=columns)

        non_norm_cols = [
            col for col in dataframe.columns if col not in self.pre_normalized_columns
        ]

        dataframe[non_norm_cols] = self.gan_scaler_a.inverse_transform(
            dataframe[non_norm_cols]
        )

        return dataframe
    def scale_dataframe(self, dataframe: DataFrame):
        full_df_norm = self.rolling_dataframe_normalise(dataframe)
        if self.use_pca_reduction:
            full_df_norm = self.apply_pca(full_df_norm)
        return full_df_norm
    def clean_for_tensor(self, df: DataFrame) -> DataFrame:
        """Apply the same column filtering as rolling_dataframe_normalise but
        SKIP the scaling step. For the post-GAN scaling pipeline where the
        GAN sees raw values and a tensor scaler is applied downstream.

        Without this, df_to_tensor on a raw dataframe trips on non-numeric
        columns (date, object dtype indicators, debug columns) with a
        std::bad_cast in mx.array.
        """
        df_clean = df.copy()
        df_clean = self.process_one_hot_columns(df_clean)

        drop_list = [c for c in df_clean.columns if c not in self.include_list]
        if drop_list:
            df_clean = df_clean.drop(columns=drop_list)

        df_clean = self.dataframeUtils.remove_debug_columns(df_clean)
        df_clean = df_clean.replace([np.inf, -np.inf], 0)
        df_clean = df_clean.fillna(0)
        return df_clean
    def descale_dataframe(self, dataframe: DataFrame):
        return dataframe
    def apply_pca(self, norm_df: pd.DataFrame) -> pd.DataFrame:
        """Apply saved whitened PCA transform to a normalised dataframe."""
        if self.pca_components is None or self.pca_feature_columns is None:
            self.debug_print(
                "    apply_pca: no PCA data loaded, returning original dataframe"
            )
            return norm_df

        passthrough = [c for c in self.pca_passthrough_columns if c in norm_df.columns]
        result = norm_df[passthrough].copy()

        missing = [c for c in self.pca_feature_columns if c not in norm_df.columns]
        if missing:
            print(f"    apply_pca WARNING: missing columns for PCA: {missing}")
            return norm_df

        X = norm_df[self.pca_feature_columns].to_numpy()
        X_centered = X - self.pca_mean
        X_pca = X_centered @ self.pca_components.T

        if self.pca_explained_variance is not None:
            X_pca = X_pca / np.sqrt(self.pca_explained_variance)

        if self.pca_col_min is not None and self.pca_col_max is not None:
            col_range = self.pca_col_max - self.pca_col_min
            col_range[col_range == 0] = 1.0
            X_pca = 2.0 * (X_pca - self.pca_col_min) / col_range - 1.0
            X_pca = np.clip(X_pca, -1.0, 1.0)

        pca_cols = [f"pca_{i}" for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, index=norm_df.index, columns=pca_cols)
        result = pd.concat(
            [result.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1
        )
        result.index = norm_df.index

        self.debug_print(
            f"    apply_pca: {len(passthrough)} passthrough + {len(pca_cols)} PCA columns "
            f"= {result.shape[1]} total"
        )
        return result
