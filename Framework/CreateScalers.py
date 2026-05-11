# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
CreateScalers - creates and savesa dataframe scaler using data from all of the pairs in the whitelist.
The saved scaler can then be used in any strategy that needs to scale a dataframe, and should work for any pair.
Also, saves PCA data for the aggregate dataframes in case a strategy wants to use that
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
import pandas as pd
import traceback

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Add parent directory to path to import NNNC
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType

from utils.DataframePopulator import DataframePopulator, DatasetType

from utils.Scalers import scaler_exists, save_scaler, load_scaler, remove_scaler

class CreateScalers(BaseNNStrategy):

    pair_count = 0
    combined_df = None
    scaler_name = "main_scaler"

    def remove_old_scaler(self, name) -> None:
        scaler_dir = self.get_storage_location()
        if scaler_exists(scaler_dir, name):
            print(f"    Removing old scaler: {name}")
            remove_scaler(scaler_dir, name)

    def create_scalers(self, combined_df: DataFrame) -> None:
        print("    Creating main scaler using aggregate dataframe of all pairs")

        # add sequential index to dataframe
        combined_df = self.add_sequential_index(combined_df)

        # normalising the dataframe will handle columns appropriately and save the scaler
        norm_df = self.rolling_dataframe_normalise(combined_df)

        # Fit PCA on all numeric columns (both pre-normalized and RobustScaler-normalized)
        pca_cols = [
            c for c in norm_df.columns
            if norm_df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]
        if pca_cols:
            X = norm_df[pca_cols].to_numpy()
            print(f"    PCA input: {len(pca_cols)} columns, {X.shape[0]} rows")
            print(f"    PCA columns: {pca_cols}")
            pca_full = PCA(n_components=None)
            pca_full.fit(X)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            for i, (ev, cv) in enumerate(zip(pca_full.explained_variance_ratio_, cumvar)):
                print(f"      PC{i:3d}: var_ratio={ev:.6f}  cumulative={cv:.6f}")
            pca = PCA(n_components=0.99)
            pca.fit(X)
            # Compute whitened training data to get per-component min/max
            X_pca = pca.transform(X)
            X_whitened = X_pca / np.sqrt(pca.explained_variance_)
            col_min = X_whitened.min(axis=0).astype(np.float64)
            col_max = X_whitened.max(axis=0).astype(np.float64)
            BaseNNStrategy.save_pca_data(
                self.get_storage_location(),
                self.pca_name,
                pca.components_.astype(np.float64),
                pca.mean_.astype(np.float64),
                pca.n_components_,
                feature_columns=pca_cols,
                explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
                explained_variance=pca.explained_variance_.astype(np.float64),
                col_min=col_min,
                col_max=col_max,
            )
            print(f"    PCA fitted and saved: n_components={pca.n_components_} (99% variance, whitened)")
            print(f"    PCA per-component min/max saved for scaling to [-1, 1]")

        print("    Creating GAN scaler (minmax normalized)")
        minmax_df = self.normalise_for_gan(norm_df)

        # DEBUG: print range of each column:
        print()
        self.print_dataframe_ranges("Original Column Ranges", combined_df)
        print()
        self.print_dataframe_ranges("Normalised Column Ranges", norm_df)
        print()        
        self.print_dataframe_ranges("GAN Column Ranges", minmax_df)
        print()

        print()
        # print("    Examining full correlation matrix")
        # print()
        # self.examine_full_correlation_matrix(combined_df)
        print()
        print("    Examining correlation matrix for normalised dataframe")
        print()
        self.examine_full_correlation_matrix(norm_df)
        print()

        # run the inverse transform and check against the original

        print("\n\n    Checking inverse transform against original\n")

        # Calculate the difference matrix E = A - B
        A = norm_df.to_numpy()
        denorm_minmax_df = self.denormalise_from_gan(minmax_df)
        B = denorm_minmax_df.to_numpy()
        E = A - B

        # Calculate the Frobenius Norm of the difference matrix E
        frob_norm = np.linalg.norm(E, ord='fro')

        print(f"Difference Matrix E:\n{E}")
        norm_result = " (Good)" if frob_norm < 1e-6 else " (Bad)"
        print(f"Frobenius Norm ||E||_F: {frob_norm}{norm_result}\n\n")

        # Analyze trading signal correlations
        # Get trading labels from original dataframe
        print("    Getting training labels...\n")
        labels = self.get_training_labels(combined_df)

        print("\n\n    Analyzing trading signal correlations (Raw dataframe):\n")
        self.analyze_trading_signal_correlations(combined_df, labels, top_n=20)
        print("\n\n    Analyzing trading signal correlations (Normed dataframe):\n")
        self.analyze_trading_signal_correlations(norm_df, labels, top_n=20)

    def analyze_trading_signal_correlations(self, normalized_df: DataFrame, labels, top_n: int = 20):
        """
        Analyze correlations between normalized features and trading signals (buy/sell).
        Helps identify which features are most predictive of trading signals.
        
        Args:
            original_df: Raw dataframe (for generating trading signals)
            normalized_df: Normalized dataframe (for correlation analysis - what model sees)
            top_n: Number of top features to display
        """

        # Extract buy/sell signals from labels (TradingAction: SELL=0, HOLD=1, BUY=2)
        buy_signals = np.where(labels == TradingAction.BUY, 1.0, 0.0)
        sell_signals = np.where(labels == TradingAction.SELL, 1.0, 0.0)

        # Add signals to normalized dataframe (signals remain 0/1, not normalized)
        df = normalized_df.copy()
        df["_buy_signal"] = buy_signals
        df["_sell_signal"] = sell_signals

        # Get feature columns (exclude debug columns and signal columns)
        feature_cols = [col for col in df.columns 
                       if not col.startswith("%") 
                       and col not in ["_buy_signal", "_sell_signal"]]

        # Compute correlations
        buy_corrs = {}
        sell_corrs = {}

        for col in feature_cols:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Pearson correlation
                buy_corr = df[col].corr(df["_buy_signal"], method="pearson")
                sell_corr = df[col].corr(df["_sell_signal"], method="pearson")

                if not np.isnan(buy_corr):
                    buy_corrs[col] = buy_corr
                if not np.isnan(sell_corr):
                    sell_corrs[col] = sell_corr

        # Sort by absolute correlation
        buy_sorted = sorted(buy_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        sell_sorted = sorted(sell_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

        # Print results
        print("\n" + "="*80)
        print("TRADING SIGNAL FEATURE CORRELATION ANALYSIS")
        print("="*80)

        print(f"\nBuy Signal Statistics:")
        print(f"  Total buy signals: {df['_buy_signal'].sum():.0f} ({100*df['_buy_signal'].mean():.2f}%)")
        print(f"\nTop {top_n} Features Correlated with BUY Signals:")
        print(f"{'Feature':<40} {'Correlation':>12} {'Abs Corr':>12}")
        print("-" * 64)
        for col, corr in buy_sorted[:top_n]:
            print(f"{col:<40} {corr:>12.4f} {abs(corr):>12.4f}")

        print(f"\nSell Signal Statistics:")
        print(f"  Total sell signals: {df['_sell_signal'].sum():.0f} ({100*df['_sell_signal'].mean():.2f}%)")
        print(f"\nTop {top_n} Features Correlated with SELL Signals:")
        print(f"{'Feature':<40} {'Correlation':>12} {'Abs Corr':>12}")
        print("-" * 64)
        for col, corr in sell_sorted[:top_n]:
            print(f"{col:<40} {corr:>12.4f} {abs(corr):>12.4f}")

        # Summary
        print(f"\nSummary:")
        if buy_sorted:
            max_buy_corr = buy_sorted[0][1]
            print(f"  Highest buy correlation: {max_buy_corr:.4f} ({buy_sorted[0][0]})")
        if sell_sorted:
            max_sell_corr = sell_sorted[0][1]
            print(f"  Highest sell correlation: {max_sell_corr:.4f} ({sell_sorted[0][0]})")

        # Warning if correlations are weak
        if buy_sorted and abs(buy_sorted[0][1]) < 0.1:
            print(f"\n  ⚠️  WARNING: Very weak correlations (< 0.1) suggest features may not be predictive")
        elif buy_sorted and abs(buy_sorted[0][1]) < 0.2:
            print(f"\n  ⚠️  WARNING: Weak correlations (< 0.2) - features may have limited predictive power")
        else:
            print(f"\n  ✓ Correlations look reasonable")

        print("="*80 + "\n")

        return buy_corrs, sell_corrs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # this is the main freqtrade loop, which is fed dataframes pair, by pair

        whitelist = self.dp.current_whitelist()

        curr_pair = metadata["pair"]

        self.iteration_init()
        dataframe = self.check_precision_columns(dataframe)
        dataframe = self.dataframePopulator.add_indicators(
            dataframe, dataset_type=DatasetType.MINIMAL
            # dataframe, dataset_type=DatasetType.CUSTOM3
        )
        dataframe = self.add_additional_indicators(dataframe)

        if self.combined_df is None:
            self.remove_old_scaler(self.main_scaler_name)
            self.remove_old_scaler(self.gan_scaler_a_name)

            print(f"    Init with: {curr_pair}")
            self.combined_df = dataframe.reset_index(drop=True)
        else:

            print(f"    Appending: {curr_pair}")
            self.combined_df = pd.concat([self.combined_df, dataframe], ignore_index=True)

        self.pair_count += 1

        if self.pair_count == len(whitelist):
            combined = self.combined_df.reset_index(drop=True)
            self.create_scalers(combined)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
