# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
DebugCheckRedundancy - identifies redundant (highly correlated) technical indicators.
Compares all indicators in the normalized dataframe against required indicators (used in
populate_entry_trend, populate_exit_trend, and derived indicators like flow, momentum, regime, risk).
Identifies indicators that are highly correlated with required ones and could be dropped.
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Set, List, Tuple, Dict

from sklearn.decomposition import PCA

# Add parent directory to path to from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType


class DebugCheckRedundancy(BaseNNStrategy):
    """
    Strategy to identify redundant technical indicators by checking correlations
    with required indicators.
    """

    pair_count = 0
    combined_df = None

    def get_required_indicators(self) -> Set[str]:
        """
        Returns the set of indicators that MUST be kept because they are used in:
        - populate_entry_trend / populate_exit_trend
        - get_market_regime
        - get_risk_level
        - get_flow
        - get_momentum
        """
        required = set()

        # From populate_entry_trend (lines 2625-2628)
        required.add("close_zscore")
        required.add("guard_metric")
        required.add("adx")  # Raw ADX used in entry conditions
        required.add("adx_scaled")  # Also used in derived indicators
        required.add("bb_width")

        # From populate_exit_trend (lines 2687-2688)
        # close_zscore and guard_metric already added above

        # From get_market_regime (lines 1107-1126)
        required.add("close_zscore")  # Already added
        required.add("adx_scaled")  # Already added
        required.add("rsi_scaled")
        required.add("macd_zscore")

        # From get_risk_level (lines 1200-1227)
        required.add("atr_norm")
        required.add("adx_scaled")  # Already added
        # required.add("close")  # Raw close used for drawdown calculation
        # required.add("volume")  # Raw volume used for volume risk

        # From get_flow (lines 1275-1282)
        required.add("adx_scaled")  # Already added

        # From get_momentum (lines 1332-1338)
        required.add("aroonosc_scaled")

        return required

    def find_redundant_indicators(
        self, normalized_df: DataFrame, correlation_threshold: float = 0.9
    ) -> List[Tuple[str, str, float]]:
        """
        Find indicators that are highly correlated with required indicators.

        Args:
            normalized_df: Normalized dataframe with all indicators
            correlation_threshold: Minimum absolute correlation to consider redundant (default: 0.9)

        Returns:
            List of tuples: (redundant_indicator, required_indicator, correlation_value)
            Sorted by absolute correlation (highest first)
        """
        # Minimum samples for meaningful correlation
        # Statistical reasoning: Pearson correlation needs sufficient data points to be reliable.
        # With < 10 samples, correlations can be very unstable and misleading (e.g., 2 points
        # can give perfect correlation by chance). 10 is a conservative threshold that ensures
        # we have enough data for a meaningful correlation estimate.
        MIN_SAMPLES_FOR_CORRELATION = 10

        required = self.get_required_indicators()

        # Get all numeric columns from normalized dataframe
        numeric_cols = [
            col
            for col in normalized_df.columns
            if normalized_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            and not col.startswith("%")  # Exclude debug columns
        ]

        # Separate required and candidate indicators
        required_cols = [col for col in numeric_cols if col in required]
        candidate_cols = [col for col in numeric_cols if col not in required]

        if not required_cols:
            print("    WARNING: No required indicators found in dataframe!")
            return []

        if not candidate_cols:
            print("    INFO: No candidate indicators to check (all are required)")
            return []

        print(
            f"    Checking {len(candidate_cols)} candidate indicators "
            f"against {len(required_cols)} required indicators..."
        )

        redundant_list = []
        insufficient_samples_count = 0
        insufficient_samples_pairs = []

        # For each candidate indicator, check correlation with all required indicators
        for candidate in candidate_cols:
            if candidate not in normalized_df.columns:
                continue

            candidate_series = normalized_df[candidate]
            if candidate_series.isna().all():
                continue  # Skip if all NaN

            max_corr = 0.0
            best_match = None

            # Check correlation with each required indicator
            for required in required_cols:
                if required not in normalized_df.columns:
                    continue

                required_series = normalized_df[required]
                if required_series.isna().all():
                    continue  # Skip if all NaN

                # Calculate Pearson correlation
                try:
                    # Align indices and drop NaN pairs
                    aligned = pd.concat([candidate_series, required_series], axis=1).dropna()
                    if len(aligned) < MIN_SAMPLES_FOR_CORRELATION:
                        insufficient_samples_count += 1
                        if len(insufficient_samples_pairs) < 10:  # Track first 10 for reporting
                            insufficient_samples_pairs.append(
                                (candidate, required, len(aligned))
                            )
                        continue

                    # Explicitly use Pearson correlation (default, but making it clear)
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='pearson')
                    if pd.isna(corr):
                        continue

                    abs_corr = abs(corr)
                    if abs_corr > max_corr:
                        max_corr = abs_corr
                        best_match = required

                except Exception:
                    # Skip if correlation calculation fails
                    continue

            # If correlation exceeds threshold, mark as redundant
            if max_corr >= correlation_threshold and best_match is not None:
                redundant_list.append((candidate, best_match, max_corr))

        # Sort by absolute correlation (highest first)
        redundant_list.sort(key=lambda x: abs(x[2]), reverse=True)

        # Report insufficient samples if any
        if insufficient_samples_count > 0:
            print(
                f"\n    WARNING: {insufficient_samples_count} indicator pairs skipped "
                f"due to insufficient samples (< {MIN_SAMPLES_FOR_CORRELATION} samples)"
            )
            if insufficient_samples_pairs:
                print("    Examples (first 10):")
                for candidate, required, sample_count in insufficient_samples_pairs[:10]:
                    print(f"      {candidate} vs {required}: {sample_count} samples")

        return redundant_list

    def find_high_correlations(
        self, normalized_df: DataFrame, correlation_threshold: float = 0.75
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find all indicators with high correlations to required indicators.

        Args:
            normalized_df: Normalized dataframe with all indicators
            correlation_threshold: Minimum absolute correlation to report (default: 0.75)

        Returns:
            Dictionary mapping required indicator -> list of (candidate_indicator, correlation)
            Sorted by absolute correlation (highest first) for each required indicator
        """
        MIN_SAMPLES_FOR_CORRELATION = 10

        required = self.get_required_indicators()

        # Get all numeric columns from normalized dataframe
        numeric_cols = [
            col
            for col in normalized_df.columns
            if normalized_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            and not col.startswith("%")  # Exclude debug columns
        ]

        # Separate required and candidate indicators
        required_cols = [col for col in numeric_cols if col in required]
        candidate_cols = [col for col in numeric_cols if col not in required]

        # Dictionary to store correlations grouped by required indicator
        correlations_by_required: Dict[str, List[Tuple[str, float]]] = {
            req: [] for req in required_cols
        }

        # For each candidate indicator, check correlation with all required indicators
        for candidate in candidate_cols:
            if candidate not in normalized_df.columns:
                continue

            candidate_series = normalized_df[candidate]
            if candidate_series.isna().all():
                continue  # Skip if all NaN

            # Check correlation with each required indicator
            for required in required_cols:
                if required not in normalized_df.columns:
                    continue

                required_series = normalized_df[required]
                if required_series.isna().all():
                    continue  # Skip if all NaN

                # Calculate Pearson correlation
                try:
                    # Align indices and drop NaN pairs
                    aligned = pd.concat([candidate_series, required_series], axis=1).dropna()
                    if len(aligned) < MIN_SAMPLES_FOR_CORRELATION:
                        continue

                    # Explicitly use Pearson correlation
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='pearson')
                    if pd.isna(corr):
                        continue

                    # If correlation exceeds threshold, add to list
                    if abs(corr) >= correlation_threshold:
                        correlations_by_required[required].append((candidate, corr))

                except Exception:
                    # Skip if correlation calculation fails
                    continue

        # Sort each list by absolute correlation (highest first)
        for required in correlations_by_required:
            correlations_by_required[required].sort(key=lambda x: abs(x[1]), reverse=True)

        return correlations_by_required

    def find_high_spearman_correlations(
        self, normalized_df: DataFrame, correlation_threshold: float = 0.75
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find all indicators with high Spearman correlations to required indicators.
        Spearman captures monotonic relationships (non-linear but monotonic), robust to outliers.

        Args:
            normalized_df: Normalized dataframe with all indicators
            correlation_threshold: Minimum absolute correlation to report (default: 0.75)

        Returns:
            Dictionary mapping required indicator -> list of (candidate_indicator, correlation)
            Sorted by absolute correlation (highest first) for each required indicator
        """
        MIN_SAMPLES_FOR_CORRELATION = 10

        required = self.get_required_indicators()

        # Get all numeric columns from normalized dataframe
        numeric_cols = [
            col
            for col in normalized_df.columns
            if normalized_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            and not col.startswith("%")  # Exclude debug columns
        ]

        # Separate required and candidate indicators
        required_cols = [col for col in numeric_cols if col in required]
        candidate_cols = [col for col in numeric_cols if col not in required]

        # Dictionary to store correlations grouped by required indicator
        correlations_by_required: Dict[str, List[Tuple[str, float]]] = {
            req: [] for req in required_cols
        }

        # For each candidate indicator, check correlation with all required indicators
        for candidate in candidate_cols:
            if candidate not in normalized_df.columns:
                continue

            candidate_series = normalized_df[candidate]
            if candidate_series.isna().all():
                continue  # Skip if all NaN

            # Check correlation with each required indicator
            for required in required_cols:
                if required not in normalized_df.columns:
                    continue

                required_series = normalized_df[required]
                if required_series.isna().all():
                    continue  # Skip if all NaN

                # Calculate Spearman correlation
                try:
                    # Align indices and drop NaN pairs
                    aligned = pd.concat([candidate_series, required_series], axis=1).dropna()
                    if len(aligned) < MIN_SAMPLES_FOR_CORRELATION:
                        continue

                    # Use Spearman correlation (monotonic relationships)
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='spearman')
                    if pd.isna(corr):
                        continue

                    # If correlation exceeds threshold, add to list
                    if abs(corr) >= correlation_threshold:
                        correlations_by_required[required].append((candidate, corr))

                except Exception:
                    # Skip if correlation calculation fails
                    continue

        # Sort each list by absolute correlation (highest first)
        for required in correlations_by_required:
            correlations_by_required[required].sort(key=lambda x: abs(x[1]), reverse=True)

        return correlations_by_required

    def find_pca_redundancy(
        self, normalized_df: DataFrame, variance_threshold: float = 0.90
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find indicators that are redundant based on PCA analysis.
        Identifies candidate indicators that load heavily on the same principal components
        as required indicators, indicating they explain similar variance.

        Args:
            normalized_df: Normalized dataframe with all indicators
            variance_threshold: Fraction of variance to explain with PCA (default: 0.90)

        Returns:
            Dictionary mapping required indicator -> list of (candidate_indicator, loading_similarity)
            Sorted by similarity (highest first) for each required indicator
        """
        required = self.get_required_indicators()

        # Get all numeric columns from normalized dataframe
        numeric_cols = [
            col
            for col in normalized_df.columns
            if normalized_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            and not col.startswith("%")  # Exclude debug columns
        ]

        # Separate required and candidate indicators
        required_cols = [col for col in numeric_cols if col in required]
        candidate_cols = [col for col in numeric_cols if col not in required]

        if not required_cols or not candidate_cols:
            return {}

        # Prepare data for PCA (drop NaN rows)
        df_clean = normalized_df[numeric_cols].dropna()
        if len(df_clean) < 10:  # Need minimum samples
            return {}

        # Run PCA
        try:
            # Determine number of components to explain variance_threshold
            pca_full = PCA()
            pca_full.fit(df_clean.values)

            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.where(cumsum_var >= variance_threshold)[0]
            n_components = n_components[0] + 1 if len(n_components) > 0 else len(cumsum_var)

            # Run PCA with selected number of components
            pca = PCA(n_components=n_components)
            pca.fit(df_clean.values)

            # Get loadings (components) - shape: (n_components, n_features)
            loadings = pd.DataFrame(
                pca.components_.T,
                index=df_clean.columns.values,
                columns=[f"PC{i+1}" for i in range(n_components)]
            )

        except Exception as e:
            print(f"    WARNING: PCA analysis failed: {e}")
            return {}

        # Dictionary to store PCA-based redundancy
        pca_redundancy: Dict[str, List[Tuple[str, float]]] = {
            req: [] for req in required_cols if req in loadings.index
        }

        # For each required indicator, find candidates with similar loadings
        for required in required_cols:
            if required not in loadings.index:
                continue

            required_loadings = loadings.loc[required].values

            # Check each candidate indicator
            for candidate in candidate_cols:
                if candidate not in loadings.index:
                    continue

                candidate_loadings = loadings.loc[candidate].values

                # Calculate cosine similarity of loadings (how similar are their PCA contributions)
                # This measures if they contribute to variance in similar ways
                dot_product = np.dot(required_loadings, candidate_loadings)
                norm_required = np.linalg.norm(required_loadings)
                norm_candidate = np.linalg.norm(candidate_loadings)

                if norm_required > 0 and norm_candidate > 0:
                    similarity = dot_product / (norm_required * norm_candidate)
                    # Use absolute value to catch both positive and negative correlations
                    abs_similarity = abs(similarity)

                    # Threshold: 0.8 means they load similarly on principal components
                    if abs_similarity >= 0.8:
                        pca_redundancy[required].append((candidate, similarity))

        # Sort each list by absolute similarity (highest first)
        for required in pca_redundancy:
            pca_redundancy[required].sort(key=lambda x: abs(x[1]), reverse=True)

        return pca_redundancy

    def analyze_redundancy(self, normalized_df: DataFrame) -> None:
        """
        Main analysis function that identifies and reports redundant indicators.
        """
        print("\n" + "=" * 80)
        print("REDUNDANT INDICATOR ANALYSIS")
        print("=" * 80)

        # Get required indicators
        required = self.get_required_indicators()
        print(f"\nRequired Indicators ({len(required)}):")
        print("  " + ", ".join(sorted(required)))

        # Find and report high Pearson correlations (>= 0.75) for each required indicator
        print("\n" + "-" * 80)
        print("HIGH PEARSON CORRELATIONS TO REQUIRED INDICATORS (>= 0.75)")
        print("(Linear relationships)")
        print("-" * 80)
        high_correlations = self.find_high_correlations(normalized_df, correlation_threshold=0.75)

        for required_ind in sorted(high_correlations.keys()):
            correlations = high_correlations[required_ind]
            if correlations:
                print(f"\n{required_ind} ({len(correlations)} high correlations):")
                print(f"  {'Indicator':<40} {'Correlation':>12}")
                print("  " + "-" * 52)
                for candidate, corr in correlations:
                    print(f"  {candidate:<40} {corr:>12.4f}")
            else:
                print(f"\n{required_ind}: No high correlations found (>= 0.75)")

        # Find and report high Spearman correlations (>= 0.75) for each required indicator
        print("\n" + "-" * 80)
        print("HIGH SPEARMAN CORRELATIONS TO REQUIRED INDICATORS (>= 0.75)")
        print("(Monotonic relationships - captures non-linear but monotonic patterns)")
        print("-" * 80)
        high_spearman = self.find_high_spearman_correlations(normalized_df, correlation_threshold=0.75)

        for required_ind in sorted(high_spearman.keys()):
            correlations = high_spearman[required_ind]
            if correlations:
                print(f"\n{required_ind} ({len(correlations)} high Spearman correlations):")
                print(f"  {'Indicator':<40} {'Correlation':>12}")
                print("  " + "-" * 52)
                for candidate, corr in correlations:
                    print(f"  {candidate:<40} {corr:>12.4f}")
            else:
                print(f"\n{required_ind}: No high Spearman correlations found (>= 0.75)")

        # Find and report PCA-based redundancy
        print("\n" + "-" * 80)
        print("PCA-BASED REDUNDANCY (Loading Similarity >= 0.8)")
        print("(Indicators that explain similar variance in principal components)")
        print("-" * 80)
        pca_redundancy = self.find_pca_redundancy(normalized_df, variance_threshold=0.90)

        if pca_redundancy:
            for required_ind in sorted(pca_redundancy.keys()):
                redundancies = pca_redundancy[required_ind]
                if redundancies:
                    print(f"\n{required_ind} ({len(redundancies)} PCA-redundant indicators):")
                    print(f"  {'Indicator':<40} {'Similarity':>12}")
                    print("  " + "-" * 52)
                    for candidate, similarity in redundancies:
                        print(f"  {candidate:<40} {similarity:>12.4f}")
                else:
                    print(f"\n{required_ind}: No PCA-redundant indicators found")
        else:
            print("\n  PCA analysis not available (insufficient data or error)")

        # Find redundant indicators with different thresholds
        thresholds = [0.95, 0.9, 0.85]
        all_redundant = {}

        for threshold in thresholds:
            redundant = self.find_redundant_indicators(normalized_df, threshold)
            all_redundant[threshold] = redundant

        # Report results
        print("\n" + "-" * 80)
        print("REDUNDANT INDICATORS BY CORRELATION THRESHOLD")
        print("-" * 80)

        for threshold in thresholds:
            redundant = all_redundant[threshold]
            print(f"\nThreshold: {threshold:.2f} ({len(redundant)} redundant indicators)")

            if redundant:
                print(f"\n{'Redundant Indicator':<40} {'Correlates With':<30} {'Correlation':>12}")
                print("-" * 82)
                for candidate, required, corr in redundant:
                    print(f"{candidate:<40} {required:<30} {corr:>12.4f}")
            else:
                print("  No redundant indicators found at this threshold.")

        # Summary and recommendations
        print("\n" + "-" * 80)
        print("SUMMARY AND RECOMMENDATIONS")
        print("-" * 80)

        # Use 0.9 threshold for recommendations (balanced)
        redundant_90 = all_redundant[0.9]
        if redundant_90:
            redundant_names = sorted(set([r[0] for r in redundant_90]))
            print(f"\nRecommended removals (correlation >= 0.9): {len(redundant_names)} indicators")
            print("\nPython list of redundant indicators to remove:")
            print(f"redundant_indicators = {redundant_names}")

            print("\nPotential space savings:")
            print(f"  - {len(redundant_names)} indicators could be removed from include_list")
            print(f"  - This would reduce normalized dataframe size by ~{len(redundant_names)} columns")
        else:
            print("\nNo redundant indicators found at 0.9 threshold.")
            print("All indicators appear to provide unique information.")

        # Show indicators that are safe to keep (not redundant at any threshold)
        all_candidates = set()
        for redundant_list in all_redundant.values():
            all_candidates.update([r[0] for r in redundant_list])

        numeric_cols = [
            col
            for col in normalized_df.columns
            if normalized_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            and not col.startswith("%")
        ]
        safe_indicators = [
            col
            for col in numeric_cols
            if col not in required and col not in all_candidates
        ]

        if safe_indicators:
            print(f"\nSafe to keep (not redundant): {len(safe_indicators)} indicators")
            print("  " + ", ".join(sorted(safe_indicators)[:20]))  # Show first 20
            if len(safe_indicators) > 20:
                print(f"  ... and {len(safe_indicators) - 20} more")

        print("\n" + "=" * 80 + "\n")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Main freqtrade loop - collects dataframes from all pairs.
        """
        whitelist = self.dp.current_whitelist()
        curr_pair = metadata["pair"]

        self.iteration_init()
        dataframe = self.check_precision_columns(dataframe)
        dataframe = self.dataframePopulator.add_indicators(
            dataframe, dataset_type=DatasetType.MINIMAL
        )
        dataframe = self.add_additional_indicators(dataframe)

        if self.combined_df is None:
            print(f"    Init with: {curr_pair}")
            self.combined_df = dataframe.reset_index(drop=True)
        else:
            print(f"    Appending: {curr_pair}")
            self.combined_df = pd.concat([self.combined_df, dataframe], ignore_index=True)

        self.pair_count += 1

        # When all pairs processed, run redundancy analysis
        if self.pair_count == len(whitelist):
            combined = self.combined_df.reset_index(drop=True)
            print(f"\n    Processing {len(combined)} rows from {self.pair_count} pairs")

            # Normalize the dataframe (required indicators are in normalized space)
            print("    Normalizing dataframe...")
            normalized_df = self.rolling_dataframe_normalise(combined)

            # Run redundancy analysis
            self.analyze_redundancy(normalized_df)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """No-op for this debug strategy"""
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """No-op for this debug strategy"""
        return dataframe
