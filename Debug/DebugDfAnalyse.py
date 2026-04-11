# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
DebugDfAnalyse - Debug Dataframe Analysis

Collects dataframes, aggregates them and analyses for corellations and dependancies
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
import pandas as pd

import sklearn.decomposition as skd
from sklearn.feature_selection import (
    mutual_info_regression,
    mutual_info_classif,
    f_regression,
    f_classif,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import spearmanr

# Add parent directory to path to import NNNC
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import (
    BaseStrategy,
    ScalerType,
    MarketRegime,
    TradingAction,
    FlowDirection,
    MomentumDirection,
    RiskLevel,
    GANType,
)
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import (
    BaseStrategy,
    ScalerType,
    MarketRegime,
    TradingAction,
    FlowDirection,
    MomentumDirection,
    RiskLevel,
    GANType,
)
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import (
    BaseStrategy,
    ScalerType,
    MarketRegime,
    TradingAction,
    FlowDirection,
    MomentumDirection,
    RiskLevel,
    GANType,
)
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import (
    BaseStrategy,
    ScalerType,
    MarketRegime,
    TradingAction,
    FlowDirection,
    MomentumDirection,
    RiskLevel,
    GANType,
)
from utils.DataframePopulator import DataframePopulator, DatasetType


class DebugDfAnalyse(BaseNNStrategy):

    pair_count = 0
    combined_df = None
    combined_buys = None
    combined_sells = None
    feature_scores = {}  # Track ranks/scores for each feature across analyses
    composite_results = []  # Store final results for redundancy analysis

    def column_ranges(self, dataframe: DataFrame) -> None:
        print()
        self.print_dataframe_ranges("Original Column Ranges", dataframe)
        print()

    def clean_dataframe(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()
        ignore_list = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_count",
            "close_count",
            "high_count",
            "low_count",
            "max_count",
        ]
        drop_list = []
        for col in df.columns:
            # drop if in ignore_list
            if col in ignore_list:
                if col not in drop_list:
                    drop_list.append(col)

            if (
                col.startswith("open_")
                or col.startswith("high_")
                or col.startswith("low_")
                or col.startswith("close_")
                # or col.startswith("volume_")
                or col.startswith("bb_up")
                or col.startswith("bb_mid")
                or col.startswith("bb_low")
            ):
                if col not in drop_list:
                    drop_list.append(col)

            for other_col in df.columns:
                if other_col.startswith(col + "_"):
                    if col not in drop_list:
                        drop_list.append(col)

        if len(drop_list) > 0:
            print(f"    Dropping columns: {drop_list}")
            df = df.drop(columns=drop_list)

        if np.shape(df)[1] == 0:
            raise ValueError(
                "    No columns left in dataframe, returning empty dataframe"
            )

        return df

    def pearson_correlation(self, dataframe: DataFrame) -> None:
        print()
        print("    Examining Pearson correlation matrix for dataframe")
        print()
        self.examine_full_correlation_matrix(dataframe)
        print()

    def check_include_list_correlations(self, dataframe: DataFrame) -> None:
        """Check features in include_list for high correlations (redundancy detection)"""
        print()
        print("=" * 80)
        print("INCLUDE_LIST CORRELATION ANALYSIS")
        print("=" * 80)
        print()

        # df_clean = self.clean_dataframe(dataframe)
        df_clean = dataframe

        # Get include_list from parent class
        include_list = getattr(self, "include_list", [])

        if not include_list:
            print("    No include_list found in strategy class")
            return

        # Filter to only features that exist in dataframe
        include_list_present = [col for col in include_list if col in df_clean.columns]

        if not include_list_present:
            print(
                f"    None of the {len(include_list)} features in include_list are present in dataframe"
            )
            print(f"    include_list: {include_list}")
            return

        print(f"    Checking {len(include_list_present)} features from include_list")
        print(
            f"    ({len(include_list) - len(include_list_present)} features not found in dataframe)"
        )
        print()

        # Calculate correlation matrix
        all_features = include_list_present + [
            c for c in df_clean.columns if c not in include_list_present
        ]
        corr_matrix = df_clean[all_features].corr(method="pearson")

        # Check correlations within include_list
        print("--- High Correlations WITHIN include_list ---")
        print("(Features that might be redundant)")
        print()

        within_correlations = []
        for i, feat1 in enumerate(include_list_present):
            for feat2 in include_list_present[i + 1 :]:
                if feat1 in corr_matrix.index and feat2 in corr_matrix.columns:
                    corr_val = corr_matrix.loc[feat1, feat2]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                        within_correlations.append((feat1, feat2, corr_val))

        if within_correlations:
            print(
                f"Found {len(within_correlations)} highly correlated pairs (|r| > 0.7):"
            )
            for feat1, feat2, corr_val in sorted(
                within_correlations, key=lambda x: abs(x[2]), reverse=True
            ):
                print(f"  {feat1:30} <-> {feat2:30}  r = {corr_val:6.3f}")
        else:
            print("  No high correlations found within include_list features")
        print()

        # Check correlations between include_list and other features
        print("--- High Correlations: include_list features vs OTHER features ---")
        print("(Other features that might be redundant with include_list features)")
        print()

        other_features = [c for c in df_clean.columns if c not in include_list_present]
        cross_correlations = []

        for feat1 in include_list_present:
            for feat2 in other_features:
                if feat1 in corr_matrix.index and feat2 in corr_matrix.columns:
                    corr_val = corr_matrix.loc[feat1, feat2]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                        cross_correlations.append((feat1, feat2, corr_val))

        if cross_correlations:
            print(
                f"Found {len(cross_correlations)} highly correlated pairs (|r| > 0.7):"
            )
            # Group by include_list feature
            by_include_feat = {}
            for feat1, feat2, corr_val in cross_correlations:
                if feat1 not in by_include_feat:
                    by_include_feat[feat1] = []
                by_include_feat[feat1].append((feat2, corr_val))

            for feat1 in sorted(by_include_feat.keys()):
                print(f"\n  {feat1} is highly correlated with:")
                for feat2, corr_val in sorted(
                    by_include_feat[feat1], key=lambda x: abs(x[1]), reverse=True
                ):
                    print(f"    {feat2:30}  r = {corr_val:6.3f}")
        else:
            print(
                "  No high correlations found between include_list and other features"
            )
        print()

        print("=" * 80)
        print()

    def suggest_features_from_pca(self, dataframe: DataFrame) -> None:
        """Suggest features for inclusion based on PCA analysis"""
        print()
        print("=" * 80)
        print("PCA-BASED FEATURE SUGGESTIONS")
        print("=" * 80)
        print()
        # df_clean = self.clean_dataframe(dataframe)
        df_clean = dataframe

        # Get include_list from parent class
        include_list = getattr(self, "include_list", [])
        include_list_present = [col for col in include_list if col in df_clean.columns]

        # Run PCA
        try:
            pca = self.get_pca(df_clean)
        except Exception as e:
            print(f"    Error running PCA: {e}")
            return

        ratios = pca.explained_variance_ratio_
        loadings = pd.DataFrame(pca.components_.T, index=df_clean.columns.values)

        # Calculate importance scores
        # Method 1: Weighted sum of absolute loadings by variance ratio
        weighted_loadings = loadings.abs().mul(ratios, axis=1)
        weighted_scores = weighted_loadings.sum(axis=1).sort_values(ascending=False)

        # Method 2: Sum of absolute loadings for top N components (covering 90% variance)
        cumsum_var = np.cumsum(ratios)
        n_components_90 = np.where(cumsum_var >= 0.90)[0]
        n_components_90 = (
            n_components_90[0] + 1 if len(n_components_90) > 0 else len(ratios)
        )

        top_components_loadings = loadings.abs().iloc[:, :n_components_90]
        top_components_scores = top_components_loadings.sum(axis=1).sort_values(
            ascending=False
        )

        print(f"    Analyzing {len(df_clean.columns)} features")
        var_explained = cumsum_var[n_components_90 - 1] if n_components_90 > 0 else 0
        print(
            f"    Top {n_components_90} components explain {var_explained:.1%} of variance"
        )
        print()

        # Features currently in include_list
        print("--- Features Currently in include_list ---")
        if include_list_present:
            print(f"Count: {len(include_list_present)}")
            # Sort by weighted PCA score (descending) instead of column name
            scored_features = []
            for col in include_list_present:
                weighted_score = weighted_scores.get(col, 0.0)
                top_score = top_components_scores.get(col, 0.0)
                scored_features.append((col, weighted_score, top_score))

            for col, weighted_score, top_score in sorted(
                scored_features, key=lambda x: x[1], reverse=True
            ):
                print(
                    f"  {col:30}  weighted={weighted_score:.4f}  top_comp={top_score:.4f}"
                )
        else:
            print("  No features from include_list found in dataframe")
        print()

        # Features NOT in include_list but important according to PCA
        features_not_in_list = [
            col for col in df_clean.columns if col not in include_list_present
        ]

        print("--- Top Features NOT in include_list (by PCA importance) ---")
        print("(Consider adding these to include_list)")
        print()

        suggestions = []
        for col in features_not_in_list:
            weighted_score = weighted_scores.get(col, 0)
            top_score = top_components_scores.get(col, 0)
            suggestions.append((col, weighted_score, top_score))

        # Sort by weighted score
        suggestions.sort(key=lambda x: x[1], reverse=True)

        # Show top 30 suggestions
        print("Top 30 suggestions (by weighted PCA score):")
        for i, (col, weighted_score, top_score) in enumerate(suggestions[:30], 1):
            print(
                f"  {i:2d}. {col:30}  weighted={weighted_score:.4f}  top_comp={top_score:.4f}"
            )

        if len(suggestions) > 30:
            print(f"\n  ... and {len(suggestions) - 30} more features")
        print()

        # Compare: average score of include_list vs suggestions
        if include_list_present and suggestions:
            avg_include_score = np.mean(
                [weighted_scores.get(col, 0) for col in include_list_present]
            )
            avg_suggestion_score = np.mean(
                [s[1] for s in suggestions[: len(include_list_present)]]
            )

            print("--- Comparison ---")
            print("Average weighted PCA score:")
            print(f"  Current include_list features: {avg_include_score:.4f}")
            print(
                f"  Top {len(include_list_present)} suggestions: {avg_suggestion_score:.4f}"
            )
            print()

        # Features in include_list with low PCA importance (might consider removing)
        print("--- Features in include_list with LOW PCA importance ---")
        print("(Might consider removing if not needed for other reasons)")
        print()

        low_importance = []
        for col in include_list_present:
            weighted_score = weighted_scores.get(col, 0)
            if weighted_score < weighted_scores.median():
                low_importance.append((col, weighted_score))

        if low_importance:
            low_importance.sort(key=lambda x: x[1])
            print(f"Found {len(low_importance)} features with below-median importance:")
            for col, score in low_importance:
                print(f"  {col:30}  score={score:.4f}")
        else:
            print("  All include_list features have above-median importance")
        print()

        # Features that could be dropped if using 95% variance threshold
        print("--- Features that could be dropped (95% variance threshold) ---")
        print("(Highest loadings on variance beyond the first 95%; safe to drop first)")
        print()
        try:
            X_num = df_clean.select_dtypes(include=[np.number])
            n_feat = X_num.shape[1]
            if n_feat > 0:
                pca_full = skd.PCA(n_components=n_feat, svd_solver="full").fit(X_num)
                cumsum = np.cumsum(pca_full.explained_variance_ratio_)
                n_95 = int(np.argmax(cumsum >= 0.95)) + 1
                n_95 = min(n_95, n_feat)
                # Loadings: components_.T is (n_features, n_components)
                loadings = pca_full.components_.T
                drop_scores = np.sum(loadings[:, n_95:] ** 2, axis=1)
                names = X_num.columns.tolist()
                by_drop = sorted(
                    zip(names, drop_scores), key=lambda x: x[1], reverse=True
                )
                n_discard = n_feat - n_95
                print(
                    f"Retaining {n_95} components (95% variance); "
                    f"{n_discard} dimensions discarded."
                )
                print(
                    "Features to consider dropping first (by loading on discarded dims):"
                )
                for col, score in by_drop[: min(50, len(by_drop))]:
                    print(f"  {col:40}  drop_score={score:.6f}")
                if len(by_drop) > 50:
                    print(f"  ... and {len(by_drop) - 50} more")
            else:
                print("  No numeric columns to analyze.")
        except Exception as e:
            print(f"  Error computing 95% drop list: {e}")
        print()

        print("=" * 80)
        print()

        # Track PCA ranks for composite scoring
        self._track_pca_ranks(df_clean, weighted_scores, top_components_scores)

    # ----------------------------

    def _track_pca_ranks(
        self,
        dataframe: DataFrame,
        weighted_scores: pd.Series,
        top_components_scores: pd.Series,
    ) -> None:
        """Track ranks from PCA analysis for composite scoring"""
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()

        # Initialize feature scores if not present
        for col in numeric_cols:
            if col not in self.feature_scores:
                self.feature_scores[col] = {}

        # Get ranks (lower rank = better score)
        weighted_ranked = weighted_scores.sort_values(ascending=False)
        top_comp_ranked = top_components_scores.sort_values(ascending=False)

        weighted_ranks = {
            col: rank + 1 for rank, col in enumerate(weighted_ranked.index)
        }
        top_comp_ranks = {
            col: rank + 1 for rank, col in enumerate(top_comp_ranked.index)
        }

        # Store ranks
        for col in numeric_cols:
            self.feature_scores[col]["pca_weighted_rank"] = weighted_ranks.get(
                col, len(numeric_cols)
            )
            self.feature_scores[col]["pca_top_comp_rank"] = top_comp_ranks.get(
                col, len(numeric_cols)
            )

    # ----------------------------

    def composite_feature_ranking(self, dataframe: DataFrame) -> None:
        """Generate composite ranking based on buy/sell correlation and PCA analyses"""
        print()
        print("=" * 80)
        print("COMPOSITE FEATURE RANKING")
        print("=" * 80)
        print()

        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()

        # First pass: calculate raw scores for all features
        max_rank = len(numeric_cols)
        raw_correlation_scores = []
        raw_pca_scores = []
        feature_data = []

        for col in numeric_cols:
            if col not in self.feature_scores:
                continue

            scores = self.feature_scores[col]

            # Calculate raw correlation score (weighted average: buy 60%, sell 40%)
            # Buy correlations are more important for trading signals
            buy_pearson_score = 1.0 / scores.get("buy_pearson_rank", max_rank)
            sell_pearson_score = 1.0 / scores.get("sell_pearson_rank", max_rank)
            buy_spearman_score = 1.0 / scores.get("buy_spearman_rank", max_rank)
            sell_spearman_score = 1.0 / scores.get("sell_spearman_rank", max_rank)
            buy_mi_score = 1.0 / scores.get("buy_mi_rank", max_rank)
            sell_mi_score = 1.0 / scores.get("sell_mi_rank", max_rank)

            buy_avg = (buy_pearson_score + buy_spearman_score + buy_mi_score) / 3.0
            sell_avg = (sell_pearson_score + sell_spearman_score + sell_mi_score) / 3.0
            correlation_score_raw = buy_avg * 0.6 + sell_avg * 0.4  # 60% buy, 40% sell

            # Calculate raw PCA score
            pca_weighted_score = 1.0 / scores.get("pca_weighted_rank", max_rank)
            pca_top_comp_score = 1.0 / scores.get("pca_top_comp_rank", max_rank)
            pca_score_raw = (pca_weighted_score + pca_top_comp_score) / 2.0

            raw_correlation_scores.append(correlation_score_raw)
            raw_pca_scores.append(pca_score_raw)
            feature_data.append(
                {
                    "feature": col,
                    "correlation_score_raw": correlation_score_raw,
                    "pca_score_raw": pca_score_raw,
                    "buy_pearson_rank": scores.get("buy_pearson_rank", max_rank),
                    "sell_pearson_rank": scores.get("sell_pearson_rank", max_rank),
                    "buy_mi_rank": scores.get("buy_mi_rank", max_rank),
                    "sell_mi_rank": scores.get("sell_mi_rank", max_rank),
                    "pca_weighted_rank": scores.get("pca_weighted_rank", max_rank),
                }
            )

        # Normalize scores to [0, 1] scale using min-max normalization
        if len(raw_correlation_scores) > 0:
            corr_min = min(raw_correlation_scores)
            corr_max = max(raw_correlation_scores)
            corr_range = corr_max - corr_min if corr_max > corr_min else 1.0

            pca_min = min(raw_pca_scores)
            pca_max = max(raw_pca_scores)
            pca_range = pca_max - pca_min if pca_max > pca_min else 1.0

            # Second pass: normalize and combine scores
            # Composite ranking uses only buy/sell correlation (PCA does not correlate to signals)
            composite_scores = []
            for i, data in enumerate(feature_data):
                # Normalize to [0, 1]
                correlation_score_norm = (
                    data["correlation_score_raw"] - corr_min
                ) / corr_range
                pca_score_norm = (data["pca_score_raw"] - pca_min) / pca_range

                # Composite score = correlation only (PCA excluded from ranking)
                composite_score = correlation_score_norm

                composite_scores.append(
                    {
                        "feature": data["feature"],
                        "composite_score": composite_score,
                        "correlation_score": correlation_score_norm,
                        "pca_score": pca_score_norm,
                        "buy_pearson_rank": data["buy_pearson_rank"],
                        "sell_pearson_rank": data["sell_pearson_rank"],
                        "buy_mi_rank": data["buy_mi_rank"],
                        "sell_mi_rank": data["sell_mi_rank"],
                        "pca_weighted_rank": data["pca_weighted_rank"],
                    }
                )
        else:
            composite_scores = []

        # Sort by composite score (descending)
        composite_scores.sort(key=lambda x: x["composite_score"], reverse=True)

        # Store for redundancy recommendations (recommend_redundancy_drops reads this)
        self.composite_results = composite_scores

        print("Top 50 features by composite ranking:")
        print(
            "(Buy/sell correlation only; PCA score shown for reference, not used in ranking)"
        )
        print()
        header = (
            f"{'Rank':<6} {'Feature':<30} {'Composite':<12} {'Corr':<12} {'PCA(ref)':<12} "
            f"{'Buy_P':<8} {'Sell_P':<8} {'Buy_MI':<8} {'Sell_MI':<8} {'PCA_W':<8}"
        )
        print(header)
        print("-" * 120)

        for rank, item in enumerate(composite_scores[:50], 1):
            print(
                f"{rank:<6} "
                f"{item['feature']:<30} "
                f"{item['composite_score']:<12.6f} "
                f"{item['correlation_score']:<12.6f} "
                f"{item['pca_score']:<12.6f} "
                f"{item['buy_pearson_rank']:<8} "
                f"{item['sell_pearson_rank']:<8} "
                f"{item['buy_mi_rank']:<8} "
                f"{item['sell_mi_rank']:<8} "
                f"{item['pca_weighted_rank']:<8}"
            )

        print()

        # Check include_list features
        include_list = getattr(self, "include_list", [])
        include_list_present = [col for col in include_list if col in numeric_cols]

        if include_list_present:
            print("--- include_list Features: Composite Rankings ---")
            print()
            include_scores = [
                item
                for item in composite_scores
                if item["feature"] in include_list_present
            ]
            include_scores.sort(key=lambda x: x["composite_score"], reverse=True)

            for rank, item in enumerate(include_scores, 1):
                overall_rank = next(
                    (
                        i + 1
                        for i, s in enumerate(composite_scores)
                        if s["feature"] == item["feature"]
                    ),
                    len(composite_scores),
                )
                print(
                    f"  Overall Rank {overall_rank:3d}: {item['feature']:<30} "
                    f"score={item['composite_score']:.6f} "
                    f"(corr={item['correlation_score']:.6f}, pca_ref={item['pca_score']:.6f})"
                )
            print()

        print("=" * 80)
        print()

    def spearman_correlation(self, dataframe: DataFrame) -> None:
        """Spearman correlation (monotonic relationships, robust to outliers)"""
        print()
        print("    Examining Spearman correlation matrix for dataframe")
        print()
        corr_matrix = dataframe.corr(method="spearman")
        print(corr_matrix.round(3))

        # Find high correlations
        abs_corr = corr_matrix.abs()
        high_corr_pairs = []
        for i in range(len(abs_corr.columns)):
            for j in range(i + 1, len(abs_corr.columns)):
                if abs_corr.iloc[i, j] > 0.7:
                    high_corr_pairs.append(
                        (abs_corr.columns[i], abs_corr.columns[j], abs_corr.iloc[i, j])
                    )

        if high_corr_pairs:
            print("\n--- Highly Correlated Feature Pairs (Spearman, |r| > 0.7) ---")
            for feat1, feat2, corr_val in sorted(
                high_corr_pairs, key=lambda x: x[2], reverse=True
            ):
                print(f"{feat1:30} {feat2:30} {corr_val:.3f}")
        print()

    def mutual_information_analysis(
        self, dataframe: DataFrame, target_col: str = None
    ) -> None:
        """Mutual Information analysis (captures non-linear relationships)"""
        print()
        print("    Examining Mutual Information scores")
        print()

        # df_clean = self.clean_dataframe(dataframe)
        df_clean = dataframe

        # Remove non-numeric columns and handle NaN
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_clean[numeric_cols].fillna(0)

        if target_col and target_col in df_numeric.columns:
            X = df_numeric.drop(columns=[target_col])
            y = df_numeric[target_col]

            # Determine if classification or regression
            unique_vals = len(y.unique())
            if unique_vals < 20:  # Likely classification
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:  # Likely regression
                mi_scores = mutual_info_regression(X, y, random_state=42)

            mi_df = pd.DataFrame(
                {"feature": X.columns, "mutual_info": mi_scores}
            ).sort_values("mutual_info", ascending=False)

            print("--- Mutual Information Scores (vs target) ---")
            print(mi_df.head(20))
        else:
            print("    No target column specified, skipping MI analysis")
        print()

    def vif_analysis(self, dataframe: DataFrame) -> None:
        """Variance Inflation Factor - detects multicollinearity"""
        print()
        print("    Examining Variance Inflation Factors (VIF)")
        print()

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            print("    statsmodels not available, skipping VIF analysis")
            return

        # df_clean = self.clean_dataframe(dataframe)
        df_clean = dataframe
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_clean[numeric_cols].fillna(0)

        # Remove constant columns
        df_numeric = df_numeric.loc[:, (df_numeric != df_numeric.iloc[0]).any()]

        if df_numeric.shape[1] == 0:
            print("    No valid numeric columns for VIF analysis")
            return

        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df_numeric.columns
        vif_data["VIF"] = [
            variance_inflation_factor(df_numeric.values, i)
            for i in range(df_numeric.shape[1])
        ]
        vif_data = vif_data.sort_values("VIF", ascending=False)

        print("--- Variance Inflation Factors ---")
        print("VIF > 5 suggests moderate multicollinearity")
        print("VIF > 10 suggests high multicollinearity")
        print()
        print(vif_data)

        high_vif = vif_data[vif_data["VIF"] > 5]
        if len(high_vif) > 0:
            print(
                f"\n    WARNING: {len(high_vif)} features with VIF > 5 (multicollinearity):"
            )
            print(high_vif[["Feature", "VIF"]])
        print()

    def feature_importance_analysis(
        self, dataframe: DataFrame, target_col: str = None
    ) -> None:
        """Feature importance from Random Forest"""
        print()
        print("    Examining Feature Importance (Random Forest)")
        print()

        # df_clean = self.clean_dataframe(dataframe)
        df_clean = dataframe
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_clean[numeric_cols].fillna(0)

        if target_col and target_col in df_numeric.columns:
            X = df_numeric.drop(columns=[target_col])
            y = df_numeric[target_col]

            # Determine if classification or regression
            unique_vals = len(y.unique())
            if unique_vals < 20:  # Classification
                rf = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            else:  # Regression
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            rf.fit(X, y)

            importance_df = pd.DataFrame(
                {"feature": X.columns, "importance": rf.feature_importances_}
            ).sort_values("importance", ascending=False)

            print("--- Random Forest Feature Importance ---")
            print(importance_df.head(20))
        else:
            print("    No target column specified, skipping feature importance")
        print()

    def univariate_feature_selection(
        self, dataframe: DataFrame, target_col: str = None
    ) -> None:
        """F-statistic for univariate feature selection"""
        print()
        print("    Examining Univariate Feature Selection (F-statistic)")
        print()

        # df_clean = self.clean_dataframe(dataframe)
        df_clean = dataframe
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_clean[numeric_cols].fillna(0)

        if target_col and target_col in df_numeric.columns:
            X = df_numeric.drop(columns=[target_col])
            y = df_numeric[target_col]

            # Determine if classification or regression
            unique_vals = len(y.unique())
            if unique_vals < 20:  # Classification
                f_scores, p_values = f_classif(X, y)
            else:  # Regression
                f_scores, p_values = f_regression(X, y)

            fstat_df = pd.DataFrame(
                {"feature": X.columns, "f_score": f_scores, "p_value": p_values}
            ).sort_values("f_score", ascending=False)

            print("--- F-Statistic Scores ---")
            print("Higher F-score = stronger relationship with target")
            print(fstat_df.head(20))
        else:
            print("    No target column specified, skipping F-statistic analysis")
        print()

    def normalization_recommendations(self, dataframe: DataFrame) -> None:
        """Provide recommendations on which variables should be normalized"""
        print()
        print("=" * 80)
        print("NORMALIZATION RECOMMENDATIONS")
        print("=" * 80)
        print()

        # df_clean = self.clean_dataframe(dataframe)
        df_clean = dataframe
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        # Categories of features
        should_normalize = []  # Features that should be normalized
        pre_normalized = []  # Features already normalized (z-score, scaled, etc.)
        exclude_from_normalization = []  # Features that shouldn't be normalized

        for col in numeric_cols:
            col_lower = col.lower()
            values = df_clean[col].dropna()

            # Check if already normalized (has _scaled, _zscore, _norm in name, or is in [-1, 1] range)
            if any(
                x in col_lower for x in ["_scaled", "_zscore", "_norm", "_normalized"]
            ):
                pre_normalized.append(col)
            elif len(values) > 0:
                value_range = values.max() - values.min()

                # Check if already in normalized range (roughly [-1, 1] or [0, 1])
                if (
                    value_range < 3.0
                    and abs(values.max()) < 2.0
                    and abs(values.min()) < 2.0
                ):
                    pre_normalized.append(col)
                # Check if it's a ratio/percentage (already normalized conceptually)
                elif value_range < 2.0 and abs(values.max()) <= 1.0:
                    pre_normalized.append(col)
                # Check if it's a one-hot encoded feature (only 0/1)
                elif set(values.unique()).issubset({0, 1}):
                    exclude_from_normalization.append(col)
                # Check if it's a categorical encoding (small integer range)
                elif value_range < 10 and values.dtype in [np.int8, np.int16, np.int32]:
                    exclude_from_normalization.append(col)
                else:
                    should_normalize.append(col)

        print("--- Features Already Normalized (include in pre_normalized_columns) ---")
        print(f"Count: {len(pre_normalized)}")
        for col in sorted(pre_normalized):
            print(f"  {col}")
        print()

        print(
            "--- Features to Normalize (include in include_list for normalization) ---"
        )
        print(f"Count: {len(should_normalize)}")
        for col in sorted(should_normalize)[:30]:  # Show first 30
            print(f"  {col}")
        if len(should_normalize) > 30:
            print(f"  ... and {len(should_normalize) - 30} more")
        print()

        print("--- Features to Exclude from Normalization ---")
        print(f"Count: {len(exclude_from_normalization)}")
        for col in sorted(exclude_from_normalization):
            print(f"  {col}")
        print()

        # Summary statistics
        print("--- Summary Statistics ---")
        print(f"Total numeric features: {len(numeric_cols)}")
        pct_pre = (
            100 * len(pre_normalized) / len(numeric_cols)
            if len(numeric_cols) > 0
            else 0
        )
        pct_norm = (
            100 * len(should_normalize) / len(numeric_cols)
            if len(numeric_cols) > 0
            else 0
        )
        pct_excl = (
            100 * len(exclude_from_normalization) / len(numeric_cols)
            if len(numeric_cols) > 0
            else 0
        )
        print(f"  - Already normalized: {len(pre_normalized)} ({pct_pre:.1f}%)")
        print(f"  - Should normalize: {len(should_normalize)} ({pct_norm:.1f}%)")
        print(
            f"  - Exclude from normalization: {len(exclude_from_normalization)} ({pct_excl:.1f}%)"
        )
        print()

        print("=" * 80)
        print()

    # get the PCA model for the supplied dataframe (dataframe must be normalised)
    def get_pca(self, df: DataFrame):

        ncols = df.shape[1]  # allow all components to get the full variance matrix
        whiten = True

        pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver="full").fit(df)
        var_ratios = pca.explained_variance_ratio_

        # if self.dbg_verbose:
        #     print ("PCA variance_ratio: ", pca.explained_variance_ratio_)

        # scan variance and only take if column contributes >x%
        ncols = 0
        var_sum = 0.0
        variance_threshold = 0.999
        # variance_threshold = 0.99
        while (var_sum < variance_threshold) & (ncols < len(var_ratios)):
            var_sum = var_sum + var_ratios[ncols]
            ncols = ncols + 1

        # if necessary, re-calculate pca with reduced column set
        if ncols != df.shape[1]:
            # pca = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=True).fit(df_norm)
            pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver="full").fit(df)

        return pca

    # does a quick for suspicious values. Separate func because we always want to call this
    def check_pca(self, pca, df):

        ratios = pca.explained_variance_ratio_
        loadings = pd.DataFrame(pca.components_.T, index=df.columns.values)

        # check variance ratios
        var_big = np.where(ratios >= 0.5)[0]
        if len(var_big) > 0:
            # print("    !!! high variance in columns: ", var_big)
            print("    !!! high variance in columns: ", df.columns.values[var_big])
            # print("    !!! variances: ", ratios)

        var_0 = np.where(ratios == 0)[0]
        if len(var_0) > 0:
            print("    !!! zero variance in columns: ", var_0)

        # check PCA rows
        inf_rows = loadings[(np.isinf(loadings)).any(axis=1)].index.values.tolist()

        if len(inf_rows) > 0:
            print("    !!! inf values in rows: ", inf_rows)

        na_rows = loadings[loadings.isna().any(axis=1)].index.values.tolist()
        if len(na_rows) > 0:
            print("    !!! na values in rows: ", na_rows)

        zero_rows = loadings[(loadings == 0).any(axis=1)].index.values.tolist()
        if len(zero_rows) > 0:
            print("    !!! No contribution from indicator(s) - remove ?! : ", zero_rows)

        return

    def analyse_pca(self, pca, df):
        print("")
        print("Variance Ratios:")
        ratios = pca.explained_variance_ratio_
        print(ratios)
        print("")

        # print matrix of weightings for selected components
        loadings = pd.DataFrame(pca.components_.T, index=df.columns.values)

        l2 = loadings.abs()
        l3 = loadings.mul(ratios)
        ranks = loadings.rank()

        loadings["Score"] = l2.sum(axis=1)
        loadings["Score0"] = loadings[loadings.columns.values[0]].abs()
        loadings["Rank"] = loadings["Score"].rank(ascending=False)
        loadings["Rank0"] = loadings["Score0"].rank(ascending=False)
        print("Loadings, by PC0:")
        print(loadings.sort_values("Rank0").head(n=30))
        print("")
        # print("Loadings, by All Columns:")
        # print(loadings.sort_values('Rank').head(n=30))
        # print("")

        # weighted by variance ratios
        l3a = l3.abs()
        l3["Score"] = l3a.sum(axis=1)
        l3["Rank"] = loadings["Score"].rank(ascending=False)
        print("Loadings, Weighted by Variance Ratio")
        print(l3.sort_values("Rank").head(n=20))

        # # rankings per column
        ranks["Score"] = ranks.sum(axis=1)
        ranks["Rank"] = ranks["Score"].rank(ascending=True)
        print("Rankings per column")
        print(ranks.sort_values("Rank", ascending=True).head(n=30))

        # print(loadings.head())
        # print(l3.head())

    def pca_analysis(self, dataframe: DataFrame) -> None:
        print()
        # df_clean = self.clean_dataframe(dataframe)
        df = dataframe
        pca = self.get_pca(df)
        self.check_pca(pca, df)
        self.analyse_pca(pca, df)

    def analyse_buy_sell_correlations(
        self, dataframe: DataFrame, buys: np.ndarray, sells: np.ndarray
    ) -> None:
        """Analyze correlations between buy/sell signals and dataframe features"""
        print()
        print("=" * 80)
        print("BUY/SELL SIGNAL CORRELATION ANALYSIS")
        print("=" * 80)
        print()

        # df_clean = self.clean_dataframe(dataframe)
        df_clean = dataframe

        # Ensure buys and sells are the same length as dataframe
        min_len = min(len(df_clean), len(buys), len(sells))
        df_clean = df_clean.iloc[:min_len]
        buys = buys[:min_len]
        sells = sells[:min_len]

        # Convert to binary if needed (handle both 0/1 and boolean)
        buys_binary = (
            np.where(buys > 0.5, 1, 0) if buys.dtype != bool else buys.astype(int)
        )
        sells_binary = (
            np.where(sells > 0.5, 1, 0) if sells.dtype != bool else sells.astype(int)
        )

        # Get numeric columns only
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_clean[numeric_cols].fillna(0)

        print(f"    Analyzing {len(numeric_cols)} features against buy/sell signals")
        print(
            f"    Buy signals: {np.sum(buys_binary)} ({100*np.mean(buys_binary):.1f}%)"
        )
        print(
            f"    Sell signals: {np.sum(sells_binary)} ({100*np.mean(sells_binary):.1f}%)"
        )
        print()

        # Pearson correlation
        print("--- Pearson Correlation with Buy Signals ---")
        buy_correlations = []
        for col in numeric_cols:
            if col in df_numeric.columns:
                corr_val = np.corrcoef(df_numeric[col].values, buys_binary)[0, 1]
                if not np.isnan(corr_val):
                    buy_correlations.append((col, corr_val))

        buy_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 30 features (by absolute correlation):")
        for i, (col, corr_val) in enumerate(buy_correlations[:30], 1):
            direction = "↑" if corr_val > 0 else "↓"
            print(f"  {i:2d}. {col:30}  r = {corr_val:7.4f} {direction}")
        print()

        print("--- Pearson Correlation with Sell Signals ---")
        sell_correlations = []
        for col in numeric_cols:
            if col in df_numeric.columns:
                corr_val = np.corrcoef(df_numeric[col].values, sells_binary)[0, 1]
                if not np.isnan(corr_val):
                    sell_correlations.append((col, corr_val))

        sell_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 30 features (by absolute correlation):")
        for i, (col, corr_val) in enumerate(sell_correlations[:30], 1):
            direction = "↑" if corr_val > 0 else "↓"
            print(f"  {i:2d}. {col:30}  r = {corr_val:7.4f} {direction}")
        print()

        # Spearman correlation (monotonic relationships)
        print("--- Spearman Correlation with Buy Signals ---")
        buy_spearman = []
        for col in numeric_cols:
            if col in df_numeric.columns:
                corr_val, p_val = spearmanr(df_numeric[col].values, buys_binary)
                if not np.isnan(corr_val):
                    buy_spearman.append((col, corr_val, p_val))

        buy_spearman.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 30 features (by absolute correlation):")
        for i, (col, corr_val, p_val) in enumerate(buy_spearman[:30], 1):
            direction = "↑" if corr_val > 0 else "↓"
            sig = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            )
            print(
                f"  {i:2d}. {col:30}  r = {corr_val:7.4f} {direction}  p={p_val:.4f} {sig}"
            )
        print()

        print("--- Spearman Correlation with Sell Signals ---")
        sell_spearman = []
        for col in numeric_cols:
            if col in df_numeric.columns:
                corr_val, p_val = spearmanr(df_numeric[col].values, sells_binary)
                if not np.isnan(corr_val):
                    sell_spearman.append((col, corr_val, p_val))

        sell_spearman.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 30 features (by absolute correlation):")
        for i, (col, corr_val, p_val) in enumerate(sell_spearman[:30], 1):
            direction = "↑" if corr_val > 0 else "↓"
            sig = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            )
            print(
                f"  {i:2d}. {col:30}  r = {corr_val:7.4f} {direction}  p={p_val:.4f} {sig}"
            )
        print()

        # Mutual Information (non-linear relationships)
        print("--- Mutual Information with Buy/Sell Signals ---")
        mi_buy_df = None
        mi_sell_df = None
        try:

            X = df_numeric.values
            mi_buy = mutual_info_classif(X, buys_binary, random_state=42)
            mi_sell = mutual_info_classif(X, sells_binary, random_state=42)

            mi_buy_df = pd.DataFrame(
                {"feature": numeric_cols, "mi_buy": mi_buy}
            ).sort_values("mi_buy", ascending=False)

            mi_sell_df = pd.DataFrame(
                {"feature": numeric_cols, "mi_sell": mi_sell}
            ).sort_values("mi_sell", ascending=False)

            print("Top 30 features for Buy signals (by Mutual Information):")
            for i, row in enumerate(mi_buy_df.head(30).itertuples(), 1):
                print(f"  {i:2d}. {row.feature:30}  MI = {row.mi_buy:.4f}")
            print()

            print("Top 30 features for Sell signals (by Mutual Information):")
            for i, row in enumerate(mi_sell_df.head(30).itertuples(), 1):
                print(f"  {i:2d}. {row.feature:30}  MI = {row.mi_sell:.4f}")
            print()

        except Exception as e:
            print(f"    Error calculating Mutual Information: {e}")
            print()

        # Compare: features highly correlated with buys but not sells (and vice versa)
        print("--- Feature Comparison: Buy vs Sell Correlations ---")
        print("Features that correlate differently with buys vs sells:")
        print()

        # Create dictionaries for quick lookup
        buy_corr_dict = {col: corr for col, corr in buy_correlations}
        sell_corr_dict = {col: corr for col, corr in sell_correlations}

        # Find features with strong buy correlation but weak sell correlation
        buy_specific = []
        for col, buy_corr in buy_correlations[:50]:  # Top 50 buy correlations
            sell_corr = sell_corr_dict.get(col, 0)
            diff = abs(buy_corr) - abs(sell_corr)
            if diff > 0.1:  # At least 0.1 difference
                buy_specific.append((col, buy_corr, sell_corr, diff))

        sell_specific = []
        for col, sell_corr in sell_correlations[:50]:  # Top 50 sell correlations
            buy_corr = buy_corr_dict.get(col, 0)
            diff = abs(sell_corr) - abs(buy_corr)
            if diff > 0.1:  # At least 0.1 difference
                sell_specific.append((col, buy_corr, sell_corr, diff))

        if buy_specific:
            print("Features more correlated with BUY signals:")
            buy_specific.sort(key=lambda x: x[3], reverse=True)
            for col, buy_corr, sell_corr, diff in buy_specific[:20]:
                print(
                    f"  {col:30}  buy_r={buy_corr:6.3f}  sell_r={sell_corr:6.3f}  diff={diff:6.3f}"
                )
            print()

        if sell_specific:
            print("Features more correlated with SELL signals:")
            sell_specific.sort(key=lambda x: x[3], reverse=True)
            for col, buy_corr, sell_corr, diff in sell_specific[:20]:
                print(
                    f"  {col:30}  buy_r={buy_corr:6.3f}  sell_r={sell_corr:6.3f}  diff={diff:6.3f}"
                )
            print()

        # Check include_list features specifically
        include_list = getattr(self, "include_list", [])
        include_list_present = [col for col in include_list if col in numeric_cols]

        if include_list_present:
            print("--- include_list Features: Correlation with Buy/Sell ---")
            print()
            for col in sorted(include_list_present):
                buy_corr = buy_corr_dict.get(col, 0)
                sell_corr = sell_corr_dict.get(col, 0)
                buy_dir = "↑" if buy_corr > 0 else "↓"
                sell_dir = "↑" if sell_corr > 0 else "↓"
                print(
                    f"  {col:30}  buy_r={buy_corr:7.4f} {buy_dir}  sell_r={sell_corr:7.4f} {sell_dir}"
                )
            print()

        print("=" * 80)
        print()

        # Track ranks for composite scoring
        self._track_buy_sell_ranks(
            numeric_cols,
            buy_correlations,
            sell_correlations,
            buy_spearman,
            sell_spearman,
            mi_buy_df,
            mi_sell_df,
        )

    def recommend_redundancy_drops(self, dataframe: DataFrame) -> None:
        """Recommend which columns to drop based on redundancy and importance"""
        print()
        print("=" * 80)
        print("REDUNDANCY PRUNING RECOMMENDATIONS")
        print("=" * 80)
        print()

        if not self.composite_results:
            print("    No composite ranking results available for analysis")
            return

        # Calculate Spearman correlation matrix for redundancy detection (robust to non-linearities)
        corr_matrix = dataframe.corr(method="spearman").abs()

        # Dictionary for fast composite score lookup
        score_lookup = {
            item["feature"]: item["composite_score"] for item in self.composite_results
        }

        # Track which features we already processed to avoid double-dipping
        processed = set()
        recommendations = []

        # Set correlation threshold for redundancy
        CORR_THRESHOLD = 0.85

        # Sort features by importance (highest first)
        important_features = [item["feature"] for item in self.composite_results]

        for i, f1 in enumerate(important_features):
            if f1 in processed or f1 not in corr_matrix.index:
                continue

            # Check all other features (even less important ones)
            for f2 in important_features[i + 1 :]:
                if f2 in processed or f2 not in corr_matrix.columns:
                    continue

                corr_val = corr_matrix.loc[f1, f2]
                if corr_val > CORR_THRESHOLD:
                    # Found redundancy. Both are processed to avoid cascading drops of all related features
                    # though in practice we only want to drop the weaker of the pair.
                    score1 = score_lookup.get(f1, 0)
                    score2 = score_lookup.get(f2, 0)

                    # Drop the one with lower composite score
                    drop_feat = f2 if score1 >= score2 else f1
                    keep_feat = f1 if score1 >= score2 else f2

                    recommendations.append(
                        {
                            "drop": drop_feat,
                            "keep": keep_feat,
                            "corr": corr_val,
                            "drop_score": min(score1, score2),
                            "keep_score": max(score1, score2),
                            "reason": f"High correlation (|r|={corr_val:.3f})",
                        }
                    )
                    processed.add(drop_feat)

        if not recommendations:
            print(f"    No features found with Spearman correlation > {CORR_THRESHOLD}")
        else:
            print(
                f"    Found {len(recommendations)} redundant pairs. Recommended drops:"
            )
            print()
            header = f"{'DROP RECOMMENDED':<30}  {'KEEP INSTEAD':<30}  {'Corr':<8}  {'Reason'}"
            print(header)
            print("-" * 100)

            # Sort by correlation strength
            for rec in sorted(recommendations, key=lambda x: x["corr"], reverse=True):
                print(
                    f"  {rec['drop']:<30}  {rec['keep']:<30}  {rec['corr']:<8.3f}  {rec['reason']}"
                )

            print()
            print("--- Actionable Drop List (for copy-paste) ---")
            drop_only = [rec["drop"] for rec in recommendations]
            print(f"    # {len(drop_only)} suggested drops:")
            print(f"    {drop_only}")
            print()

        print("=" * 80)
        print()

    # ----------------------------

    def _track_buy_sell_ranks(
        self,
        numeric_cols: list,
        buy_correlations: list,
        sell_correlations: list,
        buy_spearman: list,
        sell_spearman: list,
        mi_buy_df: pd.DataFrame,
        mi_sell_df: pd.DataFrame,
    ) -> None:
        """Track ranks from buy/sell correlation analysis for composite scoring"""
        # Initialize feature scores if not present
        for col in numeric_cols:
            if col not in self.feature_scores:
                self.feature_scores[col] = {}

        # Track Pearson correlation ranks (by absolute value)
        buy_pearson_ranks = {
            col: rank + 1 for rank, (col, _) in enumerate(buy_correlations)
        }
        sell_pearson_ranks = {
            col: rank + 1 for rank, (col, _) in enumerate(sell_correlations)
        }

        # Track Spearman correlation ranks (by absolute value)
        buy_spearman_ranks = {
            col: rank + 1 for rank, (col, _, _) in enumerate(buy_spearman)
        }
        sell_spearman_ranks = {
            col: rank + 1 for rank, (col, _, _) in enumerate(sell_spearman)
        }

        # Track Mutual Information ranks
        buy_mi_ranks = {}
        sell_mi_ranks = {}
        if mi_buy_df is not None and not mi_buy_df.empty:
            buy_mi_ranks = {
                row.feature: rank + 1 for rank, row in enumerate(mi_buy_df.itertuples())
            }
        if mi_sell_df is not None and not mi_sell_df.empty:
            sell_mi_ranks = {
                row.feature: rank + 1
                for rank, row in enumerate(mi_sell_df.itertuples())
            }

        # Store ranks in feature_scores
        for col in numeric_cols:
            self.feature_scores[col]["buy_pearson_rank"] = buy_pearson_ranks.get(
                col, len(numeric_cols)
            )
            self.feature_scores[col]["sell_pearson_rank"] = sell_pearson_ranks.get(
                col, len(numeric_cols)
            )
            self.feature_scores[col]["buy_spearman_rank"] = buy_spearman_ranks.get(
                col, len(numeric_cols)
            )
            self.feature_scores[col]["sell_spearman_rank"] = sell_spearman_ranks.get(
                col, len(numeric_cols)
            )
            self.feature_scores[col]["buy_mi_rank"] = buy_mi_ranks.get(
                col, len(numeric_cols)
            )
            self.feature_scores[col]["sell_mi_rank"] = sell_mi_ranks.get(
                col, len(numeric_cols)
            )

    # ----------------------------

    def analyse_dataframe(
        self,
        dataframe: DataFrame,
        buys: np.ndarray,
        sells: np.ndarray,
        target_col: str = None,
    ) -> None:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE DATAFRAME ANALYSIS")
        print("=" * 80)

        # df_clean = self.clean_dataframe(dataframe)
        df_clean = self.rolling_dataframe_normalise(dataframe)

        # Basic statistics
        self.column_ranges(df_clean)

        # Buy/Sell signal correlation analysis
        self.analyse_buy_sell_correlations(df_clean, buys, sells)

        # Correlation analyses
        self.pearson_correlation(df_clean)
        self.spearman_correlation(df_clean)

        # Check include_list for redundant features
        self.check_include_list_correlations(df_clean)

        # Multicollinearity detection
        self.vif_analysis(df_clean)

        # Supervised analyses (if target column provided)
        if target_col:
            self.mutual_information_analysis(df_clean, target_col)
            self.feature_importance_analysis(df_clean, target_col)
            self.univariate_feature_selection(df_clean, target_col)

        # Dimensionality reduction analysis
        self.pca_analysis(df_clean)

        # Suggest features based on PCA
        self.suggest_features_from_pca(df_clean)

        # Normalization recommendations
        self.normalization_recommendations(df_clean)

        # Composite ranking based on all analyses
        self.composite_feature_ranking(df_clean)

        # Recommend drops based on redundancy and composite score
        self.recommend_redundancy_drops(df_clean)

        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

    # ----------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # this is the main freqtrade loop, which is fed dataframes pair, by pair

        whitelist = self.dp.current_whitelist()

        curr_pair = metadata["pair"]

        self.iteration_init()
        dataframe = self.check_precision_columns(dataframe)
        dataframe = self.dataframePopulator.add_indicators(
            dataframe,
            dataset_type=DatasetType.MINIMAL,
            # dataframe, dataset_type=DatasetType.CUSTOM3
        )
        dataframe = self.add_additional_indicators(dataframe)

        if self.combined_df is None:
            print(f"    Init with: {curr_pair}")
            self.combined_df = dataframe.reset_index(drop=True)
            buys = self.get_train_buy_signals(dataframe)
            sells = self.get_train_sell_signals(dataframe)
            self.combined_buys = buys
            self.combined_sells = sells
        else:
            print(f"    Appending: {curr_pair}")
            self.combined_df = pd.concat(
                [self.combined_df, dataframe], ignore_index=True
            )
            buys = self.get_train_buy_signals(dataframe)
            sells = self.get_train_sell_signals(dataframe)
            self.combined_buys = pd.concat(
                [self.combined_buys, buys], ignore_index=True
            )
            self.combined_sells = pd.concat(
                [self.combined_sells, sells], ignore_index=True
            )

        self.pair_count += 1

        if self.pair_count == len(whitelist):
            combined = self.combined_df.reset_index(drop=True)
            self.analyse_dataframe(combined, self.combined_buys, self.combined_sells)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
