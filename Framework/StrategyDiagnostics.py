# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
StrategyDiagnostics - debug / analysis / reporting mixin.

Pure-diagnostic methods (classification assessment, probability/distribution
dumps, dataframe range + correlation analysis) extracted verbatim from
BaseStrategy / BaseNNStrategy. Mixed into BaseStrategy so every strategy keeps
calling them as `self.method(...)`. These methods only print/compute-and-print;
they do not affect training or trading outcomes. The shared gate `debug_print`
remains on BaseStrategy and is resolved via `self` through the MRO.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)


class StrategyDiagnostics:
    """Debug/analysis/reporting helpers. Stateless except for what it reads via
    `self` (e.g. `self.debug_print`). Intended to be mixed into BaseStrategy."""

    # =========================================================================
    # Assessment / Reporting Methods
    # =========================================================================

    def get_assessment_feedback(self, score: float, metric_type: str) -> str:
        """Provides qualitative feedback for a single metric score."""
        if metric_type == "MCC":
            if score >= 0.6:
                return "Excellent. Strong positive correlation."
            if score >= 0.4:
                return "Good. Reliable positive correlation."
            if score >= 0.2:
                return "Okay. Weak but meaningful correlation."
            if score >= 0.05:
                return "Poor. Correlation is barely better than random."
            return "Bad. Correlation is near zero or negative."

        if metric_type == "Kappa":
            # Landis & Koch 1977 guidelines
            if score >= 0.81:
                return "Excellent. Near-perfect agreement."
            if score >= 0.61:
                return "Substantial. Strong agreement."
            if score >= 0.41:
                return "Moderate. Meaningful agreement."
            if score >= 0.21:
                return "Fair. Weak agreement."
            if score >= 0.0:
                return "Slight. Agreement is barely above chance."
            return "Bad. Agreement is poor or non-existent."

        # Generic score for Precision/Recall/F1
        if score >= 0.8:
            return "Excellent. Very strong performance."
        if score >= 0.6:
            return "Good. Reliable performance."
        if score >= 0.4:
            return "Okay. Acceptable, but needs improvement."
        if score >= 0.2:
            return "Poor. Significant room for improvement."
        return "Very Bad. Performance is extremely low."

    def _print_assessment_header(
        self, title: str = "CLASSIFICATION PERFORMANCE ASSESSMENT"
    ) -> None:
        """Print the header section for assessment reports."""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    def _print_metrics_table_header(self) -> None:
        """Print the header for the metrics table."""
        COL_1_WIDTH = 25
        COL_2_WIDTH = 7
        COL_3_WIDTH = 40

        print("\n--- Qualitative Assessment of Key Metrics ---")
        header = f"{'Metric':<{COL_1_WIDTH}} {'Score':<{COL_2_WIDTH}} {'Assessment':<{COL_3_WIDTH}} {'Trading Context'}"
        print(header)
        print(
            "-" * COL_1_WIDTH
            + "   "
            + "-" * COL_2_WIDTH
            + "   "
            + "-" * COL_3_WIDTH
            + "   "
            + "-" * 30
        )

    def _print_metric_row(
        self, metric_name: str, score: float, metric_type: str, context: str
    ) -> None:
        """Print a single row in the metrics table."""
        COL_1_WIDTH = 25
        COL_2_WIDTH = 7
        COL_3_WIDTH = 40

        assessment = self.get_assessment_feedback(score, metric_type)
        print(
            f"{metric_name:<{COL_1_WIDTH}} {score:<{COL_2_WIDTH}.3f} {assessment:<{COL_3_WIDTH}} {context}"
        )

    def analyze_and_assess_results(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """
        Calculates essential metrics and provides qualitative feedback with fixed-width formatting.
        Binary classification version (focuses on Buy signals).

        Args:
            y_true: The ground truth labels (1D array, binary: 0/1).
            y_pred: The predicted labels (1D array, binary: 0/1).
        """

        # Calculate all metrics
        report = classification_report(
            y_true, y_pred, digits=3, zero_division=0, output_dict=True
        )
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        # FIX: Use string key '1' for the classification report dictionary lookup
        class_1_metrics = report.get("1", {})

        # Extract metrics for the positive class (1.0)
        precision_1 = class_1_metrics.get("precision", 0.0)
        recall_1 = class_1_metrics.get("recall", 0.0)
        f1_1 = class_1_metrics.get("f1-score", 0.0)

        # ------------------ PRINTING ------------------
        self._print_assessment_header("CLASSIFICATION PERFORMANCE ASSESSMENT (Binary)")

        # 1. Classification Report
        print("--- Detailed Classification Report ---")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))

        # 2. Key Metric Analysis Table
        self._print_metrics_table_header()

        # Print analysis for the classification report metrics
        self._print_metric_row(
            "Precision (1.0)",
            precision_1,
            "F1",
            "Trading objective: Avoid False Alarms",
        )
        self._print_metric_row(
            "Recall (1.0)", recall_1, "F1", "Trading objective: Find All Opportunities"
        )
        self._print_metric_row(
            "F1-Score (1.0)", f1_1, "F1", "Balanced performance on the Buy signal"
        )

        # Print analysis for the summary metrics
        self._print_metric_row(
            "MCC", mcc, "MCC", "Correlation between prediction and reality"
        )
        self._print_metric_row(
            "Cohen's Kappa", kappa, "Kappa", "Agreement better than random chance"
        )

        # 3. Final Summary Recommendation
        if precision_1 < 0.4:
            recommendation = f"🚨 WARNING: Precision (Buy Signal) is only {precision_1:.3f}. The model predicts a 'Buy' signal but is **wrong {int((1 - precision_1) * 100)}% of the time**. This is dangerous for trading. **Priority: Improve Precision.**"
        elif recall_1 < 0.5:
            recommendation = f"⚠️ CONCERN: Recall (Buy Signal) is only {recall_1:.3f}. The model is missing over {int((1 - recall_1) * 100)}% of the available 'Buy' signals. **Priority: Improve Recall and overall F1.**"
        elif f1_1 < 0.6:
            recommendation = f"✅ ACCEPTABLE: Performance is okay, but the F1-Score of {f1_1:.3f} needs to be higher for a robust trading strategy. **Focus: Fine-tune for higher F1/MCC.**"
        else:
            recommendation = f"🌟 GOOD PERFORMANCE: The model shows strong balance (F1-Score {f1_1:.3f}) and reliable correlation (MCC {mcc:.3f})."

        print("\n--- Summary and Recommendation ---")
        print(recommendation)
        print("=" * 80)

    def analyze_and_assess_results_tristate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """
        Calculates essential metrics and provides qualitative feedback with fixed-width formatting.
        Tri-state classification version (Sell=0, Hold=1, Buy=2).

        Args:
            y_true: The ground truth labels (1D array, tri-state: 0=Sell, 1=Hold, 2=Buy).
            y_pred: The predicted labels (1D array, tri-state: 0=Sell, 1=Hold, 2=Buy).
        """

        # Calculate all metrics
        report = classification_report(
            y_true, y_pred, digits=3, zero_division=0, output_dict=True
        )
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        # Get macro and micro averages
        macro_avg = report.get("macro avg", {})
        micro_avg = report.get("weighted avg", {})

        macro_precision = macro_avg.get("precision", 0.0)
        macro_recall = macro_avg.get("recall", 0.0)
        macro_f1 = macro_avg.get("f1-score", 0.0)

        micro_precision = micro_avg.get("precision", 0.0)
        micro_recall = micro_avg.get("recall", 0.0)
        micro_f1 = micro_avg.get("f1-score", 0.0)

        # Extract metrics for each class
        class_0_metrics = report.get("0.0", report.get("0", {}))  # SELL
        class_1_metrics = report.get("1.0", report.get("1", {}))  # HOLD
        class_2_metrics = report.get("2.0", report.get("2", {}))  # BUY

        precision_sell = class_0_metrics.get("precision", 0.0)
        recall_sell = class_0_metrics.get("recall", 0.0)
        f1_sell = class_0_metrics.get("f1-score", 0.0)
        support_sell = class_0_metrics.get("support", 0)

        precision_hold = class_1_metrics.get("precision", 0.0)
        recall_hold = class_1_metrics.get("recall", 0.0)
        f1_hold = class_1_metrics.get("f1-score", 0.0)
        support_hold = class_1_metrics.get("support", 0)

        precision_buy = class_2_metrics.get("precision", 0.0)
        recall_buy = class_2_metrics.get("recall", 0.0)
        f1_buy = class_2_metrics.get("f1-score", 0.0)
        support_buy = class_2_metrics.get("support", 0)

        # ------------------ PRINTING ------------------
        self._print_assessment_header(
            "CLASSIFICATION PERFORMANCE ASSESSMENT (Tri-State)"
        )

        # 1. Classification Report
        print("--- Detailed Classification Report ---")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\n--- Confusion Matrix ---")
        print("Rows = True, Columns = Predicted")
        print(f"{'':<8} {'Sell':>8} {'Hold':>8} {'Buy':>8}")
        print(f"{'Sell':>8} {cm[0,0]:>8} {cm[0,1]:>8} {cm[0,2]:>8}")
        print(f"{'Hold':>8} {cm[1,0]:>8} {cm[1,1]:>8} {cm[1,2]:>8}")
        print(f"{'Buy':>8} {cm[2,0]:>8} {cm[2,1]:>8} {cm[2,2]:>8}")

        # 3. Per-Class Metrics Table
        self._print_metrics_table_header()

        # SELL class metrics
        self._print_metric_row(
            f"Precision (Sell)",
            precision_sell,
            "F1",
            f"Avoid false sell signals (support: {support_sell})",
        )
        self._print_metric_row(
            f"Recall (Sell)",
            recall_sell,
            "F1",
            f"Find all sell opportunities (support: {support_sell})",
        )
        self._print_metric_row(
            f"F1-Score (Sell)",
            f1_sell,
            "F1",
            f"Balanced performance on Sell signal (support: {support_sell})",
        )

        # HOLD class metrics
        self._print_metric_row(
            f"Precision (Hold)",
            precision_hold,
            "F1",
            f"Correctly identify neutral periods (support: {support_hold})",
        )
        self._print_metric_row(
            f"Recall (Hold)",
            recall_hold,
            "F1",
            f"Find all neutral periods (support: {support_hold})",
        )
        self._print_metric_row(
            f"F1-Score (Hold)",
            f1_hold,
            "F1",
            f"Balanced performance on Hold signal (support: {support_hold})",
        )

        # BUY class metrics
        self._print_metric_row(
            f"Precision (Buy)",
            precision_buy,
            "F1",
            f"Avoid false buy signals (support: {support_buy})",
        )
        self._print_metric_row(
            f"Recall (Buy)",
            recall_buy,
            "F1",
            f"Find all buy opportunities (support: {support_buy})",
        )
        self._print_metric_row(
            f"F1-Score (Buy)",
            f1_buy,
            "F1",
            f"Balanced performance on Buy signal (support: {support_buy})",
        )

        # 4. Overall Metrics
        print("\n--- Overall Performance Metrics ---")
        self._print_metrics_table_header()

        self._print_metric_row(
            "Macro Avg Precision",
            macro_precision,
            "F1",
            "Average precision across all classes",
        )
        self._print_metric_row(
            "Macro Avg Recall", macro_recall, "F1", "Average recall across all classes"
        )
        self._print_metric_row(
            "Macro Avg F1-Score", macro_f1, "F1", "Average F1-score across all classes"
        )
        self._print_metric_row(
            "Weighted Avg Precision",
            micro_precision,
            "F1",
            "Support-weighted average precision",
        )
        self._print_metric_row(
            "Weighted Avg Recall", micro_recall, "F1", "Support-weighted average recall"
        )
        self._print_metric_row(
            "Weighted Avg F1-Score", micro_f1, "F1", "Support-weighted average F1-score"
        )
        self._print_metric_row(
            "MCC", mcc, "MCC", "Correlation between prediction and reality"
        )
        self._print_metric_row(
            "Cohen's Kappa", kappa, "Kappa", "Agreement better than random chance"
        )

        # 5. Final Summary Recommendation
        worst_class = None
        worst_f1 = min(f1_sell, f1_hold, f1_buy)
        if worst_f1 == f1_sell:
            worst_class = "Sell"
            worst_precision = precision_sell
            worst_recall = recall_sell
        elif worst_f1 == f1_hold:
            worst_class = "Hold"
            worst_precision = precision_hold
            worst_recall = recall_hold
        else:
            worst_class = "Buy"
            worst_precision = precision_buy
            worst_recall = recall_buy

        if worst_precision < 0.4:
            recommendation = (
                f"🚨 WARNING: {worst_class} class precision is only {worst_precision:.3f}. "
                f"The model predicts '{worst_class}' but is **wrong {int((1 - worst_precision) * 100)}% of the time**. "
                f"This is dangerous for trading. **Priority: Improve {worst_class} Precision.**"
            )
        elif worst_recall < 0.5:
            recommendation = (
                f"⚠️ CONCERN: {worst_class} class recall is only {worst_recall:.3f}. "
                f"The model is missing over {int((1 - worst_recall) * 100)}% of available '{worst_class}' signals. "
                f"**Priority: Improve {worst_class} Recall and overall F1.**"
            )
        elif macro_f1 < 0.6:
            recommendation = (
                f"✅ ACCEPTABLE: Performance is okay, but the macro F1-Score of {macro_f1:.3f} "
                f"needs to be higher for a robust trading strategy. "
                f"**Focus: Fine-tune for higher F1/MCC, especially for {worst_class} class.**"
            )
        else:
            recommendation = (
                f"🌟 GOOD PERFORMANCE: The model shows strong balance "
                f"(Macro F1-Score {macro_f1:.3f}, Weighted F1-Score {micro_f1:.3f}) "
                f"and reliable correlation (MCC {mcc:.3f}). "
                f"All classes performing reasonably well."
            )

        print("\n--- Summary and Recommendation ---")
        print(recommendation)
        print("=" * 80)

    def print_probability_stats(
        self, task: str, name: str, probabilities: np.ndarray, threshold: float = None
    ):
        """Utility function to print probability statistics and distribution"""

        num_nans = np.sum(np.isnan(probabilities))  # Should be 0 after replacement
        if num_nans > 0:
            print(
                f"    *** WARNING: {num_nans} NaN(s) in raw predictions for {name} ***"
            )
            probabilities = np.nan_to_num(probabilities, nan=0.0)

        self.debug_print(f"    {task} - {name}:")
        self.debug_print(
            f"        min: {probabilities.min():.3f} max: {probabilities.max():.3f}, mean: {probabilities.mean():.3f}"
        )

        if threshold is not None:
            signals_above_threshold = np.sum(probabilities > threshold)
            signal_percentage = 100.0 * signals_above_threshold / len(probabilities)
            self.debug_print(
                f"        signals > {threshold:.3f}: {signals_above_threshold} ({signal_percentage:.1f}%)"
            )

        # Show probability distribution as compact arrays
        bins = np.bincount(
            (probabilities * 10).astype(int), minlength=11
        )  # 0.0 to 1.0 in 0.1 steps
        percentages = bins / len(probabilities)
        cumulative = np.cumsum(percentages)
        self.debug_print(f"        counts: {bins.tolist()}")
        self.debug_print(f"        percentages: {percentages.round(2).tolist()}")
        self.debug_print(f"        cumulative: {cumulative.round(2).tolist()}")

    def print_distribution_compact(self, name: str, distribution: np.ndarray) -> None:
        counts = np.bincount(distribution, minlength=3)
        percentages = counts / len(distribution) * 100
        percent_str = (
            f"[{percentages[0]:.1f}%, {percentages[1]:.1f}%, {percentages[2]:.1f}%]"
        )
        self.debug_print(f"      {name} distribution: {counts} {percent_str}")

    # =========================================================================
    # Dataframe analysis
    # =========================================================================

    def print_dataframe_ranges(self, title: str, dataframe: DataFrame):
        """Print statistics of the columns in the dataframe as a formatted table"""
        print(f"    {title}")
        print(
            f"{'Column':<24} {'Min':>12} {'Max':>12} {'Mean':>12} {'Median':>12} {'Std':>12}"
        )
        print("-" * 96)
        for col in dataframe.columns:
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                continue
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            col_mean = dataframe[col].mean()
            col_median = dataframe[col].median()
            col_std = dataframe[col].std()
            print(
                f"{col:<24} {col_min:>12.4f} {col_max:>12.4f} {col_mean:>12.4f} {col_median:>12.4f} {col_std:>12.4f}"
            )
        print("-" * 96)

    def examine_correlation(self, dataframe: DataFrame, col_name: str):
        """Examine the correlation between the features and the labels"""
        correlation_matrix = dataframe.corr(method="pearson")
        feature_target_correlation = correlation_matrix[col_name].sort_values(
            ascending=False
        )
        feature_target_correlation = feature_target_correlation.drop(
            col_name, errors="ignore"
        )
        print()
        print("\n--- Pearson Correlation with {col_name} ---")
        print(feature_target_correlation)
        print()

    def examine_full_correlation_matrix(self, dataframe: DataFrame):
        """Calculates and prints the full feature-to-feature Pearson correlation matrix."""
        correlation_matrix = dataframe.corr(method="pearson")

        print()
        print("--- Feature-to-Feature Correlation Matrix ---")
        print(correlation_matrix.round(2))
        print()

        abs_correlation_matrix = correlation_matrix.abs()
        lower_triangle_mask = np.triu(
            np.ones(abs_correlation_matrix.shape), k=1
        ).astype(bool)
        high_corr_pairs = abs_correlation_matrix.where(lower_triangle_mask)

        threshold = 0.6
        high_corr_pairs = high_corr_pairs.stack().sort_values(ascending=False)[
            high_corr_pairs.stack() > threshold
        ]

        if not high_corr_pairs.empty:
            print(
                f"\n--- Highly Correlated Feature Pairs (Absolute Value > {threshold}) ---"
            )
            print(high_corr_pairs)
        else:
            print(f"\nNo feature pairs found with absolute correlation > {threshold}.")
