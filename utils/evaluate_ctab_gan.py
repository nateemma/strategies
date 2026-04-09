#!/usr/bin/env python3
"""
Standalone evaluation script for CTAB-GAN+ models.

Usage:
    python evaluate_ctab_gan.py <model_path> <real_data_path> [options]

Example:
    python evaluate_ctab_gan.py ./models/CTABGANs ./data/real_data.csv --num_samples 2000
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add parent directory to path to import CTABGANPlus
sys.path.insert(0, str(Path(__file__).parent))

from df_ctab_gan import CTABGANPlus


def print_metrics(metrics: dict, verbose: bool = False):
    """Print evaluation metrics in a readable format."""
    print("\n" + "=" * 80)
    print("CTAB-GAN+ EVALUATION RESULTS")
    print("=" * 80)
    
    # Overall score
    overall = metrics.get("overall_score", {})
    print("\n📊 OVERALL QUALITY SCORES:")
    print(f"  Overall Quality:     {overall.get('overall_quality', 0.0):.4f} (0-1, higher is better)")
    print(f"  Diversity Score:     {overall.get('diversity_score', 0.0):.4f} (0-1, higher is better)")
    print(f"  Correlation Score:   {overall.get('correlation_score', 0.0):.4f} (0-1, higher is better)")
    print(f"  Statistical Score:  {overall.get('statistical_score', 0.0):.4f} (0-1, higher is better)")
    print(f"  Validity Score:     {overall.get('validity_score', 0.0):.4f} (0-1, higher is better)")
    
    # Diversity metrics
    diversity = metrics.get("diversity", {})
    print("\n🎲 DIVERSITY METRICS (Critical for avoiding overfitting):")
    print(f"  Diversity Ratio:              {diversity.get('diversity_ratio', 0.0):.4f} (target: ~1.0)")
    print(f"  Gen Pairwise Distance Mean:    {diversity.get('gen_pairwise_distance_mean', 0.0):.4f}")
    print(f"  Real Pairwise Distance Mean:   {diversity.get('real_pairwise_distance_mean', 0.0):.4f}")
    print(f"  Value Space Coverage:          {diversity.get('value_space_coverage', 0.0):.4f} (0-1, higher is better)")
    print(f"  Nearest Real Distance Mean:    {diversity.get('nearest_real_distance_mean', 0.0):.4f}")
    
    if verbose and "categorical_uniqueness" in diversity:
        print("\n  Categorical Uniqueness:")
        for col, stats in diversity["categorical_uniqueness"].items():
            print(f"    {col}: real={stats['real']}, gen={stats['generated']}, ratio={stats['ratio']:.3f}")
    
    # Correlation metrics
    correlation = metrics.get("correlation", {})
    print("\n🔗 CORRELATION PRESERVATION (Critical for feature relationships):")
    print(f"  Correlation Preservation:      {correlation.get('correlation_preservation', 0.0):.4f} (0-1, higher is better)")
    print(f"  Correlation Error:            {correlation.get('correlation_error', 0.0):.4f} (lower is better)")
    
    # Statistical metrics
    statistics = metrics.get("statistics", {})
    print("\n📈 STATISTICAL SIMILARITY:")
    print(f"  Mean Error (avg):              {statistics.get('mean_error_avg', 0.0):.4f} (lower is better)")
    print(f"  Std Error (avg):               {statistics.get('std_error_avg', 0.0):.4f} (lower is better)")
    print(f"  Categorical Error (avg):       {statistics.get('categorical_error_avg', 0.0):.4f} (lower is better)")
    
    if verbose and "continuous_statistics" in statistics:
        print("\n  Continuous Column Statistics:")
        for col, stats in statistics["continuous_statistics"].items():
            print(f"    {col}:")
            print(f"      Mean: real={stats['real_mean']:.4f}, gen={stats['gen_mean']:.4f}, error={stats['mean_error']:.4f}")
            print(f"      Std:  real={stats['real_std']:.4f}, gen={stats['gen_std']:.4f}, error={stats['std_error']:.4f}")
    
    # Validity metrics
    validity = metrics.get("validity", {})
    print("\n✅ VALIDITY CHECKS:")
    print(f"  Overall Validity Score:        {validity.get('overall_validity_score', 0.0):.4f} (0-1, higher is better)")
    
    if verbose:
        if "continuous_validity" in validity:
            print("\n  Continuous Column Validity:")
            for col, stats in validity["continuous_validity"].items():
                if stats["out_of_range_count"] > 0:
                    print(f"    {col}: {stats['out_of_range_count']} out of range ({stats['out_of_range_pct']*100:.2f}%)")
        
        if "categorical_validity" in validity:
            print("\n  Categorical Column Validity:")
            for col, stats in validity["categorical_validity"].items():
                if stats["invalid_value_count"] > 0:
                    print(f"    {col}: {stats['invalid_value_count']} invalid values: {stats['invalid_values']}")
    
    print("\n" + "=" * 80)
    
    # Interpretation
    overall_quality = overall.get("overall_quality", 0.0)
    diversity_score = overall.get("diversity_score", 0.0)
    corr_score = overall.get("correlation_score", 0.0)
    
    print("\n💡 INTERPRETATION:")
    if overall_quality >= 0.8:
        print("  ✅ Excellent: Model is generating high-quality, diverse samples with preserved correlations")
    elif overall_quality >= 0.6:
        print("  ⚠️  Good: Model is generating reasonable samples, but may need improvement")
    elif overall_quality >= 0.4:
        print("  ⚠️  Fair: Model needs improvement in diversity or correlation preservation")
    else:
        print("  ❌ Poor: Model is likely overfitting or not learning the data distribution well")
    
    if diversity_score < 0.5:
        print("  ⚠️  WARNING: Low diversity - possible mode collapse detected!")
    if corr_score < 0.5:
        print("  ⚠️  WARNING: Low correlation preservation - feature relationships may be lost!")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CTAB-GAN+ model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved CTAB-GAN+ model directory"
    )
    parser.add_argument(
        "real_data_path",
        type=str,
        help="Path to CSV file with real data for comparison"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate for evaluation (default: 1000)"
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=None,
        help="Specific class label to generate (if None, uses uniform distribution)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON output of metrics (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed metrics for each column"
    )
    
    args = parser.parse_args()
    
    # Check model path
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    generator_path = os.path.join(args.model_path, "generator.keras")
    metadata_path = os.path.join(args.model_path, "metadata.pkl")
    
    if not os.path.exists(generator_path):
        print(f"ERROR: Generator model not found: {generator_path}")
        sys.exit(1)
    
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found: {metadata_path}")
        sys.exit(1)
    
    # Check real data path
    if not os.path.exists(args.real_data_path):
        print(f"ERROR: Real data file does not exist: {args.real_data_path}")
        sys.exit(1)
    
    # Load model
    print(f"Loading CTAB-GAN+ model from: {args.model_path}")
    model = CTABGANPlus()
    try:
        model.load(args.model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Load real data
    print(f"Loading real data from: {args.real_data_path}")
    try:
        real_data = pd.read_csv(args.real_data_path)
        print(f"✅ Loaded {len(real_data)} real samples with {len(real_data.columns)} columns")
    except Exception as e:
        print(f"ERROR: Failed to load real data: {e}")
        sys.exit(1)
    
    # Ensure columns match
    model_cols = set(model.column_order)
    real_cols = set(real_data.columns)
    
    if not model_cols.issubset(real_cols):
        missing = model_cols - real_cols
        print(f"ERROR: Real data missing columns: {missing}")
        sys.exit(1)
    
    # Evaluate
    print(f"\nEvaluating model with {args.num_samples} generated samples...")
    try:
        metrics = model.evaluate(
            real_data=real_data,
            num_samples=args.num_samples,
            class_label=args.class_label,
        )
        print("✅ Evaluation completed")
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    print_metrics(metrics, verbose=args.verbose)
    
    # Save output if requested
    if args.output:
        print(f"\nSaving metrics to: {args.output}")
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        metrics_json = convert_to_native(metrics)
        with open(args.output, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print("✅ Metrics saved")


if __name__ == "__main__":
    main()

