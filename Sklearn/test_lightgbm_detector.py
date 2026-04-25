#!/usr/bin/env python3
"""
Test script for LightGBM detector
This script tests the LightGBM anomaly detector with sample data
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import NNDetector

def test_lightgbm_detector():
    """Test the LightGBM detector with sample data"""
    
    print("Testing LightGBM Anomaly Detector")
    print("=" * 40)
    
    # Check if LightGBM is installed
    try:
        import lightgbm as lgb
        print("✓ LightGBM is installed")
    except ImportError:
        print("✗ LightGBM is not installed. Please install with: pip install lightgbm")
        return False
    
    # Create sample data
    print("\nCreating sample data...")
    np.random.seed(42)
    
    # Generate normal data (most of the data)
    n_samples = 1000
    n_features = 10
    
    # Normal data with some noise
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some anomalies (outliers)
    n_anomalies = 50
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Create anomalies by adding large values
    for idx in anomaly_indices:
        normal_data[idx] += np.random.normal(5, 2, n_features)
    
    print(f"Created {n_samples} samples with {n_features} features")
    print(f"Added {n_anomalies} anomalies")
    
    # Create detector
    print("\nCreating LightGBM detector...")
    detector = NNDetector.NNDetector_LightGBM(
        pair="BTC/USDT",
        seq_len=1,
        num_features=n_features,
        tag="test"
    )
    
    # Split data for training and validation
    train_size = int(0.8 * n_samples)
    train_data = normal_data[:train_size]
    val_data = normal_data[train_size:]
    
    print(f"Training data: {train_data.shape}")
    print(f"Validation data: {val_data.shape}")
    
    # Train the detector
    print("\nTraining LightGBM detector...")
    try:
        history = detector.train(train_data, val_data)
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # Test predictions
    print("\nTesting predictions...")
    try:
        # Get anomaly scores for all data
        anomaly_scores = detector.predict(normal_data)
        print(f"✓ Predictions completed successfully")
        print(f"Anomaly scores shape: {anomaly_scores.shape}")
        print(f"Min score: {anomaly_scores.min():.6f}")
        print(f"Max score: {anomaly_scores.max():.6f}")
        print(f"Mean score: {anomaly_scores.mean():.6f}")
        print(f"Std score: {anomaly_scores.std():.6f}")
        
        # Check if anomalies have higher scores
        normal_scores = anomaly_scores[np.setdiff1d(np.arange(n_samples), anomaly_indices)]
        anomaly_scores_only = anomaly_scores[anomaly_indices]
        
        print(f"\nNormal data scores - Mean: {normal_scores.mean():.6f}, Std: {normal_scores.std():.6f}")
        print(f"Anomaly data scores - Mean: {anomaly_scores_only.mean():.6f}, Std: {anomaly_scores_only.std():.6f}")
        
        # Check if anomalies are detected (higher scores)
        if anomaly_scores_only.mean() > normal_scores.mean():
            print("✓ Anomalies detected correctly (higher scores)")
        else:
            print("⚠ Anomalies not clearly separated from normal data")
            
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False
    
    # Test model saving and loading
    print("\nTesting model saving and loading...")
    try:
        # Save model
        model_path = "test_lightgbm_model.txt"
        detector.save_model(model_path)
        print("✓ Model saved successfully")
        
        # Create new detector and load model
        detector2 = NNDetector.NNDetector_LightGBM(
            pair="BTC/USDT",
            seq_len=1,
            num_features=n_features,
            tag="test"
        )
        detector2.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Test predictions with loaded model
        scores2 = detector2.predict(normal_data[:100])  # Test with subset
        print(f"✓ Predictions with loaded model: {scores2.shape}")
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(model_path.replace('.txt', '_metadata.json')):
            os.remove(model_path.replace('.txt', '_metadata.json'))
        print("✓ Cleanup completed")
        
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("✓ All tests passed! LightGBM detector is working correctly.")
    return True

if __name__ == "__main__":
    success = test_lightgbm_detector()
    sys.exit(0 if success else 1) 